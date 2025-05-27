import os
import torch
import requests
import argparse
import time
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from util import split_gpt2
from tqdm import tqdm
from evaluate import load as load_metric
import json

def setup_ddp():
    """Setup distributed training if available"""
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return local_rank, world_size, True
    else:
        return 0, 1, False

def cleanup_ddp(is_distributed):
    if is_distributed:
        dist.destroy_process_group()

def wait_for_server(server_url, max_retries=30, delay=2):
    """Wait for server to be ready"""
    # Check both server ports
    server_ports = [8000, 8001]
    
    for port in server_ports:
        url = f"http://127.0.0.1:{port}"
        for i in range(max_retries):
            try:
                response = requests.get(f"{url}/health", timeout=5)
                if response.status_code == 200:
                    print(f"Server is ready at {url}")
                    break
            except requests.exceptions.RequestException:
                if i == max_retries - 1:
                    print(f"Server at {url} not ready after {max_retries} attempts")
                continue
        else:
            continue
        break
    else:
        print("No servers ready")
        return False
    return True


class LoadedSplitModelTrainer:
    """Enhanced trainer that supports loading and incremental training"""
    
    def __init__(self, head_model, tail_model, tokenizer, server_url, local_rank, world_size):
        self.head_model = head_model
        self.tail_model = tail_model
        self.tokenizer = tokenizer
        self.server_url = server_url
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{local_rank}")
        
        # Initialize optimizers (will be replaced if loading from checkpoint)
        self.head_optimizer = optim.AdamW(
            [p for p in head_model.parameters() if p.requires_grad], lr=2e-4
        )
        self.tail_optimizer = optim.AdamW(
            [p for p in tail_model.parameters() if p.requires_grad], lr=2e-4
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def load_models_and_optimizers(self, model_path):
        """Load both models and optimizers for true incremental training"""
        
        # Load server model first
        print("Loading server model...")
        try:
            server_response = requests.post(
                f"{self.server_url}/load_model", 
                json={"path": model_path}, 
                timeout=30
            )
            server_data = server_response.json()
            print("Server model loaded:", server_data)
            
            if server_data.get("status") != "loaded":
                print("Failed to load server model")
                return False
                
        except Exception as e:
            print(f"Failed to load server model: {e}")
            return False
        
        # Load client model weights
        head_path = os.path.join(model_path, "head_model.pt")
        tail_path = os.path.join(model_path, "tail_model.pt")
        
        if os.path.exists(head_path) and os.path.exists(tail_path):
            print("Loading client model weights...")
            
            # Get actual models (unwrap DDP if needed)
            head_model_to_load = self.head_model.module if hasattr(self.head_model, "module") else self.head_model
            tail_model_to_load = self.tail_model.module if hasattr(self.tail_model, "module") else self.tail_model
            
            head_model_to_load.load_state_dict(torch.load(head_path, map_location=self.device))
            tail_model_to_load.load_state_dict(torch.load(tail_path, map_location=self.device))
            print("Client model weights loaded successfully")
        else:
            print(f"Warning: Client model weights not found at {model_path}")
            return False
        
        # Load optimizer states
        head_opt_path = os.path.join(model_path, "head_optimizer.pt")
        tail_opt_path = os.path.join(model_path, "tail_optimizer.pt")
        
        if os.path.exists(head_opt_path) and os.path.exists(tail_opt_path):
            print("Loading optimizer states...")
            try:
                self.head_optimizer.load_state_dict(torch.load(head_opt_path, map_location=self.device))
                self.tail_optimizer.load_state_dict(torch.load(tail_opt_path, map_location=self.device))
                print("Optimizer states loaded successfully")
            except Exception as e:
                print(f"Warning: Failed to load optimizer states: {e}")
                return False
        else:
            print("Warning: Optimizer states not found")
            return False
        
        # Initialize server training state
        try:
            init_response = requests.post(
                f"{self.server_url}/start_training",
                json={"learning_rate": 2e-4},
                timeout=10
            )
            print("Server training initialized:", init_response.json())
        except Exception as e:
            print(f"Warning: Failed to initialize server training: {e}")
            return False
        
        return True

    def load_e2e_dataset(self):
        dataset = load_dataset("e2e_nlg", trust_remote_code=True)
        
        def preprocess(example):
            text = example["meaning_representation"] + " " + example["human_reference"]
            enc = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=128
            )
            return {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": enc["input_ids"],
                "human_reference": example["human_reference"]
            }
        
        train = dataset["train"].map(preprocess, remove_columns=dataset["train"].column_names)
        test = dataset["test"].map(preprocess, remove_columns=dataset["test"].column_names)
        return train, test

    def create_dataloader(self, ds, batch_size, shuffle=True, sampler=None):
        def collate_fn(batch):
            return {
                "input_ids": torch.tensor([b["input_ids"] for b in batch], dtype=torch.long),
                "attention_mask": torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long),
                "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.long),
                "human_reference": [b["human_reference"] for b in batch]
            }
        
        return DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(shuffle if sampler is None else False),
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True
        )

    def train(self, dataloader, epochs, test_ds=None):
        """Training loop identical to client.py but using loaded state"""
        
        # Synchronize all processes
        if self.world_size > 1:
            dist.barrier()
        
        for epoch in range(epochs):
            self.head_model.train()
            self.tail_model.train()
            total_loss = 0.0
            
            # Set epoch for distributed sampler
            if hasattr(dataloader.sampler, 'set_epoch'):
                dataloader.sampler.set_epoch(epoch)
            
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", disable=self.local_rank != 0)):
                try:
                    input_ids = batch["input_ids"].to(self.device)
                    attn_mask = batch["attention_mask"].to(self.device).float()
                    labels = batch["labels"].to(self.device)
                    
                    # Zero optimizers
                    self.head_optimizer.zero_grad()
                    self.tail_optimizer.zero_grad()
                    
                    # Head forward
                    head_out = self.head_model(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        output_hidden_states=True
                    )
                    head_hid = head_out.hidden_states[-1]
                    
                    # Send to server with rank identifier
                    payload = {
                        "activations": head_hid.detach().cpu().tolist(),
                        "attention_mask": attn_mask.cpu().tolist(),
                        "rank_id": self.local_rank
                    }
                    
                    server_url = self.get_server_url()
                    sr = requests.post(f"{server_url}/forward_train", json=payload, timeout=30)
                    body_act = torch.tensor(sr.json()["body_activations"], device=self.device)
                    body_act.requires_grad_()
                    
                    # Tail forward and loss
                    tail_out = self.tail_model(inputs_embeds=body_act, attention_mask=attn_mask)
                    logits = tail_out.logits
                    
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = self.loss_fn(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    
                    # Backward through tail
                    loss.backward(retain_graph=True)
                    grad_output = body_act.grad.detach().cpu().tolist()
                    
                    # Send gradient to server
                    server_url = self.get_server_url()  # Use load balancing
                    br = requests.post(
                        f"{server_url}/backward",
                        json={
                            "grad_output": grad_output,
                            "loss": loss.item(),
                            "rank_id": self.local_rank
                        },
                        timeout=30
                    )

                    grad_input = torch.tensor(br.json()["grad_input"], device=self.device)
                    
                    # Backward through head
                    head_hid.backward(grad_input)
                    
                    # Optimizer steps
                    self.head_optimizer.step()
                    self.tail_optimizer.step()
                    
                    total_loss += loss.item()
                    
                except requests.exceptions.RequestException as e:
                    print(f"Rank {self.local_rank}: Server communication error: {e}")
                    continue
                except Exception as e:
                    print(f"Rank {self.local_rank}: Training error: {e}")
                    continue
            
            avg = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
            if self.local_rank == 0:
                print(f"Epoch {epoch+1} avg loss: {avg:.4f}")
            
            # End epoch (only rank 0)
            if self.local_rank == 0:
                is_final = (epoch == epochs-1)
                try:
                    requests.post(f"{self.server_url}/end_epoch", json={"is_final": is_final}, timeout=10)
                except requests.exceptions.RequestException as e:
                    print(f"Failed to signal end of epoch: {e}")
        
        # Save models (only rank 0)
        if self.local_rank == 0:
            try:
                requests.post(f"{self.server_url}/save_model", json={"path": "./server_model"}, timeout=30)
                client_save_info = self.save_models("./server_model")
                print("Client models saved:", client_save_info)
                
                # EVALUATE USING BUILT-IN HF METRICS
                print("Starting evaluation with Hugging Face metrics...")
                eval_results = self.evaluate(test_ds)
                if eval_results:
                    print(f"Final BLEU: {eval_results['bleu']:.4f}, METEOR: {eval_results['meteor']:.4f}")
                
            except requests.exceptions.RequestException as e:
                print(f"Failed to save models: {e}")

    def save_models(self, path="./server_model"):
        """Save head and tail models"""
        os.makedirs(path, exist_ok=True)
        
        # Get the actual models (unwrap DDP if needed)
        head_model_to_save = self.head_model.module if hasattr(self.head_model, "module") else self.head_model
        tail_model_to_save = self.tail_model.module if hasattr(self.tail_model, "module") else self.tail_model
        
        torch.save(head_model_to_save.state_dict(), os.path.join(path, "head_model.pt"))
        torch.save(tail_model_to_save.state_dict(), os.path.join(path, "tail_model.pt"))
        torch.save(self.head_optimizer.state_dict(), os.path.join(path, "head_optimizer.pt"))
        torch.save(self.tail_optimizer.state_dict(), os.path.join(path, "tail_optimizer.pt"))
        
        return {"head_path": os.path.join(path, "head_model.pt"), 
                "tail_path": os.path.join(path, "tail_model.pt")}

    def generate(self, input_ids, attention_mask, max_length=64):
        """Generate text for evaluation using the split model"""
        with torch.no_grad():
            try:
                generated_ids = input_ids.clone()
                
                for step in range(min(max_length - input_ids.size(1), 32)):
                    # Head forward
                    head_out = self.head_model(
                        input_ids=generated_ids,
                        attention_mask=torch.ones_like(generated_ids).float(),
                        output_hidden_states=True
                    )
                    head_hidden = head_out.hidden_states[-1]
                    
                    # Server forward with load balancing
                    payload = {
                        "activations": head_hidden.cpu().tolist(),
                        "attention_mask": torch.ones_like(generated_ids).float().cpu().tolist()
                    }
                    server_url = self.get_server_url()  # Use load balancing
                    resp = requests.post(f"{server_url}/forward", json=payload, timeout=10)
                    body_act = torch.tensor(resp.json()["body_activations"], device=self.device)
                    
                    # Tail forward
                    tail_out = self.tail_model(
                        inputs_embeds=body_act,
                        attention_mask=torch.ones_like(generated_ids).float()
                    )
                    logits = tail_out.logits
                    
                    # Get next token
                    next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)
                    
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                
                return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            except Exception as e:
                print(f"Generation error: {e}")
                return "Generation failed"


    def evaluate(self, test_ds):
        """Evaluate model using built-in Hugging Face metrics"""
        if self.local_rank != 0:  # Only rank 0 evaluates
            return
            
        self.head_model.eval()
        self.tail_model.eval()
        
        try:
            # Load built-in Hugging Face metrics
            bleu_metric = load_metric("bleu")
            meteor_metric = load_metric("meteor")
            
            preds, refs = [], []
            eval_samples = test_ds[:100]  # Limit for speed
            
            print("Starting evaluation on E2E NLG dataset using HF metrics...")
            for sample in tqdm(eval_samples, desc="Evaluating"):
                input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0).to(self.device)
                attention_mask = torch.tensor(sample["attention_mask"]).unsqueeze(0).to(self.device)
                
                # Generate prediction
                generated_text = self.generate(input_ids, attention_mask)
                preds.append(generated_text)
                refs.append([sample["human_reference"]])  # BLEU expects list of references
            
            # Calculate metrics using built-in HF methods
            try:
                # Calculate metrics using built-in HF methods
                bleu_score = bleu_metric.compute(predictions=preds, references=refs)
                
                # Fix METEOR format - it expects flat lists, not nested
                meteor_score = meteor_metric.compute(predictions=preds, references=[r[0] for r in refs])
                
                # Safe access to metric results
                bleu_value = bleu_score.get('bleu', 0.0) if isinstance(bleu_score, dict) else 0.0
                meteor_value = meteor_score.get('meteor', 0.0) if isinstance(meteor_score, dict) else 0.0
                
                print(f"E2E NLG BLEU Score: {bleu_value:.4f}")
                print(f"E2E NLG METEOR Score: {meteor_value:.4f}")
                
                # Save evaluation results
                results = {
                    "bleu": bleu_value,
                    "meteor": meteor_value
                }
                
            except Exception as eval_error:
                print(f"Evaluation metric error: {eval_error}")
                # Fallback to basic evaluation
                results = {
                    "bleu": 0.0,
                    "meteor": 0.0,
                    "error": str(eval_error)
                }
            
            with open("./server_model/evaluation_results.json", "w") as f:
                json.dump(results, f, indent=2)
            
            print("Evaluation results saved to ./server_model/evaluation_results.json")
            return results
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            return None
    
    def get_server_url(self):
        """Load balance between server processes"""
        if self.world_size > 1:
            # Alternate between servers based on client rank
            server_port = 8000 + (self.local_rank % 2)
            return f"http://127.0.0.1:{server_port}"
        return self.server_url



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./server_model")
    parser.add_argument("--server_url", type=str, default="http://localhost:8000")
    parser.add_argument("--continue_training", action="store_true", help="Continue training after loading")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")  # Match shell scripts
    args = parser.parse_args()
    
    
    # Setup distributed training
    local_rank, world_size, is_distributed = setup_ddp()
    
    # Wait for server (only rank 0)
    if local_rank == 0:
        if not wait_for_server(args.server_url):
            print("Server is not available. Please start the server first.")
            return
    
    # Synchronize all processes
    if is_distributed:
        dist.barrier()
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Model path {args.model_path} does not exist!")
        return
    
    device = torch.device(f"cuda:{local_rank}")
    
    # Initialize models with same configuration as training
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    full_model = AutoModelForCausalLM.from_pretrained("gpt2")
    head_m, _, tail_m = split_gpt2(full_model, head_layers=2, tail_layers=2)
    
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        bias="none", use_dora=True, task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj"]
    )
    
    head_m = get_peft_model(head_m, lora_cfg).to(device)
    tail_m = get_peft_model(tail_m, lora_cfg).to(device)
    
    if is_distributed:
        head_m = DDP(head_m, device_ids=[local_rank])
        tail_m = DDP(tail_m, device_ids=[local_rank])
    
    # Create loaded trainer
    trainer = LoadedSplitModelTrainer(head_m, tail_m, tokenizer, args.server_url, local_rank, world_size)
    
    # Load models and optimizers
    if trainer.load_models_and_optimizers(args.model_path):
        print("All models and optimizers loaded successfully!")
        
        # Continue training if requested
        if args.continue_training:
            print(f"Continuing training for {args.epochs} epochs...")
            
            # Load dataset and create dataloader
            train_ds, test_ds = trainer.load_e2e_dataset()
            
            if is_distributed:
                train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank)
                train_dl = trainer.create_dataloader(
                    train_ds, batch_size=args.batch_size, shuffle=False, sampler=train_sampler
                )
            else:
                train_dl = trainer.create_dataloader(
                    train_ds, batch_size=args.batch_size, shuffle=True
                )
            
            # Continue training
            trainer.train(train_dl, epochs=args.epochs, test_ds = test_ds)
            print("Incremental training completed!")
        else:
            print("Models loaded successfully. Use --continue_training to resume training.")
    else:
        print("Failed to load models properly!")
    
    cleanup_ddp(is_distributed)

if __name__ == "__main__":
    main()
