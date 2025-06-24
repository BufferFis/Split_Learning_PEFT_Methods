#client.py
import os
import argparse
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from evaluate import load as load_metric
from tqdm import tqdm
from util import split_gpt2
from peft import LoraConfig, get_peft_model
import time
from torch.amp import autocast
from torch.cuda.amp import GradScaler

import traceback





# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def setup_ddp():
    """Setup distributed training"""
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

def robust_server_request(url, json_data, max_retries=3, timeout=300):
    """Make server request with retry logic"""
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=json_data, timeout=timeout)
            return response
        except requests.exceptions.Timeout as e:
            if attempt == max_retries - 1:
                print(f"Server request failed after {max_retries} attempts: {e}")
                raise e
            else:
                print(f"Request timeout (attempt {attempt + 1}/{max_retries}), retrying...")
                time.sleep(2)  # Wait before retry
        except Exception as e:
            print(f"Server request error: {e}")
            raise e



class SplitModelTrainer:
    def __init__(self, head_model, tail_model, tokenizer, server_url, local_rank, world_size):
        self.head_model = head_model
        self.tail_model = tail_model
        self.tokenizer = tokenizer
        self.server_url = server_url
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{local_rank}")
        self.scaler = GradScaler()


        
        self.head_optimizer = optim.AdamW(
            [p for p in head_model.parameters() if p.requires_grad], lr=2e-4
        )
        self.tail_optimizer = optim.AdamW(
            [p for p in tail_model.parameters() if p.requires_grad], lr=2e-4
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def load_e2e_dataset(self):
        dataset = load_dataset("e2e_nlg", trust_remote_code=True)
        
        def preprocess(example):
            text = example["meaning_representation"] + " " + example["human_reference"]
            enc = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_attention_mask=True  # Explicitly request attention mask
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
            FIXED_LENGTH = 128
            input_ids_batch = []
            attention_mask_batch = []
            labels_batch = []
            for b in batch:
                input_ids = b["input_ids"][:FIXED_LENGTH]
                attention_mask = b["attention_mask"][:FIXED_LENGTH]
                labels = b["labels"][:FIXED_LENGTH]
                # Pad if shorter
                if len(input_ids) < FIXED_LENGTH:
                    pad_length = FIXED_LENGTH - len(input_ids)
                    input_ids.extend([self.tokenizer.pad_token_id] * pad_length)
                    attention_mask.extend([0] * pad_length)
                    labels.extend([-100] * pad_length)
                input_ids_batch.append(input_ids)
                attention_mask_batch.append(attention_mask)
                labels_batch.append(labels)
            return {
                "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.float32),
                "labels": torch.tensor(labels_batch, dtype=torch.long),
                "human_reference": [b["human_reference"] for b in batch]
            }
        return DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(shuffle if sampler is None else False),
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
            drop_last=True  # Ensures all batches are [batch_size, 128]
        )

    


    def train(self, dataloader, epochs, test_ds=None):
        # Initialize server-side training (only rank 0)
        if self.local_rank == 0:
            try:
                resp = requests.post(
                    f"{self.server_url}/start_training",
                    json={"learning_rate": 2e-4},
                    timeout=150
                )
                print("Server start_training:", resp.json())
            except requests.exceptions.RequestException as e:
                print(f"Failed to initialize server training: {e}")
                return
        
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
                    
                    # FIXED: Consistent attention mask handling for head model
                    batch_size, seq_len = attn_mask.shape
                    
                    # For head model, expand attention mask properly
                    head_attn_mask = self._create_causal_mask(batch_size, seq_len, self.device)
                    
                    self.head_optimizer.zero_grad()
                    self.tail_optimizer.zero_grad()

                    # Head forward with proper mask
                    with autocast(device_type="cuda", dtype=torch.float16):
                        head_out = self.head_model(
                            input_ids=input_ids,
                            attention_mask=head_attn_mask,
                            output_hidden_states=True,
                            use_cache=False
                        )
                        head_hid = head_out.hidden_states[-1]

                        # Validate head output
                        if torch.isnan(head_hid).any() or torch.isinf(head_hid).any():
                            print(f"Rank {self.local_rank}: ERROR: head_hid contains NaN or Inf values")
                            continue

                        payload = {
                            "activations": head_hid.detach().cpu().half().tolist(),
                            "attention_mask": attn_mask.cpu().tolist(),
                            "rank_id": self.local_rank
                        }
                        
                        server_url = self.get_server_url()
                        sr = robust_server_request(f"{server_url}/forward_train", payload)
                        body_act = torch.tensor(sr.json()["body_activations"], device=self.device, dtype=head_hid.dtype)
                        body_act.requires_grad_()

                        # Tail forward with 2D attention mask (tail model will handle expansion)
                        tail_out = self.tail_model(inputs_embeds=body_act, attention_mask=attn_mask, use_cache=False)
                        logits = tail_out.logits

                        # Compute loss
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        loss = self.loss_fn(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )

                    # Backward pass
                    self.scaler.scale(loss).backward(retain_graph=True)
                    
                    # Get gradient for server
                    if body_act.grad is not None:
                        grad_output = body_act.grad.detach().cpu().tolist()
                    else:
                        print(f"Warning: No gradient for body_act at batch {batch_idx}")
                        continue

                    # Send gradient to server
                    server_url = self.get_server_url()
                    br = requests.post(
                        f"{server_url}/backward",
                        json={
                            "grad_output": grad_output,
                            "loss": loss.item(),
                            "rank_id": self.local_rank
                        },
                        timeout=300
                    )
                    grad_input = torch.tensor(br.json()["grad_input"], device=self.device, dtype=head_hid.dtype)

                    # Backward through head
                    head_hid.backward(grad_input)

                    # Optimizer steps with gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.head_model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.tail_model.parameters(), max_norm=1.0)
                    
                    self.scaler.step(self.head_optimizer)
                    self.scaler.step(self.tail_optimizer)
                    self.scaler.update()

                    total_loss += loss.item()

                    # Enhanced memory cleanup
                    del input_ids, attn_mask, labels, head_attn_mask
                    del head_out, head_hid, body_act, tail_out, logits
                    del shift_logits, shift_labels, loss, grad_input
                    if batch_idx % 5 == 0:  # More frequent cleanup
                        torch.cuda.empty_cache()

                except requests.exceptions.RequestException as e:
                    print(f"Rank {self.local_rank}: Server communication error: {e}")
                    continue
                except Exception as e:
                    print(f"Rank {self.local_rank}: Training error: {e}")
                    traceback.print_exc()
                    continue

            avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
            if self.local_rank == 0:
                print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")
            
            # End epoch signal
            if self.local_rank == 0:
                is_final = (epoch == epochs-1)
                try:
                    requests.post(f"{self.server_url}/end_epoch", json={"is_final": is_final}, timeout=10)
                except requests.exceptions.RequestException as e:
                    print(f"Failed to signal end of epoch: {e}")
        
        # Save and evaluate
        if self.local_rank == 0:
            try:
                requests.post(f"{self.server_url}/save_model", json={"path": "./server_model"}, timeout=30)
                self.save_models("./server_model")
                print("Models saved successfully")
                
                if test_ds is not None:
                    print("Starting evaluation...")
                    eval_results = self.evaluate(test_ds)
                    if eval_results:
                        print(f"Final BLEU: {eval_results['bleu']:.4f}, METEOR: {eval_results['meteor']:.4f}")
                        
            except Exception as e:
                print(f"Failed to save/evaluate: {e}")

    def _create_causal_mask(self, batch_size, seq_len, device):
        """Create causal attention mask for head model"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        mask = mask.expand(batch_size, 12, seq_len, seq_len)  # [batch, heads, seq_len, seq_len]
        return mask.float()


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

    
    def generate(self, input_ids, attention_mask, max_length=128):  # Match preprocessing
        """Generate text for evaluation using the split model"""
        with torch.no_grad():
            try:
                self.head_model.eval()
                self.tail_model.eval()
                
                # Ensure consistent dimensions
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                if attention_mask.dim() == 1:
                    attention_mask = attention_mask.unsqueeze(0)
                    
                # Ensure proper types
                input_ids = input_ids.long()
                attention_mask = attention_mask.float()
                
                generated_ids = input_ids.clone()
                
                for step in range(min(max_length - input_ids.size(1), 32)):
                    # Create attention mask that EXACTLY matches sequence length
                    current_attention_mask = torch.ones(
                        generated_ids.size(0), 
                        generated_ids.size(1), 
                        dtype=torch.float32, 
                        device=self.device
                    )
                    
                    # Head forward with exact dimensions
                    head_out = self.head_model(
                        input_ids=generated_ids,
                        attention_mask=current_attention_mask,
                        output_hidden_states=True,
                        use_cache = False
                    )
                    head_hidden = head_out.hidden_states[-1]
                    
                    # Server forward with dimension validation
                    payload = {
                        "activations": head_hidden.cpu().tolist(),
                        "attention_mask": current_attention_mask.cpu().tolist()
                    }
                    
                    server_url = self.get_server_url()
                    resp = requests.post(f"{server_url}/forward", json=payload, timeout=120)
                    body_act = torch.tensor(resp.json()["body_activations"], device=self.device, dtype=head_hidden.dtype)
                    
                    # Tail forward
                    tail_out = self.tail_model(
                        inputs_embeds=body_act,
                        attention_mask=current_attention_mask,
                        use_cache = False
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
            """Robust evaluation with better error handling"""
            if self.local_rank != 0:
                return
                
            self.head_model.eval()
            self.tail_model.eval()
            
            try:
                from evaluate import load as load_metric
                bleu_metric = load_metric("bleu")
                meteor_metric = load_metric("meteor")
                
                preds, refs = [], []
                
                print("Starting evaluation on E2E NLG dataset...")
                
                # Safer dataset sampling
                eval_size = min(50, len(test_ds))  # Reduced for stability
                if hasattr(test_ds, 'select'):
                    eval_samples = test_ds.select(range(eval_size))
                else:
                    eval_samples = test_ds[:eval_size]
                
                successful_evals = 0
                for i, sample in enumerate(tqdm(eval_samples, desc="Evaluating")):
                    try:
                        if not isinstance(sample, dict) or "input_ids" not in sample:
                            continue
                        
                        input_ids = torch.tensor(sample["input_ids"][:64]).unsqueeze(0).to(self.device)  # Limit length
                        attention_mask = torch.ones_like(input_ids, dtype=torch.float32)
                        
                        generated_text = self.generate(input_ids, attention_mask, max_length=64)
                        
                        if generated_text and generated_text != "Generation failed":
                            preds.append(generated_text)
                            refs.append([sample["human_reference"]])
                            successful_evals += 1
                            
                    except Exception as e:
                        print(f"Error in sample {i}: {e}")
                        continue
                
                print(f"Successfully evaluated {successful_evals}/{eval_size} samples")
                
                if preds:
                    try:
                        bleu_score = bleu_metric.compute(predictions=preds, references=refs)
                        meteor_score = meteor_metric.compute(predictions=preds, references=[r[0] for r in refs])
                        
                        results = {
                            "bleu": bleu_score.get('bleu', 0.0),
                            "meteor": meteor_score.get('meteor', 0.0),
                            "successful_samples": successful_evals
                        }
                        
                        # Save results
                        os.makedirs("./server_model", exist_ok=True)
                        with open("./server_model/evaluation_results.json", "w") as f:
                            json.dump(results, f, indent=2)
                        
                        return results
                    except Exception as e:
                        print(f"Metric computation error: {e}")
                        return {"bleu": 0.0, "meteor": 0.0, "error": str(e)}
                else:
                    return {"bleu": 0.0, "meteor": 0.0, "error": "No valid predictions"}
                    
            except Exception as e:
                print(f"Evaluation error: {e}")
                return None




    def load_models_and_optimizers(self, model_path):
        """Load both models and optimizers for incremental training"""
        
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
                return True
            except Exception as e:
                print(f"Warning: Failed to load optimizer states: {e}")
                return False
        else:
            print("Warning: Optimizer states not found")
            return False
    def get_server_url(self):
        """Load balance between server processes"""
        # Only load balance if we have multiple server processes AND multiple client processes
        if self.world_size > 1:
            # Check if server is actually distributed by trying both ports
            server_port = 8000 + (self.local_rank % 2)
            return f"http://127.0.0.1:{server_port}"
        return self.server_url




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)  # Match shell scripts
    parser.add_argument("--epochs", type=int, default=1)  # Match shell scripts
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:8000")
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
    
    device = torch.device(f"cuda:{local_rank}")
    
    # Load models
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
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
    
    trainer = SplitModelTrainer(head_m, tail_m, tokenizer, args.server_url, local_rank, world_size)
    
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
    
    if not args.eval_only:
        trainer.train(train_dl, epochs=args.epochs, test_ds=test_ds)  # Pass test_ds
    
    cleanup_ddp(is_distributed)

if __name__ == "__main__":
    main()
