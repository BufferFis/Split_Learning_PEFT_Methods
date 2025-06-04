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
import sys

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
                response = requests.get(f"{url}/health", timeout=120)
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
        max_retries = 2  # Reduce retries since NCCL issues are immediate
        
        for attempt in range(max_retries):
            try:
                print(f"Server load attempt {attempt + 1}/{max_retries}...")
                server_response = requests.post(
                    f"{self.server_url}/load_model", 
                    json={"path": model_path}, 
                    timeout=300  # Reduce timeout since no barriers
                )
                server_data = server_response.json()
                print("Server model loaded:", server_data)
                
                # Accept success even if one rank had issues
                if server_data.get("status") in ["loaded", "error"]:
                    if server_data.get("status") == "error":
                        print(f"Warning: Server rank {server_data.get('server_rank', 'unknown')} had issues but continuing...")
                    break
                else:
                    print("Server returned unexpected status")
                    if attempt == max_retries - 1:
                        print("Proceeding anyway - server may still work for training")
                        break
                        
            except requests.exceptions.Timeout as e:
                print(f"Timeout on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    print("Server loading timed out - this may indicate NCCL issues")
                    return False
                print("Retrying in 10 seconds...")
                time.sleep(10)
            except Exception as e:
                print(f"Failed to load server model: {e}")
                if attempt == max_retries - 1:
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
                timeout=100
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
            num_workers=2,
            pin_memory=True,
            drop_last=True  # Ensures all batches are [batch_size, 128]
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
                    
                    # Expand attention mask to 4D [batch, heads, seq_len, seq_len]
                    attn_mask_expanded = attn_mask.unsqueeze(1).unsqueeze(2)  # [64, 1, 1, 128]
                    attn_mask_expanded = attn_mask_expanded.expand(-1, 12, 128, -1)  # [64, 12, 128, 128]
                    
                    # Validate ALL shapes
                    assert input_ids.shape == torch.Size([64, 128]), f"input_ids shape {input_ids.shape}"
                    assert attn_mask_expanded.shape == torch.Size([64, 12, 128, 128]), "Invalid mask shape"
                    assert labels.shape == torch.Size([64, 128]), f"labels shape {labels.shape}"


                    # Zero optimizers
                    self.head_optimizer.zero_grad()
                    self.tail_optimizer.zero_grad()
                    
                    # Head forward
                    head_out = self.head_model(
                        input_ids=input_ids,
                        attention_mask=attn_mask_expanded,
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
                    sr = robust_server_request(f"{server_url}/forward_train", payload)
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
                        timeout=300
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
                    requests.post(f"{self.server_url}/end_epoch", json={"is_final": is_final}, timeout=100)
                except requests.exceptions.RequestException as e:
                    print(f"Failed to signal end of epoch: {e}")
        
        # Save models (only rank 0)
        if self.local_rank == 0:
            try:
                requests.post(f"{self.server_url}/save_model", json={"path": "./server_model"}, timeout=300)
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

    def generate(self, input_ids, attention_mask, max_length=128):
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
                    
                    # Head forward (HeadModel handles mask expansion internally)
                    head_out = self.head_model(
                        input_ids=generated_ids,
                        attention_mask=current_attention_mask,
                        output_hidden_states=True
                    )
                    head_hidden = head_out.hidden_states[-1]
                    
                    # Server forward
                    payload = {
                        "activations": head_hidden.cpu().tolist(),
                        "attention_mask": current_attention_mask.cpu().tolist()
                    }
                    server_url = self.get_server_url()
                    resp = requests.post(f"{server_url}/forward", json=payload, timeout=120)
                    body_act = torch.tensor(resp.json()["body_activations"], device=self.device)
                    
                    # Tail forward (TailModel now handles mask expansion internally)
                    tail_out = self.tail_model(
                        inputs_embeds=body_act,
                        attention_mask=current_attention_mask
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
            
            print("Starting evaluation on E2E NLG dataset using HF metrics...")
            
            # FIXED: Proper dataset sampling
            if hasattr(test_ds, 'select'):
                # Use dataset.select() for HuggingFace datasets
                eval_samples = test_ds.select(range(min(100, len(test_ds))))
            else:
                # Fallback for other dataset types
                eval_samples = test_ds[:100]
            
            print(f"Evaluating on {len(eval_samples)} samples...", flush=True)
            sys.stdout.flush()

            for i, sample in enumerate(tqdm(eval_samples, desc="Evaluating")):
                if i%20 == 0:
                    print(f"processed {i}/100 samples...", flush=True)
                
                try:
                    # DEFENSIVE: Check if sample is a dictionary
                    if isinstance(sample, str):
                        print(f"Warning: Sample {i} is a string, skipping")
                        continue
                        
                    if not isinstance(sample, dict):
                        print(f"Warning: Sample {i} is not a dict, type: {type(sample)}")
                        continue
                    
                    # SAFE: Check if required keys exist
                    if "input_ids" not in sample or "human_reference" not in sample:
                        print(f"Warning: Sample {i} missing required keys")
                        continue
                    
                    # Convert to tensors safely
                    input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0).to(self.device)
                    
                    # Handle attention mask safely
                    if "attention_mask" in sample:
                        attention_mask = torch.tensor(sample["attention_mask"]).unsqueeze(0).to(self.device)
                    else:
                        attention_mask = torch.ones_like(input_ids, dtype=torch.float32)
                    
                    # Generate prediction
                    generated_text = self.generate(input_ids, attention_mask)
                    preds.append(generated_text)
                    refs.append([sample["human_reference"]])
                    
                except Exception as sample_error:
                    print(f"Error processing sample {i}: {sample_error}", flush=True)
                    continue
            
            if not preds:
                print("No valid predictions generated")
                return {"bleu": 0.0, "meteor": 0.0, "error": "No valid samples"}
            
            print(f"Generated {len(preds)} predictions, computing metrics...", flush=True)
            sys.stdout.flush()

            # Calculate metrics with error handling
            try:
                bleu_score = bleu_metric.compute(predictions=preds, references=refs)
                meteor_score = meteor_metric.compute(predictions=preds, references=[r[0] for r in refs])
                
                # Safe access to metric results
                bleu_value = bleu_score.get('bleu', 0.0) if isinstance(bleu_score, dict) else 0.0
                meteor_value = meteor_score.get('meteor', 0.0) if isinstance(meteor_score, dict) else 0.0
                
                print(f"E2E NLG BLEU Score: {bleu_value:.4f}", flush=True)
                print(f"E2E NLG METEOR Score: {meteor_value:.4f}", flush=True)
                sys.stdout.flush()
                
                results = {
                    "bleu": bleu_value,
                    "meteor": meteor_value
                }
                
            except Exception as eval_error:
                print(f"Evaluation metric error: {eval_error}")
                results = {
                    "bleu": 0.0,
                    "meteor": 0.0,
                    "error": str(eval_error)
                }
            
            # Save results
            with open("./server_model/evaluation_results.json", "w") as f:
                json.dump(results, f, indent=2)
            print("Evaluation results saved to ./server_model/evaluation_results.json")
            sys.stdout.flush()
            return results
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            sys.stdout.flush()
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
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation, no training")  # This exists
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

    # Create loaded trainer
    trainer = LoadedSplitModelTrainer(head_m, tail_m, tokenizer, args.server_url, local_rank, world_size)

    # Load models and optimizers
    if trainer.load_models_and_optimizers(args.model_path):
        print("All models and optimizers loaded successfully!")
        
        # Add this before the evaluation section in load.py
        if args.eval_only and local_rank == 0:
            print("Checking server connectivity...", flush=True)
            try:
                health_response = requests.get(f"{args.server_url}/health", timeout=10)
                if health_response.status_code == 200:
                    print(f"✅ Server is healthy: {health_response.json()}", flush=True)
                else:
                    print(f"⚠️ Server health check failed: {health_response.status_code}", flush=True)
            except Exception as health_error:
                print(f"❌ Server health check error: {health_error}", flush=True)
                print("Make sure ./server_launch.sh is running!", flush=True)
            sys.stdout.flush()

        # ADD THIS: Handle eval_only mode
        if args.eval_only:
            print("=== EVAL ONLY MODE STARTED ===", flush=True)
            print(f"Local rank: {local_rank}, World size: {world_size}", flush=True)
            sys.stdout.flush()
            
            # Load dataset for evaluation
            print("Loading E2E dataset...", flush=True)
            try:
                train_ds, test_ds = trainer.load_e2e_dataset()
                print(f"Dataset loaded: train={len(train_ds)}, test={len(test_ds)}", flush=True)
            except Exception as dataset_error:
                print(f"Dataset loading failed: {dataset_error}", flush=True)
                sys.stdout.flush()
                return
            
            # CRITICAL: Only rank 0 should evaluate, others should wait
            if local_rank == 0:
                print("=== RANK 0: Starting evaluation ===", flush=True)
                sys.stdout.flush()
                
                try:
                    print("Calling trainer.evaluate()...", flush=True)
                    sys.stdout.flush()
                    
                    eval_results = trainer.evaluate(test_ds)
                    
                    print(f"Evaluation returned: {eval_results}", flush=True)
                    sys.stdout.flush()
                    
                    if eval_results and isinstance(eval_results, dict):
                        bleu_score = eval_results.get('bleu', 0.0)
                        meteor_score = eval_results.get('meteor', 0.0)
                        
                        print("=" * 50, flush=True)
                        print("        EVALUATION RESULTS", flush=True)
                        print("=" * 50, flush=True)
                        print(f"BLEU Score:   {bleu_score:.4f}", flush=True)
                        print(f"METEOR Score: {meteor_score:.4f}", flush=True)
                        print("=" * 50, flush=True)
                        sys.stdout.flush()
                        
                        # Also save to file for verification
                        result_file = "./server_model/eval_only_results.json"
                        with open(result_file, "w") as f:
                            json.dump(eval_results, f, indent=2)
                        print(f"Results saved to: {result_file}", flush=True)
                        
                    else:
                        print("ERROR: Evaluation failed - no results returned", flush=True)
                        print(f"eval_results type: {type(eval_results)}", flush=True)
                        print(f"eval_results content: {eval_results}", flush=True)
                        sys.stdout.flush()
                        
                except Exception as eval_error:
                    print(f"EVALUATION ERROR: {eval_error}", flush=True)
                    import traceback
                    traceback.print_exc()
                    sys.stdout.flush()
            else:
                print(f"=== RANK {local_rank}: Waiting for rank 0 evaluation ===", flush=True)
                sys.stdout.flush()
            
            # Synchronize all ranks before finishing
            if is_distributed:
                print(f"Rank {local_rank}: Waiting at barrier...", flush=True)
                sys.stdout.flush()
                dist.barrier()
                print(f"Rank {local_rank}: Barrier completed", flush=True)
                sys.stdout.flush()
            
            print("=== EVAL ONLY MODE COMPLETED ===", flush=True)
            sys.stdout.flush()

            
        # Continue training if requested (existing logic)
        elif args.continue_training:
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
            trainer.train(train_dl, epochs=args.epochs, test_ds=test_ds)
            print("Incremental training completed!")
        else:
            print("Models loaded successfully. Use --continue_training to resume training or --eval_only for evaluation.")
    else:
        print("Failed to load models properly!")

    cleanup_ddp(is_distributed)

