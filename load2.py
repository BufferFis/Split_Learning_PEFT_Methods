#load.py
print("=== SCRIPT STARTING ===", flush=True)
try:
    import os
    print("os import OK", flush=True)
    import torch
    print("torch import OK", flush=True)
    import transformers
    print("transformers import OK", flush=True)
    from util import split_gpt2
    print("util import OK", flush=True)
    print("All imports successful", flush=True)
except Exception as e:
    print(f"IMPORT ERROR: {e}", flush=True)
    exit(1)


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
from torch.amp import autocast, GradScaler





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
                    print(f"Server is ready at {url}", flush=True)
                    break
            except requests.exceptions.RequestException:
                if i == max_retries - 1:
                    print(f"Server at {url} not ready after {max_retries} attempts", flush=True)
                continue
        else:
            continue
        break
    else:
        print("No servers ready", flush=True)
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
                print(f"Server request failed after {max_retries} attempts: {e}", flush=True)
                raise e
            else:
                print(f"Request timeout (attempt {attempt + 1}/{max_retries}), retrying...", flush=True)
                time.sleep(2)  # Wait before retry
        except Exception as e:
            print(f"Server request error: {e}", flush=True)
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
        print(f"[RANK {self.local_rank}] Starting load_models_and_optimizers from {model_path}", flush=True)
        
        # Load server model first
        max_retries = 2  # Reduce retries since NCCL issues are immediate
        
        for attempt in range(max_retries):
            try:
                print(f"Server load attempt {attempt + 1}/{max_retries}...", flush=True)
                server_response = requests.post(
                    f"{self.server_url}/load_model", 
                    json={"path": model_path}, 
                    timeout=300  # Reduce timeout since no barriers
                )
                server_data = server_response.json()
                print("Server model loaded:", server_data, flush=True)
                
                # Accept success even if one rank had issues
                if server_data.get("status") in ["loaded", "error"]:
                    if server_data.get("status") == "error":
                        print(f"Warning: Server rank {server_data.get('server_rank', 'unknown')} had issues but continuing...", flush=True)
                    break
                else:
                    print("Server returned unexpected status", flush=True)
                    if attempt == max_retries - 1:
                        print("Proceeding anyway - server may still work for training", flush=True)
                        break
                        
            except requests.exceptions.Timeout as e:
                print(f"Timeout on attempt {attempt + 1}: {e}", flush=True)
                if attempt == max_retries - 1:
                    print("Server loading timed out - this may indicate NCCL issues")
                    return False
                print("Retrying in 10 seconds...", flush=True)
                time.sleep(10)
            except Exception as e:
                print(f"Failed to load server model: {e}", flush=True)
                if attempt == max_retries - 1:
                    return False
        
        # Load client model weights
        head_path = os.path.join(model_path, "head_model.pt")
        tail_path = os.path.join(model_path, "tail_model.pt")
        print(f"[RANK {self.local_rank}] Checking for client models: {head_path}, {tail_path}", flush=True)
        
        if os.path.exists(head_path) and os.path.exists(tail_path):
            print("Loading client model weights...", flush=True)
            print(f"[RANK {self.local_rank}] Client model files found, loading...", flush=True)
            
            # Get actual models (unwrap DDP if needed)
            head_model_to_load = self.head_model.module if hasattr(self.head_model, "module") else self.head_model
            tail_model_to_load = self.tail_model.module if hasattr(self.tail_model, "module") else self.tail_model
            
            head_model_to_load.load_state_dict(torch.load(head_path, map_location=self.device))
            tail_model_to_load.load_state_dict(torch.load(tail_path, map_location=self.device))
            print("Client model weights loaded successfully", flush=True)
        else:
            print(f"[RANK {self.local_rank}] ERROR: Client model weights not found!", flush=True)
            print(f"[RANK {self.local_rank}] head_path exists: {os.path.exists(head_path)}", flush=True)
            print(f"[RANK {self.local_rank}] tail_path exists: {os.path.exists(tail_path)}", flush=True)
            return False
        
        # Load optimizer states
        head_opt_path = os.path.join(model_path, "head_optimizer.pt")
        tail_opt_path = os.path.join(model_path, "tail_optimizer.pt")
        print(f"[RANK {self.local_rank}] Checking for optimizer files: {head_opt_path}, {tail_opt_path}", flush=True)
        
        if os.path.exists(head_opt_path) and os.path.exists(tail_opt_path):
            print(f"[RANK {self.local_rank}] Optimizer files found, loading...", flush=True)
            try:
                self.head_optimizer.load_state_dict(torch.load(head_opt_path, map_location=self.device))
                self.tail_optimizer.load_state_dict(torch.load(tail_opt_path, map_location=self.device))
                print("Optimizer states loaded successfully", flush=True)
            except Exception as e:
                print(f"Warning: Failed to load optimizer states: {e}", flush=True)
                return False
        else:
            print(f"[RANK {self.local_rank}] ERROR: Optimizer files not found!", flush=True)
            return False
        
        # Initialize server training state
        try:
            init_response = requests.post(
                f"{self.server_url}/start_training",
                json={"learning_rate": 2e-4},
                timeout=100
            )
            print("Server training initialized:", init_response.json(), flush=True)
        except Exception as e:
            print(f"Warning: Failed to initialize server training: {e}", flush=True)
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
                    
                    batch_size, seq_len = input_ids.shape
                    assert seq_len == 128, f"input_ids shape {input_ids.shape}"
                    assert attn_mask.shape == (batch_size, seq_len), f"attn_mask shape {attn_mask.shape}"
                    assert labels.shape == (batch_size, seq_len), f"labels shape {labels.shape}"

                    # Expand attention mask to 4D [batch, num_heads, seq_len, seq_len]
                    attn_mask_expanded = attn_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
                    attn_mask_expanded = attn_mask_expanded.expand(-1, 12, seq_len, seq_len)  # [batch, 12, 128, 128]
                    assert attn_mask_expanded.shape == (batch_size, 12, seq_len, seq_len), f"attn_mask_expanded shape {attn_mask_expanded.shape}"



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
                    print(f"Rank {self.local_rank}: Server communication error: {e}", flush=True)
                    continue
                except Exception as e:
                    print(f"Rank {self.local_rank}: Training error: {e}", flush=True)
                    continue
            
            avg = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
            if self.local_rank == 0:
                print(f"Epoch {epoch+1} avg loss: {avg:.4f}", flush=True)
            
            # End epoch (only rank 0)
            if self.local_rank == 0:
                is_final = (epoch == epochs-1)
                try:
                    requests.post(f"{self.server_url}/end_epoch", json={"is_final": is_final}, timeout=100)
                except requests.exceptions.RequestException as e:
                    print(f"Failed to signal end of epoch: {e}", flush=True)
        
        # Save models (only rank 0)
        if self.local_rank == 0:
            try:
                requests.post(f"{self.server_url}/save_model", json={"path": "./server_model"}, timeout=300)
                client_save_info = self.save_models("./server_model")
                print("Client models saved:", client_save_info, flush=True)
                
                # EVALUATE USING BUILT-IN HF METRICS
                print("Starting evaluation with Hugging Face metrics...", flush=True)
                eval_results = self.evaluate(test_ds)
                if eval_results:
                    print(f"Final BLEU: {eval_results['bleu']:.4f}, METEOR: {eval_results['meteor']:.4f}", flush=True)
                
            except requests.exceptions.RequestException as e:
                print(f"Failed to save models: {e}", flush=True)

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
                print(f"Generation error: {e}", flush=True)
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
            
            print("Starting evaluation on E2E NLG dataset using HF metrics...", flush=True)
            
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
                        print(f"Warning: Sample {i} is a string, skipping", flush=True)
                        continue
                        
                    if not isinstance(sample, dict):
                        print(f"Warning: Sample {i} is not a dict, type: {type(sample)}", flush=True)
                        continue
                    
                    # SAFE: Check if required keys exist
                    if "input_ids" not in sample or "human_reference" not in sample:
                        print(f"Warning: Sample {i} missing required keys", flush=True)
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
                print("No valid predictions generated", flush=True)
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
                print(f"Evaluation metric error: {eval_error}", flush=True)
                results = {
                    "bleu": 0.0,
                    "meteor": 0.0,
                    "error": str(eval_error)
                }
            
            # Save results
            with open("./server_model/evaluation_results.json", "w") as f:
                json.dump(results, f, indent=2)
            print("Evaluation results saved to ./server_model/evaluation_results.json", flush=True)
            sys.stdout.flush()
            return results
            
        except Exception as e:
            print(f"Evaluation error: {e}", flush=True)
            sys.stdout.flush()
            return None

    
    def get_server_url(self):
        """Load balance between server processes"""
        # Always try port 8000 first for load operations
        base_url = "http://127.0.0.1:8000"
        
        # Test if server is reachable
        try:
            import requests
            resp = requests.get(f"{base_url}/health", timeout=5)
            if resp.status_code == 200:
                return base_url
        except:
            pass
        
        # Fallback to original logic
        if self.world_size > 1:
            server_port = 8000 + (self.local_rank % 2)
            return f"http://127.0.0.1:{server_port}"
        return self.server_url


def check_required_files(model_path, local_rank):
    """Check if all required model files exist"""
    required_files = [
        "head_model.pt",
        "tail_model.pt", 
        "head_optimizer.pt",
        "tail_optimizer.pt",
        "body_model",  # directory
        "body_optimizer.pt"
    ]
    
    print(f"[RANK {local_rank}] Checking required files in {model_path}:", flush=True)
    missing_files = []
    
    for file in required_files:
        path = os.path.join(model_path, file)
        exists = os.path.exists(path)
        print(f"[RANK {local_rank}] {file}: {'✓' if exists else '✗'}", flush=True)
        if not exists:
            missing_files.append(file)
    
    if missing_files:
        print(f"[RANK {local_rank}] MISSING FILES: {missing_files}", flush=True)
        return False
    
    print(f"[RANK {local_rank}] All required files found!", flush=True)
    return True



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./server_model")
    parser.add_argument("--server_url", type=str, default="http://localhost:8000")
    parser.add_argument("--continue_training", action="store_true", help="Continue training after loading")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation, no training")
    args = parser.parse_args()

    # Setup distributed training
    local_rank, world_size, is_distributed = setup_ddp()
    print(f"[RANK {local_rank}] Started load.py with args: {args}", flush=True)

    # Wait for server (only rank 0)
    if local_rank == 0:
        if not wait_for_server(args.server_url):
            print("Server is not available. Please start the server first.", flush=True)
            return

    # Synchronize all processes
    if is_distributed:
        dist.barrier()

    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"[RANK {local_rank}] Model path {args.model_path} does not exist!", flush=True)
        return

    # STEP 1: Check required files FIRST
    print(f"[RANK {local_rank}] Checking for required files...", flush=True)
    if not check_required_files(args.model_path, local_rank):
        print(f"[RANK {local_rank}] Cannot proceed - missing required files", flush=True)
        print(f"[RANK {local_rank}] You need to run ./client_launch.sh first to create initial checkpoints", flush=True)
        cleanup_ddp(is_distributed)
        return

    device = torch.device(f"cuda:{local_rank}")

    # Initialize models with same configuration as training
    print(f"[RANK {local_rank}] Initializing models...", flush=True)
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

    print(f"[RANK {local_rank}] Creating LoadedSplitModelTrainer...", flush=True)
    trainer = LoadedSplitModelTrainer(head_m, tail_m, tokenizer, args.server_url, local_rank, world_size)

    # STEP 2: Try to load models and optimizers
    print(f"[RANK {local_rank}] Attempting to load models and optimizers...", flush=True)
    load_success = trainer.load_models_and_optimizers(args.model_path)
    print(f"[RANK {local_rank}] Load result: {load_success}", flush=True)

    if load_success:
        print(f"[RANK {local_rank}] All models and optimizers loaded successfully!", flush=True)
        
        if args.eval_only:
            print(f"[RANK {local_rank}] Running evaluation only...", flush=True)
            # Load dataset for evaluation
            train_ds, test_ds = trainer.load_e2e_dataset()
            
            # Run evaluation only (no training)
            if local_rank == 0:  # Only rank 0 evaluates
                print("Starting evaluation with Hugging Face metrics...", flush=True)
                eval_results = trainer.evaluate(test_ds)
                if eval_results:
                    bleu_score = eval_results.get('bleu', 0.0)
                    meteor_score = eval_results.get('meteor', 0.0)
                    print(f"=== EVALUATION RESULTS ===", flush=True)
                    print(f"Final BLEU: {bleu_score:.4f}", flush=True)
                    print(f"Final METEOR: {meteor_score:.4f}", flush=True)
                    print(f"========================", flush=True)
            
        elif args.continue_training:
            print(f"[RANK {local_rank}] ENTERING TRAINING MODE for {args.epochs} epochs...", flush=True)
            
            # Load dataset and create dataloader
            print(f"[RANK {local_rank}] Loading E2E dataset...", flush=True)
            train_ds, test_ds = trainer.load_e2e_dataset()
            print(f"[RANK {local_rank}] Dataset loaded: train={len(train_ds)}, test={len(test_ds)}", flush=True)

            if is_distributed:
                print(f"[RANK {local_rank}] Creating distributed dataloader...", flush=True)
                train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank)
                train_dl = trainer.create_dataloader(
                    train_ds, batch_size=args.batch_size, shuffle=False, sampler=train_sampler
                )
            else:
                print(f"[RANK {local_rank}] Creating single-process dataloader...", flush=True)
                train_dl = trainer.create_dataloader(
                    train_ds, batch_size=args.batch_size, shuffle=True
                )

            print(f"[RANK {local_rank}] Starting training...", flush=True)
            trainer.train(train_dl, epochs=args.epochs, test_ds=test_ds)
            print(f"[RANK {local_rank}] Training completed!", flush=True)
        else:
            print(f"[RANK {local_rank}] No --continue_training or --eval_only flag provided", flush=True)
    else:
        print(f"[RANK {local_rank}] FAILED TO LOAD MODELS - cannot proceed!", flush=True)
        print(f"[RANK {local_rank}] Check if all required files exist in {args.model_path}", flush=True)

    cleanup_ddp(is_distributed)


