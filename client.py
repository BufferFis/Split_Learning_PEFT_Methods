import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import requests
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from evaluate import load as load_metric
from tqdm import tqdm
import json
import os
from datetime import datetime
from util import split_gpt2
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()



# Server URL configuration
SERVER_URL = "http://127.0.0.1:8000"  # Change to your server URL

# Load GPT-2 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model_name = "gpt2"
full_model = AutoModelForCausalLM.from_pretrained(model_name)

# Split the model
head_model, _, tail_model = split_gpt2(full_model, head_layers=2, tail_layers=2)

# Apply DoRA
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.05,
    bias="none",
    use_dora=True,
    task_type="CAUSAL_LM"
)

head_model = get_peft_model(head_model, lora_config)
tail_model = get_peft_model(tail_model, lora_config)

# Move models to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
head_model = head_model.to(device)
tail_model = tail_model.to(device)

print(f"Client models loaded on {device}")
print(f"Head model trainable parameters: {sum(p.numel() for p in head_model.parameters() if p.requires_grad)}")
print(f"Tail model trainable parameters: {sum(p.numel() for p in tail_model.parameters() if p.requires_grad)}")

# Client trainer
class SplitModelTrainer:
    def __init__(self, head_model, tail_model, tokenizer, server_url=SERVER_URL):
        self.head_model = head_model
        self.tail_model = tail_model
        self.tokenizer = tokenizer
        self.server_url = server_url
        self.device = device
        
        # Setup optimizers for client-side parameters
        self.head_optimizer = optim.AdamW([p for p in self.head_model.parameters() if p.requires_grad], lr=2e-4)
        self.tail_optimizer = optim.AdamW([p for p in self.tail_model.parameters() if p.requires_grad], lr=2e-4)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def load_e2e_dataset(self):
        """Load and preprocess E2E NLG dataset"""
        dataset = load_dataset("e2e_nlg", trust_remote_code=True)
        
        def preprocess(example):
            text = example["meaning_representation"] + " " + example["human_reference"]
            encodings = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=128
            )
            return {
                "input_ids": torch.tensor(encodings["input_ids"]),
                "attention_mask": torch.tensor(encodings["attention_mask"]),
                "labels": torch.tensor(encodings["input_ids"]),
                "human_reference": example["human_reference"]
            }
        
        train_dataset = dataset["train"].map(preprocess)
        test_dataset = dataset["test"].map(preprocess)
        
        return train_dataset, test_dataset
    
    def create_dataloader(self, dataset, batch_size=2048 , shuffle=True, sampler=None):
        """Create dataloader from dataset"""
        def collate_fn(batch):
            return {
                "input_ids": torch.stack([torch.tensor(item["input_ids"]) for item in batch]),
                "attention_mask": torch.stack([torch.tensor(item["attention_mask"]) for item in batch]),
                "labels": torch.stack([torch.tensor(item["labels"]) for item in batch]),
                "human_reference": [item["human_reference"] for item in batch]
            }
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle if sampler is None else False,  # <-- disables shuffle if sampler is used
            sampler=sampler,
            num_workers=8,  # Use more workers for faster loading
            pin_memory=True
        )
    
    def train(self, train_dataloader, epochs=3):
        """Train the split model"""
        # Initialize training on server
        response = requests.post(f"{self.server_url}/start_training", 
                               json={"learning_rate": 2e-4})
        
        if response.status_code != 200:
            print(f"Failed to initialize server training: {response.text}")
            return
        
        print(f"Server training initialized: {response.json()}")
        
        # Training loop
        for epoch in range(epochs):
            self.head_model.train()
            self.tail_model.train()
            total_loss = 0
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Zero gradients
                self.head_optimizer.zero_grad()
                self.tail_optimizer.zero_grad()
                
                # Forward through head model
                head_outputs = self.head_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                head_hidden_states = head_outputs.hidden_states[-1]
                
                # Send to server for middle layers processing
                payload = {"activations": head_hidden_states.detach().cpu().tolist()}
                response = requests.post(f"{self.server_url}/forward_train", json=payload)
                
                if response.status_code != 200:
                    print(f"Server error: {response.text}")
                    continue
                
                # Get processed hidden states from server
                server_hidden = torch.tensor(response.json()["body_activations"], device=self.device)
                server_hidden.requires_grad = True
                
                # Forward through tail model
                tail_outputs = self.tail_model(inputs_embeds=server_hidden, attention_mask=attention_mask)
                logits = tail_outputs.logits
                
                # Calculate loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # Backward pass
                loss.backward()
                
                # Send gradients to server
                grad_payload = {
                    "grad_output": server_hidden.grad.cpu().tolist(),
                    "loss": loss.item()
                }
                response = requests.post(f"{self.server_url}/backward", json=grad_payload)
                
                # Update client models
                self.head_optimizer.step()
                self.tail_optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})
                
                # Log every 50 batches
                if batch_idx % 50 == 0 and batch_idx > 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
            
            # Notify server about epoch end
            is_final = (epoch == epochs - 1)
            response = requests.post(f"{self.server_url}/end_epoch", 
                                   json={"is_final": is_final})
        
        # Save client models
        os.makedirs("./client_models", exist_ok=True)
        head_model.save_pretrained("./client_models/head")
        tail_model.save_pretrained("./client_models/tail")
        tokenizer.save_pretrained("./client_models")
        
        print("Training completed! Models saved.")
    
    def evaluate(self, test_dataset):
        """Evaluate the split model on test dataset"""
        self.head_model.eval()
        self.tail_model.eval()
        
        # Load metrics
        bleu_metric = load_metric("bleu")
        meteor_metric = load_metric("meteor")
        rouge_metric = load_metric("rouge")
        
        predictions = []
        references = []
        
        print("Generating predictions for evaluation...")
        for i, example in enumerate(tqdm(test_dataset)):
            input_ids = example["input_ids"].unsqueeze(0).to(self.device)
            attention_mask = example["attention_mask"].unsqueeze(0).to(self.device)
            
            # Generate text using the split model
            generated_text = self.generate(input_ids, attention_mask)
            reference = example["human_reference"]
            
            predictions.append(generated_text)
            references.append([reference])
            
            # Print a few examples
            if i < 5:
                print(f"\nExample {i+1}:")
                print(f"Reference: {reference}")
                print(f"Generated: {generated_text}")
        
        # Calculate metrics
        bleu_score = bleu_metric.compute(predictions=predictions, references=references)
        meteor_score = meteor_metric.compute(predictions=predictions, references=references)
        rouge_score = rouge_metric.compute(predictions=predictions, references=[r[0] for r in references])
        
        print("\nEvaluation Results:")
        print(f"BLEU: {bleu_score}")
        print(f"METEOR: {meteor_score}")
        print(f"ROUGE-L: {rouge_score['rougeL']}")
        
        # Save results
        results = {
            "bleu": bleu_score,
            "meteor": meteor_score,
            "rouge": rouge_score,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open("evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def generate(self, input_ids, attention_mask, max_length=128):
        """Generate text using the split model"""
        with torch.no_grad():
            # Get initial hidden states from head model
            head_outputs = self.head_model(input_ids=input_ids, attention_mask=attention_mask)
            head_hidden_states = head_outputs.last_hidden_state
            
            # Process through server
            payload = {"activations": head_hidden_states.cpu().tolist()}
            response = requests.post(f"{self.server_url}/forward", json=payload)
            
            if response.status_code != 200:
                print(f"Server error during generation: {response.text}")
                return "Error during generation"
            
            # Get processed hidden states from server
            server_hidden = torch.tensor(response.json()["body_activations"], device=self.device)
            
            # Process through tail model for initial token
            tail_outputs = self.tail_model(inputs_embeds=server_hidden, attention_mask=attention_mask)
            logits = tail_outputs.logits
            
            # Start with the input sequence
            generated_ids = input_ids.clone()
            
            # Generate tokens one by one
            for _ in range(max_length - input_ids.size(1)):
                # Get next token prediction
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # Check if EOS token was generated
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Create new attention mask
                new_attention_mask = torch.ones((1, generated_ids.size(1)), device=self.device)
                
                # Forward through head model
                head_outputs = self.head_model(input_ids=generated_ids, attention_mask=new_attention_mask)
                head_hidden_states = head_outputs.last_hidden_state
                
                # Process through server
                payload = {"activations": head_hidden_states.cpu().tolist()}
                response = requests.post(f"{self.server_url}/forward", json=payload)
                
                if response.status_code != 200:
                    break
                
                # Get processed hidden states from server
                server_hidden = torch.tensor(response.json()["body_activations"], device=self.device)
                
                # Process through tail model
                tail_outputs = self.tail_model(inputs_embeds=server_hidden, attention_mask=new_attention_mask)
                logits = tail_outputs.logits
            
            # Decode the generated tokens
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            return generated_text

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate split GPT-2 with DoRA on E2E NLG dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only")
    parser.add_argument("--server_url", type=str, default=SERVER_URL, help="URL of the server")
    
    args = parser.parse_args()
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    # Move models to correct device
    head_model_ddp = head_model.to(device)
    tail_model_ddp = tail_model.to(device)

    # Wrap with DDP
    head_model_ddp = DDP(head_model_ddp, device_ids=[local_rank])
    tail_model_ddp = DDP(tail_model_ddp, device_ids=[local_rank])

    
    # Create trainer
    trainer = SplitModelTrainer(
        head_model_ddp, tail_model_ddp, tokenizer, server_url=args.server_url
    )
    trainer.device = device
    
    # Load dataset
    train_dataset, test_dataset = trainer.load_e2e_dataset()
    print(f"Loaded {len(train_dataset)} training examples and {len(test_dataset)} test examples")
    
    if not args.eval_only:
        # Use DistributedSampler for DDP
        train_sampler = DistributedSampler(train_dataset)
        train_dataloader = trainer.create_dataloader(
            train_dataset, batch_size=64, shuffle=False, sampler=train_sampler
        )
        print(f"Starting training for {args.epochs} epochs with batch size {args.batch_size}")
        trainer.train(train_dataloader, epochs=args.epochs)
    
    
    # Evaluate
    if local_rank == 0:
        print("Starting evaluation...")
        results = trainer.evaluate(test_dataset)
        print("Done!")

    cleanup_ddp()

if __name__ == "__main__":
    main()
