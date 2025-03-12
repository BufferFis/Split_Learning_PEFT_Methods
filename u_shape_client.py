import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import requests
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import random
import numpy as np
from tqdm import tqdm

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)

# Set the split points for the model
split1 = 2  # Client will handle the first 2 layers
split2 = 6  # Server will handle layers 2-6

# Client-side model definition
class ClientModel(nn.Module):
    def __init__(self, model, split1, split2):
        super().__init__()
        self.config = model.config
        self.embeddings = model.transformer.wte
        self.position_embeddings = model.transformer.wpe
        self.initial_layers = model.transformer.h[:split1]
        self.final_layers = model.transformer.h[split2:]
        self.ln_f = model.transformer.ln_f
        self.lm_head = model.lm_head
        
    def forward(self, input_ids=None, hidden_states=None):
        if hidden_states is None:
            # Initial forward path
            inputs_embeds = self.embeddings(input_ids)
            position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
            position_embeds = self.position_embeddings(position_ids)
            hidden_states = inputs_embeds + position_embeds
            
            # Process through initial layers
            for layer in self.initial_layers:
                hidden_states = layer(hidden_states)[0]
            
            return hidden_states
        else:
            # Process from hidden states through final layers
            for layer in self.final_layers:
                hidden_states = layer(hidden_states)[0]
            
            # Final normalization and language model head
            hidden_states = self.ln_f(hidden_states)
            logits = self.lm_head(hidden_states)
            
            return logits

# Create client model
client_model = ClientModel(model, split1, split2)

# Data-to-Text dataset
class DataToTextDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128):
        """
        Args:
            csv_file (string): Path to the csv file with MR,REF columns
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
        """
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        mr = self.data.iloc[idx]['mr']  # Meaning representation
        ref = self.data.iloc[idx]['ref'] # Reference text
        
        # Create input sequence as "MR: {mr} Text:"
        input_text = f"MR: {mr} Text:"
        target_text = f"{ref}{self.tokenizer.eos_token}"
        
        # Tokenize input and target
        input_encoding = self.tokenizer(input_text, 
                                        max_length=self.max_length, 
                                        padding="max_length", 
                                        truncation=True,
                                        return_tensors="pt")
        
        target_encoding = self.tokenizer(target_text, 
                                         max_length=self.max_length, 
                                         padding="max_length", 
                                         truncation=True,
                                         return_tensors="pt")
        
        # Create labels tensor for causal language modeling
        labels = target_encoding.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding in loss
        
        return {
            "input_ids": input_encoding.input_ids.squeeze(),
            "attention_mask": input_encoding.attention_mask.squeeze(),
            "labels": labels.squeeze()
        }

# Data collator
class DataToTextCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# Client trainer
class ClientTrainer:
    def __init__(self, client_model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.client_model = client_model
        self.tokenizer = tokenizer
        self.device = device
        self.client_model.to(device)
        
        # Setup optimizer for client-side parameters
        self.optimizer = optim.AdamW([p for p in self.client_model.parameters() if p.requires_grad], lr=5e-5)
        self.loss_fn = nn.CrossEntropyLoss()
        
    def prepare_dataset(self, csv_file, batch_size=4):
        """Prepare dataset from CSV file"""
        dataset = DataToTextDataset(csv_file, self.tokenizer)
        collator = DataToTextCollator(self.tokenizer)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collator,
            shuffle=True
        )
        
        return dataloader
    
    def train(self, train_dataloader, eval_dataloader=None, epochs=3):
        """Train the U-shaped model"""
        # Initialize training on the server side
        response = requests.post("http://127.0.0.1:8000/start_training", 
                               json={"learning_rate": 2e-4, "batch_size": 4, "epochs": epochs})
        
        if response.status_code != 200:
            print("Failed to initialize training on server:", response.text)
            return
        
        print("Server training initialized:", response.json())
        
        for epoch in range(epochs):
            # Training loop
            self.client_model.train()
            total_loss = 0
            
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Clear gradients
                self.optimizer.zero_grad()
                
                # Forward pass through initial client layers
                hidden_states = self.client_model(input_ids=input_ids)
                
                # Send to server for middle layers processing
                payload = {"hidden_states": hidden_states.cpu().detach().numpy().tolist()}
                response = requests.post("http://127.0.0.1:8000/process_train", json=payload)
                
                if response.status_code != 200:
                    print(f"Server error: {response.text}")
                    continue
                
                # Get processed hidden states from server
                server_hidden = torch.tensor(response.json()["hidden_states"], 
                                          device=self.device, 
                                          requires_grad=True)
                
                # Continue through final layers and get logits
                logits = self.client_model(hidden_states=server_hidden)
                
                # Calculate loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), 
                                   shift_labels.view(-1))
                
                # Backward pass for client
                loss.backward()
                
                # Get gradients to send back to server
                server_hidden_grad = server_hidden.grad if server_hidden.grad is not None else torch.zeros_like(server_hidden)
                
                # Send gradients to server
                grad_payload = {
                    "gradients": server_hidden_grad.cpu().detach().numpy().tolist(),
                    "loss": loss.item()
                }
                response = requests.post("http://127.0.0.1:8000/backward", json=grad_payload)
                
                if response.status_code != 200:
                    print(f"Server error during backward pass: {response.text}")
                
                # Update client weights
                self.optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                
                # Check server pruning stats periodically
                if batch_idx % 20 == 0:
                    self.check_server_pruning_stats()
            
            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
            
            # Evaluate if eval_dataloader is provided
            if eval_dataloader:
                eval_loss, bleu_score = self.evaluate(eval_dataloader)
                print(f"Validation Loss: {eval_loss:.4f}, BLEU score: {bleu_score:.4f}")
            
            # Notify server about epoch end
            is_final = (epoch == epochs - 1)
            response = requests.post("http://127.0.0.1:8000/end_epoch", 
                                   json={"is_final": is_final})
        
        print("Training completed!")
    
    def evaluate(self, eval_dataloader):
        """Evaluate the model on validation data"""
        self.client_model.eval()
        eval_loss = 0
        all_bleu_scores = []
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass through initial client layers
                hidden_states = self.client_model(input_ids=input_ids)
                
                # Send to server for middle layers processing
                payload = {"hidden_states": hidden_states.cpu().detach().numpy().tolist()}
                response = requests.post("http://127.0.0.1:8000/process", json=payload)
                
                if response.status_code != 200:
                    print(f"Server error: {response.text}")
                    continue
                
                # Get processed hidden states from server
                server_hidden = torch.tensor(response.json()["hidden_states"], device=self.device)
                
                # Continue through final layers
                logits = self.client_model(hidden_states=server_hidden)
                
                # Calculate loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), 
                                   shift_labels.view(-1))
                
                eval_loss += loss.item()
                
                # Generate text for BLEU score calculation
                generated_ids = self.generate_from_input_ids(input_ids)
                
                # Calculate BLEU scores for each example in the batch
                for i in range(input_ids.shape[0]):
                    target_text = self.tokenizer.decode(labels[i][labels[i] != -100], skip_special_tokens=True)
                    generated_text = self.tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                    
                    # Calculate BLEU score
                    bleu = self.calculate_bleu(target_text, generated_text)
                    all_bleu_scores.append(bleu)
        
        avg_eval_loss = eval_loss / len(eval_dataloader)
        avg_bleu = sum(all_bleu_scores) / len(all_bleu_scores) if all_bleu_scores else 0
        
        return avg_eval_loss, avg_bleu
    
    def generate_from_input_ids(self, input_ids, max_new_tokens=50):
        """Generate text using the trained model"""
        self.client_model.eval()
        
        # Start with the input ids
        current_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass through initial client layers
                hidden_states = self.client_model(input_ids=current_ids)
                
                # Send to server for middle layers processing
                payload = {"hidden_states": hidden_states.cpu().detach().numpy().tolist()}
                response = requests.post("http://127.0.0.1:8000/process", json=payload)
                
                if response.status_code != 200:
                    print(f"Server error: {response.text}")
                    break
                
                # Get processed hidden states from server
                server_hidden = torch.tensor(response.json()["hidden_states"], device=self.device)
                
                # Continue through final layers
                logits = self.client_model(hidden_states=server_hidden)
                
                # Get the next token prediction (only look at the last position)
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                
                # Concatenate with the current sequence
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
                # Stop if all sequences have EOS token
                if (next_token == self.tokenizer.eos_token_id).all():
                    break
        
        return current_ids
    
    def calculate_bleu(self, reference, candidate):
        """Calculate BLEU score between reference and candidate"""
        reference_tokens = nltk.word_tokenize(reference.lower())
        candidate_tokens = nltk.word_tokenize(candidate.lower())
        
        # Handle edge cases
        if len(candidate_tokens) == 0:
            return 0.0
        
        # Use smoothing to avoid 0 scores when there are no 4-gram matches
        smoothie = SmoothingFunction().method1
        
        # Calculate BLEU score with weights for 1-grams only since texts may be short
        return sentence_bleu([reference_tokens], candidate_tokens, 
                            weights=(1, 0, 0, 0), 
                            smoothing_function=smoothie)
    
    def evaluate_nlg_challenge(self, csv_file):
        """Evaluate on NLG challenge prompts"""
        # Load data
        data = pd.read_csv(csv_file)
        results = []
        
        for i, row in data.iterrows():
            mr = row['mr']
            ref = row['ref']
            
            # Create input sequence
            input_text = f"MR: {mr} Text:"
            
            # Generate text from the input
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            generated_ids = self.generate_from_input_ids(input_ids)
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Calculate BLEU score
            bleu = self.calculate_bleu(ref, generated_text)
            
            results.append({
                "mr": mr,
                "reference": ref,
                "generated": generated_text,
                "bleu": bleu
            })
            
            # Print some examples
            if i % 10 == 0:
                print(f"\nExample {i}:")
                print(f"MR: {mr}")
                print(f"Reference: {ref}")
                print(f"Generated: {generated_text}")
                print(f"BLEU: {bleu:.4f}")
        
        avg_bleu = sum(r["bleu"] for r in results) / len(results)
        print(f"\nAverage BLEU score: {avg_bleu:.4f}")
        
        return results
    
    def check_server_pruning_stats(self):
        """Check pruning statistics from server"""
        try:
            response = requests.get("http://127.0.0.1:8000/prune_stats")
            if response.status_code == 200:
                stats = response.json()
                active = stats.get("total_active_ranks", 0)
                potential = stats.get("potential_ranks", 0)
                if potential > 0:
                    print(f"Pruning stats: {active}/{potential} active ranks ({active/potential:.1%})")
            else:
                print(f"Failed to get pruning stats: {response.status_code}")
        except Exception as e:
            print(f"Error checking pruning stats: {e}")

def main():
    parser = argparse.ArgumentParser(description="Train U-shape client-server model with adaptive LoRA")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training data file")
    parser.add_argument("--eval_file", type=str, help="Path to evaluation data file")
    parser.add_argument("--nlg_prompts", type=str, help="Path to NLG challenge prompts file")
    parser.add_argument("--block_size", type=int, default=128, help="Block size for tokenization")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create trainer instance
    trainer = ClientTrainer(client_model, tokenizer)
    
    # Train the model
    if args.train_file:
        print(f"Preparing training data from {args.train_file}")
        train_dataloader = trainer.prepare_dataset(args.train_file, batch_size=args.batch_size)
        
        eval_dataloader = None
        if args.eval_file:
            print(f"Preparing evaluation data from {args.eval_file}")
            eval_dataloader = trainer.prepare_dataset(args.eval_file, batch_size=args.batch_size)
        
        print("Starting training...")
        trainer.train(train_dataloader, eval_dataloader, epochs=args.epochs)
    
    # Evaluate on NLG challenge if provided
    if args.nlg_prompts:
        print(f"Evaluating on NLG challenge prompts from {args.nlg_prompts}")
        results = trainer.evaluate_nlg_challenge(args.nlg_prompts)
        
        # Save results
        import json
        with open('nlg_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("NLG evaluation results saved to nlg_results.json")

if __name__ == "__main__":
    main()