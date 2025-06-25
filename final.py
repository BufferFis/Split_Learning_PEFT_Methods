# splitlora_single.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from evaluate import load as load_metric
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
import json
import argparse
from typing import Dict, List, Tuple, Optional
import traceback
from datetime import datetime

# Set GPU device to A1000 (GPU 1)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def split_gpt2(model, head_layers=2, tail_layers=2):
    """Split GPT2 model into head, body, and tail parts"""
    total_layers = len(model.transformer.h)
    body_layers = total_layers - head_layers - tail_layers
    
    if body_layers <= 0:
        raise ValueError(f"Not enough layers to split. Total: {total_layers}, Head: {head_layers}, Tail: {tail_layers}")
    
    print(f"Splitting model: Head({head_layers}) + Body({body_layers}) + Tail({tail_layers}) = {total_layers}")
    
    # Head Model (embedding + first few layers)
    class HeadModel(nn.Module):
        def __init__(self, original_model, num_layers):
            super().__init__()
            self.wte = original_model.transformer.wte
            self.wpe = original_model.transformer.wpe
            self.drop = original_model.transformer.drop
            self.h = nn.ModuleList(original_model.transformer.h[:num_layers])
            self.config = original_model.config
            
            # Add missing generation attributes for PEFT compatibility
            self.generation_config = getattr(original_model, 'generation_config', None)
            self.main_input_name = getattr(original_model, 'main_input_name', 'input_ids')
            
            # FIX: Add the missing prepare_inputs_for_generation method
            if hasattr(original_model, 'prepare_inputs_for_generation'):
                self.prepare_inputs_for_generation = original_model.prepare_inputs_for_generation
            else:
                self.prepare_inputs_for_generation = self._prepare_inputs_for_generation
                
            # Add other missing attributes that PEFT might need
            for attr in ['_get_resized_embeddings', 'get_input_embeddings', 'set_input_embeddings', 
                        'get_output_embeddings', 'set_output_embeddings', 'resize_token_embeddings']:
                if hasattr(original_model, attr):
                    setattr(self, attr, getattr(original_model, attr))
            
        def _prepare_inputs_for_generation(self, input_ids, **kwargs):
            """Default implementation for prepare_inputs_for_generation"""
            return {"input_ids": input_ids}
            
        def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **kwargs):
            inputs_embeds = self.wte(input_ids)
            seq_length = input_ids.size(-1)
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
            position_embeds = self.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds
            hidden_states = self.drop(hidden_states)
            
            # Handle attention mask
            if attention_mask is not None and attention_mask.dim() == 2:
                batch_size, seq_len = attention_mask.shape
                causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=attention_mask.device))
                attention_mask = attention_mask.unsqueeze(-1).unsqueeze(-1) * causal_mask
                attention_mask = attention_mask.unsqueeze(1).expand(-1, self.config.n_head, -1, -1)
                attention_mask = attention_mask.float()
                attention_mask = (1.0 - attention_mask) * -10000.0
            
            all_hidden_states = ()
            for block in self.h:
                hidden_states = block(hidden_states, attention_mask=attention_mask)[0]
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            if output_hidden_states:
                return type('HeadOutput', (), {
                    'last_hidden_state': hidden_states,
                    'hidden_states': all_hidden_states
                })()
            else:
                return type('HeadOutput', (), {'last_hidden_state': hidden_states})()
    
    # Body Model (middle layers)
    class BodyModel(nn.Module):
        def __init__(self, original_model, start_layer, num_layers):
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList(
                original_model.transformer.h[start_layer:start_layer + num_layers]
            )
            self.transformer.ln_f = original_model.transformer.ln_f
            self.config = original_model.config
            
            # Add missing generation attributes for PEFT compatibility
            self.generation_config = getattr(original_model, 'generation_config', None)
            self.main_input_name = getattr(original_model, 'main_input_name', 'input_ids')
            
            # FIX: Add the missing prepare_inputs_for_generation method
            if hasattr(original_model, 'prepare_inputs_for_generation'):
                self.prepare_inputs_for_generation = original_model.prepare_inputs_for_generation
            else:
                self.prepare_inputs_for_generation = self._prepare_inputs_for_generation
                
            # Add other missing attributes that PEFT might need
            for attr in ['_get_resized_embeddings', 'get_input_embeddings', 'set_input_embeddings', 
                        'get_output_embeddings', 'set_output_embeddings', 'resize_token_embeddings']:
                if hasattr(original_model, attr):
                    setattr(self, attr, getattr(original_model, attr))
            
        def _prepare_inputs_for_generation(self, input_ids, **kwargs):
            """Default implementation for prepare_inputs_for_generation"""
            return {"input_ids": input_ids}
            
        def forward(self, hidden_states=None, attention_mask=None, **kwargs):
            if attention_mask is not None and attention_mask.dim() == 2:
                batch_size, seq_len = attention_mask.shape
                num_heads = self.config.n_head
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(3)
                attention_mask = attention_mask.expand(batch_size, num_heads, seq_len, seq_len)
                attention_mask = attention_mask.float()
                attention_mask = (1.0 - attention_mask) * -10000.0
            
            for block in self.transformer.h:
                hidden_states = block(hidden_states, attention_mask=attention_mask, use_cache=False)[0]
            
            hidden_states = self.transformer.ln_f(hidden_states)
            return type('BodyOutput', (), {'last_hidden_state': hidden_states})()
    
    # Tail Model (last few layers + LM head)
    class TailModel(nn.Module):
        def __init__(self, original_model, start_layer):
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList(original_model.transformer.h[start_layer:])
            self.lm_head = original_model.lm_head
            self.config = original_model.config
            
            # Add missing generation attributes for PEFT compatibility
            self.generation_config = getattr(original_model, 'generation_config', None)
            self.main_input_name = getattr(original_model, 'main_input_name', 'input_ids')
            
            # FIX: Add the missing prepare_inputs_for_generation method
            if hasattr(original_model, 'prepare_inputs_for_generation'):
                self.prepare_inputs_for_generation = original_model.prepare_inputs_for_generation
            else:
                self.prepare_inputs_for_generation = self._prepare_inputs_for_generation
                
            # Add other missing attributes that PEFT might need
            for attr in ['_get_resized_embeddings', 'get_input_embeddings', 'set_input_embeddings', 
                        'get_output_embeddings', 'set_output_embeddings', 'resize_token_embeddings']:
                if hasattr(original_model, attr):
                    setattr(self, attr, getattr(original_model, attr))
            
        def _prepare_inputs_for_generation(self, input_ids, **kwargs):
            """Default implementation for prepare_inputs_for_generation"""
            return {"input_ids": input_ids}
            
        def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
            hidden_states = inputs_embeds
            
            if attention_mask is not None and attention_mask.dim() == 2:
                batch_size, seq_len = attention_mask.shape
                num_heads = self.config.n_head
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(3)
                attention_mask = attention_mask.expand(batch_size, num_heads, seq_len, seq_len)
                attention_mask = attention_mask.float()
                attention_mask = (1.0 - attention_mask) * -10000.0
            
            for block in self.transformer.h:
                hidden_states = block(hidden_states, attention_mask=attention_mask, use_cache=False)[0]
            
            logits = self.lm_head(hidden_states)
            return type('TailOutput', (), {'logits': logits})()
    
    head_model = HeadModel(model, head_layers)
    body_model = BodyModel(model, head_layers, body_layers)
    tail_model = TailModel(model, head_layers + body_layers)
    
    return head_model, body_model, tail_model

class ServerModel:
    """Server component handling the body layers"""
    def __init__(self, body_model, learning_rate=2e-4):
        self.body_model = body_model.to(device)
        self.optimizer = optim.AdamW(
            [p for p in self.body_model.parameters() if p.requires_grad], 
            lr=learning_rate
        )
        self.step_count = 0
        self.stored_activations = {}
        
    def forward(self, activations, attention_mask=None):
        """Forward pass through body layers (inference mode)"""
        self.body_model.eval()
        with torch.no_grad():
            output = self.body_model(hidden_states=activations, attention_mask=attention_mask)
            return output.last_hidden_state
    
    def forward_train(self, activations, attention_mask=None):
        """Forward pass during training"""
        self.body_model.train()
        activations.requires_grad_(True)
        output = self.body_model(hidden_states=activations, attention_mask=attention_mask)
        
        # Store activations for backward pass
        activation_id = id(activations)
        self.stored_activations[activation_id] = activations
        
        return output.last_hidden_state, activation_id
    
    def backward(self, grad_output, activation_id):
        """Backward pass through body layers"""
        self.optimizer.zero_grad()
        
        # Get stored activations
        activations = self.stored_activations.get(activation_id)
        if activations is None:
            raise ValueError("No stored activations found for this ID")
        
        # Compute gradients
        torch.autograd.backward(
            tensors=[grad_output],
            grad_tensors=[torch.ones_like(grad_output)],
            retain_graph=False
        )
        
        # Get input gradient
        input_grad = activations.grad if activations.grad is not None else torch.zeros_like(activations)
        
        # Update parameters
        self.optimizer.step()
        self.step_count += 1
        
        # Cleanup
        del self.stored_activations[activation_id]
        
        return input_grad

class HeadClient:
    """Client component handling head layers"""
    def __init__(self, head_model, learning_rate=2e-4):
        self.head_model = head_model.to(device)
        self.optimizer = optim.AdamW(
            [p for p in self.head_model.parameters() if p.requires_grad], 
            lr=learning_rate
        )
        
    def forward(self, input_ids, attention_mask=None):
        """Forward pass through head layers"""
        output = self.head_model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            output_hidden_states=True
        )
        return output.hidden_states[-1]
    
    def backward(self, grad_input):
        """Backward pass through head layers"""
        self.optimizer.zero_grad()
        # Gradient computation happens during the full backward pass
        self.optimizer.step()

class TailClient:
    """Client component handling tail layers"""
    def __init__(self, tail_model, learning_rate=2e-4):
        self.tail_model = tail_model.to(device)
        self.optimizer = optim.AdamW(
            [p for p in self.tail_model.parameters() if p.requires_grad], 
            lr=learning_rate
        )
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, body_activations, attention_mask=None):
        """Forward pass through tail layers"""
        output = self.tail_model(inputs_embeds=body_activations, attention_mask=attention_mask)
        return output.logits
    
    def compute_loss_and_backward(self, body_activations, labels, attention_mask=None):
        """Compute loss and perform backward pass"""
        self.optimizer.zero_grad()
        
        # Forward pass
        body_activations.requires_grad_(True)
        logits = self.tail_model(inputs_embeds=body_activations, attention_mask=attention_mask).logits
        
        # Compute loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # Backward pass
        loss.backward(retain_graph=True)
        
        # Get gradient for body activations
        body_grad = body_activations.grad if body_activations.grad is not None else torch.zeros_like(body_activations)
        
        self.optimizer.step()
        
        return loss.item(), body_grad

class SplitLoRATrainer:
    """Main trainer class combining all components"""
    def __init__(self, model_name="gpt2", head_layers=2, tail_layers=2, learning_rate=2e-4):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load and split model
        full_model = AutoModelForCausalLM.from_pretrained(model_name)
        head_model, body_model, tail_model = split_gpt2(full_model, head_layers, tail_layers)
        
        # Apply LoRA/DoRA - Now supported with Python 3.11.9!
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            use_dora=True,  # DoRA is now fully supported!
            task_type="CAUSAL_LM",
            target_modules=["c_attn", "c_proj"]
        )
        
        head_model = get_peft_model(head_model, lora_config)
        body_model = get_peft_model(body_model, lora_config)
        tail_model = get_peft_model(tail_model, lora_config)
        
        # Initialize components
        self.server = ServerModel(body_model, learning_rate)
        self.head_client = HeadClient(head_model, learning_rate)
        self.tail_client = TailClient(tail_model, learning_rate)
        
        self.metrics = {"loss": []}
        
    def load_e2e_dataset(self):
        """Load and preprocess E2E NLG dataset"""
        dataset = load_dataset("e2e_nlg", trust_remote_code=True)
        
        def preprocess(example):
            text = example["meaning_representation"] + " " + example["human_reference"]
            enc = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_attention_mask=True
            )
            return {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": enc["input_ids"],
                "human_reference": example["human_reference"]
            }
        
        train_ds = dataset["train"].map(preprocess, remove_columns=dataset["train"].column_names)
        test_ds = dataset["test"].map(preprocess, remove_columns=dataset["test"].column_names)
        
        return train_ds, test_ds
    
    def create_dataloader(self, dataset, batch_size=8, shuffle=True):
        """Create DataLoader with proper collation"""
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
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )
    
    def train(self, train_dataloader, epochs=1):
        """Train the split model"""
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                try:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    
                    # Forward pass through head
                    head_activations = self.head_client.forward(input_ids, attention_mask)
                    
                    # Forward pass through server (body)
                    body_activations, activation_id = self.server.forward_train(
                        head_activations, attention_mask
                    )
                    
                    # Forward pass through tail and compute loss
                    loss, body_grad = self.tail_client.compute_loss_and_backward(
                        body_activations, labels, attention_mask
                    )
                    
                    # Backward through server (body)
                    head_grad = self.server.backward(body_grad, activation_id)
                    
                    # Backward through head
                    head_activations.backward(head_grad, retain_graph=False)
                    self.head_client.backward(head_grad)
                    
                    total_loss += loss
                    num_batches += 1
                    
                except Exception as e:
                    print(f"Training error: {e}")
                    traceback.print_exc()
                    continue
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            self.metrics["loss"].append(avg_loss)
            print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        print("Training completed!")
    
    def generate(self, input_ids, attention_mask, max_length=128):
        """Generate text using the split model"""
        with torch.no_grad():
            try:
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                if attention_mask.dim() == 1:
                    attention_mask = attention_mask.unsqueeze(0)
                
                generated_ids = input_ids.clone()
                
                for step in range(min(max_length - input_ids.size(1), 32)):
                    current_attention_mask = torch.ones(
                        generated_ids.size(0),
                        generated_ids.size(1),
                        dtype=torch.float32,
                        device=device
                    )
                    
                    # Forward through head
                    head_activations = self.head_client.forward(generated_ids, current_attention_mask)
                    
                    # Forward through server
                    body_activations = self.server.forward(head_activations, current_attention_mask)
                    
                    # Forward through tail
                    logits = self.tail_client.forward(body_activations, current_attention_mask)
                    
                    # Get next token
                    next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)
                    
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                
                return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            except Exception as e:
                print(f"Generation error: {e}")
                return "Generation failed"
    
    def evaluate(self, test_dataset):
        """Evaluate model using SplitLoRA paper metrics"""
        print("Starting evaluation...")
        
        try:
            # Load metrics as used in SplitLoRA paper
            bleu_metric = load_metric("bleu")
            meteor_metric = load_metric("meteor")
            
            preds, refs = [], []
            
            # Sample evaluation data
            eval_samples = test_dataset.select(range(min(100, len(test_dataset))))
            
            for i, sample in enumerate(tqdm(eval_samples, desc="Evaluating")):
                try:
                    if not isinstance(sample, dict) or "input_ids" not in sample:
                        continue
                    
                    input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0).to(device)
                    attention_mask = torch.tensor(sample["attention_mask"]).unsqueeze(0).to(device)
                    
                    # Generate prediction
                    generated_text = self.generate(input_ids, attention_mask)
                    preds.append(generated_text)
                    refs.append([sample["human_reference"]])
                    
                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    continue
            
            if not preds:
                print("No valid predictions generated")
                return {"bleu": 0.0, "meteor": 0.0, "error": "No valid samples"}
            
            # Calculate metrics
            try:
                bleu_score = bleu_metric.compute(predictions=preds, references=refs)
                meteor_score = meteor_metric.compute(predictions=preds, references=[r[0] for r in refs])
                
                bleu_value = bleu_score.get('bleu', 0.0) if isinstance(bleu_score, dict) else 0.0
                meteor_value = meteor_score.get('meteor', 0.0) if isinstance(meteor_score, dict) else 0.0
                
                print(f"BLEU Score: {bleu_value:.4f}")
                print(f"METEOR Score: {meteor_value:.4f}")
                
                results = {
                    "bleu": bleu_value,
                    "meteor": meteor_value,
                    "num_samples": len(preds)
                }
                
                return results
                
            except Exception as eval_error:
                print(f"Evaluation metric error: {eval_error}")
                return {"bleu": 0.0, "meteor": 0.0, "error": str(eval_error)}
                
        except Exception as e:
            print(f"Evaluation error: {e}")
            return None
    
    def save_checkpoint(self, path="./splitlora_checkpoint"):
        """Save model and optimizer states"""
        os.makedirs(path, exist_ok=True)
        
        # Save models
        self.head_client.head_model.save_pretrained(os.path.join(path, "head_model"))
        self.server.body_model.save_pretrained(os.path.join(path, "body_model"))
        self.tail_client.tail_model.save_pretrained(os.path.join(path, "tail_model"))
        
        # Save optimizers
        torch.save(self.head_client.optimizer.state_dict(), os.path.join(path, "head_optimizer.pt"))
        torch.save(self.server.optimizer.state_dict(), os.path.join(path, "body_optimizer.pt"))
        torch.save(self.tail_client.optimizer.state_dict(), os.path.join(path, "tail_optimizer.pt"))
        
        # Save metrics
        with open(os.path.join(path, "metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"Checkpoint saved to {path}")
        return path
    
    def load_checkpoint(self, path="./splitlora_checkpoint"):
        """Load model and optimizer states"""
        if not os.path.exists(path):
            print(f"Checkpoint path {path} does not exist")
            return False
        
        try:
            # Load models
            # Recreate base models
            full_model = AutoModelForCausalLM.from_pretrained("gpt2")
            head_model, body_model, tail_model = split_gpt2(full_model, 2, 2)
            
            # Load PEFT models
            head_model = PeftModel.from_pretrained(head_model, os.path.join(path, "head_model"), is_trainable=True)
            body_model = PeftModel.from_pretrained(body_model, os.path.join(path, "body_model"), is_trainable=True)
            tail_model = PeftModel.from_pretrained(tail_model, os.path.join(path, "tail_model"), is_trainable=True)
            
            # Update components
            self.head_client.head_model = head_model.to(device)
            self.server.body_model = body_model.to(device)
            self.tail_client.tail_model = tail_model.to(device)
            
            # Load optimizers
            self.head_client.optimizer.load_state_dict(torch.load(os.path.join(path, "head_optimizer.pt"), map_location=device))
            self.server.optimizer.load_state_dict(torch.load(os.path.join(path, "body_optimizer.pt"), map_location=device))
            self.tail_client.optimizer.load_state_dict(torch.load(os.path.join(path, "tail_optimizer.pt"), map_location=device))
            
            # Load metrics
            with open(os.path.join(path, "metrics.json"), "r") as f:
                self.metrics = json.load(f)
            
            print(f"Checkpoint loaded from {path}")
            return True
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            traceback.print_exc()
            return False

def main():
    parser = argparse.ArgumentParser(description="SplitLoRA Single File Implementation")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to checkpoint to load")
    parser.add_argument("--save_path", type=str, default="./splitlora_checkpoint", help="Path to save checkpoint")
    parser.add_argument("--gpu_device", type=str, default="1", help="GPU device to use")
    
    args = parser.parse_args()
    
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    
    # Initialize trainer
    trainer = SplitLoRATrainer(learning_rate=args.learning_rate)
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        trainer.load_checkpoint(args.load_checkpoint)
    
    # Load dataset
    train_ds, test_ds = trainer.load_e2e_dataset()
    
    if not args.eval_only:
        # Create dataloader and train
        train_dl = trainer.create_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
        trainer.train(train_dl, epochs=args.epochs)
        
        # Save checkpoint
        trainer.save_checkpoint(args.save_path)
    
    # Evaluate
    results = trainer.evaluate(test_ds)
    if results:
        print(f"Final Results: BLEU={results['bleu']:.4f}, METEOR={results['meteor']:.4f}")
        
        # Save evaluation results
        with open(os.path.join(args.save_path, "evaluation_results.json"), "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
