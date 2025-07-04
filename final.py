# splitlora_single.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"     # or the bus-id / UUID of AF:00.0
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
import math
from peft import PeftModel
# after the other imports in final.py
from split_beam_wrapper import SplitGPT2ForGeneration   # NEW
import copy
from transformers import LogitsProcessorList, MinLengthLogitsProcessor
import numpy as np
from sacrebleu.metrics import BLEU as SBLEU 
from itertools import zip_longest
import subprocess, tempfile, pathlib, json
from transformers import get_linear_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def diagnose_tokenizer_corruption(trainer):
    """Check if tokenizer is corrupted"""
    print("=== TOKENIZER DIAGNOSIS ===")
    print(f"Vocab size: {len(trainer.tokenizer)}")
    print(f"Original GPT-2 vocab size: 50257")
    print(f"Added tokens: {len(trainer.tokenizer) - 50257}")
    
    # Test basic tokens
    test_words = ["the", "restaurant", "food", "good", "city"]
    for word in test_words:
        token_ids = trainer.tokenizer.encode(word, add_special_tokens=False)
        decoded = trainer.tokenizer.decode(token_ids)
        print(f"'{word}' -> {token_ids} -> '{decoded}' {'❌ CORRUPTED' if decoded != word else '✅'}")
    
    # Test special tokens
    print(f"DELIM token: '{trainer.DELIM}' -> {trainer.tokenizer.encode(trainer.DELIM)}")
    print(f"PAD token: '{trainer.PAD}' -> {trainer.tokenizer.pad_token_id}")

def create_clean_model_and_tokenizer():
    """Create a clean model with properly added tokens"""
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    
    # Load clean tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # CRITICAL: Set pad_token BEFORE adding custom tokens
    tokenizer.pad_token = tokenizer.eos_token  # Use existing token first
    
    # Add custom tokens properly
    special_tokens = {
        'additional_special_tokens': ['<|gen|>']
    }
    
    num_added = tokenizer.add_special_tokens(special_tokens)
    print(f"Added {num_added} special tokens")
    
    # CRITICAL: Resize embeddings properly
    model.resize_token_embeddings(len(tokenizer))
    
    # Initialize new token embeddings properly
    with torch.no_grad():
        # Get the new token embeddings
        new_tokens_start = len(tokenizer) - num_added
        
        # Initialize new embeddings as average of existing embeddings
        existing_embeddings = model.transformer.wte.weight[:new_tokens_start]
        avg_embedding = existing_embeddings.mean(dim=0)
        
        # Set new token embeddings
        for i in range(new_tokens_start, len(tokenizer)):
            model.transformer.wte.weight[i] = avg_embedding + torch.randn_like(avg_embedding) * 0.01
    
    # Verify tokenizer works
    test_text = "The restaurant serves food"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    assert decoded == test_text, f"Tokenizer corrupted: '{test_text}' != '{decoded}'"
    
    print("✅ Clean model and tokenizer created successfully")
    return model, tokenizer



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
            super().__init__()  # CRITICAL: Call parent constructor
            self.wte = original_model.transformer.wte
            self.wpe = original_model.transformer.wpe
            self.drop = original_model.transformer.drop
            self.h = nn.ModuleList(original_model.transformer.h[:num_layers])
            self.config = original_model.config
            
            # Add missing generation attributes for PEFT compatibility
            self.generation_config = getattr(original_model, 'generation_config', None)
            self.main_input_name = getattr(original_model, 'main_input_name', 'input_ids')
            
            # Add the missing prepare_inputs_for_generation method
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

            # FIXED: Pass attention_mask to each block
            all_hidden_states = ()
            for block in self.h:
                # Convert attention_mask to the format GPT-2 blocks expect
                if attention_mask is not None:
                    # GPT-2 uses 4D attention mask: [batch, 1, seq_len, seq_len]
                    extended_attention_mask = attention_mask[:, None, None, :]
                    extended_attention_mask = extended_attention_mask.to(dtype=self.wte.weight.dtype)
                    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
                else:
                    extended_attention_mask = None
                    
                hidden_states = block(hidden_states, attention_mask=extended_attention_mask, use_cache=False)[0]
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
            super().__init__()  # CRITICAL: Call parent constructor
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList(
                original_model.transformer.h[start_layer:start_layer + num_layers]
            )
            self.config = original_model.config
            
            # Add missing generation attributes for PEFT compatibility
            self.generation_config = getattr(original_model, 'generation_config', None)
            self.main_input_name = getattr(original_model, 'main_input_name', 'input_ids')
            
            # Add the missing prepare_inputs_for_generation method
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
            # FIXED: Pass attention_mask to each block
            for block in self.transformer.h:
                if attention_mask is not None:
                    # GPT-2 uses 4D attention mask: [batch, 1, seq_len, seq_len]
                    extended_attention_mask = attention_mask[:, None, None, :]
                    extended_attention_mask = extended_attention_mask.to(dtype=hidden_states.dtype)
                    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
                else:
                    extended_attention_mask = None
                    
                hidden_states = block(hidden_states, attention_mask=extended_attention_mask, use_cache=False)[0]
            return type('BodyOutput', (), {'last_hidden_state': hidden_states})()

    # Tail Model (last few layers + LM head)
    class TailModel(nn.Module):
        def __init__(self, original_model, start_layer):
            super().__init__()  # CRITICAL: Call parent constructor
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList(original_model.transformer.h[start_layer:])
            self.transformer.ln_f = original_model.transformer.ln_f
            self.lm_head = original_model.lm_head
            self.config = original_model.config
            
            # Add missing generation attributes for PEFT compatibility
            self.generation_config = getattr(original_model, 'generation_config', None)
            self.main_input_name = getattr(original_model, 'main_input_name', 'input_ids')
            
            # Add the missing prepare_inputs_for_generation method
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

            # FIXED: Pass attention_mask to each block
            for block in self.transformer.h:
                if attention_mask is not None:
                    # GPT-2 uses 4D attention mask: [batch, 1, seq_len, seq_len]
                    extended_attention_mask = attention_mask[:, None, None, :]
                    extended_attention_mask = extended_attention_mask.to(dtype=hidden_states.dtype)
                    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
                else:
                    extended_attention_mask = None
                    
                hidden_states = block(hidden_states, attention_mask=extended_attention_mask, use_cache=False)[0]

            hidden_states = self.transformer.ln_f(hidden_states)
            logits = self.lm_head(hidden_states)

            return type('TailOutput', (), {'logits': logits})()

    
    head_model = HeadModel(model, head_layers)
    body_model = BodyModel(model, head_layers, body_layers)
    tail_model = TailModel(model, head_layers + body_layers)

    tail_model.lm_head.weight = head_model.wte.weight
    
    return head_model, body_model, tail_model



class ServerModel:
    """Server component handling the body layers"""
    def __init__(self, body_model, learning_rate=2e-4):
        self.body_model = body_model.to(device)
        self.optimizer = optim.AdamW(
            [p for p in self.body_model.parameters() if p.requires_grad], 
            lr=learning_rate
        )
        
    def forward(self, activations, attention_mask=None):
        """Forward pass through body layers (inference mode)"""
        self.body_model.eval()
        with torch.no_grad():
            output = self.body_model(hidden_states=activations, attention_mask=attention_mask)
            return output.last_hidden_state
    
    def forward_train(self, activations, attention_mask=None):
        """Forward pass during training"""
        self.body_model.train()
        # Don't detach - maintain gradient connection
        activations.requires_grad_(True)
        output = self.body_model(hidden_states=activations, attention_mask=attention_mask)
        return output.last_hidden_state, activations
    
    def backward(self, body_output, body_grad, head_activations):
        """FIXED: Add retain_grad() and gradient clipping"""
        self.optimizer.zero_grad()
        
        # FIX: Add retain_grad() BEFORE accessing .grad
        head_activations.requires_grad_(True)
        head_activations.retain_grad()  # CRITICAL: Add this line
        
        if body_grad is not None:
            # Backward through body layers
            torch.autograd.backward(
                tensors=[body_output],
                grad_tensors=[body_grad],
                retain_graph=True
            )
            
            # Now safely access .grad
            head_grad = head_activations.grad.clone() if head_activations.grad is not None else torch.zeros_like(head_activations)
        else:
            head_grad = torch.zeros_like(head_activations)
        
        # FIX: Add gradient clipping
        torch.nn.utils.clip_grad_norm_(self.body_model.parameters(), max_norm=1.0)
        
        # Update body parameters
        self.optimizer.step()
        
        return head_grad

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
    
    def backward(self, head_activations, head_grad):
        """ESSENTIAL: Backward pass for split learning"""
        self.optimizer.zero_grad()
        
        # Apply gradients received from body
        if head_grad is not None:
            torch.autograd.backward(
                tensors=[head_activations],
                grad_tensors=[head_grad],
                retain_graph=False
            )
        
        # Update head parameters
        self.optimizer.step()


class TailClient:
    """Client component handling tail layers"""
    def __init__(self, tail_model, learning_rate=2e-4):
        self.tail_model = tail_model.to(device)
        self.optimizer = optim.AdamW(
            [p for p in self.tail_model.parameters() if p.requires_grad], 
            lr=learning_rate
        )
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        
    def forward(self, body_activations, attention_mask=None):
        """Forward pass through tail layers"""
        output = self.tail_model(inputs_embeds=body_activations, attention_mask=attention_mask)
        return output.logits
    
    def compute_loss_and_backward(self, body_activations, labels, attention_mask=None):
        """FIXED: Add retain_grad() for non-leaf tensors"""
        self.optimizer.zero_grad()
        
        # FIX: Add retain_grad() BEFORE accessing .grad
        body_activations.requires_grad_(True)
        body_activations.retain_grad()  # CRITICAL: Add this line
        
        # Forward pass
        logits = self.tail_model(inputs_embeds=body_activations, attention_mask=attention_mask).logits
        logits = torch.clamp(logits, -50.0, 50.0)

        # Compute loss 
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()               # existing
        shift_labels[shift_labels == -100] = self.loss_fn.ignore_index

        
        # Check for NaN in logits
        if torch.isnan(shift_logits).any():
            print("WARNING: NaN detected in logits!")
            return 0.0, torch.zeros_like(body_activations)
        
        loss = self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # Check for NaN loss
        if torch.isnan(loss):
            print("WARNING: NaN loss detected!")
            return 0.0, torch.zeros_like(body_activations)
        
        # Backward pass
        loss.backward(retain_graph=True)
        
        # Now safely access .grad
        body_grad = body_activations.grad.clone() if body_activations.grad is not None else torch.zeros_like(body_activations)
        
        # Update tail parameters
        self.optimizer.step()
        
        return loss.item(), body_grad



class SplitLoRATrainer:
    def __init__(self,
                model_name="gpt2",
                head_layers=2,
                tail_layers=2,
                learning_rate=2e-4,
                warmup_steps=500,
                max_epochs=5):
        
        # FIXED: Use existing tokens only - no custom token addition
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        full_model = GPT2LMHeadModel.from_pretrained('gpt2')
        
        # Use existing tokens instead of adding new ones
        self.DELIM = " ||| "  # Use existing characters - no vocab corruption
        self.PAD = self.tokenizer.eos_token  # Use eos_token as pad
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"
        
        # Set generation config with existing vocabulary
        full_model.config.eos_token_id = self.tokenizer.eos_token_id
        full_model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Keep vocab unchanged - no resize_token_embeddings!
        vocab = len(self.tokenizer)  # Should be exactly 50257
        full_model.config.vocab_size = vocab
        
        # Split model (no custom tokens to corrupt)
        head_model, body_model, tail_model = split_gpt2(full_model, head_layers, tail_layers)
        
        # Apply LoRA/DoRA to clean models
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            use_dora=True,
            task_type="CAUSAL_LM",
            target_modules=["c_attn", "c_proj"]
        )
        
        head_model = get_peft_model(head_model, lora_config)
        body_model = get_peft_model(body_model, lora_config)
        tail_model = get_peft_model(tail_model, lora_config)
        
        # Standard weight tying
        tail_model.base_model.lm_head.weight = head_model.base_model.wte.weight
        
        # Initialize components
        self.server = ServerModel(body_model, learning_rate)
        self.head_client = HeadClient(head_model, learning_rate)
        self.tail_client = TailClient(tail_model, learning_rate)
        
        self.metrics = {"loss": []}
        self._sched_steps = None
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        self.schedulers = []

     
        
    def load_e2e_dataset(self, debug_mode=False):
        """Improved preprocessing with optional debug mode"""
        dataset = load_dataset("e2e_nlg", trust_remote_code=True)
        
        def preprocess(example):
            
            SEQUENCE_LENGTH = 128
            mr_text = example["meaning_representation"]
            ref_text = example["human_reference"]
            space_delim = " " + self.DELIM + " "  # Now uses " ||| " instead of " <|gen|> "
            full_text = mr_text + space_delim + ref_text
    
    
      
            # Tokenize full sequence
            encoding = self.tokenizer(
                full_text,
                max_length=SEQUENCE_LENGTH,
                truncation=True,
                padding="max_length",
                return_attention_mask=True
            )
            # Simple masking
            labels = encoding["input_ids"].copy()
            mr_tokens = self.tokenizer.encode(mr_text + space_delim, add_special_tokens=False)
            labels[:len(mr_tokens)] = [-100] * len(mr_tokens)
            labels = [ -100 if tok == self.tokenizer.pad_token_id else tok
                        for tok in labels ]        
            return {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "labels": labels,
                "human_reference": example["human_reference"],
                "meaning_representation": mr_text
            }
        
        train_ds = dataset["train"].map(preprocess, remove_columns=dataset["train"].column_names)
        test_ds = dataset["test"].map(preprocess, remove_columns=dataset["test"].column_names)
        
        # DEBUG MODE: Use only tiny subset
        if debug_mode:
            print("🐛 DEBUG MODE: Using tiny dataset subset")
            print(f"   - Sequence length: 64")
            print(f"   - Training samples: 2000 (instead of {len(train_ds):,})")
            print(f"   - Test samples: 200 (instead of {len(test_ds):,})")
            train_ds = train_ds.select(range(2000))  # Only 2000 samples!
            test_ds = test_ds.select(range(200))     # Only 200 samples!
        
        return train_ds, test_ds


    
    def create_dataloader(self, dataset, batch_size=8, shuffle=True, debug_mode=False):
        """FIXED: Consistent sequence length with debug support"""
        def collate_fn(batch):
            FIXED_LENGTH = 128  # Match preprocessing length!
            
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
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    
    def attach_schedulers(self, train_dataloader):
        if self._sched_steps is None:
            total_steps = len(train_dataloader) * self.max_epochs
            self._sched_steps = total_steps
            for opt in (self.head_client.optimizer,
                        self.server.optimizer,
                        self.tail_client.optimizer):
                sched = get_linear_schedule_with_warmup(
                            opt,
                            num_warmup_steps=self.warmup_steps,
                            num_training_steps=total_steps)
                self.schedulers.append(sched)


    def train(self, train_dataloader, epochs=1):
        """FIXED: Add gradient clipping and NaN checking"""
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
                # DEBUG: inspect the first batch once per epoch
                if batch_idx == 0:
                    # decode first two label sequences (remove -100 paddings)
                    for k in range(min(2, batch["labels"].size(0))):
                        lbl_ids = [t for t in batch["labels"][k].tolist() if t != -100]
                        print("   LABEL:", self.tokenizer.decode(lbl_ids))
                try:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    print(f"Input attention mask sum: {attention_mask.sum()}")
                    print(f"Attention mask shape: {attention_mask.shape}")
                    print(f"Pad token positions: {(input_ids == self.tokenizer.pad_token_id).sum()}")

                    
                    # FIX: Check for NaN inputs
                    if torch.isnan(input_ids.float()).any() or torch.isnan(labels.float()).any():
                        print(f"Skipping batch {batch_idx} due to NaN inputs")
                        continue
                    
                    # Forward through pipeline
                    head_activations = self.head_client.forward(input_ids, attention_mask)
                    body_activations, head_activations_stored = self.server.forward_train(
                        head_activations, attention_mask
                    )

                    loss, body_grad = self.tail_client.compute_loss_and_backward(
                        body_activations, labels, attention_mask
                    )

                    
                    # FIX: Check for NaN loss
                    if math.isnan(loss):
                        print(f"NaN loss at batch {batch_idx}, skipping...")
                        continue
                    
                    # Backward through body and head
                    
                    # FIXED: Use the correct variable name
                    head_grad = self.server.backward(body_activations, body_grad, head_activations_stored)

                    self.head_client.backward(head_activations, head_grad)
                    for sched in self.schedulers:
                        sched.step()
                    
                    # FIX: Add gradient clipping to all components
                    torch.nn.utils.clip_grad_norm_(self.head_client.head_model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.tail_client.tail_model.parameters(), max_norm=1.0)
                    
                    total_loss += loss
                    num_batches += 1
                    
                    
                    if batch_idx % 100 == 0:
                        print(f"Batch {batch_idx}, Loss: {loss:.4f}")
                    
                except Exception as e:
                    print(f"Training error: {e}")
                    traceback.print_exc()
                    continue
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            self.metrics["loss"].append(avg_loss)
            print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        print("Training completed!")    
    
            

    
    def save_checkpoint(self, path="./splitlora_checkpoint"):
        """FIXED: Proper PEFT model saving"""
        os.makedirs(path, exist_ok=True)
        
        try:
            # Save PEFT models properly (not base models)
            self.head_client.head_model.save_pretrained(
                os.path.join(path, "head_model"),
                save_embedding_layers=True,  # Important for custom tokens
                save_config=True
            )
            
            self.server.body_model.save_pretrained(
                os.path.join(path, "body_model"),
                save_embedding_layers=False,  # No embeddings in body
                save_config=True
            )
            
            self.tail_client.tail_model.save_pretrained(
                os.path.join(path, "tail_model"),
                save_embedding_layers=True,  # Important for LM head
                save_config=True
            )
            
            # Save optimizers
            torch.save(self.head_client.optimizer.state_dict(), os.path.join(path, "head_optimizer.pt"))
            torch.save(self.server.optimizer.state_dict(), os.path.join(path, "body_optimizer.pt"))
            torch.save(self.tail_client.optimizer.state_dict(), os.path.join(path, "tail_optimizer.pt"))
            
            # Save tokenizer (critical for vocab consistency)
            self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))
            
            # Save training metadata
            metadata = {
                "vocab_size": len(self.tokenizer),
                "delim_token": self.DELIM,
                "pad_token": self.PAD,
                "metrics": self.metrics,
                "model_config": {
                    "head_layers": 2,
                    "tail_layers": 2,
                    "body_layers": 8
                }
            }
            
            with open(os.path.join(path, "training_metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Checkpoint saved successfully to {path}")
            return path
            
        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")
            traceback.print_exc()
            return None

    
    def load_checkpoint(self, path="./splitlora_checkpoint"):
        """FIXED: Proper PEFT model loading without nesting"""
        if not os.path.exists(path):
            print(f"❌ Checkpoint path {path} does not exist")
            return False

        try:
            # Load metadata first to verify compatibility
            metadata_path = os.path.join(path, "training_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                print(f"Loading checkpoint with vocab_size: {metadata['vocab_size']}")
            
            # CRITICAL: Don't use PeftModel.from_pretrained on already-PEFT models
            # Instead, load the adapter weights directly
            
            # Method 1: Load adapter weights manually
            head_adapter_path = os.path.join(path, "head_model", "adapter_model.safetensors")
            body_adapter_path = os.path.join(path, "body_model", "adapter_model.safetensors") 
            tail_adapter_path = os.path.join(path, "tail_model", "adapter_model.safetensors")
            
            if all(os.path.exists(p) for p in [head_adapter_path, body_adapter_path, tail_adapter_path]):
                # Load adapter weights directly
                from safetensors.torch import load_file
                
                head_weights = load_file(head_adapter_path)
                body_weights = load_file(body_adapter_path)
                tail_weights = load_file(tail_adapter_path)
                
                # Load weights into existing PEFT models
                self.head_client.head_model.load_state_dict(head_weights, strict=False)
                self.server.body_model.load_state_dict(body_weights, strict=False)
                self.tail_client.tail_model.load_state_dict(tail_weights, strict=False)
                
            else:
                print("⚠️  Adapter files not found, trying alternative loading method...")
                return False
            
            # Ensure weight tying after loading
            self.tail_client.tail_model.base_model.lm_head.weight = self.head_client.head_model.base_model.wte.weight
            
            # Load optimizers
            self.head_client.optimizer.load_state_dict(torch.load(os.path.join(path, "head_optimizer.pt"), map_location=device))
            self.server.optimizer.load_state_dict(torch.load(os.path.join(path, "body_optimizer.pt"), map_location=device))
            self.tail_client.optimizer.load_state_dict(torch.load(os.path.join(path, "tail_optimizer.pt"), map_location=device))

            print(f"✅ Checkpoint loaded successfully from {path}")
            return True

        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            traceback.print_exc()
            return False


    


# ─── Beam-search helpers ──────────────────────────────────────────────
from evaluate import load as load_metric

def generate_with_beam(trainer, wrapper, mr_text, max_new_tokens=64):
    """Return one realisation for a single MR using SplitFM-style beam search."""
    prompt = mr_text + " " + trainer.DELIM + " "
    enc    = trainer.tokenizer(prompt, return_tensors="pt")
    ids, m = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    
    procs = LogitsProcessorList([
        MinLengthLogitsProcessor(
            min_length=10,  # Ensure at least 10 tokens generated
            eos_token_id=trainer.tokenizer.eos_token_id
        )
    ])

    with torch.no_grad():
        out = wrapper.generate(
                ids, 
                attention_mask=m,
                # FIXED PARAMETERS:
                max_new_tokens=64,          # Keep this
                min_length=ids.size(1) + 10,  # Ensure minimum output length
                length_penalty=0.7,
                early_stopping=True,
                no_repeat_ngram_size=3,     # Reduced from 4
                repetition_penalty=1.1,    # Reduced from 1.2
                diversity_penalty=0.0,
                
                # FIXED TOKEN IDs:
                eos_token_id=trainer.tokenizer.eos_token_id,  # Proper EOS
                pad_token_id=trainer.tokenizer.pad_token_id,  # Your custom pad
                bos_token_id=trainer.tokenizer.bos_token_id,  # Add BOS
                
                # REMOVE PROBLEMATIC PROCESSOR:
                # Remove the MinLengthLogitsProcessor from here
                remove_invalid_values=True,
                do_sample=False,            # Ensure deterministic beam search
                temperature=1.0,            # Keep neutral
            )
    
    return trainer.tokenizer.decode(out[0, ids.size(1):],
                                    skip_special_tokens=True).strip()


def evaluate_official(preds,
                      ref_file="references/e2e_refs.tsv"):
    # 1) write system outputs to a temp file
    with tempfile.NamedTemporaryFile('w', delete=False) as f:
        f.write("\n".join(preds) + "\n")
        sys_file = f.name

    repo = pathlib.Path(__file__).resolve().parent / "e2e-metrics"
    ref_path = repo / ref_file

    try:
        out = subprocess.check_output(
            ["python",
             str(repo / "measure_scores.py"),
             "--python",          # pure-Python BLEU/NIST
             "-t",               # TSV one-line output
             str(ref_path),
             sys_file],
            text=True
        )
    finally:
        #os.unlink(sys_file)
        pass  # keep the temp file for debugging

    # 2) take the last non-empty line
    last = [l for l in out.splitlines() if l.strip()][-1]
    fields = last.split("\t")          # 0 = sys name, 1..5 = scores

    return {
        "bleu":     float(fields[1]),
        "nist":     float(fields[2]),
        "meteor":   float(fields[3]),
        "rouge_l":  float(fields[4]),
        "cider":    float(fields[5])
    }


def generate_with_sampling(trainer, wrapper, mr_text, ref_text, max_new_tokens=40):
    """Use sampling instead of beam search to avoid repetition"""
    prompt = mr_text + " " + trainer.DELIM + " "
    enc = trainer.tokenizer(prompt, return_tensors="pt")
    ids, m = enc["input_ids"].to(device), enc["attention_mask"].to(device)

    with torch.no_grad():
        output = wrapper.generate(
            ids,
            attention_mask=m,
            num_beams = 5,  # Use single beam for sampling
            # SAMPLING APPROACH (avoids beam search repetition)
            max_new_tokens=30,
            do_sample=False,
            num_beam_groups = 2,
            # Sampling parameters
            top_k=40,
            top_p=0.85,
            temperature=1.2,
            
            # Still use some repetition control
            repetition_penalty=2.0,
            no_repeat_ngram_size=4,
            length_penalty=1.1,         # Encourage longer outputs
            diversity_penalty=1.0,      # Force diversity
            
            eos_token_id=trainer.tokenizer.eos_token_id,
            pad_token_id=trainer.tokenizer.pad_token_id,
        )
    
    result = trainer.tokenizer.decode(output[0][ids.size(1):], skip_special_tokens=True).strip()
    return result


def evaluate_beam(trainer, wrapper, dataset, n_samples=100):
    """Compute BLEU & METEOR on `n_samples` examples using beam search."""
    bleu   = load_metric("bleu")
    meteor = load_metric("meteor")
    eval_split = dataset.select(range(min(n_samples, len(dataset))))

    # ------------- generate once per MR, collect refs ----------------
    store = {}                                   # mr → {"pred": str, "refs": [str]}
    for sample in tqdm(eval_split, desc="Evaluating", unit="sample"):
        mr   = sample["meaning_representation"]
        ref  = sample["human_reference"]         # singular in the dataset
        # generate only the first time we meet this MR
        if mr not in store:
            try:
                pred = generate_with_beam_mbr(
                           trainer, wrapper, mr, ref)      # any ref is fine
            except Exception as e:
                print("generation failed:", e)
                pred = "empty"
            store[mr] = {"pred": pred, "refs": [ref]}
        else:
            store[mr]["refs"].append(ref)

    # ------------- prepare lists for evaluate.compute ----------------

    preds, refs, fails = [], [], 0
    

    for mr, bundle in store.items():
            if bundle["pred"] == "empty":
                fails += 1
                continue
            preds.append(bundle["pred"])
            refs.append(bundle["refs"])          # list-of-refs for this MR

    if not preds:
        return {"bleu": 0.0, "meteor": 0.0, "failed": len(store)}
    
    ref_sets = list(map(list, zip(*refs)))      # shape: n_refs × n_sents

    max_refs = max(len(r) for r in refs)
    ref_sets = [
        [sent_refs[i] if i < len(sent_refs) else ""          # fillvalue=""
        for sent_refs in refs]                              # traverse sentences
        for i in range(max_refs)                             # traverse ref indices
    ]

    official = evaluate_official(preds)           # uses the 9 refs
    print(f"OFFICIAL BLEU: {official['bleu']:.2f} • "
        f"NIST {official['nist']:.4f} • "
        f"ROUGE-L {official['rouge_l']:.2f}  •"
        f"Meteor {official['meteor']:.2f} • ")


    sb = SBLEU(tokenize="13a",             # WMT / leaderboard tokeniser
            smooth_method="exp",        # same smoothing as HF metric
            smooth_value=0.0,
            effective_order=True)       # ignore higher n if sent < n words

    sacre_score = sb.corpus_score(preds, ref_sets).score / 100

    # ------------- corpus-level multi-reference BLEU -----------------
    bleu_score = bleu.compute(predictions=preds,
                            references=refs,   # <-- list-of-lists
                            smooth=True)["bleu"]

    meteor_score = meteor.compute(predictions=preds,
                                references=[r[0] for r in refs])["meteor"]

    print(f"BLEU  : {bleu_score:.4f}  •  METEOR: {meteor_score:.4f}  •  SBLUE {sacre_score:.4f}  • failed {fails}/{len(store)}")
    return {"bleu": bleu_score, "meteor": meteor_score, "failed": fails, "sacrebleu": sacre_score, **official}

# ─── MBR beam search ────────────────────────────────────────────────
def generate_with_beam_mbr(trainer, wrapper, mr_text, ref_text,
                           max_new_tokens=64, k=10):
    """
    Return the BLEU-best candidate among the top-k beams (MBR reranking).
    """
    prompt = mr_text + " " + trainer.DELIM + " "
    enc    = trainer.tokenizer(prompt, return_tensors="pt")
    ids, m = enc["input_ids"].to(device), enc["attention_mask"].to(device)

    with torch.no_grad():
        beams = wrapper.generate(
            ids, 
            attention_mask=m,
            max_new_tokens=30,
            min_length=ids.size(1) + 8, 
            num_beams=k,
            num_return_sequences=k,
            early_stopping=False,        # Don't stop early
            no_repeat_ngram_size=4,      # Reduced to allow some repetition
            repetition_penalty=2.00,     # Mild penalty
            length_penalty=0.9,             # Slight length preference
            diversity_penalty=1.5,          # Force beam diversit
            # FIXED: Token handling
            eos_token_id=trainer.tokenizer.eos_token_id,
            pad_token_id=trainer.tokenizer.pad_token_id,
            
            # FIXED: Generation behavior
            do_sample=False,
            temperature=1.0,
            remove_invalid_values=True,
        )
    # DEBUG: Check what was generated
    print(f"Generated {len(beams)} total sequences")
    # Extract only the generated parts (after the prompt)
    candidates = []
    for i, beam in enumerate(beams):
        generated_part = beam[ids.size(1):]  # Remove prompt
        candidate = trainer.tokenizer.decode(generated_part, skip_special_tokens=True).strip()
        
        print(f"Beam {i}: '{candidate}'")
        
        # FIXED: Filter out very short candidates
        if len(candidate.split()) >= 3:  # At least 3 words
            candidates.append(candidate)
        else:
            print(f"  -> Filtered out (too short)")
    
    # If no valid candidates, return fallback
    if not candidates:
        print("No valid candidates found, returning fallback")
        return "the restaurant has a high rating"
    
    # FIXED: Safe MBR reranking
    if len(candidates) == 1:
        return candidates[0]
    
    # decode beams to strings
    prompt_len = ids.size(1)              # tokens that belong to the MR
    cand_txt = [trainer.tokenizer.decode(
                seq[prompt_len:],       # keep everything *after* the prompt
                skip_special_tokens=True).strip()
                for seq in beams]

    bleu = load_metric("bleu")            # already imported in final.py
    scores = [bleu.compute(predictions=[c], references=[[ref_text]])["bleu"]
              for c in cand_txt]
    
    print(f"Generated {len(beams)} beams")
    for i, beam in enumerate(beams[:3]):  # Show first 3 beams
        decoded = trainer.tokenizer.decode(beam, skip_special_tokens=False)
        print(f"Beam {i}: {decoded}")
        
        # Show just the generated part
        gen_part = trainer.tokenizer.decode(beam[ids.size(1):], skip_special_tokens=True)
        print(f"Generated part {i}: '{gen_part}'")

    best = cand_txt[int(np.argmax(scores))]
    return best


def test_simple_greedy(trainer, wrapper, mr_text):
    """Test with simple greedy generation"""
    prompt = mr_text + " " + trainer.DELIM + " "
    enc = trainer.tokenizer(prompt, return_tensors="pt")
    ids, m = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    
    print(f"Greedy input: '{prompt}'")
    
    with torch.no_grad():
        output = wrapper.generate(
            ids,
            attention_mask=m,
            max_new_tokens=20,
            do_sample=False,
            repetition_penalty=1.8,     # Strong penalty even for greedy
            no_repeat_ngram_size=3,
            eos_token_id=trainer.tokenizer.eos_token_id,
            pad_token_id=trainer.tokenizer.pad_token_id,
        )
    
    result = trainer.tokenizer.decode(output[0][ids.size(1):], skip_special_tokens=True).strip()
    print(f"Greedy result: '{result}'")
    return result





def main():
    parser = argparse.ArgumentParser(description="SplitLoRA Single File Implementation")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_epochs",  type=int, default=5)
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to checkpoint to load")
    parser.add_argument("--save_path", type=str, default="./splitlora_checkpoint", help="Path to save checkpoint")
    parser.add_argument("--gpu_device", type=str, default="1", help="GPU device to use")
    args = parser.parse_args()
    
    trainer = SplitLoRATrainer(model_name="gpt2",
                           head_layers=2,
                           tail_layers=2,
                           learning_rate=args.learning_rate,
                           warmup_steps=args.warmup_steps,
                           max_epochs=args.max_epochs)
    
    diagnose_tokenizer_corruption(trainer)
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        trainer.load_checkpoint(args.load_checkpoint)
    
    wrapper = SplitGPT2ForGeneration(
            tokenizer   = trainer.tokenizer,
            head_client = trainer.head_client,
            server      = trainer.server,
            tail_client = trainer.tail_client,
            base_config = trainer.head_client.head_model.config
         ).to(device).eval()

    wrapper.config.vocab_size = len(trainer.tokenizer)          # 50 259
    if hasattr(wrapper, "generation_config"):
        wrapper.generation_config.vocab_size = len(trainer.tokenizer)
    
    # Load dataset (regular mode)
    train_ds, test_ds = trainer.load_e2e_dataset(debug_mode=False)
    
    if args.eval_only:
        # quick manual check on one MR
        example_mr = "name[Blue Spice], eatType[coffee shop], area[city centre]"
        example_ref = "Blue Spice is a coffee shop in the city centre."
        print("Sampling output:", generate_with_sampling(trainer, wrapper, example_mr, example_ref))
        print("=== SIMPLE GREEDY TEST ===")
        greedy_result = test_simple_greedy(trainer, wrapper, example_mr)

        train_ds, test_ds = trainer.load_e2e_dataset(debug_mode=False)
        
        results = evaluate_beam(trainer, wrapper, test_ds, n_samples=len(test_ds))
        # Save evaluation results
        with open(os.path.join(args.save_path, "evaluation_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        return



    if not args.eval_only:
        # Create dataloader and train
        train_dl = trainer.create_dataloader(train_ds, batch_size=args.batch_size, shuffle=True, debug_mode=False)
        trainer.attach_schedulers(train_dl)
        print(f"Vocab size: {len(trainer.tokenizer)}")
        print(f"DELIM token: '{trainer.DELIM}' -> {trainer.tokenizer.encode(trainer.DELIM)}")
        print(f"PAD token: '{trainer.PAD}' -> {trainer.tokenizer.pad_token_id}")
        print(f"EOS token: '{trainer.tokenizer.eos_token}' -> {trainer.tokenizer.eos_token_id}")
        trainer.train(train_dl, epochs=args.epochs)
        
        # Save checkpoint
        trainer.save_checkpoint(args.save_path)


if __name__ == "__main__":
    main()

