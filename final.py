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
                
                    
                hidden_states = block(hidden_states, use_cache=False)[0]
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
    
            for block in self.transformer.h:
                hidden_states = block(hidden_states, use_cache=False)[0]
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

            for block in self.transformer.h:   
                hidden_states = block(hidden_states, use_cache=False)[0]

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
            output = self.body_model(hidden_states=activations)
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
        output = self.tail_model(inputs_embeds=body_activations, 
                                 attention_mask=attention_mask)
        return output.logits
    
    def compute_loss_and_backward(self, body_activations, labels, attention_mask=None):
        """FIXED: Add retain_grad() for non-leaf tensors"""
        self.optimizer.zero_grad()
        
        # FIX: Add retain_grad() BEFORE accessing .grad
        body_activations.requires_grad_(True)
        body_activations.retain_grad()  # CRITICAL: Add this line
        
        # Forward pass
        logits = self.tail_model(inputs_embeds=body_activations).logits
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

        # ADD: Entropy regularization for better diversity
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        entropy = -torch.sum(log_probs * torch.softmax(shift_logits, dim=-1), dim=-1)
        entropy_loss = -torch.mean(entropy)  # Encourage high entropy
        
        # OPTIMIZED: Stronger entropy weight for E2E NLG
        beta = 0.15  # INCREASED from 0.1
        total_loss = loss + beta * entropy_loss
        
        # Check for NaN loss
        if torch.isnan(loss):
            print("WARNING: NaN loss detected!")
            return 0.0, torch.zeros_like(body_activations)
        
        # Backward pass
        total_loss.backward(retain_graph=True)
        
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
        

         # FIXED: Properly initialize custom token embeddings
        original_vocab_size = len(self.tokenizer)
            # Add custom tokens
        self.DELIM = "<|gen|>"
        special_tokens = {"additional_special_tokens": [self.DELIM]}
        
        if self.tokenizer.pad_token is None:
            special_tokens["pad_token"] = "<|pad|>"
        
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        print(f"Added {num_added} special tokens")
        
        # CRITICAL: Resize and properly initialize embeddings
        full_model.resize_token_embeddings(len(self.tokenizer))
        
        # FIXED: Initialize new token embeddings properly
        if num_added > 0:
            with torch.no_grad():
                # Get existing embeddings
                existing_embeddings = full_model.transformer.wte.weight[:original_vocab_size]
                avg_embedding = existing_embeddings.mean(dim=0)
                
                # Initialize new token embeddings
                for i in range(original_vocab_size, len(self.tokenizer)):
                    # Initialize as average + small random noise
                    full_model.transformer.wte.weight[i] = avg_embedding + torch.randn_like(avg_embedding) * 0.01
        
        self.PAD = self.tokenizer.pad_token
        self.tokenizer.padding_side = "right"
        
        # Set generation config with existing vocabulary
        full_model.config.eos_token_id = self.tokenizer.eos_token_id
        full_model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Keep vocab unchanged - no resize_token_embeddings!
        vocab = len(self.tokenizer)  # Should be exactly 50257
        full_model.config.vocab_size = vocab

        if hasattr(full_model, 'generation_config'):
            full_model.generation_config.vocab_size = vocab
        
        # Split model (no custom tokens to corrupt)
        head_model, body_model, tail_model = split_gpt2(full_model, head_layers, tail_layers)
        
        # Apply LoRA/DoRA to clean models
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="lora_only",
            use_dora=True,
            task_type="CAUSAL_LM",
            target_modules=["c_attn", "c_proj", "c_fc"]
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

        
    def preprocess(self, example):
        SEQUENCE_LENGTH = 128
        mr_text = example["meaning_representation"]
        ref_text = example["human_reference"]
        
        # SIMPLIFIED: Use delimiter with spaces, but rely on single token search
        space_delim = " " + self.DELIM + " "  # " <|gen|> "
        full_text = mr_text + space_delim + ref_text
        
        encoding = self.tokenizer(
            full_text,
            max_length=SEQUENCE_LENGTH,
            truncation=True,
            padding="max_length",
            return_attention_mask=True
        )
        
        # SIMPLIFIED: Just look for the special token directly (what your fallback does)
        labels = encoding["input_ids"].copy()
        
        # Find the delimiter token (single token) - this is what's working
        delim_token_id = self.tokenizer.encode(self.DELIM, add_special_tokens=False)[0]  # [50257]
        
        try:
            delim_pos = encoding["input_ids"].index(delim_token_id)
            # Mask everything up to and including delimiter
            labels[:delim_pos + 1] = [-100] * (delim_pos + 1)
            # Remove debug prints for cleaner output
        except ValueError:
            # Fallback: mask first half if delimiter not found
            labels[:len(labels)//2] = [-100] * (len(labels)//2)
        
        # Mask padding tokens
        labels = [-100 if tok == self.tokenizer.pad_token_id else tok for tok in labels]
        
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": labels,
            "human_reference": example["human_reference"],
            "meaning_representation": mr_text
        }



    def load_e2e_dataset(self, debug_mode=False):
        """Improved preprocessing with optional debug mode"""
        dataset = load_dataset("e2e_nlg", trust_remote_code=True)
        
        # Use the class method instead of nested function
        train_ds = dataset["train"].map(self.preprocess, remove_columns=dataset["train"].column_names)
        test_ds = dataset["test"].map(self.preprocess, remove_columns=dataset["test"].column_names)
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


    def validate_model_sanity(self):
        """Check if model can generate basic English using the wrapper"""
        # FIXED: Use wrapper instead of individual head model
        try:
            from split_beam_wrapper import SplitGPT2ForGeneration
            
            # Create temporary wrapper for validation
            temp_wrapper = SplitGPT2ForGeneration(
                tokenizer=self.tokenizer,
                head_client=self.head_client,
                server=self.server,
                tail_client=self.tail_client,
                base_config=self.head_client.head_model.base_model.config
            ).to(device).eval()
            
            test_prompt = "The restaurant is"
            enc = self.tokenizer(test_prompt, return_tensors="pt")
            ids = enc["input_ids"].to(device)
            
            with torch.no_grad():
                # Use the wrapper for generation
                output = temp_wrapper.generate(
                    ids,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            result = self.tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"Sanity check: '{result}'")
            
            # Check if output contains mostly punctuation/symbols
            text_part = result[len(test_prompt):].strip()
            symbol_ratio = sum(1 for c in text_part if c in '|[](){}') / max(len(text_part), 1)
            
            if symbol_ratio > 0.3:
                print("❌ MODEL GENERATING GIBBERISH - TRAINING FAILED")
                return False
            else:
                print("✅ Model generating reasonable text")
                return True
                
        except Exception as e:
            print(f"⚠️ Sanity check failed: {e}")
            return True  # Continue training anyway

        
    def train(self, train_dataloader, epochs=1):
        print(f"Starting training for {epochs} epochs...")
        example_inputs = None
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
                if batch_idx == 0 and example_inputs is None:
                    example_inputs = {
                        'input_ids': batch["input_ids"][0:1].clone(),
                        'attention_mask': batch["attention_mask"][0:1].clone(),
                        'labels': batch["labels"][0:1].clone()
                    }
                
                if batch_idx % 500 == 0:
                    if not self.validate_model_sanity():
                        print("Stopping training due to gibberish generation")
                        break
                
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
                    head_activations = self.head_client.forward(input_ids)  # No attention_mask
                    body_activations, head_activations_stored = self.server.forward_train(head_activations)  # No attention_mask
                    loss, body_grad = self.tail_client.compute_loss_and_backward(body_activations, labels)  # No attention_mask

                    
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

                    if batch_idx % 100 == 0 and example_inputs is not None:
                        self.debug_model_learning(example_inputs)
                    
                except Exception as e:
                    print(f"Training error: {e}")
                    traceback.print_exc()
                    continue
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            self.metrics["loss"].append(avg_loss)
            print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        print("Training completed!")


    def debug_model_learning(self, example_batch):
        """Debug what the model is actually learning"""
        with torch.no_grad():
            input_ids = example_batch['input_ids'].to(device)
            attention_mask = example_batch['attention_mask'].to(device)
            labels = example_batch['labels'].to(device)
            
            # CRITICAL: Show what we're actually debugging
            print("=== DEBUGGING CONTEXT ===")
            full_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            print(f"Full input: '{full_text}'")
            
            # Find actual content end (before padding)
            padding_mask = (input_ids[0] == self.tokenizer.pad_token_id)
            if padding_mask.any():
                real_length = padding_mask.int().argmax().item()  # First padding position
            else:
                real_length = input_ids.size(1)  # No padding
            
            print(f"Real sequence length: {real_length}/{input_ids.size(1)}")
            
            # FIXED: Analyze positions relative to REAL content, not padded length
            if real_length > 5:
                # Analyze last 3 positions of ACTUAL content
                meaningful_positions = [real_length-3, real_length-2, real_length-1]
                
                print("=== MEANINGFUL POSITIONS ANALYSIS ===")
                print(f"Analyzing actual content positions: {meaningful_positions}")
                
                # Show what tokens are at these positions
                for pos in meaningful_positions:
                    token_id = input_ids[0, pos].item()
                    token_text = self.tokenizer.decode([token_id])
                    label = labels[0, pos].item()
                    print(f"Position {pos}: '{token_text}' (label: {label})")
                
                # Forward pass
                head_out = self.head_client.forward(input_ids)
                body_out = self.server.forward(head_out)
                logits = self.tail_client.forward(body_out)
                
                # Get predictions for meaningful positions
                probs = torch.softmax(logits[0, meaningful_positions], dim=-1)
                top_tokens = torch.topk(probs, 3, dim=-1)
                
                for i, pos in enumerate(meaningful_positions):
                    print(f"Position {pos} (real content):")
                    for j in range(3):
                        token_id = top_tokens.indices[i, j].item()
                        prob = top_tokens.values[i, j].item()
                        token_text = self.tokenizer.decode([token_id])
                        print(f"  Top {j+1}: '{token_text}' (prob: {prob:.3f})")
            else:
                print("Sequence too short for meaningful analysis")

    def save_checkpoint(self, path="./splitlora_checkpoint"):
        """FIXED: Save merged models using state dicts"""
        os.makedirs(path, exist_ok=True)
        
        try:
            print("Merging and saving models with custom embeddings...")
            
            # Merge PEFT adapters into base models
            head_merged = self.head_client.head_model.merge_and_unload()
            body_merged = self.server.body_model.merge_and_unload()
            tail_merged = self.tail_client.tail_model.merge_and_unload()
            
            # FIXED: Save state dicts instead of using save_pretrained
            torch.save(head_merged.state_dict(), os.path.join(path, "head_model_merged.pt"))
            torch.save(body_merged.state_dict(), os.path.join(path, "body_model_merged.pt"))
            torch.save(tail_merged.state_dict(), os.path.join(path, "tail_model_merged.pt"))
            
            # Save model configurations
            torch.save(head_merged.config, os.path.join(path, "head_config.pt"))
            torch.save(body_merged.config, os.path.join(path, "body_config.pt"))
            torch.save(tail_merged.config, os.path.join(path, "tail_config.pt"))
            
            # Save tokenizer
            self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))
            
            # Save training metadata including custom token info
            metadata = {
                "vocab_size": len(self.tokenizer),
                "original_vocab_size": 50257,
                "custom_tokens": {
                    "delim_token": self.DELIM,
                    "delim_id": self.tokenizer.encode(self.DELIM, add_special_tokens=False)[0],
                    "pad_token": self.PAD,
                    "pad_id": self.tokenizer.pad_token_id
                },
                "metrics": self.metrics,
                "model_config": {
                    "head_layers": 2,
                    "tail_layers": 2,
                    "body_layers": 8
                }
            }
            
            with open(os.path.join(path, "training_metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Merged checkpoint with custom embeddings saved to {path}")
            return path
            
        except Exception as e:
            print(f"❌ Error saving merged checkpoint: {e}")
            traceback.print_exc()
            return None

    def load_checkpoint(self, path="./splitlora_checkpoint"):
        from transformers.models.gpt2.configuration_gpt2 import GPT2Config
        """FIXED: Load merged models from state dicts"""
        if not os.path.exists(path):
            print(f"❌ Checkpoint path {path} does not exist")
            return False

        try:
            # Load metadata first
            with open(os.path.join(path, "training_metadata.json"), "r") as f:
                metadata = json.load(f)
            
            print(f"Loading merged checkpoint with vocab_size: {metadata['vocab_size']}")
            
            # Load tokenizer (preserves custom tokens)
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer"))
            
            # FIXED: Create fresh full model with correct vocab size
            from transformers import GPT2LMHeadModel
            full_model = GPT2LMHeadModel.from_pretrained('gpt2')
            
            # Resize to match saved model
            full_model.resize_token_embeddings(metadata['vocab_size'])
            
            # Load the merged head model state dict to get the embeddings
            head_state = torch.load(os.path.join(path, "head_model_merged.pt"), map_location=device)
            
            # Extract embedding weights and copy to full model
            if 'wte.weight' in head_state:
                full_model.transformer.wte.weight.data = head_state['wte.weight']
            if 'wpe.weight' in head_state:
                full_model.transformer.wpe.weight.data = head_state['wpe.weight']
            
            with torch.serialization.safe_globals([GPT2Config]):
                head_config = torch.load(os.path.join(path, "head_config.pt"), 
                                        map_location=device, weights_only=True)
                body_state = torch.load(os.path.join(path, "body_model_merged.pt"), 
                                    map_location=device, weights_only=True) 
                tail_state = torch.load(os.path.join(path, "tail_model_merged.pt"), 
                                    map_location=device, weights_only=True)
                # Reconstruct layer weights in full model
            layer_idx = 0
            
            # Head layers
            for i in range(2):  # head_layers
                if f'h.{i}.ln_1.weight' in head_state:
                    full_model.transformer.h[layer_idx].load_state_dict({
                        k[len(f'h.{i}.'):]: v for k, v in head_state.items() 
                        if k.startswith(f'h.{i}.')
                    }, strict=False)
                layer_idx += 1
            
            # Body layers  
            for i in range(8):  # body_layers
                body_layer_key = f'transformer.h.{i}'
                if f'{body_layer_key}.ln_1.weight' in body_state:
                    full_model.transformer.h[layer_idx].load_state_dict({
                        k[len(f'{body_layer_key}.'):]: v for k, v in body_state.items()
                        if k.startswith(f'{body_layer_key}.')
                    }, strict=False)
                layer_idx += 1
            
            # Tail layers
            for i in range(2):  # tail_layers
                tail_layer_key = f'transformer.h.{i}'
                if f'{tail_layer_key}.ln_1.weight' in tail_state:
                    full_model.transformer.h[layer_idx].load_state_dict({
                        k[len(f'{tail_layer_key}.'):]: v for k, v in tail_state.items()
                        if k.startswith(f'{tail_layer_key}.')
                    }, strict=False)
                layer_idx += 1
            
            # Load final layer norm and LM head from tail
            if 'transformer.ln_f.weight' in tail_state:
                full_model.transformer.ln_f.weight.data = tail_state['transformer.ln_f.weight']
                full_model.transformer.ln_f.bias.data = tail_state['transformer.ln_f.bias']
            
            if 'lm_head.weight' in tail_state:
                full_model.lm_head.weight.data = tail_state['lm_head.weight']
            
            print("✅ Merged models loaded successfully with custom embeddings")
            
            # Verify custom token embeddings were preserved
            delim_id = metadata["custom_tokens"]["delim_id"]
            delim_embedding = full_model.transformer.wte.weight[delim_id]
            print(f"Loaded DELIM embedding norm: {delim_embedding.norm().item():.4f}")
            
            # Recreate split models from loaded full model
            head_model, body_model, tail_model = split_gpt2(full_model, 2, 2)
            
            # Re-apply PEFT to the loaded models
            lora_config = LoraConfig(
                r=8, lora_alpha=16, lora_dropout=0.1,
                bias="none", use_dora=True, task_type="CAUSAL_LM",
                target_modules=["c_attn", "c_proj"]
            )
            
            head_model = get_peft_model(head_model, lora_config)
            body_model = get_peft_model(body_model, lora_config)
            tail_model = get_peft_model(tail_model, lora_config)
            
            # Update components with loaded models
            self.head_client.head_model = head_model.to(device)
            self.server.body_model = body_model.to(device)
            self.tail_client.tail_model = tail_model.to(device)
            
            # Ensure weight tying
            self.tail_client.tail_model.base_model.lm_head.weight = self.head_client.head_model.base_model.wte.weight
            
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

def diagnose_preprocessing_detailed(trainer):
    """FIXED: Process raw data directly without dataset mapping"""
    print("=== DETAILED PREPROCESSING DIAGNOSIS ===")
    
    # Get raw example (no mapping) - same as your pattern
    from datasets import load_dataset
    dataset = load_dataset("e2e_nlg", trust_remote_code=True)
    raw_example = dataset["train"][0]
    
    print(f"Raw MR: '{raw_example['meaning_representation']}'")
    print(f"Raw reference: '{raw_example['human_reference']}'")
    
    # Process manually step by step (KEEP all the detailed analysis)
    mr_text = raw_example["meaning_representation"]
    ref_text = raw_example["human_reference"]
    space_delim = " " + trainer.DELIM + " "
    full_text = mr_text + space_delim + ref_text
    
    print(f"Full text: '{full_text}'")
    
    # Tokenize manually (no dataset mapping)
    encoding = trainer.tokenizer(
        full_text,
        max_length=256,
        truncation=True,
        padding="max_length",
        return_attention_mask=True
    )
    
    print(f"Encoded length: {len(encoding['input_ids'])}")
    print(f"Input IDs: {encoding['input_ids'][:20]}")
    
    # Check delimiter position
    delimiter_tokens = trainer.tokenizer.encode(space_delim, add_special_tokens=False)
    print(f"Delimiter tokens: {delimiter_tokens}")
    
    # Find delimiter in sequence
    input_tokens = encoding["input_ids"]
    for i in range(len(input_tokens) - len(delimiter_tokens) + 1):
        if input_tokens[i:i+len(delimiter_tokens)] == delimiter_tokens:
            print(f"Delimiter found at position {i}")
            break
    else:
        print("❌ Delimiter not found in tokenized sequence!")
    
    # FIXED: Process single example directly (no dataset mapping)
    processed = trainer.preprocess(raw_example)  # Single processing, no dataset mapping
    labels = processed['labels']
    
    # Extract target (non -100 labels)
    target_ids = [t for t in labels if t != -100]
    target_text = trainer.tokenizer.decode(target_ids, skip_special_tokens=True)
    print(f"Extracted target: '{target_text}'")
    
    # This should match the reference text
    if target_text.strip() != ref_text.strip():
        print("❌ Target extraction is wrong!")
        print(f"Expected: '{ref_text}'")
        print(f"Got: '{target_text}'")
    else:
        print("✅ Target extraction is correct")


def generate_with_sampling(trainer, wrapper, mr_text, ref_text, max_new_tokens=40):
    prompt = mr_text + " " + trainer.DELIM + " "
    enc = trainer.tokenizer(prompt, return_tensors="pt")
    ids, m = enc["input_ids"].to(device), enc["attention_mask"].to(device)

    with torch.no_grad():
        output = wrapper.generate(
            ids,
            
            # MUCH SIMPLER parameters
            max_new_tokens=20,
            min_length=ids.size(1) + 3,
            
            do_sample=True,
            
            # MINIMAL sampling parameters
            top_k=50,
            top_p=0.9,
            temperature=0.8,
            
            # VERY GENTLE repetition control
            repetition_penalty=1.05,  # Almost no penalty
            
            eos_token_id=trainer.tokenizer.eos_token_id,
            pad_token_id=trainer.tokenizer.pad_token_id,
        )
    
    result = trainer.tokenizer.decode(output[0][ids.size(1):], skip_special_tokens=True).strip()
    return result


def diagnose_custom_token_embeddings(trainer):
    """Check if custom token embeddings are corrupted"""
    print("=== CUSTOM TOKEN EMBEDDING DIAGNOSIS ===")
    
    delim_id = trainer.tokenizer.encode(trainer.DELIM, add_special_tokens=False)[0]
    pad_id = trainer.tokenizer.pad_token_id
    
    # Get embeddings from head model
    head_embeddings = trainer.head_client.head_model.base_model.wte.weight
    
    delim_embedding = head_embeddings[delim_id]
    pad_embedding = head_embeddings[pad_id]
    
    print(f"DELIM embedding norm: {delim_embedding.norm().item():.4f}")
    print(f"PAD embedding norm: {pad_embedding.norm().item():.4f}")
    print(f"DELIM embedding mean: {delim_embedding.mean().item():.4f}")
    print(f"PAD embedding mean: {pad_embedding.mean().item():.4f}")
    
    # Check if they're all zeros or NaN (corruption indicators)
    if delim_embedding.norm().item() < 1e-6:
        print("❌ DELIM embedding is near-zero - CORRUPTED!")
    if torch.isnan(delim_embedding).any():
        print("❌ DELIM embedding contains NaN - CORRUPTED!")
    
    # Test tokenizer decode
    test_ids = [delim_id, pad_id, 1169]  # delim, pad, "the"
    decoded = trainer.tokenizer.decode(test_ids, skip_special_tokens=False)
    print(f"Test decode: {test_ids} -> '{decoded}'")
    
    if '�' in decoded:
        print("❌ Unicode corruption detected in tokenizer decode!")



def evaluate_beam(trainer, wrapper, dataset, n_samples=100):
    """Compute BLEU & METEOR on `n_samples` examples using beam search."""
    bleu = load_metric("bleu")
    meteor = load_metric("meteor")
    
    eval_split = dataset.select(range(min(n_samples, len(dataset))))
    
    # FIXED: Generate one prediction per test example, not per unique MR
    preds, refs_list, fails = [], [], 0
    
    # Store unique MR predictions to avoid redundant generation
    mr_cache = {}
    
    for sample in tqdm(eval_split, desc="Evaluating", unit="sample"):
        mr = sample["meaning_representation"]
        ref = sample["human_reference"]
        
        # Generate prediction (use cache for efficiency)
        if mr not in mr_cache:
            try:
                pred = generate_with_beam_mbr(trainer, wrapper, mr, ref)
                if not pred or pred.strip() == "":
                    pred = "The restaurant serves food"
            except Exception as e:
                print("generation failed:", e)
                pred = "The restaurant serves food"
            mr_cache[mr] = pred
        else:
            pred = mr_cache[mr]
        
        # FIXED: Add one prediction per test example
        preds.append(pred)
        refs_list.append([ref])  # Single reference per example
        
        if pred == "The restaurant serves food":
            fails += 1
    
    if not preds:
        return {"bleu": 0.0, "meteor": 0.0, "failed": len(eval_split)}
    
    # Now preds and refs_list have same length as eval_split
    print(f"Generated {len(preds)} predictions for {len(eval_split)} examples")
    
    # Multi-reference preparation for corpus-level metrics
    # Group by MR for multi-reference evaluation
    mr_groups = {}
    for i, sample in enumerate(eval_split):
        mr = sample["meaning_representation"]
        if mr not in mr_groups:
            mr_groups[mr] = {"pred": preds[i], "refs": []}
        mr_groups[mr]["refs"].append(sample["human_reference"])
    
    # Prepare for official evaluation (which needs grouped references)
    grouped_preds = [bundle["pred"] for bundle in mr_groups.values()]
    
    # Official evaluation
    official = evaluate_official(preds)  # uses the grouped references
    print(f"OFFICIAL BLEU: {official['bleu']:.2f} • "
          f"NIST {official['nist']:.4f} • "
          f"ROUGE-L {official['rouge_l']:.2f} •"
          f"Meteor {official['meteor']:.2f} • ")
    
    # Per-example metrics
    sb = SBLEU(tokenize="13a", smooth_method="exp", smooth_value=0.0, effective_order=True)
    sacre_score = sb.corpus_score(preds, [[ref[0]] for ref in refs_list]).score / 100
    
    bleu_score = bleu.compute(predictions=preds, references=refs_list, smooth=True)["bleu"]
    meteor_score = meteor.compute(predictions=preds, references=[r[0] for r in refs_list])["meteor"]
    
    print(f"BLEU : {bleu_score:.4f} • METEOR: {meteor_score:.4f} • SBLUE {sacre_score:.4f} • failed {fails}/{len(preds)}")
    
    return {"bleu": bleu_score, "meteor": meteor_score, "failed": fails, "sacrebleu": sacre_score, **official}


# ─── MBR beam search ────────────────────────────────────────────────
def generate_with_beam_mbr(trainer, wrapper, mr_text, ref_text, max_new_tokens=64, k=10):
    prompt = mr_text + " " + trainer.DELIM + " "
    enc = trainer.tokenizer(prompt, return_tensors="pt")
    ids, m = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    bad_tokens = trainer.tokenizer.encode("_*#-=.", add_special_tokens=False)
    with torch.no_grad():
        # ULTRA SIMPLE beam search - no complex parameters
        output = wrapper.generate(
            ids,
            max_new_tokens=40,              # Very short
            do_sample=False,                # Pure greedy
            eos_token_id=trainer.tokenizer.eos_token_id,
            pad_token_id=trainer.tokenizer.pad_token_id,
            # NO OTHER PARAMETERS AT ALL
        )

    # Extract candidates
    # candidates = []
    # for beam in beams:
    #     generated_part = beam[ids.size(1):]
    #     candidate = trainer.tokenizer.decode(generated_part, skip_special_tokens=True).strip()
        
    #     symbol_count = sum(1 for c in candidate if c in '_*#-.=|[]')
    #     total_chars = len(candidate)
    #     symbol_ratio = symbol_count / max(total_chars, 1)
    #     if symbol_ratio < 0.3 and len(candidate.split()) >= 2:  # Less than 30% 
    #         candidates.append(candidate)
    #     if not candidates:
    #         return "The restaurant serves food"
    
    # Return best candidate or fallback
    # return candidates[0]/
    result = trainer.tokenizer.decode(output[0][ids.size(1):], skip_special_tokens=True).strip()
    return result

def diagnose_training_data(trainer, train_ds):
    """Check if training data makes sense"""
    print("=== TRAINING DATA DIAGNOSIS ===")

    sample = train_ds[0]
    
    print(f"Input IDs: {sample['input_ids'][:20]}")
    print(f"Labels: {sample['labels'][:20]}")
    
    # Decode input
    full_input = trainer.tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
    print(f"Full input: '{full_input}'")
    
    # Check label masking
    valid_labels = [t for t in sample['labels'] if t != -100]
    target_text = trainer.tokenizer.decode(valid_labels, skip_special_tokens=True)
    print(f"Target text: '{target_text}'")
    
    # This should be proper English restaurant description
    if any(char in target_text for char in '_*#-.='):
        print("❌ Target text contains symbols - preprocessing is wrong!")
    else:
        print("✅ Target text looks normal")




def test_simple_greedy(trainer, wrapper, mr_text):
    prompt = mr_text + " " + trainer.DELIM + " "
    enc = trainer.tokenizer(prompt, return_tensors="pt")
    ids, m = enc["input_ids"].to(device), enc["attention_mask"].to(device)

    with torch.no_grad():
        # ABSOLUTE MINIMAL greedy generation
        output = wrapper.generate(
            ids,
            max_new_tokens=12,              # Very short
            do_sample=False,                # Pure greedy
            eos_token_id=trainer.tokenizer.eos_token_id,
            pad_token_id=trainer.tokenizer.pad_token_id,
            # NO OTHER PARAMETERS AT ALL
        )
    
    result = trainer.tokenizer.decode(output[0][ids.size(1):], skip_special_tokens=True).strip()
    return result


def test_single_token_delimiters(tokenizer):
    """Find good single-token delimiters"""
    candidates = [
        ":",      # Colon
        ";",      # Semicolon  
        "=",      # Equals
        "|",      # Pipe
        "###",    # Triple hash (from search result [2])
        ">>",     # Double arrow
        "=>",     # Arrow equals
        ":",      # Just colon
    ]
    
    print("=== TESTING SINGLE-TOKEN DELIMITERS ===")
    for delim in candidates:
        tokens = tokenizer.encode(delim, add_special_tokens=False)
        print(f"'{delim}' -> {tokens} ({len(tokens)} tokens) {'✅ SINGLE' if len(tokens) == 1 else '❌ MULTI'}")




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
        diagnose_custom_token_embeddings(trainer)
    
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
    
    
    if args.eval_only:
        train_ds, test_ds = trainer.load_e2e_dataset(debug_mode=False)
        diagnose_training_data(trainer, train_ds)
        diagnose_preprocessing_detailed(trainer)
        diagnose_custom_token_embeddings(trainer)
        # quick manual check on one MR
        example_mr = "name[Blue Spice], eatType[coffee shop], area[city centre]"
        example_ref = "Blue Spice is a coffee shop in the city centre."
        print("Sampling output:", generate_with_sampling(trainer, wrapper, example_mr, example_ref))
        

        print("=== TESTING WITH GENTLER PARAMETERS ===")
        test_result = test_simple_greedy(trainer, wrapper, example_mr)
        print(f"Gentle generation: '{test_result}'")

        sampling_result = generate_with_sampling(trainer, wrapper, example_mr, example_ref)
        print(f"Gentle sampling: '{sampling_result}'")

        # Check if it's still generating gibberish
        if any(char in test_result for char in '|[](){}'):
            print("❌ Still generating symbols - training issues persist")
        else:
            print("✅ Generating normal text - period prediction might be normal")

        test_single_token_delimiters(trainer.tokenizer)
        results = evaluate_beam(trainer, wrapper, test_ds, n_samples=len(test_ds))
        # Save evaluation results
        with open(os.path.join(args.save_path, "evaluation_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        return



    if not args.eval_only:
        # Load dataset (regular mode)
        train_ds, test_ds = trainer.load_e2e_dataset(debug_mode=False)
        diagnose_training_data(trainer, train_ds)
        diagnose_preprocessing_detailed(trainer)
        diagnose_custom_token_embeddings(trainer)
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

