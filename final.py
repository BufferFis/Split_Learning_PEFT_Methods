# splitlora_single.py
import os
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
            
            # SIMPLIFIED: Let GPT-2 handle attention masking internally
            all_hidden_states = ()
            for block in self.h:
                hidden_states = block(hidden_states, use_cache=False)[0]  # No attention_mask!
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
            self.transformer.ln_f = original_model.transformer.ln_f
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
            # SIMPLIFIED: Remove complex attention mask handling
            for block in self.transformer.h:
                hidden_states = block(hidden_states, use_cache=False)[0]  # No attention_mask!
            
            hidden_states = self.transformer.ln_f(hidden_states)
            return type('BodyOutput', (), {'last_hidden_state': hidden_states})()
    
    # Tail Model (last few layers + LM head)
    class TailModel(nn.Module):
        def __init__(self, original_model, start_layer):
            super().__init__()  # CRITICAL: Call parent constructor
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList(original_model.transformer.h[start_layer:])
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
            
            # SIMPLIFIED: Remove complex attention mask handling
            for block in self.transformer.h:
                hidden_states = block(hidden_states, use_cache=False)[0]  # No attention_mask!
            
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
        self.loss_fn = nn.CrossEntropyLoss()
        
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
    """Main trainer class combining all components"""
    def __init__(self, model_name="gpt2", head_layers=2, tail_layers=2, learning_rate=2e-4):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        
        # Load and split model
        full_model = AutoModelForCausalLM.from_pretrained(model_name)
        added = 0
        
        self.DELIM = "<|gen|>"                          # ONE token
        if self.DELIM not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [self.DELIM]}
            )
            full_model.resize_token_embeddings(len(self.tokenizer))
        
        self.PAD = "<|pad|>"
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": self.PAD})
            full_model.resize_token_embeddings(len(self.tokenizer))
        
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
        
    def load_e2e_dataset(self, debug_mode=False):
        """Improved preprocessing with optional debug mode"""
        dataset = load_dataset("e2e_nlg", trust_remote_code=True)
        
        def preprocess(example):
            SEQUENCE_LENGTH = 128
            
            mr_text = example["meaning_representation"]
            ref_text = example["human_reference"]
            
            space_delim = " " + self.DELIM + " "

            full_text = mr_text + space_delim + ref_text          
            
            
            # Tokenize full sequence
            encoding = self.tokenizer(
                full_text,
                max_length=SEQUENCE_LENGTH,
                truncation=True,
                padding="max_length"
            )
            
            # Simple masking
            labels = encoding["input_ids"].copy()
            mr_tokens = self.tokenizer.encode(mr_text + space_delim, add_special_tokens=False)
            labels[:len(mr_tokens)] = [-100] * len(mr_tokens)
            labels = [ -100 if tok == self.tokenizer.pad_token_id else tok
                        for tok in labels ]                       # NEW

            
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
                    
                    # FIX: Check for NaN inputs
                    if torch.isnan(input_ids.float()).any() or torch.isnan(labels.float()).any():
                        print(f"Skipping batch {batch_idx} due to NaN inputs")
                        continue
                    
                    # Forward through pipeline
                    head_activations = self.head_client.forward(input_ids, attention_mask)
                    body_activations, stored_head_activations = self.server.forward_train(
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
                    head_grad = self.server.backward(body_activations, body_grad, stored_head_activations)
                    self.head_client.backward(head_activations, head_grad)
                    
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
    
    def generate(self, prompt_ids, prompt_attention_mask, max_length=128, greedy=False):
        """FIXED: Generate continuation from MR prompt with proper causal masking"""
        with torch.no_grad():
            try:
                # ----- 1.  start with the prompt ---------------------------------
                # Start with the prompt (MR only)
                generated_ids = prompt_ids.clone()
                
                # Initialize attention mask with causal structure
                attention_mask = torch.ones_like(prompt_ids) 
                
                for step in range(max_length):
                    # ----- 2. forward through split pipeline ---------------------
                    head_acts = self.head_client.forward(generated_ids, attention_mask)
                    body_acts = self.server.forward(head_acts, attention_mask)
                    logits    = self.tail_client.forward(body_acts, attention_mask)
                    NO_REPEAT_N = 3 

                    # ----- 3. pick next token ------------------------------------
                    next_token_logits = logits[:, -1, :]                   # (1, vocab)
                    NO_REPEAT_N = 3          # size of n-gram ban
                    MAX_GREEDY_LEN = 70      # hard stop (GPT-2 tokens)
                    if greedy:
                        bs, cur_len = generated_ids.size()

                        # ------ 3-gram ban but keep EOS ‑----------------------------
                        if cur_len >= NO_REPEAT_N - 1:
                            for b in range(bs):
                                prefix = tuple(generated_ids[b, -(NO_REPEAT_N-1):].tolist())
                                blocked = {
                                    generated_ids[b, i+NO_REPEAT_N-1].item()
                                    for i in range(len(generated_ids[b]) - NO_REPEAT_N + 1)
                                    if tuple(generated_ids[b, i:i+NO_REPEAT_N-1].tolist()) == prefix
                                }
                                blocked.discard(self.tokenizer.eos_token_id)        # allow EOS
                                if blocked:
                                    next_token_logits[b, list(blocked)] = -float("inf")

                        # ------ one-shot presence penalty (last token only) ---------
                        last_tok = generated_ids[:, -1]                              # (bs,)
                        next_token_logits[torch.arange(bs), last_tok] -= 1.0

                        # ------ pick next token  -----------------------------------
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                        # ------ length cap *before* we append ----------------------
                        if cur_len >= MAX_GREEDY_LEN:
                            break
                    else:
                        # ---------------------------------------------------------
                        #  A. global repetition-penalty  (1.3 works well for GPT-2)
                        # ---------------------------------------------------------
                        REP_PENALTY = 1.3
                        for b in range(generated_ids.size(0)):               # bs is 1 here
                            prev_tokens = generated_ids[b]
                            next_token_logits[b, prev_tokens] /= REP_PENALTY

                        # ---------------------------------------------------------
                        #  B. no-repeat n-gram  (n = 3)
                        #     block any token that would form an already-seen trigram
                        # ---------------------------------------------------------
                        N = 3
                        for b in range(generated_ids.size(0)):
                            if generated_ids.size(1) >= N - 1:
                                prefix = tuple(generated_ids[b, - (N - 1):].tolist())
                                # collect all tokens that have followed this prefix before
                                blocked = set()
                                history = generated_ids[b].tolist()
                                for i in range(len(history) - N + 1):
                                    if tuple(history[i : i + N - 1]) == prefix:
                                        blocked.add(history[i + N - 1])
                                if blocked:
                                    next_token_logits[b, list(blocked)] = -float("inf")
                        
                        # ---------------------------------------------------------
                        #  final sampling step
                        # ---------------------------------------------------------
                        probs      = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, 1)


                    generated_ids  = torch.cat([generated_ids,  next_token], dim=1)

                    # ----- 4. extend the 2-D mask by one position ----------------
                    pad = torch.ones(1, 1, dtype=attention_mask.dtype, device=attention_mask.device)
                    attention_mask = torch.cat([attention_mask, pad], dim=1)

                    # ----- 5. stop conditions ------------------------------------
                    if next_token.item() in {self.tokenizer.eos_token_id,
                                            self.tokenizer.pad_token_id}:
                        break

                generated_tokens = generated_ids[0, prompt_ids.size(1):].tolist()
                return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                
                    
                
            except Exception as e:
                print(f"Generation error: {e}")
                traceback.print_exc()
                return "generation failed"


    
    def evaluate(self, test_dataset):
        """Evaluate model with proper error handling"""
        print("Starting evaluation...")
        
        try:
            # Load metrics
            bleu_metric = load_metric("bleu")
            meteor_metric = load_metric("meteor")
            
            preds, refs = [], []
            failed_generations = 0
            
            # Sample evaluation data
            eval_samples = test_dataset.select(range(100))
            
            for i, sample in enumerate(tqdm(eval_samples, desc="Evaluating")):
                try:
                    mr_text = sample["meaning_representation"]

                    space_delim = " " + self.DELIM + " "
                    prompt_text = mr_text + space_delim 
                    
                    # Tokenize only the MR
                    encoding = self.tokenizer(prompt_text, return_tensors="pt", padding=False, truncation=False)
                    input_ids = encoding["input_ids"].to(device)
                    attention_mask = encoding["attention_mask"].to(device)
                    
                    # Generate prediction from MR only
                    generated_text = self.generate(input_ids, attention_mask, max_length=80, greedy=True)
                    
                    # FIX: Check for valid generation
                    if generated_text and len(generated_text.strip()) > 0:
                        preds.append(generated_text.strip())
                        refs.append([sample["human_reference"]])
                    else:
                        failed_generations += 1
                        preds.append("empty")
                        refs.append([sample["human_reference"]])
                    
                    # Debug first few samples
                    if i < 3:
                        print(f"\nSample {i}:")
                        print(f"MR Input: {mr_text}")  # Show only MR
                        print(f"Reference: {sample['human_reference']}")
                        print(f"Generated: {generated_text}")
                        print("---")
                    
                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    failed_generations += 1
                    continue
            
            print(f"Failed generations: {failed_generations}/{len(eval_samples)}")
            
            if not preds or len([p for p in preds if p != "empty"]) == 0:
                print("No valid predictions generated")
                return {"bleu": 0.0, "meteor": 0.0, "error": "No valid samples"}
            
            # FIX: Filter out empty predictions for metric calculation
            valid_preds = []
            valid_refs = []
            for pred, ref in zip(preds, refs):
                if pred != "empty" and len(pred.strip()) > 0:
                    valid_preds.append(pred)
                    valid_refs.append(ref)
            
            if not valid_preds:
                return {"bleu": 0.0, "meteor": 0.0, "error": "No valid predictions after filtering"}
            
            print(f"Computing metrics on {len(valid_preds)} valid predictions...")
            
            # Calculate metrics with error handling
            try:
                # FIX: Add smoothing for BLEU to handle edge cases
                bleu_score = bleu_metric.compute(
                    predictions=valid_preds, 
                    references=valid_refs,
                    smooth=True  # Enable smoothing
                )
                bleu_value = bleu_score.get('bleu', 0.0) if isinstance(bleu_score, dict) else 0.0
            except Exception as bleu_error:
                print(f"BLEU computation failed: {bleu_error}")
                bleu_value = 0.0
            
            try:
                meteor_score = meteor_metric.compute(
                    predictions=valid_preds, 
                    references=[r[0] for r in valid_refs]
                )
                meteor_value = meteor_score.get('meteor', 0.0) if isinstance(meteor_score, dict) else 0.0
            except Exception as meteor_error:
                print(f"METEOR computation failed: {meteor_error}")
                meteor_value = 0.0
            
            print(f"BLEU Score: {bleu_value:.4f}")
            print(f"METEOR Score: {meteor_value:.4f}")
            
            results = {
                "bleu": bleu_value,
                "meteor": meteor_value,
                "num_samples": len(valid_preds),
                "failed_generations": failed_generations
            }
            
            return results
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            traceback.print_exc()
            return {"bleu": 0.0, "meteor": 0.0, "error": str(e)}
    
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
            # 1. be sure the tokenizer already contains the two tokens
            if "<|gen|>" not in self.tokenizer.get_vocab():
                self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|gen|>"]})
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

            # 2. load base model once
            full_model = AutoModelForCausalLM.from_pretrained("gpt2")

            # 3. ALWAYS resize if sizes differ
            if len(self.tokenizer) != full_model.get_input_embeddings().num_embeddings:
                full_model.resize_token_embeddings(len(self.tokenizer))
            
            # 4. now split and load LoRA weights
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
    

    def debug_generate_sample(self, sample_mr, max_length=64):
        """Quick generation test during training"""
        with torch.no_grad():
            try:
                # Tokenize the MR
                              # SAME delimiter as above
                space_delim = " " + self.DELIM + " "
                prompt = sample_mr + space_delim
                encoding = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
                input_ids = encoding["input_ids"].to(device)
                attention_mask = encoding["attention_mask"].to(device)
                
                # Generate using your existing generate method but shorter
                generated_text = self.generate(input_ids, attention_mask, max_length=max_length)
                
                return generated_text if generated_text else "[EMPTY]"
                
            except Exception as e:
                return f"[ERROR: {str(e)}]"

    def debug_train_and_test(self, train_dataloader, max_batches=50):
        """SUPER FAST DEBUG: Train a few batches and test generation"""
        max_batches = 50
        print(f"🐛 DEBUG TRAINING: Max {max_batches} batches")
        
        # Test samples for consistent monitoring
        test_samples = [
            ("name[Blue Spice], eatType[coffee shop], area[city centre]", 
            "Blue Spice is a coffee shop in the city centre."),
            ("name[Aromi], food[English], area[riverside]", 
            "Aromi is an English restaurant in the riverside area."),
        ]
        
        # Test initial generation (should be random)
        print("\n🔬 INITIAL GENERATION TEST (should be random):")
        for mr, expected in test_samples:
            generated = self.debug_generate_sample(mr, max_length=64)
            print(f"  MR: {mr}")
            print(f"  Generated: {generated}")
            print("---")
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Debug Training")):
            if batch_idx >= max_batches:  # Stop after max_batches
                break
                
            try:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                # Check for NaN inputs
                if torch.isnan(input_ids.float()).any() or torch.isnan(labels.float()).any():
                    print(f"❌ NaN inputs in batch {batch_idx}")
                    continue
                
                # Forward through pipeline
                head_activations = self.head_client.forward(input_ids, attention_mask)
                body_activations, stored_head_activations = self.server.forward_train(
                    head_activations, attention_mask
                )
                loss, body_grad = self.tail_client.compute_loss_and_backward(
                    body_activations, labels, attention_mask
                )
                
                # Check for NaN loss
                if math.isnan(loss):
                    print(f"❌ NaN loss at batch {batch_idx}")
                    continue
                
                # Backward through body and head
                head_grad = self.server.backward(body_activations, body_grad, stored_head_activations)
                self.head_client.backward(head_activations, head_grad)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.head_client.head_model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.tail_client.tail_model.parameters(), max_norm=1.0)
                
                total_loss += loss
                num_batches += 1
                
                print(f"✅ Batch {batch_idx}, Loss: {loss:.4f}")
                
            except Exception as e:
                print(f"❌ Training error at batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"\n🐛 Debug training completed! Average loss: {avg_loss:.4f}")
        
        # Test post-training generation
        print("\n🔬 POST-TRAINING GENERATION TEST:")
        for mr, expected in test_samples:
            generated = self.debug_generate_sample(mr, max_length=16)
            print(f"  MR: {mr}")
            print(f"  Expected: {expected}")
            print(f"  Generated: {generated}")
            
            # Quick quality check
            if generated == "[EMPTY]":
                print("  ❌ EMPTY - Model broken!")
            elif "wiz" in generated.lower() or "mcgee" in generated.lower():
                print("  ❌ REPETITIVE TOKENS - Model collapsed!")
            elif len(set(generated.split())) <= 2:
                print("  ❌ TOO REPETITIVE")
            else:
                print("  ✅ Shows variety")
            print("---")
        
        return avg_loss


# ──────────────────────────────────────────────────────────────
#  ONE helper = bullet-proof way to build a fused GPT-2 that
#  already contains the three Split-LoRA (+DoRA) adapters
# ──────────────────────────────────────────────────────────────


# ─── helper: build a single GPT-2 that already contains the
#             three LoRA + DoRA slices  ─────────────────────────


# -----------------------------------------------------------------
#  bullet-proof loader for the three SplitLoRA slices
# -----------------------------------------------------------------


def build_fused_splitlora(model_name: str,
                          ckpt_root : str,
                          tokenizer,
                          device     ):

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # resize once so <|gen|> / <|pad|> never overflow the embedding
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    # ― add HEAD adapter  → creates the first PEFT wrapper
    model = PeftModel.from_pretrained(
        model,
        os.path.join(ckpt_root, "head_model"),
        is_trainable=False,
        output_loading_info=False)

    # ― add BODY and TAIL without re-wrapping
    for name in ("body_model", "tail_model"):
        model.load_adapter(os.path.join(ckpt_root, name),
                           adapter_name=name,
                           is_trainable=False)

    # fuse low-rank + DoRA magnitudes into the base weights
    model = model.merge_and_unload().eval()

    # fix pad / eos ids so HF can build the attention mask
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    return model

def beam_generate_from_mr(mr        : str,
                          tokenizer,
                          ckpt_root : str,
                          max_new   : int = 64):

    prompt = mr + " " + "<|gen|>" + " "
    enc    = tokenizer(prompt, return_tensors="pt")
    ids    = enc["input_ids"].to(device)
    mask   = enc["attention_mask"].to(device)

    model  = build_fused_splitlora("gpt2", ckpt_root, tokenizer, device)

    with torch.no_grad():
        out = model.generate(
                ids,
                attention_mask      = mask,            # ← avoids warning
                max_new_tokens      = max_new,
                num_beams           = 10,
                length_penalty      = 0.8,
                no_repeat_ngram_size= 4,
                repetition_penalty  = 1.0,
                early_stopping      = True,
                eos_token_id        = tokenizer.eos_token_id,
                pad_token_id        = tokenizer.pad_token_id)

    return tokenizer.decode(
             out[0, ids.size(1):], skip_special_tokens=True).strip()

def main():
    parser = argparse.ArgumentParser(description="SplitLoRA Single File Implementation")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to checkpoint to load")
    parser.add_argument("--save_path", type=str, default="./splitlora_checkpoint", help="Path to save checkpoint")
    parser.add_argument("--gpu_device", type=str, default="1", help="GPU device to use")
    parser.add_argument("--debug", action="store_true", help="Ultra-fast debug mode")  # NEW
    
    args = parser.parse_args()
    
    
    
    
    # DEBUG MODE
    if args.debug:
        print("🐛 STARTING ULTRA-FAST DEBUG MODE")
        
        # Initialize trainer
        trainer = SplitLoRATrainer(learning_rate=3e-4)  # Slightly higher LR for debug
       

        # Load tiny dataset
        train_ds, test_ds = trainer.load_e2e_dataset(debug_mode=True)
        
        # Create dataloader with debug settings
        train_dl = trainer.create_dataloader(train_ds, batch_size=4, shuffle=True, debug_mode=True)
        
        # Debug training (10 batches only)
        avg_loss = trainer.debug_train_and_test(train_dl, max_batches=10)
        
        if not math.isnan(avg_loss) and avg_loss > 0:
            print("✅ Training is working! You can now run full training.")
        else:
            print("❌ Training failed - check your implementation.")
        
        return
    
    # REGULAR MODE (existing code)
    # Initialize trainer
    trainer = SplitLoRATrainer(learning_rate=1e-4)
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        trainer.load_checkpoint(args.load_checkpoint)
    
    wrapper = SplitGPT2ForGeneration(
            tokenizer   = trainer.tokenizer,
            head_client = trainer.head_client,
            server      = trainer.server,
            tail_client = trainer.tail_client,
            base_config = AutoModelForCausalLM.from_pretrained("gpt2").config
         ).to(device).eval()
    
    # Load dataset (regular mode)
    train_ds, test_ds = trainer.load_e2e_dataset(debug_mode=False)
    
    if args.eval_only:
        mr_text = "name[Blue Spice], eatType[coffee shop], area[city centre]"
        prompt  = mr_text + " " + trainer.DELIM + " "
        enc     = trainer.tokenizer(prompt, return_tensors="pt")
        ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)

        with torch.no_grad():
            out = wrapper.generate(
                    ids,
                    attention_mask        = mask,
                    max_new_tokens        = 64,
                    num_beams             = 10,
                    length_penalty        = 0.8,
                    no_repeat_ngram_size  = 4,
                    early_stopping        = True,
                    eos_token_id          = trainer.tokenizer.eos_token_id,
                    pad_token_id          = trainer.tokenizer.pad_token_id
                )

        print("Beam-10 output:",
            trainer.tokenizer.decode(out[0, ids.size(1):], skip_special_tokens=True).strip())
        return


    if not args.eval_only:
        # Create dataloader and train
        train_dl = trainer.create_dataloader(train_ds, batch_size=args.batch_size, shuffle=True, debug_mode=False)
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

