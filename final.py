# splitlora_single.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"     # or the bus-id / UUID of AF:00.0
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from evaluate import load as load_metric
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
import json
import argparse
import traceback
import math
from peft import PeftModel
# after the other imports in final.py
from split_beam_wrapper import SplitGPT2ForGeneration   # NEW
import copy
from transformers import LogitsProcessorList, MinLengthLogitsProcessor
import numpy as np
from sacrebleu.metrics import BLEU as SBLEU 
import subprocess, tempfile, pathlib, json
from transformers import get_linear_schedule_with_warmup
import random

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
                
                    
                hidden_states = block(hidden_states, attention_mask=attention_mask,use_cache=False)[0]
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
                hidden_states = block(hidden_states, attention_mask=attention_mask,use_cache=False)[0]
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
                hidden_states = block(hidden_states, attention_mask=attention_mask,use_cache=False)[0]

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
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
        
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
        beta = 0.01 
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
        
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # Re-use EOS as PAD so no new embedding is introduced
        self.tokenizer.pad_token = self.tokenizer.eos_token
        full_model = AutoModelForCausalLM.from_pretrained("gpt2")
        full_model.config.pad_token_id = self.tokenizer.eos_token_id
        self.max_seq_len = 256

         # FIXED: Properly initialize custom token embeddings
        original_vocab_size = len(self.tokenizer)
        self.PAD = self.tokenizer.pad_token
        self.tokenizer.padding_side = "right"
        self.DELIM         = ";"              # no spaces
        self.DELIM_TOKENS  = self.tokenizer.encode(self.DELIM, add_special_tokens=False)
        assert len(self.DELIM_TOKENS) == 1
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
        tail_model.lm_head.weight = head_model.wte.weight 
        # Apply LoRA/DoRA to clean models
        lora_config = LoraConfig(
            r=2,
            lora_alpha=32,
            lora_dropout=0.1,
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

    def _find_delim(self, seq):
        """Return index of LAST delimiter token or None if absent."""
        pat = self.DELIM_TOKENS
        plen = len(pat)
        for i in range(len(seq) - plen + 1):
            if seq[i:i + plen] == pat:
                return i + plen - 1                # last token index
        return None


    def preprocess(self, example, sequence_length=None):
        """
        Build a single training instance.
        The MR tokens come first, followed by the delimiter tokens,
        followed by the reference tokens.

        Anything beyond `sequence_length` is hard-truncated from the *end*,
        so the delimiter is never lost.
        """
        SEQ_LEN = sequence_length if sequence_length is not None else self.max_seq_len

        mr  = example["meaning_representation"]
        ref = example["human_reference"]

        # --- tokenise pieces independently ---------------------------------
        ids_mr   = self.tokenizer.encode(mr,  add_special_tokens=False)
        ids_ref  = self.tokenizer.encode(ref, add_special_tokens=False)
        ids_delim = self.DELIM_TOKENS                      # already built in __init__

        # -------------------------------------------------------------------
        # [MR] + [DELIM] + [REF]
        # -------------------------------------------------------------------
        input_ids = ids_mr + ids_delim + ids_ref

        # hard truncation from the right – ensures delimiter is always kept
        if len(input_ids) > SEQ_LEN:
            input_ids = input_ids[:SEQ_LEN]
            # if we chopped off part of the reference we must chop the same
            # amount from the labels to keep alignment
            chop = max(0, len(ids_mr) + len(ids_delim) + len(ids_ref) - SEQ_LEN)
            ids_ref = ids_ref[:-chop] if chop else ids_ref

        # -------------------------------------------------------------------
        # labels: mask everything up to and including the delimiter
        # -------------------------------------------------------------------
        labels = [-100] * (len(ids_mr) + len(ids_delim)) + ids_ref
        labels = labels[:SEQ_LEN]                        # may already be correct

        attention_mask = [1] * len(input_ids)            # no pad yet

        return {
            "input_ids":        input_ids,
            "attention_mask":   attention_mask,
            "labels":           labels,
            "human_reference":  ref,
            "meaning_representation": mr,
        }



    def load_e2e_dataset(
            self,
            *,                       # force keyword arguments
            debug_mode: bool = False,
            sequence_length: int = 512,
            cycle_refs: bool = True,
            seed: Optional[int] = None
        ):
        """
        Load and preprocess the E2E-NLG dataset.

        Parameters
        ----------
        debug_mode       – if True, return a 1 % slice for quick debugging  
        sequence_length  – fixed length used by `self.preprocess`  
        cycle_refs       – if True, shuffle training rows on every call
                        so different references are seen across epochs  
        seed             – optional shuffle seed (useful for reproducibility)

        Returns
        -------
        train_ds, test_ds – HuggingFace datasets ready for DataLoader
        """
        cycle_refs = True  # always shuffle train split by default
        ds = load_dataset("e2e_nlg", trust_remote_code=True)

        # ── optional tiny slice for lightning-fast tests ─────────────────
        if debug_mode:
            ds["train"] = ds["train"].select(range(max(50, len(ds["train"]) // 100)))
            ds["test"]  = ds["test"].select(range(max(20, len(ds["test"])  // 100)))

        # ── shuffle train split once per call (reference cycling) ────────
        if cycle_refs:
            # use caller-supplied seed or pick a new one each time
            seed = seed if seed is not None else random.randint(0, 2**31 - 1)
            ds["train"] = ds["train"].shuffle(seed=seed)

        # ── small wrapper so we can pass the length into preprocess ──────
        def _preprocess(ex):
            return self.preprocess(ex, sequence_length=sequence_length)

        train_ds = ds["train"].map(
            _preprocess,
            remove_columns=ds["train"].column_names,
            desc="Tokenising train split"
        )
        test_ds = ds["test"].map(
            _preprocess,
            remove_columns=ds["test"].column_names,
            desc="Tokenising test split"
        )

        return train_ds, test_ds



    
    def create_dataloader(self, dataset, batch_size=8, shuffle=True, sequence_length=None):
        """FIXED: Consistent sequence length with debug support"""
        from torch.nn.utils.rnn import pad_sequence   # add at top of file

        def collate_fn(batch):
            # turn lists into tensors, but keep variable length
            ids   = [torch.tensor(b["input_ids"],  dtype=torch.long) for b in batch]
            lbls  = [torch.tensor(b["labels"],     dtype=torch.long) for b in batch]

            # right-pad to the longest sequence in *this* minibatch
            ids  = pad_sequence(ids,  batch_first=True,
                                padding_value=self.tokenizer.eos_token_id)
            lbls = pad_sequence(lbls, batch_first=True,
                                padding_value=-100)                 # ignore in loss

            # build attention mask on-the-fly (1 = real token, 0 = pad/eos padding)
            
            attn = (ids != self.tokenizer.pad_token_id).long()

            return {"input_ids": ids,
                    "attention_mask": attn,
                    "labels": lbls,
                    "human_reference": [b["human_reference"] for b in batch]}
        
        return DataLoader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle,
                  collate_fn=collate_fn)
    
    def attach_schedulers(self, train_dataloader):
        if self._sched_steps is None:
            total_steps = len(train_dataloader) * self.max_epochs
            self._sched_steps = total_steps

        # Import the combined scheduler
        from transformers import get_cosine_schedule_with_warmup
        
        for opt in (self.head_client.optimizer,
                    self.server.optimizer,
                    self.tail_client.optimizer):
            # ENHANCED: Warmup + Cosine annealing
            sched = get_cosine_schedule_with_warmup(
                opt,
                num_warmup_steps=self.warmup_steps,  # Use your existing warmup_steps
                num_training_steps=total_steps,
                num_cycles=0.5,  # Half cosine cycle
                last_epoch=-1
            )
            for g in opt.param_groups: g['lr'] = max(g['lr'], 1e-4)
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

                    #print(f"Input attention mask sum: {attention_mask.sum()}")
                    #print(f"Attention mask shape: {attention_mask.shape}")
                    #print(f"Pad token positions: {(input_ids == self.tokenizer.pad_token_id).sum()}")

                    
                    # FIX: Check for NaN inputs
                    if torch.isnan(input_ids.float()).any() or torch.isnan(labels.float()).any():
                        print(f"Skipping batch {batch_idx} due to NaN inputs")
                        continue
                    
                    # Forward through pipeline
                    head_activations = self.head_client.forward(input_ids, attention_mask=attention_mask)  # No attention_mask
                    body_activations, head_activations_stored = self.server.forward_train(head_activations, attention_mask)  # No attention_mask
                    loss, body_grad = self.tail_client.compute_loss_and_backward(body_activations, labels, attention_mask)  # No attention_mask

                    
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
                    torch.nn.utils.clip_grad_norm_(self.head_client.head_model.parameters(), max_norm=0.5)
                    torch.nn.utils.clip_grad_norm_(self.tail_client.tail_model.parameters(), max_norm=0.5)
                    
                    total_loss += loss
                    num_batches += 1
                    
                    
                    if batch_idx % 50 == 0:
                        print(f"Batch {batch_idx}, Loss: {loss:.4f}")
                    
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

    def save_checkpoint(self, path: str = "checkpoint.pt", epoch: int = 0):
        """
        Save everything needed to  resume or to run inference later.
        If `path` is a directory, a file called `checkpoint.pt` is written inside it.
        """
        if os.path.isdir(path):
            path = os.path.join(path, "checkpoint.pt")

        torch.save({
            "epoch":      epoch,
            "head":       self.head_client.head_model.state_dict(),
            "body":       self.server.body_model.state_dict(),
            "tail":       self.tail_client.tail_model.state_dict(),
            "opt_head":   self.head_client.optimizer.state_dict(),
            "opt_body":   self.server.optimizer.state_dict(),
            "opt_tail":   self.tail_client.optimizer.state_dict(),
            "sch_head":   self.schedulers[0].state_dict() if self.schedulers else None,
            "sch_body":   self.schedulers[1].state_dict() if self.schedulers else None,
            "sch_tail":   self.schedulers[2].state_dict() if self.schedulers else None,
            "rng_state":  torch.random.get_rng_state(),
        }, path)
        print(f"✅ checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str = "checkpoint.pt", *, eval_only: bool = False) -> int:
        """
        Load weights; return the epoch *after* the one stored (for easy resumption).
        Set `eval_only=True` to skip optimiser/scheduler and switch all parts to .eval().
        """
        ckpt = torch.load(path, map_location=device)

        self.head_client.head_model.load_state_dict(ckpt["head"])
        self.server.body_model.load_state_dict(ckpt["body"])
        self.tail_client.tail_model.load_state_dict(ckpt["tail"])

        if eval_only:
            self.head_client.head_model.eval()
            self.server.body_model.eval()
            self.tail_client.tail_model.eval()
            print("✅ model loaded for evaluation only")
            return ckpt.get("epoch", 0) + 1

        # training resume: load optimiser & scheduler
        self.head_client.optimizer.load_state_dict(ckpt["opt_head"])
        self.server.optimizer.load_state_dict(ckpt["opt_body"])
        self.tail_client.optimizer.load_state_dict(ckpt["opt_tail"])

        if ckpt["sch_head"] is not None:
            self.schedulers[0].load_state_dict(ckpt["sch_head"])
            self.schedulers[1].load_state_dict(ckpt["sch_body"])
            self.schedulers[2].load_state_dict(ckpt["sch_tail"])

        torch.random.set_rng_state(ckpt["rng_state"])
        print(f"✅ checkpoint loaded from {path} – resuming training")
        return ckpt.get("epoch", 0) + 1
    
    # def save_checkpoint(self, path="./splitlora_checkpoint"):
    #     """FIXED: Save merged models using state dicts"""
    #     os.makedirs(path, exist_ok=True)
        
    #     try:
    #         print("Merging and saving models with custom embeddings...")
            
    #         # Merge PEFT adapters into base models
    #         head_merged = self.head_client.head_model.merge_and_unload()
    #         body_merged = self.server.body_model.merge_and_unload()
    #         tail_merged = self.tail_client.tail_model.merge_and_unload()
            
    #         # FIXED: Save state dicts instead of using save_pretrained
    #         torch.save(head_merged.state_dict(), os.path.join(path, "head_model_merged.pt"))
    #         torch.save(body_merged.state_dict(), os.path.join(path, "body_model_merged.pt"))
    #         torch.save(tail_merged.state_dict(), os.path.join(path, "tail_model_merged.pt"))
            
    #         # Save model configurations
    #         torch.save(head_merged.config, os.path.join(path, "head_config.pt"))
    #         torch.save(body_merged.config, os.path.join(path, "body_config.pt"))
    #         torch.save(tail_merged.config, os.path.join(path, "tail_config.pt"))
            
    #         # Save tokenizer
    #         self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))
            
    #         # Save training metadata including custom token info
    #         metadata = {
    #             "vocab_size": len(self.tokenizer),
    #             "original_vocab_size": 50257,
    #             "custom_tokens": {
    #                 "delim_token": self.DELIM,
    #                 "delim_id": self.tokenizer.encode(self.DELIM, add_special_tokens=False)[0],
    #                 "pad_token": self.PAD,
    #                 "pad_id": self.tokenizer.pad_token_id
    #             },
    #             "metrics": self.metrics,
    #             "model_config": {
    #                 "head_layers": 2,
    #                 "tail_layers": 2,
    #                 "body_layers": 8
    #             }
    #         }
            
    #         with open(os.path.join(path, "training_metadata.json"), "w") as f:
    #             json.dump(metadata, f, indent=2)
            
    #         print(f"✅ Merged checkpoint with custom embeddings saved to {path}")
    #         return path
            
    #     except Exception as e:
    #         print(f"❌ Error saving merged checkpoint: {e}")
    #         traceback.print_exc()
    #         return None

    # def load_checkpoint(self, path="./splitlora_checkpoint"):
    #     from transformers.models.gpt2.configuration_gpt2 import GPT2Config
    #     """FIXED: Load merged models from state dicts"""
    #     if not os.path.exists(path):
    #         print(f"❌ Checkpoint path {path} does not exist")
    #         return False

    #     try:
    #         # Load metadata first
    #         with open(os.path.join(path, "training_metadata.json"), "r") as f:
    #             metadata = json.load(f)
            
    #         print(f"Loading merged checkpoint with vocab_size: {metadata['vocab_size']}")
            
    #         # Load tokenizer (preserves custom tokens)
    #         self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer"))
            
    #         # FIXED: Create fresh full model with correct vocab size
    #         from transformers import GPT2LMHeadModel
    #         full_model = GPT2LMHeadModel.from_pretrained('gpt2')
            
    #         # Resize to match saved model
    #         full_model.resize_token_embeddings(metadata['vocab_size'])
            
    #         # Load the merged head model state dict to get the embeddings
    #         head_state = torch.load(os.path.join(path, "head_model_merged.pt"), map_location=device)
            
    #         # Extract embedding weights and copy to full model
    #         if 'wte.weight' in head_state:
    #             full_model.transformer.wte.weight.data = head_state['wte.weight']
    #         if 'wpe.weight' in head_state:
    #             full_model.transformer.wpe.weight.data = head_state['wpe.weight']
            
    #         with torch.serialization.safe_globals([GPT2Config]):
    #             head_config = torch.load(os.path.join(path, "head_config.pt"), 
    #                                     map_location=device, weights_only=True)
    #             body_state = torch.load(os.path.join(path, "body_model_merged.pt"), 
    #                                 map_location=device, weights_only=True) 
    #             tail_state = torch.load(os.path.join(path, "tail_model_merged.pt"), 
    #                                 map_location=device, weights_only=True)
    #             # Reconstruct layer weights in full model
    #         layer_idx = 0
            
    #         # Head layers
    #         for i in range(2):  # head_layers
    #             if f'h.{i}.ln_1.weight' in head_state:
    #                 full_model.transformer.h[layer_idx].load_state_dict({
    #                     k[len(f'h.{i}.'):]: v for k, v in head_state.items() 
    #                     if k.startswith(f'h.{i}.')
    #                 }, strict=False)
    #             layer_idx += 1
            
    #         # Body layers  
    #         for i in range(8):  # body_layers
    #             body_layer_key = f'transformer.h.{i}'
    #             if f'{body_layer_key}.ln_1.weight' in body_state:
    #                 full_model.transformer.h[layer_idx].load_state_dict({
    #                     k[len(f'{body_layer_key}.'):]: v for k, v in body_state.items()
    #                     if k.startswith(f'{body_layer_key}.')
    #                 }, strict=False)
    #             layer_idx += 1
            
    #         # Tail layers
    #         for i in range(2):  # tail_layers
    #             tail_layer_key = f'transformer.h.{i}'
    #             if f'{tail_layer_key}.ln_1.weight' in tail_state:
    #                 full_model.transformer.h[layer_idx].load_state_dict({
    #                     k[len(f'{tail_layer_key}.'):]: v for k, v in tail_state.items()
    #                     if k.startswith(f'{tail_layer_key}.')
    #                 }, strict=False)
    #             layer_idx += 1
            
    #         # Load final layer norm and LM head from tail
    #         if 'transformer.ln_f.weight' in tail_state:
    #             full_model.transformer.ln_f.weight.data = tail_state['transformer.ln_f.weight']
    #             full_model.transformer.ln_f.bias.data = tail_state['transformer.ln_f.bias']
            
    #         if 'lm_head.weight' in tail_state:
    #             full_model.lm_head.weight.data = tail_state['lm_head.weight']
            
    #         print("✅ Merged models loaded successfully with custom embeddings")
            
    #         # Verify custom token embeddings were preserved
    #         delim_id = metadata["custom_tokens"]["delim_id"]
    #         delim_embedding = full_model.transformer.wte.weight[delim_id]
    #         print(f"Loaded DELIM embedding norm: {delim_embedding.norm().item():.4f}")
            
    #         # Recreate split models from loaded full model
    #         head_model, body_model, tail_model = split_gpt2(full_model, 2, 2)
            
    #         # Re-apply PEFT to the loaded models
    #         lora_config = LoraConfig(
    #             r=2, lora_alpha=32, lora_dropout=0.1,
    #             bias="lora_only", use_dora=True, task_type="CAUSAL_LM",
    #             target_modules=["c_attn", "c_proj", "c_fc"]
    #         )
            
    #         head_model = get_peft_model(head_model, lora_config)
    #         body_model = get_peft_model(body_model, lora_config)
    #         tail_model = get_peft_model(tail_model, lora_config)
            
    #         # Update components with loaded models
    #         self.head_client.head_model = head_model.to(device)
    #         self.server.body_model = body_model.to(device)
    #         self.tail_client.tail_model = tail_model.to(device)
            
    #         # Ensure weight tying
    #         self.tail_client.tail_model.base_model.lm_head.weight = self.head_client.head_model.base_model.wte.weight
            
    #         print(f"✅ Checkpoint loaded successfully from {path}")
    #         return True

    #     except Exception as e:
    #         print(f"❌ Error loading checkpoint: {e}")
    #         traceback.print_exc()
    #         return False

def analyze_sequence_lengths(trainer, dataset):
    """Analyze your actual data to choose optimal fixed length"""
    lengths = []
    
    for i in range(min(2000, len(dataset))):  # Sample 1000 examples
        example = dataset[i]
        mr_text = example["meaning_representation"]
        ref_text = example["human_reference"]
        full_text = mr_text + " " + trainer.DELIM + " " + ref_text
        length = len(trainer.tokenizer.encode(full_text, add_special_tokens=False))
        lengths.append(length)
    
    # Statistics
    avg_length = sum(lengths) / len(lengths)
    p95_length = sorted(lengths)[int(0.95 * len(lengths))]
    p99_length = sorted(lengths)[int(0.99 * len(lengths))]
    
    print(f"Average length: {avg_length:.1f}")
    print(f"95th percentile: {p95_length}")
    print(f"99th percentile: {p99_length}")
    
    # Recommend fixed length
    recommended = min(p95_length + 16, 256)  # 95% coverage + buffer, capped at 256
    print(f"Recommended fixed length: {recommended}")
    
    return recommended


import urllib.request
import os
import pandas as pd
import pathlib

def download_official_e2e_dataset():
    """Download the official E2E testset_w_refs.csv file"""
    url = "https://raw.githubusercontent.com/tuetschek/e2e-dataset/master/testset_w_refs.csv"
    csv_file = "testset_w_refs.csv"
    
    if not os.path.exists(csv_file):
        print("Downloading official E2E testset_w_refs.csv...")
        urllib.request.urlretrieve(url, csv_file)
        print(f"✅ Downloaded {csv_file}")
    else:
        print(f"✅ {csv_file} already exists")
    
    return csv_file


def create_e2e_reference_file_from_official_csv():
    """Create e2e_refs.tsv from the official CSV file"""
    
    # Download the official CSV file
    csv_file = download_official_e2e_dataset()
    
    # Read the official CSV file
    print("Reading official E2E CSV file...")
    df = pd.read_csv(csv_file, encoding='utf-8')
    
    # Verify the structure
    print(f"CSV columns: {list(df.columns)}")
    print(f"Total rows: {len(df)}")
    print(f"Sample data:\n{df.head()}")
    
    # Create the references directory
    refs_dir = pathlib.Path(__file__).resolve().parent / "e2e-metrics" / "references"
    refs_dir.mkdir(exist_ok=True)
    
    # Create the reference TSV file
    ref_file = refs_dir / "e2e_refs.tsv"
    
    with open(ref_file, 'w', encoding='utf-8') as f:
        # Write TSV header (required for E2E metrics script)
        f.write("mr\tref\n")
        
        # Write all MR-reference pairs
        for _, row in df.iterrows():
            mr = str(row['mr']).replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
            ref = str(row['ref']).replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
            f.write(f"{mr}\t{ref}\n")
    
    print(f"✅ Created reference file: {ref_file}")
    print(f"📊 Total MR-reference pairs: {len(df)}")
    
    # Count unique MRs for verification
    unique_mrs = df['mr'].nunique()
    print(f"📊 Unique MRs: {unique_mrs}")
    
    return str(ref_file)


def debug_full_tokenization(trainer, example):
    """Debug the complete tokenization process - FIXED VERSION"""
    mr_text = example["meaning_representation"]
    ref_text = example["human_reference"]
    space_delim = " " + trainer.DELIM + " "
    full_text = mr_text + space_delim + ref_text
    
    # Tokenize and show COMPLETE sequence
    encoding = trainer.tokenizer(
        full_text,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_attention_mask=True
    )
    
    print(f"=== FULL TOKENIZATION DEBUG ===")
    print(f"Full text: '{full_text}'")
    print(f"Tokenized length: {len(encoding['input_ids'])} tokens")
    
    input_ids = encoding["input_ids"]
    non_padding = [token for token in input_ids if token != trainer.tokenizer.pad_token_id]
    print(f"Non-padding tokens: {len(non_padding)}")
    
    # FIXED: Search for just the core delimiter token (50257)
    delim_core_token = 50257  # <|gen|>
    
    delimiter_positions = []
    for i, token in enumerate(input_ids):
        if token == delim_core_token:
            delimiter_positions.append(i)
    
    if delimiter_positions:
        delim_pos = delimiter_positions[0]  # Use first occurrence
        print(f"✅ Delimiter found at position {delim_pos}")
        
        # Show context around delimiter
        start_ctx = max(0, delim_pos - 3)
        end_ctx = min(len(input_ids), delim_pos + 4)
        context_tokens = input_ids[start_ctx:end_ctx]
        context_text = trainer.tokenizer.decode(context_tokens, skip_special_tokens=True)
        print(f"Context around delimiter: {context_tokens}")
        print(f"Context text: '{context_text}'")
        
        return delim_pos
    else:
        print("❌ Delimiter token 50257 not found anywhere in sequence!")
        print(f"Unique tokens in sequence: {set(input_ids[:60])}")  # Show first 60 unique tokens
        return None



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
                
                # REMOVE PROBLEMATIC PROCESSOR:
                # Remove the MinLengthLogitsProcessor from here
                remove_invalid_values=True,
                do_sample=True,            # Ensure deterministic beam search
                temperature=1.0,            # Keep neutral
            )
    
    return trainer.tokenizer.decode(out[0, ids.size(1):],
                                    skip_special_tokens=True).strip()

import re
import subprocess
import tempfile
import pathlib

def evaluate_official(preds, ref_file="references/e2e_refs.tsv"):
    """FIXED: Use official E2E reference file for evaluation"""
    
    # Clean predictions (collapse whitespace)
    clean_preds = [re.sub(r'\s+', ' ', p).strip() for p in preds]
    
    # Write system outputs to temp file
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.txt', encoding='utf-8') as f:
        f.write("\n".join(clean_preds) + "\n")
        sys_file = f.name
    
    repo = pathlib.Path(__file__).resolve().parent / "e2e-metrics"
    ref_path = repo / ref_file
    
    try:
        # Check if reference file exists
        if not ref_path.exists():
            print(f"❌ Reference file {ref_path} not found")
            print("Creating reference file from official dataset...")
            create_e2e_reference_file_from_official_csv()
        
        # Run the official E2E evaluation script
        out = subprocess.check_output(
            ["python",
             str(repo / "measure_scores.py"),
             "--python", "-t",
             str(ref_path),
             sys_file],
            text=True
        )
        
        # Parse output
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        if not lines:
            return {"bleu": 0.0, "nist": 0.0, "meteor": 0.0, "rouge_l": 0.0, "cider": 0.0}
        
        # Find the metrics line (usually the last non-empty line)
        metrics_line = lines[-1]
        fields = metrics_line.split("\t")
        
        if len(fields) >= 6:
            return {
                "bleu": float(fields[1]),
                "nist": float(fields[2]), 
                "meteor": float(fields[3]),
                "rouge_l": float(fields[4]),
                "cider": float(fields[5])
            }
        else:
            print(f"Warning: Unexpected output format: {metrics_line}")
            return {"bleu": 0.0, "nist": 0.0, "meteor": 0.0, "rouge_l": 0.0, "cider": 0.0}
            
    except subprocess.CalledProcessError as e:
        print(f"Official evaluation failed: {e}")
        return {"bleu": 0.0, "nist": 0.0, "meteor": 0.0, "rouge_l": 0.0, "cider": 0.0}
    except Exception as e:
        print(f"Official evaluation error: {e}")
        return {"bleu": 0.0, "nist": 0.0, "meteor": 0.0, "rouge_l": 0.0, "cider": 0.0}
    finally:
        # Clean up temp files
        try:
            os.unlink(sys_file)
        except:
            pass


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
    delim_str = trainer.DELIM 
    full_text = mr_text + " " + delim_str + " " + ref_text
    
    print(f"Full text: '{full_text}'")
    
    # Tokenize manually (no dataset mapping)
    encoding = trainer.tokenizer(
        full_text,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_attention_mask=True
    )
    
    print(f"Encoded length: {len(encoding['input_ids'])}")
    print(f"Input IDs: {encoding['input_ids'][:20]}")
    
    # Check delimiter position
    delimiter_tokens = trainer.tokenizer.encode(
        " " + delim_str + " ", add_special_tokens=False)
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

def create_grouped_reference_file(grouped_mrs, grouped_refs, temp_path):
    """Create a reference file that matches grouped evaluation structure"""
    with open(temp_path, 'w') as f:
        for refs in grouped_refs:
            # Use the first reference for each MR group
            f.write(refs[0] + "\n")
    return temp_path


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
        for sent_refs in refs]                              # traverse sequences
        for i in range(max_refs)                             # traverse ref indices
    ]

    official = evaluate_official(preds)           # uses the existing refs
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
def generate_with_beam_mbr(trainer, wrapper, mr_text, ref_text, max_new_tokens=64, k=10):
    prompt = mr_text + " " + trainer.DELIM + " "
    enc = trainer.tokenizer(prompt, return_tensors="pt")
    ids, m = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    bad_tokens = trainer.tokenizer.encode("_*#-=.", add_special_tokens=False)
    with torch.no_grad():
        max_out = min(64, trainer.max_seq_len // 3)
        # ULTRA SIMPLE beam search - no complex parameters
        output = wrapper.generate(
            ids,
            num_beams=10,
            length_penalty=0.7,
            no_repeat_ngram_size=4,
            max_new_tokens=64,
            early_stopping=True,
            eos_token_id=trainer.tokenizer.eos_token_id,
            pad_token_id=trainer.tokenizer.eos_token_id)
        

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
            #max_new_tokens=12,              # Very short
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

def count_zero_target_rows(dataset):
    """
    Print how many examples have *no* real target tokens
    (i.e. every label == -100).
    """
    bad = sum(1 for ex in dataset if all(t == -100 for t in ex["labels"]))
    total = len(dataset)
    print(f"⚠️  rows with ZERO target tokens: {bad}/{total} "
          f"({bad/total*100:.2f} %)")


def main():
    parser = argparse.ArgumentParser(description="SplitLoRA Single File Implementation")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_epochs",  type=int, default=5)
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to checkpoint to load")
    parser.add_argument("--save_path", type=str, default="./splitlora_checkpoint", help="Path to save checkpoint")
    parser.add_argument("--gpu_device", type=str, default="1", help="GPU device to use")
    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    trainer = SplitLoRATrainer(model_name="gpt2",
                           head_layers=2,
                           tail_layers=2,
                           learning_rate=args.learning_rate,
                           warmup_steps=args.warmup_steps,
                           max_epochs=args.max_epochs)
    
    print("Analyzing sequence lengths to optimize training...")
    train_ds_temp, _ = trainer.load_e2e_dataset(debug_mode=False)
    optimal_length = analyze_sequence_lengths(trainer, train_ds_temp)
    trainer.max_seq_len = optimal_length
    start_epoch = 0
    # Load checkpoint if specified
    if args.load_checkpoint:
        start_epoch = trainer.load_checkpoint(args.load_checkpoint,
                                            eval_only=args.eval_only) 
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
        train_ds, test_ds = trainer.load_e2e_dataset(debug_mode=False, 
                                                        sequence_length=trainer.max_seq_len)
        print("Setting up E2E evaluation with official dataset...")
        create_e2e_reference_file_from_official_csv()
        diagnose_training_data(trainer, train_ds)
        diagnose_preprocessing_detailed(trainer)
        diagnose_custom_token_embeddings(trainer)
        count_zero_target_rows(train_ds)
        #NEW FIX
        debug_full_tokenization(trainer, train_ds[0])
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
        # Run evaluation
        results = evaluate_beam(trainer, wrapper, test_ds, n_samples=len(test_ds))
        results_file = os.path.join(args.save_path, "evaluation_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {results_file}")

        # Save evaluation results
        with open(os.path.join(args.save_path, "evaluation_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        return



    
    # Load dataset (regular mode)
    train_ds, test_ds = trainer.load_e2e_dataset(debug_mode=False, 
                                                    sequence_length=trainer.max_seq_len)
    diagnose_training_data(trainer, train_ds)
    diagnose_preprocessing_detailed(trainer)
    diagnose_custom_token_embeddings(trainer)
    count_zero_target_rows(train_ds)
    # Create dataloader and train
    train_dl = trainer.create_dataloader(train_ds, batch_size=args.batch_size, shuffle=True, sequence_length=trainer.max_seq_len)
    trainer.attach_schedulers(train_dl)
    print(f"Vocab size: {len(trainer.tokenizer)}")
    print(f"DELIM token: '{trainer.DELIM}' -> {trainer.tokenizer.encode(trainer.DELIM)}")
    print(f"PAD token: '{trainer.PAD}' -> {trainer.tokenizer.pad_token_id}")
    print(f"EOS token: '{trainer.tokenizer.eos_token}' -> {trainer.tokenizer.eos_token_id}")
    for ep in range(start_epoch, start_epoch + args.epochs):
        trainer.train(train_dl, epochs=1)
        trainer.save_checkpoint(args.save_path, epoch=ep)
    
    
    


if __name__ == "__main__":
    main()

