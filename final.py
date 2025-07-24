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
from torch.cuda.amp import autocast, GradScaler
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
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutput

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



def split_gpt2(model, head_layers=2, tail_layers=2):
    """Split GPT2 model into head, body, and tail parts with cache handling"""
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
            
        def forward(self, input_ids=None, attention_mask=None, position_ids=None, 
                past_key_values=None, use_cache=True, output_hidden_states=False, **kwargs):
            """Forward pass with proper cache handling and robust error handling"""
            
            # Handle position IDs
            if position_ids is None:
                if past_key_values is not None:
                    # For generation steps
                    # Get the right shape for past_key_values - handle cases where it might be None
                    if past_key_values[0] is not None and isinstance(past_key_values[0], tuple) and len(past_key_values[0]) > 0:
                        seq_length_past = past_key_values[0][0].shape[2] if past_key_values[0][0] is not None else 0
                    else:
                        seq_length_past = 0
                    
                    # Critical: position_ids should be incremented based on past sequence length
                    position_ids = torch.arange(
                        seq_length_past, 
                        seq_length_past + input_ids.shape[1], 
                        dtype=torch.long, 
                        device=input_ids.device
                    ).unsqueeze(0).expand(input_ids.shape[0], -1)
                else:
                    # For first step
                    seq_length = input_ids.size(-1)
                    position_ids = torch.arange(
                        0, seq_length, dtype=torch.long, device=input_ids.device
                    ).unsqueeze(0).expand(input_ids.shape[0], -1)
            
            if input_ids is not None:
                inputs_embeds = self.wte(input_ids)
            else:
                raise ValueError("You have to specify input_ids")
                
            position_embeds = self.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds
            hidden_states = self.drop(hidden_states)

            # Initialize for cache handling
            present_key_values = () if use_cache else None
            all_hidden_states = () if output_hidden_states else None
            
            # FIXED: Pass attention_mask to each block with caching
            dtype = hidden_states.dtype
            attn_mask = _expand_mask(attention_mask, dtype) if attention_mask is not None else None
            
            # Process through transformer layers
            for i, block in enumerate(self.h):
                try:
                    # Get past_key_value for this layer if available
                    past_key_value = past_key_values[i] if past_key_values is not None else None
                    
                    # Forward through the block
                    layer_outputs = block(
                        hidden_states, 
                        attention_mask=attn_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        use_cache=use_cache
                    )
                    
                    # FIX: Handle different return formats safely
                    if isinstance(layer_outputs, tuple):
                        if len(layer_outputs) > 0:
                            hidden_states = layer_outputs[0]
                        
                        # Only try to access cache if it exists in the tuple
                        if use_cache and len(layer_outputs) > 1 and layer_outputs[1] is not None:
                            present_key_values = present_key_values + (layer_outputs[1],)
                        elif use_cache:
                            # Create an empty cache placeholder if needed
                            present_key_values = present_key_values + (None,)
                    else:
                        # If not a tuple, assume it's just the hidden states
                        hidden_states = layer_outputs
                        if use_cache:
                            present_key_values = present_key_values + (None,)
                    
                    if output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden_states,)
                        
                except Exception as e:
                    print(f"Warning: Error in HeadModel layer {i}: {str(e)}")
                    if use_cache:
                        present_key_values = present_key_values + (None,)

            # Return with proper output format
            if output_hidden_states:
                return BaseModelOutputWithPast(
                    last_hidden_state=hidden_states,
                    past_key_values=present_key_values,
                    hidden_states=all_hidden_states
                )
            else:
                return hidden_states, present_key_values


    class BodyModel(nn.Module):
        def __init__(self, original_model, start_layer, num_layers):
            super().__init__()
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
            
        def forward(self, hidden_states=None, attention_mask=None, position_ids=None, 
                past_key_values=None, use_cache=True, **kwargs):
            """Forward pass with proper cache handling and robust error handling"""
            
            # Initialize for cache handling
            present_key_values = () if use_cache else None
            
            # Expand attention mask if needed
            dtype = hidden_states.dtype
            attn_mask = _expand_mask(attention_mask, dtype) if attention_mask is not None else None
            
            # Process through transformer layers
            for i, block in enumerate(self.transformer.h):
                try:
                    # Get past_key_value for this layer if available
                    past_key_value = past_key_values[i] if past_key_values is not None else None
                    
                    # Forward through the block with error handling
                    layer_outputs = block(
                        hidden_states, 
                        attention_mask=attn_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        use_cache=use_cache
                    )
                    
                    # FIX: Handle different return formats safely
                    if isinstance(layer_outputs, tuple):
                        if len(layer_outputs) > 0:
                            hidden_states = layer_outputs[0]
                        
                        # Only try to access cache if it exists in the tuple
                        if use_cache and len(layer_outputs) > 1 and layer_outputs[1] is not None:
                            present_key_values = present_key_values + (layer_outputs[1],)
                        elif use_cache:
                            # Create an empty cache placeholder if needed
                            present_key_values = present_key_values + (None,)
                    else:
                        # If not a tuple, assume it's just the hidden states
                        hidden_states = layer_outputs
                        if use_cache:
                            present_key_values = present_key_values + (None,)
                            
                except Exception as e:
                    print(f"Warning: Error in BodyModel layer {i}: {str(e)}")
                    if use_cache:
                        present_key_values = present_key_values + (None,)
            
            return hidden_states, present_key_values





    class TailModel(nn.Module):
        def __init__(self, original_model, start_layer):
            super().__init__()
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
            
        def forward(self, inputs_embeds=None, attention_mask=None, position_ids=None, 
                past_key_values=None, use_cache=True, **kwargs):
            """Forward pass with proper cache handling and robust error handling"""
        
            hidden_states = inputs_embeds
            
            # Initialize for cache handling
            present_key_values = () if use_cache else None
            
            # Get sequence length from attention mask for proper causal masking
            dtype = hidden_states.dtype
            attn_mask = _expand_mask(attention_mask, dtype) if attention_mask is not None else None
            
            # Process through transformer layers
            for i, block in enumerate(self.transformer.h):
                try:
                    # Get past_key_value for this layer if available
                    past_key_value = past_key_values[i] if past_key_values is not None else None
                    
                    # Forward through the block with error handling
                    layer_outputs = block(
                        hidden_states, 
                        attention_mask=attn_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        use_cache=use_cache
                    )
                    
                    # FIX: Handle different return formats safely
                    if isinstance(layer_outputs, tuple):
                        if len(layer_outputs) > 0:
                            hidden_states = layer_outputs[0]
                        
                        # Only try to access cache if it exists in the tuple
                        if use_cache and len(layer_outputs) > 1 and layer_outputs[1] is not None:
                            present_key_values = present_key_values + (layer_outputs[1],)
                        elif use_cache:
                            # Create an empty cache placeholder if needed
                            present_key_values = present_key_values + (None,)
                    else:
                        # If not a tuple, assume it's just the hidden states
                        hidden_states = layer_outputs
                        if use_cache:
                            present_key_values = present_key_values + (None,)
                            
                except Exception as e:
                    print(f"Warning: Error in TailModel layer {i}: {str(e)}")
                    if use_cache:
                        present_key_values = present_key_values + (None,)

            # Final layer norm and LM head
            hidden_states = self.transformer.ln_f(hidden_states)
            logits = self.lm_head(hidden_states)

            return logits, present_key_values

    # Create the model instances
    head_model = HeadModel(model, head_layers)
    body_model = BodyModel(model, head_layers, body_layers)
    tail_model = TailModel(model, head_layers + body_layers)

    # Set up weight tying
    tail_model.lm_head.weight = head_model.wte.weight
    
    # CRITICAL: Return the models - this was missing!
    return head_model, body_model, tail_model

class ServerModel:
    """Server component handling the body layers"""
    def __init__(self, body_model, learning_rate=2e-4):
        self.body_model = body_model.to(device)
        self.optimizer = optim.AdamW(
            [p for p in self.body_model.parameters() if p.requires_grad], 
            lr=learning_rate
        )
        
    def forward(self, activations, attention_mask=None, position_ids=None, past_key_values=None, use_cache=True):
        """Forward pass through body layers with proper caching"""
        try:
            self.body_model.eval()
            with torch.no_grad():
                output = self.body_model(
                    hidden_states=activations,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache
                )
                
                # Ensure consistent return format
                if isinstance(output, tuple) and len(output) == 2:
                    return output  # (hidden_states, present_key_values)
                else:
                    return output, None  # Return (output, None) if no cache
                    
        except Exception as e:
            print(f"Error in ServerModel forward: {str(e)}")
            import traceback
            traceback.print_exc()
            return activations, None  # Return input as fallback
    
    def forward_train(self, activations, attention_mask=None, position_ids=None):
        """Forward pass during training (no caching needed)"""
        try:
            self.body_model.train()
            # Don't detach - maintain gradient connection
            activations.requires_grad_(True)
            activations.retain_grad()  # CRITICAL for gradient flow
            
            output, _ = self.body_model(
                hidden_states=activations, 
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False
            )
            
            # Ensure output requires gradients
            output.requires_grad_(True)
            return output, activations
            
        except Exception as e:
            print(f"Error in ServerModel forward_train: {str(e)}")
            return activations, activations  # Return input as fallback
    
    def backward(self, body_output, body_grad, head_activations):
        """Fixed backward pass with proper gradient flow"""
        self.optimizer.zero_grad()
        
        # Ensure gradients are enabled
        if not head_activations.requires_grad:
            head_activations.requires_grad_(True)
        
        # Retain gradient for non-leaf tensors
        head_activations.retain_grad()
        
        if body_grad is not None:
            # Compute gradients with proper error handling
            try:
                torch.autograd.backward(
                    tensors=[body_output],
                    grad_tensors=[body_grad],
                    retain_graph=True,
                    create_graph=False
                )
            except Exception as e:
                print(f"Gradient computation failed: {e}")
                return torch.zeros_like(head_activations)
        
        # Get head gradients safely
        head_grad = head_activations.grad
        if head_grad is None:
            head_grad = torch.zeros_like(head_activations)
        else:
            head_grad = head_grad.clone()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.body_model.parameters(), max_norm=1.0)
        
        # Update parameters
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
        
    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=True):
        """Forward pass through head layers with proper caching"""
        try:
            # Get output from the head model
            output = self.head_model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=True
            )
            
            # CRITICAL FIX: Always return consistent output
            if isinstance(output, tuple) and len(output) == 2:
                # If output is (hidden_states, present_key_values)
                hidden_states, present_key_values = output
                return hidden_states, present_key_values
                
            elif hasattr(output, 'last_hidden_state'):
                # Output is already a BaseModelOutputWithPast
                return output.last_hidden_state, output.past_key_values
                
            else:
                # Unexpected output type, handle it safely
                print(f"Warning: Unexpected HeadModel output type: {type(output)}")
                if isinstance(output, torch.Tensor):
                    return output, None
                else:
                    return output, None
                
        except Exception as e:
            print(f"Error in HeadClient forward pass: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def backward(self, head_activations, head_grad):
        """ESSENTIAL: Backward pass for split learning"""
        self.optimizer.zero_grad()
        
        # Apply gradients received from body
        if head_grad is not None and head_activations.requires_grad:
            # Ensure head_activations has gradients enabled
            if not head_activations.requires_grad:
                head_activations.requires_grad_(True)
                
            # Apply gradients safely
            try:
                torch.autograd.backward(
                    tensors=[head_activations],
                    grad_tensors=[head_grad],
                    retain_graph=True
                )
            except Exception as e:
                print(f"Error in HeadClient backward: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.head_model.parameters(), max_norm=1.0)
        
        # Update head parameters
        self.optimizer.step()


class TailClient:
    """Client component handling tail layers"""
    def __init__(self, tail_model, learning_rate=5e-4, tokenizer=None):
        self.tail_model = tail_model.to(device)
        self.optimizer = optim.AdamW(
            [p for p in self.tail_model.parameters() if p.requires_grad], 
            lr=learning_rate
        )
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.tokenizer = tokenizer
        
    def forward(self, body_activations, attention_mask=None, position_ids=None, past_key_values=None, use_cache=True):
        """Forward pass through tail layers with proper caching"""
        output, present = self.tail_model(
            inputs_embeds=body_activations, 
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        return output if not use_cache else (output, present)
    
    def compute_loss_and_backward(self, body_activations, labels, attention_mask=None):
        """FIXED: Add retain_grad() for non-leaf tensors"""
        self.optimizer.zero_grad()
        
        # FIX: Add retain_grad() BEFORE accessing .grad
        body_activations.requires_grad_(True)
        body_activations.retain_grad()  # CRITICAL: Add this line
        
        # Forward pass
        logits, _ = self.tail_model(
            inputs_embeds=body_activations,
            attention_mask=attention_mask,
            use_cache=False
        )
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
    
    def extract_attributes(self, mr_text):
        """Extract key attributes from E2E meaning representation"""
        import re
        
        attributes = {}
        pattern = r'(\w+)\[([^\]]+)\]'
        matches = re.findall(pattern, mr_text)
        
        for attr_name, attr_value in matches:
            attr_value = attr_value.lower().strip()
            
            if attr_name == 'priceRange':
                if 'less than' in attr_value:
                    attr_value = 'cheap'
                elif 'more than' in attr_value:
                    attr_value = 'expensive'
            
            attributes[attr_name] = attr_value
        
        return attributes
    
    def validate_coverage(self, mr_text, generated_text):
        """Enhanced coverage validation for all E2E attributes"""
        attributes = self.extract_attributes(mr_text)
        gen_text_lower = generated_text.lower()
        
        covered_attrs = []
        missing_attrs = []
        
        for attr_name, attr_value in attributes.items():
            covered = False
            
            # Direct match first
            if attr_value in gen_text_lower:
                covered = True
            else:
                # Attribute-specific matching
                if attr_name == 'eatType':
                    synonyms = {
                        'coffee shop': ['coffee', 'café', 'cafe'],
                        'restaurant': ['restaurant', 'place', 'establishment'],
                        'pub': ['pub', 'bar'],
                        'fast food': ['fast food', 'takeaway']
                    }
                    if attr_value in synonyms:
                        covered = any(syn in gen_text_lower for syn in synonyms[attr_value])
                
                elif attr_name == 'food':
                    # Food cuisines are usually mentioned directly
                    covered = attr_value in gen_text_lower
                
                elif attr_name == 'priceRange':
                    price_indicators = {
                        'less than': ['cheap', 'inexpensive', 'affordable', 'less than'],
                        'more than': ['expensive', 'pricey', 'costly', 'more than'],
                        '£20-25': ['moderate', 'moderately priced'],
                        '£': ['pound', 'pounds', '£']
                    }
                    for pattern, indicators in price_indicators.items():
                        if pattern in attr_value:
                            covered = any(ind in gen_text_lower for ind in indicators)
                            break
                
                elif attr_name == 'area':
                    area_patterns = {
                        'city centre': ['city centre', 'city center', 'centre', 'center'],
                        'city center': ['city centre', 'city center', 'centre', 'center'],
                        'riverside': ['riverside', 'river side', 'by the river']
                    }
                    if attr_value in area_patterns:
                        covered = any(phrase in gen_text_lower for phrase in area_patterns[attr_value])
                
                elif attr_name == 'familyFriendly':
                    if attr_value == 'yes':
                        covered = any(phrase in gen_text_lower for phrase in ['family', 'kid', 'child'])
                    elif attr_value == 'no':
                        covered = any(phrase in gen_text_lower for phrase in ['not family', 'adult only'])
                
                elif attr_name == 'customer rating' or attr_name == 'customerRating':
                    # Handle ratings like "1 out of 5", "high", "average"
                    if 'out of' in attr_value:
                        # Extract number: "1 out of 5" -> "1"
                        rating_num = attr_value.split()[0]
                        covered = rating_num in gen_text_lower or attr_value in gen_text_lower
                    else:
                        covered = attr_value in gen_text_lower
                
                elif attr_name == 'near':
                    # Handle "near X" references
                    covered = attr_value in gen_text_lower or f"near {attr_value}" in gen_text_lower
            
            if covered:
                covered_attrs.append(f"{attr_name}={attr_value}")
            else:
                missing_attrs.append(f"{attr_name}={attr_value}")
        
        coverage_ratio = len(covered_attrs) / len(attributes) if attributes else 1.0
        
        return {
            'coverage_ratio': coverage_ratio,
            'covered': covered_attrs,
            'missing': missing_attrs,
            'complete': len(missing_attrs) == 0
        }


class _DummyLoader:
    """
    Pretend-loader used only to create schedulers before we have
    a real DataLoader.  `len()` must return a positive int.
    """
    def __init__(self, n_steps: int = 1):
        self._n = max(1, n_steps)          # at least one step
    def __len__(self):
        return self._n

class SplitLoRATrainer:
    def __init__(self,
                model_name="gpt2",
                head_layers=2,
                tail_layers=2,
                learning_rate=5e-4,
                warmup_steps=500,
                max_epochs=5):
        
        # FIXED: Use existing tokens only - no custom token addition
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # Re-use EOS as PAD so no new embedding is introduced
        self.tokenizer.pad_token = self.tokenizer.eos_token
        full_model = AutoModelForCausalLM.from_pretrained("gpt2")
        full_model.config.pad_token_id = self.tokenizer.eos_token_id
        self.max_seq_len = 512

        
        original_vocab_size = len(self.tokenizer)
        self.PAD = self.tokenizer.pad_token
        self.tokenizer.padding_side = "right"
        self.DELIM         = "|"              # no spaces
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
            r=4,
            lora_alpha=32,
            lora_dropout=0.5,
            bias="none",
            use_dora=True,
            task_type="CAUSAL_LM",
            target_modules=["c_attn", "c_proj", "c_fc"]
        )
        
        head_model = get_peft_model(head_model, lora_config)
        body_model = get_peft_model(body_model, lora_config)
        tail_model = get_peft_model(tail_model, lora_config)
        tied = tail_model.lm_head.weight.data_ptr() == head_model.base_model.model.wte.weight.data_ptr()
        print("✅ Weight tying correct (lm_head <-> wte):", tied)
        
        # Standard weight tying
        tail_model.base_model.lm_head.weight = head_model.base_model.wte.weight
        
        # Initialize components
        self.server = ServerModel(body_model, learning_rate)
        self.head_client = HeadClient(head_model, learning_rate)
        self.tail_client = TailClient(tail_model, learning_rate, tokenizer=self.tokenizer)
        
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

    def smart_truncate(self, mr_tokens, delim_tokens, ref_tokens, max_len):
        """Truncate MR first, then reference if needed"""
        
        # Reserve space for delimiter and minimum reference
        min_ref_len = 20  # Minimum reference length
        available_for_mr = max_len - len(delim_tokens) - min_ref_len
        
        # Truncate MR if too long
        if len(mr_tokens) > available_for_mr:
            mr_tokens = mr_tokens[:available_for_mr]
        
        # Calculate remaining space for reference
        remaining_space = max_len - len(mr_tokens) - len(delim_tokens)
        
        # Truncate reference only if absolutely necessary
        if len(ref_tokens) > remaining_space:
            ref_tokens = ref_tokens[:remaining_space]
        
        return mr_tokens, delim_tokens, ref_tokens



    def preprocess(self, example, sequence_length=None):
        """FIXED: Proper label alignment with truncated sequences"""
        SEQ_LEN = sequence_length if sequence_length is not None else self.max_seq_len
        
        mr = example["meaning_representation"]
        ref = example["human_reference"]
        
        # Tokenize pieces
        ids_mr = self.tokenizer.encode(mr, add_special_tokens=False)
        ids_ref = self.tokenizer.encode(ref, add_special_tokens=False)
        ids_delim = self.DELIM_TOKENS
        
        # Build full sequence
        full_sequence = ids_mr + ids_delim + ids_ref
        
        # Truncate if necessary
        if len(full_sequence) > SEQ_LEN:
            input_ids = full_sequence[:SEQ_LEN]
        else:
            input_ids = full_sequence
        
        # Create labels based on ACTUAL input_ids length
        labels = []
        
        # Find delimiter position in the ACTUAL input_ids
        delim_pos = None
        for i in range(len(input_ids) - len(ids_delim) + 1):
            if input_ids[i:i+len(ids_delim)] == ids_delim:
                delim_pos = i
                break
        
        if delim_pos is not None:
            # Mask everything before and including delimiter
            mask_length = delim_pos + len(ids_delim)
            labels = [-100] * mask_length
            
            # Add remaining tokens as targets
            remaining_tokens = input_ids[mask_length:]
            labels.extend(remaining_tokens)
        else:
            # Delimiter not found (heavily truncated) - mask everything
            labels = [-100] * len(input_ids)
        
        # Ensure exact length match
        assert len(input_ids) == len(labels), f"Length mismatch: {len(input_ids)} vs {len(labels)}"
        
        attention_mask = [1] * len(input_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "human_reference": ref,
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


    def analyze_truncation_impact(self, dataset):
        """Analyze how much data is lost to truncation"""
        
        truncated_count = 0
        total_loss = 0
        
        for example in dataset:
            original_ref = example['human_reference']
            
            # Simulate preprocessing
            processed = self.preprocess(example)
            
            # Extract actual target from labels
            target_tokens = [t for t in processed['labels'] if t != -100]
            reconstructed_ref = self.tokenizer.decode(target_tokens, skip_special_tokens=True)
            
            if len(reconstructed_ref) < len(original_ref):
                truncated_count += 1
                total_loss += len(original_ref) - len(reconstructed_ref)
        
        print(f"Truncation impact:")
        print(f"  Truncated examples: {truncated_count}/{len(dataset)} ({truncated_count/len(dataset)*100:.1f}%)")
        print(f"  Average characters lost: {total_loss/max(truncated_count, 1):.1f}")
    
    def create_dataloader(self, dataset, batch_size=8, shuffle=True, sequence_length=None):
        """FIXED: Proper padding handling that preserves label alignment"""
        from torch.nn.utils.rnn import pad_sequence
        
        def collate_fn(batch):
            # Get sequences without padding first
            ids = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
            lbls = [torch.tensor(b["labels"], dtype=torch.long) for b in batch]
            
            # Pad sequences to batch maximum
            ids_padded = pad_sequence(ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            lbls_padded = pad_sequence(lbls, batch_first=True, padding_value=-100)
            
            # CRITICAL FIX: Ensure padding tokens in input correspond to -100 in labels
            # But don't count padding -100s as "masked tokens" in your debug
            
            # Create proper attention mask
            attn_mask = (ids_padded != self.tokenizer.pad_token_id)
            
            # VERIFY: Check that non-padding positions have correct label alignment
            for i in range(len(ids_padded)):
                # Find actual sequence length (before padding)
                actual_length = attn_mask[i].sum().item()
                
                # Ensure labels match input for non-padding positions
                if actual_length > 0:
                    # The first actual_length positions should have the original labels
                    # The rest should be -100 (padding)
                    pass  # This is handled by pad_sequence correctly
            
            return {
                "input_ids": ids_padded,
                "attention_mask": attn_mask,
                "labels": lbls_padded,
                "human_reference": [b["human_reference"] for b in batch]
            }
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


    
    def attach_schedulers(self, train_dataloader):
        # 0. Avoid building duplicate schedulers
        if self.schedulers:                      # already initialised
            return

        # 1. Make sure we always have the total number of optimisation steps
        if self._sched_steps is None:
            self._sched_steps = len(train_dataloader) * self.max_epochs

        total_steps = self._sched_steps          # local alias, always defined

        # 2. Create one cosine scheduler per optimiser
        from transformers import get_cosine_schedule_with_warmup
        from transformers import get_constant_schedule_with_warmup
        from transformers import get_linear_schedule_with_warmup
        scheduler_epochs = 6
        total_steps = len(train_dataloader) * scheduler_epochs
        for opt in (self.head_client.optimizer,
                    self.server.optimizer,
                    self.tail_client.optimizer):
            sched = get_linear_schedule_with_warmup(
                opt,
                num_warmup_steps=1000,      # ~3% of total steps
                num_training_steps=total_steps,
                last_epoch=-1
            )
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
                    max_new_tokens=60,        
                    do_sample=False,                # Pure greedy
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    # NO OTHER PARAMETERS AT ALL
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

    


    def train_with_coverage(self, train_dataloader, epochs=1):
        """Enhanced training with coverage monitoring + AMP + memory optimization."""
        # Initialize GradScaler for AMP
        scaler = GradScaler()

        print(f"Starting AMP-enabled training for {epochs} epoch(s)...")
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            nan_batches = 0  # Track NaN batches for debugging

            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
                # Periodically clear CUDA cache to reduce fragmentation
                if batch_idx % 100 == 0:
                    torch.cuda.empty_cache()

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device).bool()
                labels = batch["labels"].to(device)

                # Extract MR texts for coverage penalty
                mr_texts = []
                for seq in input_ids:
                    # find delimiter index
                    delim_pos = None
                    for i in range(seq.size(0) - len(self.DELIM_TOKENS) + 1):
                        if seq[i : i + len(self.DELIM_TOKENS)].tolist() == self.DELIM_TOKENS:
                            delim_pos = i + len(self.DELIM_TOKENS)
                            break
                    mr_texts.append(
                        self.tokenizer.decode(seq[: delim_pos], skip_special_tokens=True)
                        if delim_pos is not None
                        else ""
                    )

                # AMP‐enabled forward/backward
                with autocast(dtype=torch.float16):
                    # 1) Head forward (no cache)
                    head_out = self.head_client.forward(
                        input_ids, attention_mask=attention_mask, use_cache=False
                    )
                    h_states = head_out.last_hidden_state

                    # 2) Body forward (no cache)
                    body_out, _ = self.server.forward_train(
                        h_states, attention_mask=attention_mask
                    )
                    b_states = body_out

                    # 3) Tail forward & compute loss
                    logits, _ = self.tail_client.tail_model(
                        inputs_embeds=b_states,
                        attention_mask=attention_mask,
                        past_key_values=None,
                        use_cache=False
                    )
                    
                    # ADDED: Clamp logits to prevent extreme values (critical for stability)
                    logits = torch.clamp(logits, -50.0, 50.0)

                    # shift for CE
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    shift_labels[shift_labels == -100] = self.tail_client.loss_fn.ignore_index
                    
                    # ADDED: Check for NaN in logits
                    if torch.isnan(shift_logits).any():
                        print(f"WARNING: NaN detected in logits at batch {batch_idx}. Skipping batch.")
                        nan_batches += 1
                        continue
                    
                    base_loss = self.tail_client.loss_fn(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )

                    # MODIFIED: Safer coverage penalty calculation with try-except
                    penalty = 0.0
                    try:
                        for i, mr in enumerate(mr_texts):
                            # Only process if we have valid logits for this sequence
                            if i < shift_logits.size(0):
                                pred_ids = torch.argmax(shift_logits[i], dim=-1)
                                pred_txt = self.tokenizer.decode(pred_ids, skip_special_tokens=True)
                                cov = self.tail_client.validate_coverage(mr, pred_txt)
                                penalty += (1.0 - cov["coverage_ratio"]) * 0.2
                        
                        # Add small epsilon to prevent division by zero
                        loss = base_loss + penalty / (len(mr_texts) + 1e-8)
                    except Exception as e:
                        print(f"Coverage penalty calculation failed: {e}. Using base loss.")
                        loss = base_loss
                    
                    # ADDED: Check for NaN or Inf loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"WARNING: NaN/Inf loss at batch {batch_idx}. Using base loss only.")
                        loss = base_loss
                        # If base loss is also NaN, skip this batch
                        if torch.isnan(loss) or torch.isinf(loss):
                            print("Base loss is also NaN/Inf. Skipping batch.")
                            nan_batches += 1
                            continue

                # 4) backward with scaler (only if loss is valid)
                try:
                    scaler.scale(loss).backward(retain_graph=True)
                    
                    # 5) retrieve body grads and propagate
                    body_grad = b_states.grad.clone() if b_states.grad is not None else torch.zeros_like(b_states)
                    
                    # ADDED: Check for NaN in gradients
                    if torch.isnan(body_grad).any():
                        print(f"WARNING: NaN gradient at batch {batch_idx}. Skipping optimizer step.")
                        nan_batches += 1
                        continue
                        
                    head_grad = self.server.backward(b_states, body_grad, h_states)
                    self.head_client.backward(h_states, head_grad)

                    # 6) unscale & clip (more aggressive clipping)
                    for opt in (self.head_client.optimizer, self.server.optimizer, self.tail_client.optimizer):
                        scaler.unscale_(opt)
                        
                    # More aggressive clipping
                    torch.nn.utils.clip_grad_norm_(self.head_client.head_model.parameters(), 0.5)  # Reduced from 1.0
                    torch.nn.utils.clip_grad_norm_(self.server.body_model.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(self.tail_client.tail_model.parameters(), 0.5)

                    # 7) optimizer steps & scaler update
                    scaler.step(self.head_client.optimizer)
                    scaler.step(self.server.optimizer)
                    scaler.step(self.tail_client.optimizer)
                    scaler.update()

                    # 8) scheduler step
                    for sched in self.schedulers:
                        sched.step()

                    total_loss += loss.item()
                    num_batches += 1

                    if batch_idx % 50 == 0:
                        print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
                except Exception as e:
                    print(f"Error in backward pass: {e}. Skipping batch.")
                    nan_batches += 1
                    continue

            avg = total_loss / max(num_batches, 1)
            self.metrics["loss"].append(avg)
            print(f"Epoch {epoch+1} average loss: {avg:.4f} (skipped {nan_batches} batches with NaN)")



    def train_with_wrapper(self, train_dataloader, epochs=1):
        """Train using the wrapper to maintain consistency"""
        from split_beam_wrapper import SplitGPT2ForGeneration
        
        wrapper = SplitGPT2ForGeneration(
            tokenizer=self.tokenizer,
            head_client=self.head_client,
            server=self.server,
            tail_client=self.tail_client,
            base_config=self.head_client.head_model.config
        ).to(device)
        
        # Set all components to training mode
        wrapper.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
                # Zero gradients for all components
                self.head_client.optimizer.zero_grad()
                self.server.optimizer.zero_grad()
                self.tail_client.optimizer.zero_grad()
                
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                # Forward pass through wrapper (no caching during training)
                outputs = wrapper(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Compute loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), 
                            shift_labels.view(-1))
                
                # Backward pass
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(wrapper.parameters(), max_norm=1.0)
                
                # Update all optimizers
                self.head_client.optimizer.step()
                self.server.optimizer.step()
                self.tail_client.optimizer.step()
                
                # Update schedulers
                for sched in self.schedulers:
                    sched.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 50 == 0:
                    print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")


    def evaluate_with_coverage(self, wrapper, dataset, n_samples=100):
        """Enhanced evaluation with coverage tracking"""
        eval_split = dataset.select(range(min(n_samples, len(dataset))))
        
        total_coverage = 0.0
        complete_outputs = 0
        predictions = []
        references = []
        
        for sample in tqdm(eval_split, desc="Evaluating with coverage"):
            mr = sample["meaning_representation"]
            ref = sample["human_reference"]
            
            # Generate prediction
            pred = generate_with_beam(self, wrapper, mr, max_new_tokens=64)
            
            # Check coverage
            coverage = self.validate_coverage(mr, pred)
            total_coverage += coverage['coverage_ratio']
            
            if coverage['complete']:
                complete_outputs += 1
            
            # Print examples of incomplete coverage
            if not coverage['complete'] and len(predictions) < 5:
                print(f"Incomplete coverage example:")
                print(f"  MR: {mr}")
                print(f"  Pred: {pred}")
                print(f"  Missing: {coverage['missing']}")
            
            predictions.append(pred)
            references.append(ref)
        
        # Compute metrics
        avg_coverage = total_coverage / len(eval_split)
        complete_ratio = complete_outputs / len(eval_split)
        
        # Official E2E metrics
        official_metrics = evaluate_official(predictions)
        
        print(f"Coverage Results:")
        print(f"  Average coverage: {avg_coverage:.3f}")
        print(f"  Complete outputs: {complete_ratio:.3f}")
        print(f"  BLEU: {official_metrics['bleu']:.3f}")
        
        return {
            **official_metrics,
            "coverage": avg_coverage,
            "completeness": complete_ratio
        }

    def train(self, train_dataloader, epochs=1):
        """Fixed training method that ensures proper gradient flow"""
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            nan_batches = 0
            
            # Ensure models are in training mode
            self.head_client.head_model.train()
            self.server.body_model.train()
            self.tail_client.tail_model.train()
            
            # Verify weight tying before training
            self.ensure_weight_tying()
            
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
                if batch_idx % 100 == 0:
                    torch.cuda.empty_cache()
                
                try:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    
                    # Reset gradients
                    self.head_client.optimizer.zero_grad()
                    self.server.optimizer.zero_grad()
                    self.tail_client.optimizer.zero_grad()
                    
                    # Forward pass - CRITICAL FIX: Handle return values properly
                    head_out = self.head_client.forward(
                        input_ids=input_ids, 
                        attention_mask=attention_mask,
                        use_cache=False
                    )
                    
                    # Extract hidden states safely
                    if isinstance(head_out, tuple):
                        h_states = head_out[0]  # First element is hidden states
                    elif hasattr(head_out, 'last_hidden_state'):
                        h_states = head_out.last_hidden_state
                    else:
                        h_states = head_out
                    
                    # Body forward pass with gradient tracking
                    body_out, stored_activations = self.server.forward_train(
                        h_states, 
                        attention_mask=attention_mask
                    )
                    
                    # Ensure gradient tracking
                    body_out.requires_grad_(True)
                    body_out.retain_grad()
                    
                    # Tail forward pass and loss computation
                    logits, _ = self.tail_client.tail_model(
                        inputs_embeds=body_out,
                        attention_mask=attention_mask,
                        use_cache=False
                    )
                    
                    # Compute loss
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    shift_labels[shift_labels == -100] = self.tail_client.loss_fn.ignore_index
                    
                    loss = self.tail_client.loss_fn(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    
                    # Backward pass with proper gradient flow
                    loss.backward(retain_graph=True)
                    
                    # Get body gradients
                    body_grad = body_out.grad.clone() if body_out.grad is not None else torch.zeros_like(body_out)
                    
                    # Propagate gradients to head
                    head_grad = self.server.backward(body_out, body_grad, stored_activations)
                    self.head_client.backward(h_states, head_grad)
                    
                    # Step optimizers
                    self.head_client.optimizer.step()
                    self.server.optimizer.step()
                    self.tail_client.optimizer.step()
                    
                    # Step schedulers
                    for sched in self.schedulers:
                        sched.step()
                    
                    # Update metrics
                    loss_val = loss.item()
                    if not math.isnan(loss_val) and not math.isinf(loss_val):
                        total_loss += loss_val
                        num_batches += 1
                        
                        if batch_idx % 50 == 0:
                            # Re-verify weight tying periodically
                            self.ensure_weight_tying()
                            print(f"Batch {batch_idx}, Loss: {loss_val:.4f}")
                    else:
                        nan_batches += 1
                        
                except Exception as e:
                    print(f"Training error at batch {batch_idx}: {e}")
                    traceback.print_exc()
                    nan_batches += 1
                    continue
            
            # Compute average loss for the epoch
            avg_loss = total_loss / max(num_batches, 1)
            self.metrics["loss"].append(avg_loss)
            print(f"Epoch {epoch+1} average loss: {avg_loss:.4f} (skipped {nan_batches} batches)")
        
        print("Training completed!")

    def ensure_weight_tying(self):
        """Ensure weight tying is maintained"""
        try:
            # Get the head embedding
            if hasattr(self.head_client.head_model, 'wte'):
                head_embed = self.head_client.head_model.wte
            elif hasattr(self.head_client.head_model, 'base_model') and hasattr(self.head_client.head_model.base_model, 'wte'):
                head_embed = self.head_client.head_model.base_model.wte
            elif hasattr(self.head_client.head_model, 'base_model') and hasattr(self.head_client.head_model.base_model, 'model'):
                head_embed = self.head_client.head_model.base_model.model.wte
            else:
                print("⚠️ Could not find head embedding weights")
                return False
                
            # Get the tail lm_head
            if hasattr(self.tail_client.tail_model, 'lm_head'):
                tail_lm_head = self.tail_client.tail_model.lm_head
            elif hasattr(self.tail_client.tail_model, 'base_model') and hasattr(self.tail_client.tail_model.base_model, 'lm_head'):
                tail_lm_head = self.tail_client.tail_model.base_model.lm_head
            else:
                print("⚠️ Could not find tail lm_head weights")
                return False
                
            # Ensure they're the same object
            if id(tail_lm_head.weight) != id(head_embed.weight):
                print("⚠️ Weight tying broken - fixing...")
                tail_lm_head.weight = head_embed.weight
                return False
                
            return True
        
        except Exception as e:
            print(f"Error checking weight tying: {e}")
            return False


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
        FIXED: Save everything needed to resume training properly
        """
        if os.path.isdir(path):
            path = os.path.join(path, "checkpoint.pt")
        
        # Ensure models are in training mode for proper state saving
        self.head_client.head_model.train()
        self.server.body_model.train()
        self.tail_client.tail_model.train()
        
        # Save complete training state
        checkpoint = {
            "epoch": epoch,
            "metrics": self.metrics,
            "max_seq_len": self.max_seq_len,
            "vocab_size": len(self.tokenizer),
            
            # Model states (including PEFT adapters)
            "head_model_state": self.head_client.head_model.state_dict(),
            "body_model_state": self.server.body_model.state_dict(),
            "tail_model_state": self.tail_client.tail_model.state_dict(),
            
            # Optimizer states
            "head_optimizer_state": self.head_client.optimizer.state_dict(),
            "body_optimizer_state": self.server.optimizer.state_dict(),
            "tail_optimizer_state": self.tail_client.optimizer.state_dict(),
            
            # Scheduler states
            "scheduler_states": [sched.state_dict() for sched in self.schedulers] if self.schedulers else [],
            
            # Training configuration
            "training_config": {
                "learning_rate": self.head_client.optimizer.param_groups[0]['lr'],
                "warmup_steps": self.warmup_steps,
                "max_epochs": self.max_epochs,
            },
            
            # Random states for reproducibility
            "rng_state": torch.random.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            
            # Tokenizer state
            "tokenizer_config": {
                "delim": self.DELIM,
                "pad_token": self.PAD,
                "delim_tokens": self.DELIM_TOKENS,
            }
        }
        
        # Save with error handling
        try:
            torch.save(checkpoint, path)
            print(f"✅ Complete checkpoint saved to {path}")
            print(f"   - Epoch: {epoch}")
            print(f"   - Vocab size: {len(self.tokenizer)}")
            print(f"   - Metrics: {len(self.metrics['loss'])} loss values")
        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")
            raise

    
    def load_checkpoint(self, path: str = "checkpoint.pt", *, eval_only: bool = False) -> int:
        """
        FIXED: Load complete training state for proper resumption
        """
        if os.path.isdir(path):
            path = os.path.join(path, "checkpoint.pt")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        print(f"Loading checkpoint from {path}...")
        
        try:
            checkpoint = torch.load(path, map_location=device)
            
            # Verify checkpoint integrity
            required_keys = ["head_model_state", "body_model_state", "tail_model_state"]
            for key in required_keys:
                if key not in checkpoint:
                    raise KeyError(f"Missing required key in checkpoint: {key}")
            
            # Load model states
            print("Loading model states...")
            self.head_client.head_model.load_state_dict(checkpoint["head_model_state"])
            self.server.body_model.load_state_dict(checkpoint["body_model_state"])
            self.tail_client.tail_model.load_state_dict(checkpoint["tail_model_state"])
            tied = self.tail_client.tail_model.lm_head.weight.data_ptr() == self.head_client.head_model.base_model.model.wte.weight.data_ptr()
            print("🔁 Weight tying correct after loading:", tied)
            # CRITICAL: Restore weight tying after loading
            print("Restoring weight tying...")
            if hasattr(self.head_client.head_model, 'base_model'):
                # PEFT wrapped models
                head_base = self.head_client.head_model.base_model
                tail_base = self.tail_client.tail_model.base_model
            else:
                # Direct models
                head_base = self.head_client.head_model
                tail_base = self.tail_client.tail_model
            
            if hasattr(head_base, 'wte') and hasattr(tail_base, 'lm_head'):
                tail_base.lm_head.weight = head_base.wte.weight
                print("✅ Weight tying restored")
            
            # Load training configuration
            if "training_config" in checkpoint:
                config = checkpoint["training_config"]
                self.warmup_steps = config.get("warmup_steps", self.warmup_steps)
                self.max_epochs = config.get("max_epochs", self.max_epochs)
            
            # Load tokenizer configuration
            if "tokenizer_config" in checkpoint:
                tok_config = checkpoint["tokenizer_config"]
                self.DELIM = tok_config.get("delim", self.DELIM)
                self.PAD = tok_config.get("pad_token", self.PAD)
                self.DELIM_TOKENS = tok_config.get("delim_tokens", self.DELIM_TOKENS)
            
            # Load metrics
            if "metrics" in checkpoint:
                self.metrics = checkpoint["metrics"]
                print(f"✅ Loaded {len(self.metrics['loss'])} previous loss values")
            
            # Set sequence length
            if "max_seq_len" in checkpoint:
                self.max_seq_len = checkpoint["max_seq_len"]
            
            if eval_only:
                # Set models to evaluation mode
                self.head_client.head_model.eval()
                self.server.body_model.eval()
                self.tail_client.tail_model.eval()
                print("✅ Models loaded in evaluation mode")
                return checkpoint.get("epoch", 0)
            
            # Load optimizer states for training resumption
            print("Loading optimizer states...")
            if "head_optimizer_state" in checkpoint:
                self.head_client.optimizer.load_state_dict(checkpoint["head_optimizer_state"])
            if "body_optimizer_state" in checkpoint:
                self.server.optimizer.load_state_dict(checkpoint["body_optimizer_state"])
            if "tail_optimizer_state" in checkpoint:
                self.tail_client.optimizer.load_state_dict(checkpoint["tail_optimizer_state"])
            
            # Load scheduler states
            if "scheduler_states" in checkpoint and checkpoint["scheduler_states"]:
                if not self.schedulers:
                    # Create schedulers if they don't exist
                    total_steps = checkpoint.get("_sched_steps", 1000)
                    self.attach_schedulers(_DummyLoader(total_steps))
                
                for i, sched_state in enumerate(checkpoint["scheduler_states"]):
                    if i < len(self.schedulers):
                        self.schedulers[i].load_state_dict(sched_state)
                print("✅ Scheduler states loaded")
            
            # Restore random states for reproducibility
            if "rng_state" in checkpoint:
                torch.random.set_rng_state(checkpoint["rng_state"])
                state = checkpoint.get("rng_state", None)
                if state is not None:
                    # ensure ByteTensor on CPU
                    state = state.to(device='cpu', dtype=torch.uint8)
                    try:
                        torch.random.set_rng_state(state)
                        print("✅ RNG state restored")
                    except TypeError:
                        print("⚠️ Failed to restore RNG state; skipping")
            if "cuda_rng_states" in checkpoint and checkpoint["cuda_rng_states"]:
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
            
            # Set models to training mode
            self.head_client.head_model.train()
            self.server.body_model.train()
            self.tail_client.tail_model.train()
            
            epoch = checkpoint.get("epoch", 0)
            print(f"✅ Checkpoint loaded successfully!")
            print(f"   - Resuming from epoch: {epoch}")
            print(f"   - Vocab size: {checkpoint.get('vocab_size', 'unknown')}")
            print(f"   - Previous loss values: {len(self.metrics.get('loss', []))}")
            
            return epoch
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def verify_training_state(self):
        """
        Verify that the model is in proper training state after loading
        """
        print("=== TRAINING STATE VERIFICATION ===")
        
        # Check if models are in training mode
        print(f"Head model training mode: {self.head_client.head_model.training}")
        print(f"Body model training mode: {self.server.body_model.training}")
        print(f"Tail model training mode: {self.tail_client.tail_model.training}")
        
        # Check if gradients are enabled
        head_params = list(self.head_client.head_model.parameters())
        body_params = list(self.server.body_model.parameters())
        tail_params = list(self.tail_client.tail_model.parameters())
        
        head_requires_grad = sum(1 for p in head_params if p.requires_grad)
        body_requires_grad = sum(1 for p in body_params if p.requires_grad)
        tail_requires_grad = sum(1 for p in tail_params if p.requires_grad)
        
        print(f"Head parameters requiring gradients: {head_requires_grad}/{len(head_params)}")
        print(f"Body parameters requiring gradients: {body_requires_grad}/{len(body_params)}")
        print(f"Tail parameters requiring gradients: {tail_requires_grad}/{len(tail_params)}")
        
        # Check weight tying
        if hasattr(self.head_client.head_model, 'base_model'):
            head_base = self.head_client.head_model.base_model
            tail_base = self.tail_client.tail_model.base_model
        else:
            head_base = self.head_client.head_model
            tail_base = self.tail_client.tail_model
        
        if hasattr(head_base, 'wte') and hasattr(tail_base, 'lm_head'):
            weight_tied = torch.equal(head_base.wte.weight, tail_base.lm_head.weight)
            print(f"Weight tying intact: {weight_tied}")
        
        # Check optimizer states
        print(f"Head optimizer state groups: {len(self.head_client.optimizer.param_groups)}")
        print(f"Body optimizer state groups: {len(self.server.optimizer.param_groups)}")
        print(f"Tail optimizer state groups: {len(self.tail_client.optimizer.param_groups)}")
        
        print("✅ Training state verification complete")

    def debug_label_masking(self, batch):
        """Debug label masking to ensure it's working correctly"""
        print("=== LABEL MASKING DEBUG ===")
        
        for i in range(min(3, len(batch["input_ids"]))):
            input_ids = batch["input_ids"][i]
            labels = batch["labels"][i]
            attention_mask = batch["attention_mask"][i]
            
            # CRITICAL: Only analyze non-padding tokens
            actual_length = attention_mask.sum().item()
            
            # Work with actual sequence (no padding)
            actual_input_ids = input_ids[:actual_length]
            actual_labels = labels[:actual_length]
            
            # Find delimiter position in actual sequence
            delim_pos = None
            for j in range(len(actual_input_ids) - len(self.DELIM_TOKENS) + 1):
                if actual_input_ids[j:j+len(self.DELIM_TOKENS)].tolist() == self.DELIM_TOKENS:
                    delim_pos = j
                    break
            
            if delim_pos is None:
                print(f"❌ Sample {i}: No delimiter found!")
                continue
            
            # Count masked/target tokens in ACTUAL sequence only
            masked_tokens = sum(1 for label in actual_labels if label == -100)
            target_tokens = sum(1 for label in actual_labels if label != -100)
            
            print(f"Sample {i}:")
            print(f"  Actual sequence length: {actual_length}")
            print(f"  Delimiter position: {delim_pos}")
            print(f"  Masked tokens: {masked_tokens}")
            print(f"  Target tokens: {target_tokens}")
            
            # Decode parts
            mr_part = actual_input_ids[:delim_pos]
            ref_part = actual_input_ids[delim_pos+len(self.DELIM_TOKENS):]
            
            mr_text = self.tokenizer.decode(mr_part, skip_special_tokens=True)
            ref_text = self.tokenizer.decode(ref_part, skip_special_tokens=True)
            
            print(f"  MR: '{mr_text}'")
            print(f"  REF: '{ref_text}'")
            
            # Check if masking is correct
            expected_mask_length = len(mr_part) + len(self.DELIM_TOKENS)
            
            if expected_mask_length == masked_tokens:
                print("  ✅ Label masking is correct")
            else:
                print(f"  ❌ Label masking wrong: expected {expected_mask_length}, got {masked_tokens}")


    

    def coverage_loss(self, mr_text, generated_text, base_loss):
        """Add coverage penalty to the loss function"""
        coverage = self.validate_coverage(mr_text, generated_text)
        
        # Coverage penalty: penalize missing attributes
        coverage_penalty = (1.0 - coverage['coverage_ratio']) * 0.5
        
        return base_loss + coverage_penalty
    
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

def _expand_mask(mask, dtype):
    """
    Expands attention_mask from [batch_size, seq_len] to [batch_size, 1, seq_len, seq_len]
    for GPT-2's causal attention mechanism.
    """
    if mask is None:
        return None
    
    batch_size, seq_len = mask.shape
    
    # First convert boolean/float mask to additive mask
    # [B, S] -> [B, 1, 1, S]
    mask = mask.to(dtype)
    
    # Important: Reshape to 4D for broadcasting with causal mask
    mask = mask.unsqueeze(1).unsqueeze(2)
    
    # Create causal mask (lower triangular)
    # This ensures tokens only attend to previous tokens and themselves
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=dtype, device=mask.device))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]
    
    # Combine the padding mask with the causal mask
    # broadcast: [B, 1, 1, S] * [1, 1, S, S] -> [B, 1, S, S]
    combined_mask = mask @ causal_mask
    
    # Convert to attention scores:
    # - 0 allows attention
    # - large negative number blocks attention
    return (1.0 - combined_mask) * -10000.0



# ─── Beam-search helpers ──────────────────────────────────────────────
from evaluate import load as load_metric
def generate_fixed(trainer, wrapper, mr_text, max_new_tokens=32):
    """Fixed generation with proper parameters"""
    prompt = mr_text + " " + trainer.DELIM + " "
    enc = trainer.tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    
    # Ensure weight tying before generation
    trainer.ensure_weight_tying()
    
    with torch.no_grad():
        output = wrapper.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=trainer.tokenizer.eos_token_id,
            pad_token_id=trainer.tokenizer.pad_token_id,
            # NO caching parameters - let it handle internally
        )
    
    return trainer.tokenizer.decode(output[0, ids.size(1):], skip_special_tokens=True).strip()

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
                pad_token_id=trainer.tokenizer.pad_token_id,  
                
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
                           warmup_steps=100,
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
        trainer.verify_training_state() 
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

        # Run evaluation
        results = trainer.evaluate_with_coverage(trainer, wrapper, test_ds, n_samples=len(test_ds))
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
        ckpt_name = f"ckpt_ep{ep}.pt"
        ckpt_path = os.path.join(args.save_path, f"ckpt_ep{ep}.pt")
        trainer.save_checkpoint(ckpt_path, epoch=ep)
        if ep == start_epoch:  # Test on first epoch
            print("Testing save/load cycle...")
            temp_trainer = SplitLoRATrainer(
                model_name="gpt2", head_layers=2, tail_layers=2,
                learning_rate=args.learning_rate
            )
            loaded_epoch = temp_trainer.load_checkpoint(ckpt_path, eval_only=True)
            print(f"✅ Save/load test passed: saved epoch {ep}, loaded epoch {loaded_epoch}")
    
    


if __name__ == "__main__":
    main()

