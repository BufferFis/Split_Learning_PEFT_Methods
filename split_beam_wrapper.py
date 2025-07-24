import torch, torch.nn as nn
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutput, BaseModelOutputWithPast
from typing import Optional, Tuple, List, Dict, Union, Any

class SplitGPT2ForGeneration(PreTrainedModel, GenerationMixin):
    def __init__(self, tokenizer,
                 head_client, server, tail_client,
                 base_config):
        super().__init__(base_config)

        # --- 0. dummy parameter so .device is defined -----------------
        self.dummy = nn.Parameter(torch.empty(0), requires_grad=False)

        # --- 1. keep handles to the three split parts -----------------
        self.tokenizer = tokenizer
        self.head_client = head_client
        self.server = server
        self.tail_client = tail_client
        self.config.max_length = 256
        self.config.min_length = 10
        self.config.do_sample = False

        # --- 2. make HF aware of the LM-head / embeddings -------------
        # 2-a expose the lm_head itself
        if hasattr(tail_client.tail_model, 'lm_head'):
            self.lm_head = tail_client.tail_model.lm_head
        elif hasattr(tail_client.tail_model, 'base_model') and hasattr(tail_client.tail_model.base_model, 'lm_head'):
            self.lm_head = tail_client.tail_model.base_model.lm_head
        
        # 2-b retrieve embedding weights from head model
        if hasattr(head_client.head_model, 'wte'):
            self.wte = head_client.head_model.wte
        elif hasattr(head_client.head_model, 'base_model') and hasattr(head_client.head_model.base_model, 'wte'):
            self.wte = head_client.head_model.base_model.wte
        else:
            self.wte = head_client.head_model.base_model.model.wte
            
        # 2-c ensure weight tying is maintained
        self.lm_head.weight = self.wte.weight
            
        # 2-d tie vocab-size to the actual weight matrix
        vocab = self.lm_head.weight.size(0)
        self.config.vocab_size = vocab
        self.generation_config.vocab_size = vocab
        self.config.pad_token_id = tokenizer.pad_token_id
        self.config.eos_token_id = tokenizer.eos_token_id
        
        # Initialize caches - important to track which parts belong to which component
        self.head_past_key_values = None
        self.body_past_key_values = None
        self.tail_past_key_values = None

    # GenerationMixin hook with proper growing attention mask
    # Replace the prepare_inputs_for_generation method in SplitGPT2ForGeneration
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        """Properly prepare inputs with consistent attention masks and position IDs"""
        
        # 1. Handle past key values and determine sequence length
        if past_key_values is not None:
            # Get sequence length from past cache - properly handle nested tuple structure
            if isinstance(past_key_values, tuple) and len(past_key_values) >= 1:
                # Extract head component's past
                head_past = past_key_values[0]
                if isinstance(head_past, tuple) and len(head_past) > 0 and head_past[0] is not None:
                    # Access first layer's key cache size (batch, head, seq, dim)
                    if isinstance(head_past[0], tuple) and len(head_past[0]) >= 2:
                        seq_length_past = head_past[0][0].shape[2]  # [batch, head, seq, dim]
                    else:
                        seq_length_past = 1
                else:
                    seq_length_past = 0
                
                # Only use the last token from input_ids for incremental generation
                input_ids = input_ids[:, -1:]
            else:
                # No valid cache found
                seq_length_past = 0
        else:
            # No past provided
            seq_length_past = 0
        
        # 2. Fix attention mask - CRITICAL for preventing repetition
        if attention_mask is not None and past_key_values is not None and seq_length_past > 0:
            # We need to grow the attention mask to include both past and current tokens
            batch_size = input_ids.shape[0]
            new_seq_length = seq_length_past + input_ids.shape[1]
            
            # Create new attention mask with right dimensions
            new_attention_mask = torch.ones(
                (batch_size, new_seq_length), 
                dtype=attention_mask.dtype, 
                device=attention_mask.device
            )
            
            # Copy old attention values for past positions
            if attention_mask.size(1) >= seq_length_past:
                new_attention_mask[:, :seq_length_past] = attention_mask[:, :seq_length_past]
            
            attention_mask = new_attention_mask
        
        # 3. Fix position IDs - CRITICAL for preventing repetition
        if kwargs.get("use_cache", True) and past_key_values is not None:
            # Create position IDs that point to the positions after past sequence
            position_ids = torch.arange(
                seq_length_past,
                seq_length_past + input_ids.shape[1], 
                dtype=torch.long, 
                device=input_ids.device
            ).unsqueeze(0).expand(input_ids.shape[0], -1)
        else:
            # For first generation step or when not using cache
            position_ids = torch.arange(
                0, 
                input_ids.shape[1], 
                dtype=torch.long, 
                device=input_ids.device
            ).unsqueeze(0).expand(input_ids.shape[0], -1)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            "use_cache": kwargs.get("use_cache", True)
        }

    def can_generate(self):  # required by ≥4.39
        return True

    # ---------- Fixed forward with proper cache handling --------------
    def forward(
        self, 
        input_ids=None, 
        attention_mask=None, 
        past_key_values=None,
        position_ids=None,
        use_cache=True,
        labels=None,
        **kwargs
    ):
        """Completely rewritten forward pass with proper cache handling"""
        
        # Create attention mask if not provided
        if attention_mask is None and input_ids is not None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device)
        
        # Unpack component-specific caches if provided
        head_past = None
        body_past = None
        tail_past = None
        
        if past_key_values is not None:
            if isinstance(past_key_values, tuple) and len(past_key_values) == 3:
                head_past, body_past, tail_past = past_key_values
            else:
                print(f"Warning: Invalid past_key_values format. Got {type(past_key_values)} with length {len(past_key_values) if hasattr(past_key_values, '__len__') else 'unknown'}")
        
        try:
            # Forward through head
            head_output = self.head_client.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=head_past,
                use_cache=use_cache
            )
            
            # Extract hidden states and past from head output
            if isinstance(head_output, tuple) and len(head_output) == 2:
                # If head returns (hidden_states, past_key_values)
                head_hidden, head_past = head_output
            elif hasattr(head_output, 'last_hidden_state'):
                # If head returns BaseModelOutputWithPast object
                head_hidden = head_output.last_hidden_state
                head_past = head_output.past_key_values
            else:
                # Fallback
                head_hidden = head_output
                head_past = None
            
            # Forward through body
            body_output = self.server.forward(
                activations=head_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=body_past,
                use_cache=use_cache
            )
            
            # Extract hidden states and past from body output
            if isinstance(body_output, tuple) and len(body_output) == 2:
                body_hidden, body_past = body_output
            else:
                body_hidden = body_output
                body_past = None
            
            # Forward through tail
            tail_output = self.tail_client.forward(
                body_activations=body_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=tail_past,
                use_cache=use_cache
            )
            
            # Extract logits and past from tail output
            if isinstance(tail_output, tuple) and len(tail_output) == 2:
                logits, tail_past = tail_output
            else:
                logits = tail_output
                tail_past = None
            
            # Combine all past key values
            present = (head_past, body_past, tail_past) if use_cache else None
            
            # Calculate loss if labels are provided
            loss = None
            if labels is not None:
                # Shift logits and labels for causal language modeling
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Get rid of -100 labels
                mask = shift_labels != -100
                shift_labels = torch.where(mask, shift_labels, 0)
                
                # Calculate loss
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                                shift_labels.view(-1))
            
            # Return proper CausalLMOutput - FIXED FOR COMPATIBILITY
            # Check which parameters are accepted by CausalLMOutput in your version
            try:
                return CausalLMOutput(
                    loss=loss,
                    logits=logits,
                    past_key_values=present
                )
            except TypeError:
                # Fall back to a version without past_key_values
                self.cached_past_key_values = present  # Store manually if not supported
                return CausalLMOutput(
                    loss=loss,
                    logits=logits
                )
            
        except Exception as e:
            print(f"Error in split model forward pass: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        """Enhanced generate method with better error handling"""
        try:
            if attention_mask is None and input_ids is not None:
                # Create attention mask if needed
                attention_mask = (input_ids != self.tokenizer.pad_token_id)
            
            # Reset all caches
            self.head_past_key_values = None
            self.body_past_key_values = None
            self.tail_past_key_values = None
            
            # Ensure key parameters are set correctly
            generation_kwargs = {
                "max_new_tokens": kwargs.get("max_new_tokens", 64),
                "do_sample": kwargs.get("do_sample", False),
                "use_cache": True,  # Must be True for generation
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            # Update with user kwargs
            generation_kwargs.update(kwargs)
            
            return super().generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs
            )
            
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            import traceback
            traceback.print_exc()
            raise