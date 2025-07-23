import torch, torch.nn as nn
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutput, BaseModelOutputWithPast
from typing import Optional, Tuple, List, Dict, Union, Any

class SplitGPT2ForGeneration(PreTrainedModel, GenerationMixin):
    def __init__(self, tokenizer,
                 head_client, server, tail_client,
                 base_config):
        super().__init__(base_config)

        # --- 0.  dummy parameter so .device is defined -----------------
        self.dummy = nn.Parameter(torch.empty(0), requires_grad=False)

        # --- 1.  keep handles to the three split parts -----------------
        self.tokenizer   = tokenizer
        self.head_client = head_client
        self.server      = server
        self.tail_client = tail_client
        self.config.max_length = 256
        self.config.min_length = 10
        self.config.do_sample = False

        # --- 2.  make HF aware of the LM-head / embeddings -------------
        # 2-a  expose the lm_head itself
        self.lm_head = tail_client.tail_model.lm_head
        self.lm_head.weight = (
            head_client.head_model.base_model.wte.weight
        )
        # 2-b  tie vocab-size to the actual weight matrix
        vocab = self.lm_head.weight.size(0)
        self.config.vocab_size              = vocab                   
        self.generation_config.vocab_size   = vocab                   
        self.config.pad_token_id            = tokenizer.pad_token_id
        self.config.eos_token_id            = tokenizer.eos_token_id
        
        # Initialize caches
        self.past_key_values = None

    # GenerationMixin hook with proper growing attention mask
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        # Only the last token for inputs_ids if past is defined
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            
            # CRITICAL FIX: Grow attention mask to match the full sequence length
            if attention_mask is not None:
                # Get the sequence length from past key values
                seq_length_past = past_key_values[0][0].shape[2]
                batch_size = input_ids.shape[0]
                
                # Create a new attention mask of the correct total length
                total_length = seq_length_past + input_ids.shape[1]
                
                # Create new attention mask of the right size
                new_attention_mask = torch.ones(
                    (batch_size, total_length), 
                    dtype=attention_mask.dtype, 
                    device=attention_mask.device
                )
                
                # Copy over the old values and set new position to 1
                if seq_length_past > 0:
                    new_attention_mask[:, :seq_length_past] = attention_mask[:, :seq_length_past]
                
                attention_mask = new_attention_mask
        
        # Prepare position IDs (critical for generation)
        position_ids = None
        if past_key_values is not None:
            # For generation steps after the first, set position IDs to the last position + 1
            position_ids = torch.full(
                (input_ids.shape[0], input_ids.shape[1]),
                past_key_values[0][0].shape[2],  # Get seq_length from past
                dtype=torch.long,
                device=input_ids.device
            )
        else:
            # First generation step, create position IDs for the full sequence
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(input_ids.shape[0], -1)
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            "use_cache": True,  # Always enable caching during generation
        }

    def can_generate(self):                       # required by ≥4.39
        return True

    # ---------- standard forward with proper caching and attention --------------
    def forward(
        self, 
        input_ids: torch.LongTensor, 
        attention_mask: Optional[torch.FloatTensor] = None, 
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = True,
        **kwargs
    ):
        """Fixed forward pass with proper attention mask and caching"""
        
        # Create attention mask if not provided
        if attention_mask is None:
            # When generating, we need to create the full attention mask
            if past_key_values is not None:
                # For generation steps after first, create full attention mask
                seq_length_past = past_key_values[0][0].shape[2]
                total_length = seq_length_past + input_ids.shape[1]
                attention_mask = torch.ones(
                    (input_ids.shape[0], total_length), 
                    dtype=torch.float, 
                    device=input_ids.device
                )
            else:
                # For first step or training
                attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
        
        # Create position_ids if not provided
        if position_ids is None:
            if past_key_values is not None:
                # For generation steps after first
                # Get position of the new tokens (equal to seq_length of past)
                seq_length_past = past_key_values[0][0].shape[2]
                position_ids = torch.full(
                    (input_ids.shape[0], input_ids.shape[1]),
                    seq_length_past,
                    dtype=torch.long,
                    device=input_ids.device
                )
            else:
                # For first step or training
                position_ids = torch.arange(
                    0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
                ).unsqueeze(0).expand(input_ids.shape[0], -1)
        
        # Forward through head with cache passing
        head_output, head_past = self.head_client.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values[0] if past_key_values else None,
            use_cache=use_cache
        )
        
        # Forward through body with cache passing
        body_output, body_past = self.server.forward(
            hidden_states=head_output,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values[1] if past_key_values else None,
            use_cache=use_cache
        )
        
        # Forward through tail
        logits, tail_past = self.tail_client.forward(
            inputs_embeds=body_output,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values[2] if past_key_values else None,
            use_cache=use_cache
        )
        
        # Combine past key values from all components
        if use_cache:
            past_key_values = (head_past, body_past, tail_past)
        else:
            past_key_values = None
        
        return CausalLMOutput(
            logits=logits,
            past_key_values=past_key_values
        )

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        if attention_mask is None:
            pad = self.tokenizer.pad_token_id
            attention_mask = (input_ids != pad)
        
        # Ensure caches are reset
        self.past_key_values = None
        
        return super().generate(input_ids=input_ids,
                                attention_mask=attention_mask,
                                use_cache=True,  # Enable caching
                                **kwargs)