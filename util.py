import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2LMHeadModel

def split_gpt2(model, head_layers=2, tail_layers=2):
    """
    Properly split GPT2 model into head, body, and tail parts
    """
    total_layers = len(model.transformer.h)
    body_layers = total_layers - head_layers - tail_layers
    
    if body_layers <= 0:
        raise ValueError(f"Not enough layers to split. Total: {total_layers}, Head: {head_layers}, Tail: {tail_layers}")
    
    print(f"Splitting model: Head({head_layers}) + Body({body_layers}) + Tail({tail_layers}) = {total_layers}")
    
    # Create head model (embedding + first few layers)
    class HeadModel(nn.Module):
        def __init__(self, original_model, num_layers):
            super().__init__()
            self.wte = original_model.transformer.wte
            self.wpe = original_model.transformer.wpe
            self.drop = original_model.transformer.drop
            self.h = nn.ModuleList(original_model.transformer.h[:num_layers])
            self.config = original_model.config
            
            # CRITICAL FIX: Add missing generation attributes
            self.generation_config = getattr(original_model, 'generation_config', None)
            self.main_input_name = getattr(original_model, 'main_input_name', 'input_ids')
            
            # Copy essential methods from original model to prevent PEFT errors
            if hasattr(original_model, 'prepare_inputs_for_generation'):
                self.prepare_inputs_for_generation = original_model.prepare_inputs_for_generation
            if hasattr(original_model, 'can_generate'):
                self.can_generate = original_model.can_generate
            if hasattr(original_model, '_reorder_cache'):
                self._reorder_cache = original_model._reorder_cache
            if hasattr(original_model, 'get_input_embeddings'):
                self.get_input_embeddings = original_model.get_input_embeddings
            if hasattr(original_model, 'get_output_embeddings'):
                self.get_output_embeddings = original_model.get_output_embeddings
                
        def __getattr__(self, name: str):
            """Forward missing attributes to prevent PEFT errors"""
            try:
                return super().__getattr__(name)
            except AttributeError:
                # Return a dummy function for generation methods to prevent errors
                if 'generation' in name or 'prepare_inputs' in name:
                    return lambda *args, **kwargs: None
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        def _expand_attention_mask(self, attention_mask, hidden_states):
            # Only expand if not already 4D
            if attention_mask is not None and attention_mask.dim() == 2:
                batch_size, seq_len = attention_mask.shape
                num_heads = self.config.n_head
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(3)  # [batch, 1, seq_len, 1]
                attention_mask = attention_mask.expand(batch_size, num_heads, seq_len, seq_len)  # [batch, num_heads, seq_len, seq_len]
            return attention_mask


        def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **kwargs):
            # Token + position embeddings
            inputs_embeds = self.wte(input_ids)
            seq_length = input_ids.size(-1)
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
            position_embeds = self.wpe(position_ids)
            
            hidden_states = inputs_embeds + position_embeds
            hidden_states = self.drop(hidden_states)
            attention_mask = self._expand_attention_mask(attention_mask, hidden_states)

             # Ensure consistent dtype
            if attention_mask is not None and attention_mask.dtype != hidden_states.dtype:
                attention_mask = attention_mask.to(hidden_states.dtype)

            all_hidden_states = ()
            # Process through head layers
            for block in self.h:
                expanded_mask = self._expand_attention_mask(attention_mask, hidden_states)
                hidden_states = block(hidden_states, attention_mask=expanded_mask)[0]
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            if output_hidden_states:
                return type('HeadOutput', (), {
                    'last_hidden_state': hidden_states,
                    'hidden_states': all_hidden_states
                })()
            else:
                return type('HeadOutput', (), {'last_hidden_state': hidden_states})()
    
    # Create body model (middle layers only) - FIXED FOR PEFT
    class BodyModel(nn.Module):
        def __init__(self, original_model, start_layer, num_layers):
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList(
                original_model.transformer.h[start_layer:start_layer + num_layers]
            )
            self.transformer.ln_f = original_model.transformer.ln_f
            self.config = original_model.config
            
            # CRITICAL FIX: Add missing generation attributes
            self.generation_config = getattr(original_model, 'generation_config', None)
            self.main_input_name = getattr(original_model, 'main_input_name', 'input_ids')
            
            # Copy essential methods from original model to prevent PEFT errors
            if hasattr(original_model, 'prepare_inputs_for_generation'):
                self.prepare_inputs_for_generation = original_model.prepare_inputs_for_generation
            if hasattr(original_model, 'can_generate'):
                self.can_generate = original_model.can_generate
            if hasattr(original_model, '_reorder_cache'):
                self._reorder_cache = original_model._reorder_cache
            if hasattr(original_model, 'get_input_embeddings'):
                self.get_input_embeddings = original_model.get_input_embeddings
            if hasattr(original_model, 'get_output_embeddings'):
                self.get_output_embeddings = original_model.get_output_embeddings
                
        def __getattr__(self, name: str):
            """Forward missing attributes to prevent PEFT errors"""
            # This is the key fix from search results
            try:
                return super().__getattr__(name)
            except AttributeError:
                # Return a dummy function for generation methods to prevent errors
                if 'generation' in name or 'prepare_inputs' in name:
                    return lambda *args, **kwargs: None
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        def _expand_attention_mask(self, attention_mask, hidden_states):
            # Same as HeadModel's version
            if attention_mask is not None and attention_mask.dim() == 2:
                batch_size, seq_len = attention_mask.shape
                num_heads = self.config.n_head
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(3)
                attention_mask = attention_mask.expand(batch_size, num_heads, seq_len, seq_len)
            return attention_mask



        def forward(self, hidden_states=None, attention_mask=None, **kwargs):
            # Handle both direct hidden_states and input_ids (for PEFT compatibility)
            if hidden_states is None and 'input_ids' in kwargs:
                # This shouldn't happen on server, but handle gracefully
                raise ValueError("BodyModel received input_ids instead of hidden_states")
                
            # Process through body layers
            for block in self.transformer.h:
                expanded_mask = self._expand_attention_mask(attention_mask, hidden_states)
                if expanded_mask is not None:
                    hidden_states = block(hidden_states, attention_mask=expanded_mask)[0]
                else:
                    hidden_states = block(hidden_states)[0]
            
            # Apply final layer norm
            hidden_states = self.transformer.ln_f(hidden_states)
            return type('BodyOutput', (), {'last_hidden_state': hidden_states})()
    
    # Create tail model (last few layers + language modeling head)
    class TailModel(nn.Module):
        def __init__(self, original_model, start_layer):
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList(original_model.transformer.h[start_layer:])
            self.lm_head = original_model.lm_head
            self.config = original_model.config
            
            # CRITICAL FIX: Add missing generation attributes
            self.generation_config = getattr(original_model, 'generation_config', None)
            self.main_input_name = getattr(original_model, 'main_input_name', 'input_ids')
            
            # Copy essential methods from original model to prevent PEFT errors
            if hasattr(original_model, 'prepare_inputs_for_generation'):
                self.prepare_inputs_for_generation = original_model.prepare_inputs_for_generation
            if hasattr(original_model, 'can_generate'):
                self.can_generate = original_model.can_generate
            if hasattr(original_model, '_reorder_cache'):
                self._reorder_cache = original_model._reorder_cache
            if hasattr(original_model, 'get_input_embeddings'):
                self.get_input_embeddings = original_model.get_input_embeddings
            if hasattr(original_model, 'get_output_embeddings'):
                self.get_output_embeddings = original_model.get_output_embeddings
                
        def __getattr__(self, name: str):
            """Forward missing attributes to prevent PEFT errors"""
            try:
                return super().__getattr__(name)
            except AttributeError:
                # Return a dummy function for generation methods to prevent errors
                if 'generation' in name or 'prepare_inputs' in name:
                    return lambda *args, **kwargs: None
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
            
        def _expand_attention_mask(self, attention_mask, hidden_states):
                if attention_mask is not None and attention_mask.dim() == 2:
                    batch_size, seq_len = attention_mask.shape
                    num_heads = self.config.n_head
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(3)
                    attention_mask = attention_mask.expand(batch_size, num_heads, seq_len, seq_len)
                return attention_mask
        
        def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
            hidden_states = inputs_embeds
            
            
            # Process through tail layers
            for block in self.transformer.h:
                expanded_mask = self._expand_attention_mask(attention_mask, hidden_states)
                hidden_states = block(hidden_states, attention_mask=expanded_mask)[0]
            
            # Generate logits
            logits = self.lm_head(hidden_states)
            return type('TailOutput', (), {'logits': logits})()

    head_model = HeadModel(model, head_layers)
    body_model = BodyModel(model, head_layers, body_layers)
    tail_model = TailModel(model, head_layers + body_layers)

    return head_model, body_model, tail_model