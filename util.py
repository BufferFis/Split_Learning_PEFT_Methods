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
            
        def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **kwargs):
            # Token + position embeddings
            inputs_embeds = self.wte(input_ids)
            seq_length = input_ids.size(-1)
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
            position_embeds = self.wpe(position_ids)
            
            hidden_states = inputs_embeds + position_embeds
            hidden_states = self.drop(hidden_states)
            
            all_hidden_states = ()
            # Process through head layers
            for block in self.h:
                if attention_mask is not None:
                    hidden_states = block(hidden_states, attention_mask=attention_mask)[0]
                else:
                    hidden_states = block(hidden_states)[0]
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            if output_hidden_states:
                return type('HeadOutput', (), {
                    'last_hidden_state': hidden_states,
                    'hidden_states': all_hidden_states
                })()
            else:
                return type('HeadOutput', (), {'last_hidden_state': hidden_states})()
    
    # Create body model (middle layers only)
    class BodyModel(nn.Module):
        def __init__(self, original_model, start_layer, num_layers):
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList(
                original_model.transformer.h[start_layer:start_layer + num_layers]
            )
            self.transformer.ln_f = original_model.transformer.ln_f
            self.config = original_model.config
            
        def forward(self, hidden_states, attention_mask=None, **kwargs):
            # Process through body layers
            for block in self.transformer.h:
                if attention_mask is not None:
                    hidden_states = block(hidden_states, attention_mask=attention_mask)[0]
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
            
        def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
            hidden_states = inputs_embeds
            
            # Process through tail layers
            for block in self.transformer.h:
                if attention_mask is not None:
                    hidden_states = block(hidden_states, attention_mask=attention_mask)[0]
                else:
                    hidden_states = block(hidden_states)[0]
            
            # Generate logits
            logits = self.lm_head(hidden_states)
            return type('TailOutput', (), {'logits': logits})()
    
    # Create the three models
    head_model = HeadModel(model, head_layers)
    body_model = BodyModel(model, head_layers, body_layers)
    tail_model = TailModel(model, head_layers + body_layers)
    
    return head_model, body_model, tail_model
