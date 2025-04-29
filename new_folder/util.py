def split_gpt2(model, head_layers=2, tail_layers=2):
    blocks = model.transformer.h
    n_layers = len(blocks)
    head = blocks[:head_layers]
    body = blocks[head_layers:n_layers-tail_layers]
    tail = blocks[n_layers-tail_layers:]
    from copy import deepcopy
    import torch.nn as nn
    head_model = deepcopy(model)
    body_model = deepcopy(model)
    tail_model = deepcopy(model)
    head_model.transformer.h = nn.ModuleList(head)
    body_model.transformer.h = nn.ModuleList(body)
    tail_model.transformer.h = nn.ModuleList(tail)
    
    # Remove embeddings from body and tail
    class IdentityModule(nn.Module):
        def forward(self, x):
            return x
    
    # Replace embeddings with identity
    body_model.transformer.wte = IdentityModule()
    body_model.transformer.wpe = IdentityModule()
    tail_model.transformer.wte = IdentityModule()
    tail_model.transformer.wpe = IdentityModule()
    
    return head_model, body_model, tail_model