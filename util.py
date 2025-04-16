from transformers import AutoModelForCausalLM

def split_gpt2(model, head_layers=2, tail_layers=2):
    # Get all transformer blocks
    blocks = model.transformer.h
    n_layers = len(blocks)
    # Indices
    head = blocks[:head_layers]
    body = blocks[head_layers:n_layers-tail_layers]
    tail = blocks[n_layers-tail_layers:]
    # Create submodules
    from copy import deepcopy
    import torch.nn as nn
    head_model = deepcopy(model)
    body_model = deepcopy(model)
    tail_model = deepcopy(model)
    # Assign only relevant layers
    head_model.transformer.h = nn.ModuleList(head)
    body_model.transformer.h = nn.ModuleList(body)
    tail_model.transformer.h = nn.ModuleList(tail)
    return head_model, body_model, tail_model
