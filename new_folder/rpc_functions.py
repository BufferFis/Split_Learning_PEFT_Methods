# rpc_functions.py

import torch

body_model = None
optimizer = None

def init(body, opt):
    global body_model, optimizer
    body_model = body
    optimizer = opt

def forward_train(activations, attention_mask):
    activations = activations.to("cuda").requires_grad_()
    mask = attention_mask.to("cuda")
    out = body_model(inputs_embeds=activations, attention_mask=mask)
    return out[0].cpu(), activations.cpu()


def backward(ctx, grad_output):
    activations = ctx.to("cuda").requires_grad_()
    mask = torch.ones(activations.size()[:2], device="cuda").long()
    optimizer.zero_grad(set_to_none=True)
    out = body_model(inputs_embeds=activations, attention_mask=mask)
    out.backward(grad_output.to("cuda"))
    optimizer.step()
    return activations.grad.cpu()


