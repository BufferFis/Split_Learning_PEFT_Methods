import os
import torch
import torch.optim as optim
import torch.distributed.rpc as rpc
from util import split_gpt2
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.amp import autocast, GradScaler
import rpc_functions as rf

# RPC setup
os.environ["MASTER_ADDR"] = "0.0.0.0"
os.environ["MASTER_PORT"] = "29500"
rpc.init_rpc(name="client", rank=1, world_size=2,
             rpc_backend_options=rpc.TensorPipeRpcBackendOptions(num_worker_threads=16))

# Load & split model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
full = AutoModelForCausalLM.from_pretrained("gpt2").eval()
head, _, tail = split_gpt2(full, head_layers=2, tail_layers=2)
head.config = full.config
tail.config = full.config

# Apply LoRA
lora_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                      use_dora=True, task_type="CAUSAL_LM",
                      target_modules=["c_attn", "c_proj"])
head = get_peft_model(head, lora_cfg).to("cuda").train()
tail = get_peft_model(tail, lora_cfg).to("cuda").train()

# Optimizer and mixed precision scaler
head_opt = optim.AdamW(head.parameters(), lr=2e-4)
tail_opt = optim.AdamW(tail.parameters(), lr=2e-4)
scaler = GradScaler()

# Dataset preprocessing
ds = load_dataset("e2e_nlg", trust_remote_code=True)

def preprocess(examples):
    return tokenizer(
        [mr + " " + ref for mr, ref in zip(examples["meaning_representation"], examples["human_reference"])],
        truncation=True, padding="max_length", max_length=128
    )

train = ds["train"].map(preprocess, batched=True)
loader = torch.utils.data.DataLoader(
    train, batch_size=8, shuffle=True,
    collate_fn=lambda xs: {
        "input_ids": torch.tensor([x["input_ids"] for x in xs]),
        "attention_mask": torch.tensor([x["attention_mask"] for x in xs])
    }
)

# Training loop
loss_fn = torch.nn.CrossEntropyLoss()
for epoch in range(3):
    total_loss = 0.0
    for batch in loader:
        ids  = batch["input_ids"].to("cuda")
        mask = batch["attention_mask"].to("cuda")

        head_opt.zero_grad(set_to_none=True)
        tail_opt.zero_grad(set_to_none=True)

        # HEAD: no autocast, because embedding layers expect int64 tokens
        h = head.transformer(ids, attention_mask=mask)[0]

        h = h.detach().requires_grad_()

        # BODY via RPC
        body_out, ctx = rpc.rpc_sync("server", rf.forward_train, args=(h.cpu(), mask.cpu()))
        b = body_out.to("cuda").requires_grad_()

        # TAIL + LOSS: autocast is safe here
        with autocast("cuda"):
            logits = tail(inputs_embeds=b, attention_mask=mask).logits


            loss = loss_fn(
                logits[..., :-1, :].reshape(-1, logits.size(-1)),
                ids[..., 1:].reshape(-1)
            )

        # Backward + optimizer
        scaler.scale(loss).backward(retain_graph=True)
        grad_b = b.grad.cpu()
        grad_h = rpc.rpc_sync("server", rf.backward, args=(ctx, grad_b))
        scaler.unscale_(head_opt)
        h.backward(grad_h.to("cuda"))
        scaler.step(tail_opt)
        scaler.step(head_opt)
        scaler.update()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} avg loss: {total_loss/len(loader):.4f}")

rpc.shutdown()
