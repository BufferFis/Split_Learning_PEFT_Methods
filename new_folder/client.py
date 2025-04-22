import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed.rpc as rpc
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from util import split_gpt2

def run_client(rank, world_size, port, epochs, batch_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # Initialize RPC
    rpc.init_rpc(
        "client",
        rank=rank,
        world_size=world_size
    )

    # Get server's ServerRPC RRef
    # The server must expose a get_server_rref() function that returns its ServerRPC RRef
    server_rref = rpc.rpc_sync("server", get_server_rref, args=())
    head_m.gradient_checkpointing_enable()
    tail_m.gradient_checkpointing_enable()
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    full_model = AutoModelForCausalLM.from_pretrained("gpt2")
    head_m, _, tail_m = split_gpt2(full_model, head_layers=2, tail_layers=2)

    # Apply DoRA (LoRA with magnitude update)
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        use_dora=True, task_type="CAUSAL_LM", target_modules=["c_attn", "c_proj"]
    )
    head_m = get_peft_model(head_m, lora_cfg).to(rank)
    tail_m = get_peft_model(tail_m, lora_cfg).to(rank)
    head_ddp = DDP(head_m, device_ids=[rank])
    tail_ddp = DDP(tail_m, device_ids=[rank])

    # Dataset
    dataset = load_dataset("e2e_nlg", trust_remote_code=True)
    def preprocess(example):
        text = example["meaning_representation"] + " " + example["human_reference"]
        enc = tokenizer(text, padding="max_length", truncation=True, max_length=128)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": enc["input_ids"]
        }
    train = dataset["train"].map(preprocess, remove_columns=dataset["train"].column_names)
    train_sampler = DistributedSampler(train)
    train_dl = DataLoader(
        train, batch_size=batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=True
    )

    # Optimizers and scaler
    head_optimizer = optim.AdamW([p for p in head_ddp.parameters() if p.requires_grad], lr=2e-4)
    tail_optimizer = optim.AdamW([p for p in tail_ddp.parameters() if p.requires_grad], lr=2e-4)
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        head_ddp.train()
        tail_ddp.train()
        total_loss = 0.0
        for batch in train_dl:
            input_ids = batch["input_ids"].to(rank)
            attn_mask = batch["attention_mask"].to(rank)
            labels = batch["labels"].to(rank)

            head_optimizer.zero_grad()
            tail_optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                head_out = head_ddp(
                    input_ids=input_ids, attention_mask=attn_mask,
                    output_hidden_states=True
                )
                head_hid = head_out.hidden_states[-1]

                # Forward to server via RRef
                body_act = server_rref.rpc_sync().forward_train(head_hid.cpu())
                body_act = body_act.to(rank).requires_grad_()

                tail_out = tail_ddp(inputs_embeds=body_act, attention_mask=attn_mask)
                logits = tail_out.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

            scaler.scale(loss).backward()
            grad_output = body_act.grad.cpu()

            # Backward to server via RRef
            input_grad = server_rref.rpc_sync().backward(grad_output)

            # Backprop to head
            head_hid.backward(input_grad.to(rank))

            scaler.step(head_optimizer)
            scaler.step(tail_optimizer)
            scaler.update()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} avg loss: {total_loss/len(train_dl):.4f}")

    rpc.shutdown()

# Helper function to get the server's ServerRPC RRef
def get_server_rref():
    global server_rpc_rref
    return server_rpc_rref

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--port", type=int, default=29500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    run_client(args.rank, args.world_size, args.port, args.epochs, args.batch_size)
