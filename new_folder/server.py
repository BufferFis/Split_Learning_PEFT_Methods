import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from util import split_gpt2

def run_server(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.cuda.set_device(rank)

    # Initialize RPC
    rpc.init_rpc(
        f"server",
        rank=rank,
        world_size=world_size
    )

    # Load and split model
    model_name = "gpt2"
    full_model = AutoModelForCausalLM.from_pretrained(model_name)
    _, body_model, _ = split_gpt2(full_model, head_layers=2, tail_layers=2)
    body_model.gradient_checkpointing_enable()
    # Apply DoRA (LoRA with magnitude update)
    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        use_dora=True, task_type="CAUSAL_LM", target_modules=["c_attn", "c_proj"]
    )
    body_model = get_peft_model(body_model, lora_config)
    body_model = body_model.to(rank)
    body_model = DDP(body_model, device_ids=[rank])

    optimizer = optim.AdamW([p for p in body_model.parameters() if p.requires_grad], lr=2e-4)
    scaler = torch.cuda.amp.GradScaler()

    # Server state
    server_state = {
        "body_model": body_model,
        "optimizer": optimizer,
        "scaler": scaler
    }

    class ServerRPC:
        def forward_train(self, activations):
            body_model.train()
            activations = activations.detach().to(rank).requires_grad_()
            with torch.cuda.amp.autocast():
                out = body_model(inputs_embeds=activations, output_hidden_states=True)
            last_hidden = out.hidden_states[-1]
            # Store for backward
            self.last_input = activations
            self.last_output = last_hidden
            return last_hidden.detach().cpu()

        def backward(self, grad_output):
            optimizer.zero_grad()
            self.last_output.backward(grad_output.to(rank))
            scaler.step(optimizer)
            scaler.update()
            input_grad = self.last_input.grad.detach().cpu()
            return input_grad

    rpc.RpcBackendOptions(num_worker_threads=16)
    server_rpc = ServerRPC()
    rpc.shutdown()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--port", type=int, default=29500)
    args = parser.parse_args()
    run_server(args.rank, args.world_size, args.port)
