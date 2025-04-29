import os
import torch
import torch.optim as optim
import torch.distributed.rpc as rpc

from util import split_gpt2
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import rpc_functions as rf

# RPC setup
os.environ["MASTER_ADDR"] = "0.0.0.0"
os.environ["MASTER_PORT"] = "29500"

def run_server():
    # Init RPC first
    rpc.init_rpc(name="server", rank=0, world_size=2,
                 rpc_backend_options=rpc.TensorPipeRpcBackendOptions(num_worker_threads=16))

    # Load & split model
    full = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    _, body, _ = split_gpt2(full, head_layers=2, tail_layers=2)
    body.prepare_inputs_for_generation = full.prepare_inputs_for_generation

    # Apply LoRA
    lora_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                          use_dora=True, task_type="CAUSAL_LM",
                          target_modules=["c_attn", "c_proj"])
    body_model = get_peft_model(body, lora_cfg).to("cuda").train()
    optimizer = optim.AdamW(body_model.parameters(), lr=2e-4)

    # Register shared model and optimizer in RPC functions
    rf.init(body_model, optimizer)

    print(">>> RPC server (rank 0) UP on 0.0.0.0:29500", flush=True)

    try:
        while True:
            pass
    except KeyboardInterrupt:
        rpc.shutdown()

if __name__ == "__main__":
    run_server()
