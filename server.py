from fastapi import FastAPI, Request
import torch
import torch.optim as optim
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM
import uvicorn
import json
from datetime import datetime
import os
from util import split_gpt2
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ----- DDP setup (if used) -----
def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

app = FastAPI()

# Load and split model
model_name = "gpt2"
full_model = AutoModelForCausalLM.from_pretrained(model_name)

# Split out body (middle) layers
_, body_model, _ = split_gpt2(full_model, head_layers=2, tail_layers=2)

# Apply LoRA/Dora
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_dora=True,
    task_type="CAUSAL_LM",
    target_modules=["c_attn", "c_proj"]
)
body_model = get_peft_model(body_model, lora_config)
body_model = torch.nn.DataParallel(body_model)

# Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
body_model = body_model.to(device)
print(f"Body model loaded on {device}")

# Server state
server_state = {
    "last_activations": None,
    "requires_backward": False,
    "optimizer": None,
    "step_count": 0,
    "epoch_count": 0,
    "metrics": {"loss": []},
    "training_active": False
}

# Helper: run hidden states through transformer blocks (no embedding)
def run_body_layers(activations, attention_mask):
    # activations: [batch, seq_len, hidden_dim]
    # attention_mask: [batch, seq_len]
    body = body_model.module if hasattr(body_model, "module") else body_model
    hidden = activations
    # iterate each block in this body slice
    for block in body.transformer.h:
        hidden, _ = block(hidden, attention_mask=attention_mask)
    # final layer norm
    hidden = body.transformer.ln_f(hidden)
    return hidden

@app.post("/forward")
async def forward(request: Request):
    """
    Inference forward: client sends hidden activations + mask, returns next hidden
    """
    payload = await request.json()
    activations = torch.tensor(payload["activations"], device=device)
    attention_mask = torch.tensor(payload.get("attention_mask"), device=device)
    body_model.eval()
    with torch.no_grad():
        last_hidden = run_body_layers(activations, attention_mask)
    return {"body_activations": last_hidden.cpu().tolist()}

@app.post("/forward_train")
async def forward_train(request: Request):
    """
    Training forward: store activations for backward, return new hidden
    """
    data = await request.json()
    activations = torch.tensor(data["activations"], requires_grad=True, device=device)
    attention_mask = torch.tensor(data.get("attention_mask"), device=device)

    body_model.train()
    # forward through body blocks
    last_hidden = run_body_layers(activations, attention_mask)

    # stash for backward
    server_state["last_activations"] = (activations, attention_mask)
    server_state["requires_backward"] = True

    return {"body_activations": last_hidden.detach().cpu().tolist()}

@app.post("/backward")
async def backward(request: Request):
    """
    Training backward: receives dL/d(hidden), updates body weights, returns dL/d(activations)
    """
    data = await request.json()
    if not server_state["requires_backward"]:
        return {"status": "error", "message": "No forward state"}

    grad_output = torch.tensor(data["grad_output"], device=device)
    loss_val = data.get("loss", 0.0)

    activations, attention_mask = server_state["last_activations"]

    # init optimizer if needed
    if server_state["optimizer"] is None:
        server_state["optimizer"] = optim.AdamW(
            [p for p in body_model.parameters() if p.requires_grad], lr=2e-4
        )

    opt = server_state["optimizer"]
    opt.zero_grad()

    # backward manually: run forward to get last_hidden with grad
    last_hidden = run_body_layers(activations, attention_mask)
    last_hidden.backward(grad_output)

    opt.step()

    # gradient wrt input activations
    input_grad = activations.grad if activations.grad is not None else torch.zeros_like(activations)

    # cleanup
    server_state["last_activations"] = None
    server_state["requires_backward"] = False
    server_state["step_count"] += 1
    server_state["metrics"]["loss"].append(loss_val)

    return {"grad_input": input_grad.cpu().tolist(), "step": server_state["step_count"]}

@app.post("/start_training")
async def start_training(request: Request):
    data = await request.json()
    lr = data.get("learning_rate", 2e-4)
    server_state.update({
        "step_count": 0,
        "epoch_count": 0,
        "metrics": {"loss": []},
        "training_active": True,
        "optimizer": optim.AdamW(
            [p for p in body_model.parameters() if p.requires_grad], lr=lr
        )
    })
    return {"status": "initialized", "trainable_params": sum(p.numel() for p in body_model.parameters() if p.requires_grad)}

@app.post("/end_epoch")
async def end_epoch(request: Request):
    data = await request.json()
    server_state["epoch_count"] += 1
    avg_loss = (sum(server_state["metrics"]["loss"]) / len(server_state["metrics"]["loss"])) if server_state["metrics"]["loss"] else 0.0
    if data.get("is_final", False):
        server_state["training_active"] = False
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"server_metrics_{ts}.json", "w") as f:
            json.dump({
                "epochs": server_state["epoch_count"],
                "steps": server_state["step_count"],
                "final_loss": avg_loss
            }, f, indent=2)
    server_state["metrics"]["loss"] = []
    return {"status": "ok", "epoch": server_state["epoch_count"], "avg_loss": avg_loss}

@app.post("/save_model")
async def save_model(request: Request):
    data = await request.json()
    path = data.get("path", "./server_model")
    os.makedirs(path, exist_ok=True)
    # save the peft-wrapped body
    if hasattr(body_model, "module"):
        body_model.module.save_pretrained(path)
    else:
        body_model.save_pretrained(path)
    # save optimizer
    torch.save(server_state["optimizer"].state_dict(), os.path.join(path, "optimizer.pt"))
    return {"status": "saved", "path": path}

@app.post("/load_model")
async def load_model(request: Request):
    data = await request.json()
    path = data.get("path")
    if not os.path.isdir(path):
        return {"status": "error", "message": f"Path {path} not found"}
    # load peft model
    if hasattr(body_model, "module"):
        body_model.module = PeftModel.from_pretrained(body_model.module, path)
    else:
        body_model = PeftModel.from_pretrained(body_model, path)
    # load optimizer
    opt_path = os.path.join(path, "optimizer.pt")
    if os.path.exists(opt_path) and server_state.get("optimizer") is not None:
        server_state["optimizer"].load_state_dict(torch.load(opt_path))
    return {"status": "loaded", "path": path}

@app.get("/model_info")
async def model_info():
    return {
        "model_name": model_name,
        "device": str(device),
        "trainable_params": sum(p.numel() for p in body_model.parameters() if p.requires_grad),
        "steps": server_state["step_count"],
        "epochs": server_state["epoch_count"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
