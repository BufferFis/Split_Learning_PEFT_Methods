from fastapi import FastAPI, Request
import torch
import torch.optim as optim
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
import uvicorn
import json
from datetime import datetime
import os
from util import split_gpt2
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import torch.nn as nn

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

app = FastAPI()

# Load and split model
model_name = "gpt2"
full_model = AutoModelForCausalLM.from_pretrained(model_name)


# Split the model
_, body_model, _ = split_gpt2(full_model, head_layers=2, tail_layers=2)

# Apply DoRA
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
body_model = nn.DataParallel(body_model)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
body_model = body_model.to(device)
print(f"Model loaded on {device}")

# Server state management
server_state = {
    "last_hidden_states": None,
    "requires_backward": False,
    "optimizer": None,
    "step_count": 0,
    "epoch_count": 0,
    "metrics": {"loss": [], "pruning_stats": []},
    "training_active": False
}

@app.post("/forward")
async def forward(request: Request):
    """Process forward pass during inference (no gradient tracking)"""
    data = await request.json()
    # Receive activations from client
    activations = torch.tensor(data["activations"]).to(device)
    
    body_model.eval()
    with torch.no_grad():
        output = body_model(inputs_embeds=activations, output_hidden_states=True)
        last_hidden = output.hidden_states[-1]
    
    # Send activations to client
    return {"body_activations": last_hidden.cpu().tolist()}

@app.post("/forward_train")
async def forward_train(request: Request):
    """Process forward pass during training (with gradient tracking)"""
    global server_state
    data = await request.json()
    
    # Receive activations from client
    activations = torch.tensor(data["activations"], requires_grad=True).to(device)
    
    # Set model to training mode
    body_model.train()
    
    # Forward pass with gradient tracking
    output = body_model(inputs_embeds=activations, output_hidden_states=True)
    last_hidden = output.hidden_states[-1]
    
    # Store for backward pass
    server_state["last_hidden_states"] = (activations, last_hidden)
    server_state["requires_backward"] = True
    
    # Send activations to client
    return {"body_activations": last_hidden.detach().cpu().tolist()}

@app.post("/backward")
async def backward(request: Request):
    """Process backward pass with gradients from client"""
    global server_state
    data = await request.json()
    
    if not server_state["requires_backward"] or server_state["last_hidden_states"] is None:
        return {"status": "error", "message": "No tensors available for backward pass"}
    
    # Extract the gradients sent from client
    grad_output = torch.tensor(data["grad_output"]).to(device)
    loss_value = data.get("loss", 0.0)
    
    # Get stored tensors from forward pass
    input_activations, output_activations = server_state["last_hidden_states"]
    
    # Ensure optimizer exists
    if server_state["optimizer"] is None:
        server_state["optimizer"] = optim.AdamW(
            [p for p in body_model.parameters() if p.requires_grad], 
            lr=2e-4
        )
    
    # Zero gradients
    server_state["optimizer"].zero_grad()
    
    # Backward pass
    output_activations.backward(grad_output)
    
    # Update weights
    server_state["optimizer"].step()
    
    # Get input gradients to send back to client
    input_grad = input_activations.grad if input_activations.grad is not None else torch.zeros_like(input_activations)
    
    # Update metrics
    server_state["metrics"]["loss"].append(loss_value)
    server_state["step_count"] += 1
    
    # Reset state
    server_state["last_hidden_states"] = None
    server_state["requires_backward"] = False

    if server_state["step_count"] % 100 == 0:
        torch.cuda.empty_cache()
    
    return {
        "grad_input": input_grad.cpu().tolist(),
        "step": server_state["step_count"]
    }

@app.post("/start_training")
async def start_training(request: Request):
    """Initialize training state on server"""
    global server_state
    data = await request.json()
    
    # Reset training state
    server_state["step_count"] = 0
    server_state["epoch_count"] = 0
    server_state["metrics"] = {"loss": [], "pruning_stats": []}
    server_state["training_active"] = True
    
    # Initialize optimizer with learning rate from client
    lr = data.get("learning_rate", 2e-4)
    server_state["optimizer"] = optim.AdamW(
        [p for p in body_model.module.parameters() if p.requires_grad], 
        lr=lr
    )
    
    trainable_params = sum(p.numel() for p in body_model.parameters() if p.requires_grad)
    
    return {
        "status": "Training initialized", 
        "trainable_params": trainable_params,
        "device": str(device)
    }

@app.post("/end_epoch")
async def end_epoch(request: Request):
    """Handle end of epoch, save metrics"""
    global server_state
    data = await request.json()
    
    server_state["epoch_count"] += 1
    
    # Calculate average loss for epoch
    if server_state["metrics"]["loss"]:
        avg_loss = sum(server_state["metrics"]["loss"]) / len(server_state["metrics"]["loss"])
    else:
        avg_loss = 0.0
    
    # Save metrics at end of training
    if data.get("is_final", False):
        server_state["training_active"] = False
        
        # Save metrics to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = f"server_metrics_{timestamp}.json"
        
        with open(metrics_file, "w") as f:
            json.dump({
                "epochs": server_state["epoch_count"],
                "steps": server_state["step_count"],
                "final_loss": avg_loss,
                "pruning_stats": server_state["metrics"]["pruning_stats"]
            }, f, indent=2)
    
    # Reset loss tracking for next epoch
    server_state["metrics"]["loss"] = []
    
    return {
        "status": "ok", 
        "epoch": server_state["epoch_count"], 
        "avg_loss": avg_loss
    }

@app.post("/save_model")
async def save_model(request: Request):
    """Save the server model"""
    data = await request.json()
    save_path = data.get("path", "./server_model")
    
    os.makedirs(save_path, exist_ok=True)
    body_model.module.save_pretrained(save_path)  # Use .module
    return {"status": "Model saved", "path": save_path}

@app.post("/load_model")
async def load_model(request: Request):
    """Load a saved server model"""
    global body_model
    data = await request.json()
    load_path = data.get("path")
    
    if not os.path.exists(load_path):
        return {"status": "error", "message": f"Path {load_path} does not exist"}
    
    if hasattr(body_model, "module"):
        body_model.module = PeftModel.from_pretrained(body_model.module, load_path)
    else:
        body_model = PeftModel.from_pretrained(body_model, load_path)
    return {"status": "Model loaded", "path": load_path}

@app.get("/model_info")
async def model_info():
    """Get information about the server model"""
    trainable_params = sum(p.numel() for p in body_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in body_model.parameters())
    
    return {
        "model_name": model_name,
        "device": str(device),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "training_active": server_state["training_active"],
        "steps": server_state["step_count"],
        "epochs": server_state["epoch_count"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
