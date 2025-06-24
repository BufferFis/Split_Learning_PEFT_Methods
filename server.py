#server.py
from fastapi import FastAPI, Request, HTTPException
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM
import uvicorn
from datetime import datetime
import os
from util import split_gpt2
import traceback
import asyncio
import json
from datetime import timedelta  
from torch.amp import autocast
from torch.cuda.amp import GradScaler




app = FastAPI()

# Global variables
body_model = None
device = None
server_state = {
    "last_activations": {},  # Store per-rank activations
    "requires_backward": {},
    "optimizer": None,
    "step_count": 0,
    "epoch_count": 0,
    "metrics": {"loss": []},
    "training_active": False,
    "is_distributed": False,
    "local_rank": 0,
    "world_size": 1
}

def setup_distributed():
    """Setup distributed training for server with extended timeout"""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        # Initialize distributed process group with extended timeout
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(hours=2)  # 2 hour timeout for heavy operations
        )
        torch.cuda.set_device(local_rank)
        
        server_state["is_distributed"] = True
        server_state["local_rank"] = local_rank
        server_state["world_size"] = world_size
        
        print(f"Server rank {local_rank}/{world_size} initialized with extended timeout")
        return local_rank, world_size
    else:
        return 0, 1


def initialize_model():
    global body_model, device
    
    local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
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
    body_model = body_model.to(device)
    
    # Wrap with DDP for distributed training
    if server_state["is_distributed"]:
        body_model = DDP(body_model, device_ids=[local_rank])
        print(f"Body model wrapped with DDP on GPU {local_rank}")
    
    print(f"Body model loaded on {device}, distributed: {server_state['is_distributed']}")

# Initialize model on startup
initialize_model()

def ensure_device_consistency():
    """Ensure all model parameters are on correct device"""
    if body_model is not None:
        for param in body_model.parameters():
            if param.device != device:
                param.data = param.data.to(device)
                if param.grad is not None:
                    param.grad = param.grad.to(device)

# Call this after model loading
ensure_device_consistency()


def get_server_url_for_rank(client_rank, base_url="http://127.0.0.1:8000"):
    """Load balance client requests across server ranks"""
    # Simple round-robin assignment
    server_rank = client_rank % server_state["world_size"] if server_state["is_distributed"] else 0
    port = 8000 + server_rank
    return f"http://127.0.0.1:{port}"

def get_model():
    """Get the actual model (unwrap DDP if needed)"""
    if hasattr(body_model, "module"):
        return body_model.module
    return body_model

def ensure_no_generation_calls():
    """Prevent any generation method calls on server"""
    model = get_model()
    
    # Disable generation methods on server side
    if hasattr(model, 'generate'):
        original_generate = model.generate
        def disabled_generate(*args, **kwargs):
            raise RuntimeError("Generation should not be called on server side")
        model.generate = disabled_generate
    
    if hasattr(model, 'prepare_inputs_for_generation'):
        def disabled_prepare(*args, **kwargs):
            raise RuntimeError("prepare_inputs_for_generation should not be called on server side")
        model.prepare_inputs_for_generation = disabled_prepare

ensure_no_generation_calls()



# new function to prevent generation calls
def safe_model_forward(model, *args, **kwargs):
    """Safely call forward without triggering generation methods"""
    # Only call forward, never generate
    if hasattr(model, 'forward'):
        return model.forward(*args, **kwargs)
    else:
        return model(*args, **kwargs)

def run_body_layers(activations, attention_mask=None):
    """Run hidden states through transformer blocks with consistent attention mask handling"""
    model = get_model()
    hidden = activations
    config = model.config

    def expand_mask_consistently(mask, hidden_states):
        """Consistent 4D attention mask expansion"""
        if mask is not None:
            if mask.dim() == 2:
                batch_size, seq_len = mask.shape
                num_heads = config.n_head
                # Create causal mask
                mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
                mask = mask.expand(batch_size, num_heads, seq_len, seq_len)
            elif mask.dim() == 3:
                # Handle case where mask is [batch, seq_len, seq_len]
                mask = mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
                mask = mask.expand(-1, config.n_head, -1, -1)
            
            # Ensure mask is float32 and on correct device
            mask = mask.float().to(hidden_states.device)
            
            # Convert to attention weights (0 for masked, large negative for unmasked positions)
            mask = (1.0 - mask) * -10000.0
        return mask

    try:
        # Expand attention mask once
        expanded_mask = expand_mask_consistently(attention_mask, hidden)
        
        for block in model.transformer.h:
            # ADD: Explicitly disable use_cache
            hidden = block(
                hidden, 
                attention_mask=expanded_mask, 
                use_cache=False  # ADD THIS LINE
            )[0]
            
        # Final layer norm
        hidden = model.transformer.ln_f(hidden)
        return hidden
    except Exception as e:
        print(f"Error in run_body_layers: {e}")
        traceback.print_exc()
        raise


    





@app.post("/forward")
async def forward(request: Request):
    try:
        payload = await request.json()
        activations = torch.tensor(payload["activations"], device=device)
        attention_mask = None
        if "attention_mask" in payload and payload["attention_mask"] is not None:
            attention_mask = torch.tensor(payload["attention_mask"], device=device, dtype=torch.float32)
            
            # CRITICAL FIX: Ensure attention mask matches activations sequence length
            if attention_mask.size(-1) != activations.size(1):
                print(f"Attention mask length mismatch: {attention_mask.size(-1)} vs {activations.size(1)}")
                # Resize attention mask to match activations
                if attention_mask.size(-1) > activations.size(1):
                    attention_mask = attention_mask[:, :activations.size(1)]
                else:
                    # Pad with zeros
                    pad_length = activations.size(1) - attention_mask.size(-1)
                    padding = torch.zeros(attention_mask.size(0), pad_length, device=device, dtype=torch.float32)
                    attention_mask = torch.cat([attention_mask, padding], dim=1)
        
        body_model.eval()
        with torch.no_grad(): #change
            last_hidden = run_body_layers(activations, attention_mask)
        
        return {"body_activations": last_hidden.cpu().tolist()}
    except Exception as e:
        print(f"Error in forward: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forward_train")
async def forward_train(request: Request):
    try:
        data = await request.json()
        client_rank_id = data.get("rank_id", 0)
        rank_key = f"client_{client_rank_id}_server_{server_state['local_rank']}"
        
        activations = torch.tensor(data["activations"], requires_grad=True, device=device)
        attention_mask = None
        if "attention_mask" in data and data["attention_mask"] is not None:
            attention_mask = torch.tensor(data["attention_mask"], device=device, dtype=torch.float32)
            
            # CRITICAL FIX: Ensure consistent dimensions
            if attention_mask.size(-1) != activations.size(1):
                print(f"Training: Attention mask length mismatch: {attention_mask.size(-1)} vs {activations.size(1)}")
                if attention_mask.size(-1) > activations.size(1):
                    attention_mask = attention_mask[:, :activations.size(1)]
                else:
                    pad_length = activations.size(1) - attention_mask.size(-1)
                    padding = torch.zeros(attention_mask.size(0), pad_length, device=device, dtype=torch.float32)
                    attention_mask = torch.cat([attention_mask, padding], dim=1)
        
        body_model.train()
        last_hidden = run_body_layers(activations, attention_mask)
        
        server_state["last_activations"][rank_key] = (activations, attention_mask, last_hidden)
        server_state["requires_backward"][rank_key] = True
        
        return {"body_activations": last_hidden.detach().cpu().tolist()}
    except Exception as e:
        print(f"Error in forward_train: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))




#change
"""@app.post("/backward")
async def backward(request: Request):
    try:
        data = await request.json()
        client_rank_id = data.get("rank_id", 0)
        rank_key = f"client_{client_rank_id}_server_{server_state['local_rank']}"
        
        if rank_key not in server_state["requires_backward"] or not server_state["requires_backward"][rank_key]:
            return {"status": "error", "message": "No forward state for this rank"}
        
        grad_output = torch.tensor(data["grad_output"], device=device)
        loss_val = data.get("loss", 0.0)
        
        activations, attention_mask, last_hidden = server_state["last_activations"][rank_key]
        
        # Initialize optimizer if needed
        if server_state["optimizer"] is None:
            server_state["optimizer"] = optim.AdamW(
                [p for p in body_model.parameters() if p.requires_grad], lr=2e-4
            )
        
        opt = server_state["optimizer"]
        opt.zero_grad()
        
        # Backward pass - DDP will handle gradient synchronization
        #last_hidden.backward(grad_output, retain_graph=True)
        #changed line above
        torch.autograd.backward(
            tensors=[last_hidden],
            grad_tensors=[grad_output],
            retain_graph=True
        )

        # Before opt.step():
        if server_state["is_distributed"]:
            dist.all_reduce(input_grad)  # Sync gradients

        # Step optimizer
        opt.step()
        
        # Get gradient w.r.t. input activations
        input_grad = activations.grad if activations.grad is not None else torch.zeros_like(activations)
        
        # Cleanup this rank's state
        del server_state["last_activations"][rank_key]
        server_state["requires_backward"][rank_key] = False
        server_state["step_count"] += 1
        server_state["metrics"]["loss"].append(loss_val)
        
        
        return {"grad_input": input_grad.cpu().tolist(), "step": server_state["step_count"]}
    except Exception as e:
        print(f"Error in backward: {e}")
        raise HTTPException(status_code=500, detail=str(e))
"""

@app.post("/backward")
async def backward(request: Request):
    try:
        data = await request.json()
        client_rank_id = data.get("rank_id", 0)
        rank_key = f"client_{client_rank_id}_server_{server_state['local_rank']}"
        
        if rank_key not in server_state["requires_backward"] or not server_state["requires_backward"][rank_key]:
            return {"status": "error", "message": "No forward state for this rank"}
        
        grad_output = torch.tensor(data["grad_output"], device=device)
        loss_val = data.get("loss", 0.0)
        
        activations, attention_mask, last_hidden = server_state["last_activations"][rank_key]
        
        # Initialize optimizer if needed
        if server_state["optimizer"] is None:
            server_state["optimizer"] = optim.AdamW(
                [p for p in body_model.parameters() if p.requires_grad], lr=2e-4
            )
        
        opt = server_state["optimizer"]
        opt.zero_grad()
        
        # FIXED: Backward pass with proper gradient computation
        torch.autograd.backward(
            tensors=[last_hidden],
            grad_tensors=[grad_output],
            retain_graph=False  # Changed to False to free memory
        )
        
        # FIXED: Get input gradient BEFORE optimizer step
        input_grad = activations.grad if activations.grad is not None else torch.zeros_like(activations)
        
        # FIXED: Gradient synchronization for DDP (if distributed)
        if server_state["is_distributed"]:
            # Sync model gradients (DDP handles this automatically for model params)
            # But we need to manually sync input gradients across ranks
            dist.all_reduce(input_grad, op=dist.ReduceOp.SUM)
            input_grad = input_grad / server_state["world_size"]
        
        # Step optimizer
        opt.step()
        
        # Cleanup this rank's state
        del server_state["last_activations"][rank_key]
        server_state["requires_backward"][rank_key] = False
        server_state["step_count"] += 1
        server_state["metrics"]["loss"].append(loss_val)
        
        # Clear memory
        del activations, attention_mask, last_hidden, grad_output
        torch.cuda.empty_cache()
        
        return {"grad_input": input_grad.cpu().tolist(), "step": server_state["step_count"]}
    except Exception as e:
        print(f"Error in backward: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# FIXED: Improved run_body_layers with better attention mask handling
def run_body_layers(activations, attention_mask=None):
    """Run hidden states through transformer blocks with consistent attention mask handling"""
    model = get_model()
    hidden = activations
    config = model.config

    def expand_mask_consistently(mask, hidden_states):
        """Consistent 4D attention mask expansion"""
        if mask is not None:
            if mask.dim() == 2:
                batch_size, seq_len = mask.shape
                num_heads = config.n_head
                # Create causal mask
                mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
                mask = mask.expand(batch_size, num_heads, seq_len, seq_len)
            elif mask.dim() == 3:
                # Handle case where mask is [batch, seq_len, seq_len]
                mask = mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
                mask = mask.expand(-1, config.n_head, -1, -1)
            
            # Ensure mask is float32 and on correct device
            mask = mask.float().to(hidden_states.device)
            
            # Convert to attention weights (0 for masked, large negative for unmasked positions)
            mask = (1.0 - mask) * -10000.0
        return mask

    try:
        # Expand attention mask once
        expanded_mask = expand_mask_consistently(attention_mask, hidden)
        
        for block in model.transformer.h:
            hidden = block(hidden, attention_mask=expanded_mask)[0]
            
        # Final layer norm
        hidden = model.transformer.ln_f(hidden)
        return hidden
    except Exception as e:
        print(f"Error in run_body_layers: {e}")
        traceback.print_exc()
        raise


@app.post("/start_training")
async def start_training(request: Request):
    try:
        data = await request.json()
        lr = data.get("learning_rate", 2e-4)
        
        # Only initialize on rank 0 to avoid conflicts
        if server_state["local_rank"] == 0:
            server_state.update({
                "step_count": 0,
                "epoch_count": 0,
                "metrics": {"loss": []},
                "training_active": True
            })
        
        # Each rank gets its own optimizer
        server_state["optimizer"] = optim.AdamW(
            [p for p in body_model.parameters() if p.requires_grad], lr=lr
        )
        
        trainable_params = sum(p.numel() for p in body_model.parameters() if p.requires_grad)
        return {
            "status": "initialized", 
            "trainable_params": trainable_params,
            "server_rank": server_state["local_rank"]
        }
    except Exception as e:
        print(f"Error in start_training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/end_epoch")
async def end_epoch(request: Request):
    try:
        data = await request.json()
        
        # Only rank 0 handles epoch counting
        if server_state["local_rank"] == 0:
            server_state["epoch_count"] += 1
            
            avg_loss = 0.0
            if server_state["metrics"]["loss"]:
                avg_loss = sum(server_state["metrics"]["loss"]) / len(server_state["metrics"]["loss"])
            
            if data.get("is_final", False):
                server_state["training_active"] = False
            
            server_state["metrics"]["loss"] = []
            
            return {"status": "ok", "epoch": server_state["epoch_count"], "avg_loss": avg_loss}
        else:
            return {"status": "ok", "epoch": -1, "avg_loss": 0.0}
    except Exception as e:
        print(f"Error in end_epoch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save_model")
async def save_model(request: Request):
    try:
        data = await request.json()
        path = data.get("path", "./server_model")
        
        # Only rank 0 saves the model
        if server_state["local_rank"] == 0:
            os.makedirs(path, exist_ok=True)
            
            # Save the model (unwrap DDP if needed)
            model_to_save = get_model()
            model_to_save.save_pretrained(os.path.join(path, "body_model"))
            
            # Save optimizer state
            if server_state["optimizer"] is not None:
                torch.save(server_state["optimizer"].state_dict(), os.path.join(path, "body_optimizer.pt"))
            
            return {"status": "saved", "path": path, "server_rank": server_state["local_rank"]}
        else:
            return {"status": "skipped", "server_rank": server_state["local_rank"]}
    except Exception as e:
        print(f"Error in save_model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load_model")
async def load_model(request: Request):
    try:
        global body_model
        data = await request.json()
        path = data.get("path")
        
        print(f"[SERVER RANK {server_state['local_rank']}] Loading model from {path}", flush=True)
        
        if not os.path.isdir(path):
            return {"status": "error", "message": f"Path {path} not found", "server_rank": server_state['local_rank']}
        
        body_model_path = os.path.join(path, "body_model")
        if os.path.exists(body_model_path):
            print(f"[SERVER RANK {server_state['local_rank']}] Found body model at {body_model_path}", flush=True)
            
            # Create base model
            model_name = "gpt2"
            full_model = AutoModelForCausalLM.from_pretrained(model_name)
            _, new_body_model, _ = split_gpt2(full_model, head_layers=2, tail_layers=2)
            
            print(f"[SERVER RANK {server_state['local_rank']}] Loading PEFT model...", flush=True)
            # Load PEFT model
            loaded_peft_model = PeftModel.from_pretrained(
                new_body_model, 
                body_model_path,
                is_trainable=True
            )
            loaded_peft_model = loaded_peft_model.to(device)
            
            # Update global model
            if server_state["is_distributed"]:
                body_model = DDP(loaded_peft_model, device_ids=[server_state["local_rank"]])
                print(f"[SERVER RANK {server_state['local_rank']}] Wrapped with DDP", flush=True)
            else:
                body_model = loaded_peft_model
            
            body_model.train()
            
            # Load optimizer
            opt_path = os.path.join(path, "body_optimizer.pt")
            if os.path.exists(opt_path):
                print(f"[SERVER RANK {server_state['local_rank']}] Loading optimizer...", flush=True)
                server_state["optimizer"] = optim.AdamW(
                    [p for p in body_model.parameters() if p.requires_grad], lr=2e-4
                )
                try:
                    server_state["optimizer"].load_state_dict(
                        torch.load(opt_path, map_location=device), strict=False
                    )
                    print(f"[SERVER RANK {server_state['local_rank']}] Optimizer loaded", flush=True)
                except Exception as opt_error:
                    print(f"[SERVER RANK {server_state['local_rank']}] Optimizer load failed: {opt_error}", flush=True)
            else:
                print(f"[SERVER RANK {server_state['local_rank']}] No optimizer file found at {opt_path}", flush=True)
            
            print(f"[SERVER RANK {server_state['local_rank']}] Model loading completed successfully", flush=True)
            return {"status": "loaded", "path": path, "server_rank": server_state["local_rank"]}
        else:
            print(f"[SERVER RANK {server_state['local_rank']}] ERROR: Body model not found at {body_model_path}", flush=True)
            return {"status": "error", "message": f"Body model not found at {body_model_path}", "server_rank": server_state["local_rank"]}
            
    except Exception as e:
        print(f"[SERVER RANK {server_state['local_rank']}] ERROR in load_model: {e}", flush=True)
        traceback.print_exc()
        return {"status": "error", "message": str(e), "server_rank": server_state["local_rank"]}



@app.get("/model_info")
async def model_info():
    try:
        trainable_params = sum(p.numel() for p in body_model.parameters() if p.requires_grad)
        return {
            "model_name": "gpt2",
            "device": str(device),
            "trainable_params": trainable_params,
            "steps": server_state["step_count"],
            "epochs": server_state["epoch_count"],
            "distributed": server_state["is_distributed"],
            "local_rank": server_state["local_rank"],
            "world_size": server_state["world_size"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "device": str(device),
        "server_rank": server_state["local_rank"]
    }

if __name__ == "__main__":
    # Each process runs on a different port for distributed setup
    port = 8000 + server_state["local_rank"]
    # INCREASE SERVER TIMEOUTS
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        timeout_keep_alive=1200,  # Increase from default 5 to 120 seconds
        timeout_graceful_shutdown=1200,  # Add graceful shutdown timeout
    )
