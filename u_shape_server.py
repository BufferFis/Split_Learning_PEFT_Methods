from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
import uvicorn
import transformers
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import json
import os
from datetime import datetime

from onlyDORA import Adaptive_Lora_Linear

# Load GPT-2 model and configuration
config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)

split1 = 2  # Match with client
split2 = 6  # Match with client

# Configuration for Adaptive LoRA
class AdaptiveLoraConfig:
    def __init__(self):
        self.lora_dropout = 0.1
        self.lora_alpha = 32
        self.adaptive_lora_start_rank = 8
        self.adaptive_lora_eps = 1e-6
        self.adaptive_lora_sensitivity_beta = 0.9

# Adapted version of AdaptiveLoraLinear for Conv1D layers
class AdaptiveLoraConv1D(nn.Module):
    def __init__(self, config, conv_module):
        super().__init__()
        self.config = config
        self.conv = conv_module
        self.conv.weight.requires_grad = False
        if hasattr(self.conv, 'bias') and self.conv.bias is not None:
            self.conv.bias.requires_grad = False
        
        # Get the parent module path to determine layer type
        parent_name = ""
        for name, module in model.named_modules():
            if module is conv_module:
                parent_name = name
                break
        
        # Set layer type directly from parent path for precise identification
        if 'attn.c_attn' in parent_name:
            self.layer_type = 'attn.c_attn'
            print(f"Identified layer as attn.c_attn: {parent_name}")
        elif 'attn.c_proj' in parent_name:
            self.layer_type = 'attn.c_proj'
            print(f"Identified layer as attn.c_proj: {parent_name}")
        elif 'mlp.c_fc' in parent_name:
            self.layer_type = 'mlp.c_fc'
            print(f"Identified layer as mlp.c_fc: {parent_name}")
        elif 'mlp.c_proj' in parent_name:
            self.layer_type = 'mlp.c_proj'
            print(f"Identified layer as mlp.c_proj: {parent_name}")
        else:
            self.layer_type = 'unknown'
            print(f"Unknown layer type: {parent_name}")
        
        # Corrected: Get weight dimensions correctly for Conv1D (nx=input, nf=output)
        self.in_features = self.conv.weight.shape[0]  # First dimension is input (nx)
        self.out_features = self.conv.weight.shape[1]  # Second dimension is output (nf)
        
        # Initialize LoRA parameters with proper dimensions
        rank = config.adaptive_lora_start_rank
        self.lora_dropout = nn.Dropout(config.lora_dropout)
        self.lora_a = nn.Parameter(torch.zeros(rank, self.in_features))  # [rank, in_features]
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank))  # [out_features, rank]
        self.lora_scaler = nn.Parameter(torch.ones(rank, dtype=torch.float32))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_a)
        nn.init.zeros_(self.lora_b)
        
        self.lora_scaling = config.lora_alpha / rank
        
        print(f"LoRA initialized for {self.layer_type} - in: {self.in_features}, out: {self.out_features}")

    def forward(self, x):
        # Original output from Conv1D
        original_output = self.conv(x)
        
        # Skip LoRA for very large tensors to avoid OOM
        if x.numel() > 1_000_000:
            return original_output
        
        try:
            # Get original shape for reference
            orig_shape = x.shape
            batch_size = orig_shape[0] if len(orig_shape) >= 1 else 1
            seq_len = orig_shape[1] if len(orig_shape) >= 2 else 1
            
            # Apply dropout
            lora_input = self.lora_dropout(x)
            
            # Reshape input to [batch*seq_len, in_features]
            lora_input_reshaped = lora_input.reshape(-1, self.in_features)
            
            # Apply LoRA transformations
            lora_output = lora_input_reshaped @ self.lora_a.t()  # [batch*seq, rank]
            lora_output = lora_output * self.lora_scaler.unsqueeze(0)  # Scale each rank
            lora_output = lora_output @ self.lora_b.t()  # [batch*seq, out_features]
            lora_output = lora_output.reshape(batch_size, seq_len, self.out_features)
            lora_output = lora_output * self.lora_scaling  # Apply scaling factor
            
            # Add LoRA contribution to original output
            return original_output + lora_output
            
        except Exception as e:
            print(f"Error in LoRA ({self.layer_type}): {e}")
            print(f"Input shape: {x.shape}, LoRA A: {self.lora_a.shape}, LoRA B: {self.lora_b.shape}")
            print(f"Layer info - in_features: {self.in_features}, out_features: {self.out_features}")
            return original_output
        
    def inspect_input_shape(self, x):
        """Print the shape and size of the input tensor"""
        print(f"Input shape: {x.shape}, Size: {x.numel()}, Features: {x.shape[-1]}")
        return x

    def log_forward_shapes(self, x):
        """Instrument forward pass with shape logging"""
        print(f"=== Forward for {self.layer_type} ===")  # Changed from self.layer_name to self.layer_type
        print(f"Input: {x.shape}")
        
        # Track original forward
        original = self.conv(x)
        print(f"Original output: {original.shape}")
        
        # Reshape as we would in LoRA
        reshaped = x.reshape(-1, self.in_features)
        print(f"Reshaped: {reshaped.shape}")
        
        # Show first LoRA step
        lora_a_out = reshaped @ self.lora_a.t()
        print(f"After lora_a: {lora_a_out.shape}")
        
        return original

    def debug_shapes(self, x):
        """Print detailed shape information for debugging"""
        print(f"\n=== Shape Debug for {self.layer_type} ===")
        print(f"Input: {x.shape}")
        
        # Original forward pass
        original = self.conv(x)
        print(f"Conv output: {original.shape}")
        
        # Show expected LoRA shapes
        print(f"Expected LoRA input shape after reshape: [-1, {self.in_features}]")
        print(f"Expected LoRA A output shape: [-1, {self.lora_a.shape[0]}]")
        print(f"Expected LoRA B output shape: [-1, {self.out_features}]")
        print(f"Expected final output shape: {original.shape}")
        
        return original

# Server model with adaptive LoRA
class ServerModel(torch.nn.Module):
    def __init__(self, model, split1, split2):
        super().__init__()
        self.config = model.config
        self.middle_layers = model.transformer.h[split1:split2]
        
        # Initialize importance tracking for pruning
        self.sensitivity_score_dict = {}
        self.finally_mask_dict = {}
        self.step_counter = 0
        self.max_steps = 2000
        self.lora_config = AdaptiveLoraConfig()
        
        # Debug GPT-2 structure first
        print("GPT-2 block structure example:")
        if len(self.middle_layers) > 0:
            self.print_module_structure(self.middle_layers[0])
        
        # Apply Adaptive LoRA to GPT-2 layers
        print("Applying LoRA to GPT-2 layers...")
        self.apply_adaptive_lora()
        print(f"Initialized {self.count_lora_layers()} Adaptive LoRA modules")
        self.debug_lora_layers()  # Add this line to print debugging information

    def print_module_structure(self, module, prefix=''):
        """Print the structure of a module with its attributes and submodules"""
        for name, child in module._modules.items():
            print(f"{prefix}├── {name}: {type(child).__name__}")
            if child._modules:
                self.print_module_structure(child, prefix + '│   ')
            elif isinstance(child, transformers.pytorch_utils.Conv1D):
                print(f"{prefix}│   └── Conv1D Shape: in={child.weight.size(1)}, out={child.weight.size(0)}")

    def count_lora_layers(self):
        """Count number of AdaptiveLoraLinear/Conv1D layers initialized"""
        count = 0
        for block_idx, block in enumerate(self.middle_layers):
            for name, module in block.named_modules():
                if isinstance(module, (Adaptive_Lora_Linear, AdaptiveLoraConv1D)):
                    count += 1
        return count

    def apply_adaptive_lora(self):
        """Apply Adaptive LoRA to GPT-2 layers with more robust path detection"""
        total_applied = 0
        
        # Keep track of which layers were processed
        processed_layers = {}
        
        # First identify all Conv1D modules and their full paths
        for block_idx, block in enumerate(self.middle_layers):
            print(f"\nProcessing block {block_idx}")
            
            # More accurate way to track module paths
            for name, module in block.named_modules():
                if isinstance(module, transformers.pytorch_utils.Conv1D):
                    full_name = f"block_{block_idx}_{name}"
                    print(f"  Found Conv1D at {full_name}")
                    
                    # Create LoRA wrapper
                    lora_module = AdaptiveLoraConv1D(self.lora_config, module)
                    
                    # Find parent module to replace the Conv1D
                    parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                    child_name = name.rsplit(".", 1)[1] if "." in name else name
                    
                    if parent_name:
                        parent = block
                        for part in parent_name.split("."):
                            parent = getattr(parent, part)
                        setattr(parent, child_name, lora_module)
                    else:
                        setattr(block, name, lora_module)
                    
                    print(f"  ✓ Successfully applied LoRA to {name}")
                    processed_layers[full_name] = lora_module
                    total_applied += 1
        
        print(f"Applied LoRA to {total_applied} layers")
        return processed_layers

    def forward(self, hidden_states=None, **kwargs):
        if hidden_states is None:
            hidden_states = kwargs.get("hidden_states", None)
        if hidden_states is None:
            raise ValueError("hidden_states is required for processing.")
            
        print(f"Input hidden state shape: {hidden_states.shape}")
        
        try:
            for i, layer in enumerate(self.middle_layers):
                print(f"Processing layer {i}, input shape: {hidden_states.shape}")
                hidden_states = layer(hidden_states)[0]
                print(f"Layer {i} output shape: {hidden_states.shape}")
                
            # Update step and possibly prune after processing
            self.step_counter += 1
            if self.step_counter % 50 == 0:  # Prune every 5 steps
                print(f"Step {self.step_counter}: Running pruning...")
                self.prune_lora_scaler()
                
            return {"hidden_states": hidden_states.tolist()}
        except Exception as e:
            print(f"Error in forward pass: {e}")
            # Additional debugging:
            print(f"Hidden states shape: {hidden_states.shape}")
            import traceback
            traceback.print_exc()
            raise
        
    def update_importance_score(self):
        with torch.no_grad():
            for block_idx, block in enumerate(self.middle_layers):
                for name, module in block.named_modules():
                    if isinstance(module, AdaptiveLoraConv1D):
                        full_name = f"block_{block_idx}_{name}"
                        
                        try:
                            # Calculate individual rank contributions
                            scores = []
                            for i in range(module.lora_a.size(0)):
                                # For each rank, compute the contribution
                                scale_i = module.lora_scaler[i].item()
                                a_i = module.lora_a[i:i+1, :]
                                b_i = module.lora_b[:, i:i+1]
                                
                                # Calculate rank contribution to weight matrix
                                contribution = b_i @ a_i
                                norm_i = contribution.norm(p='fro')
                                scores.append(norm_i.unsqueeze(0))
                                
                            # Combine all scores
                            if not scores:
                                continue
                                
                            # Calculate total weight norm
                            total_contrib = module.lora_b @ (module.lora_a * module.lora_scaler.unsqueeze(1))
                            total_norm = total_contrib.norm(p='fro') + self.lora_config.adaptive_lora_eps
                            
                            # Normalize scores
                            norm_i = torch.cat(scores)
                            score_i = norm_i / total_norm
                            
                            # Initialize or update the score dictionary
                            if full_name not in self.sensitivity_score_dict:
                                self.sensitivity_score_dict[full_name] = torch.zeros_like(score_i)
                                self.finally_mask_dict[full_name] = torch.zeros_like(score_i, dtype=torch.bool)
                                
                            # Update exponential moving average
                            beta = self.lora_config.adaptive_lora_sensitivity_beta
                            self.sensitivity_score_dict[full_name] = beta * self.sensitivity_score_dict[full_name] + (1 - beta) * score_i
                        
                        except Exception as e:
                            print(f"Error calculating importance for {full_name}: {e}")
                            continue

    def prune_lora_scaler(self):
        # Update importance scores
        self.update_importance_score()
        
        # Print count of modules with scores
        print(f"Found {len(self.sensitivity_score_dict)} modules with importance scores")
        
        # Determine pruning threshold
        all_scores = []
        for name, scores in self.sensitivity_score_dict.items():
            all_scores.append(scores)
            
        if not all_scores:
            print("No scores to prune, skipping")
            return
            
        all_scores = torch.cat(all_scores)
        print(f"Total scores: {all_scores.numel()}, Non-zero: {(all_scores > 0).sum().item()}")
        
        prune_percent = min(0.1, 0.3 * self.step_counter / self.max_steps)
        k = max(1, int(all_scores.numel() * prune_percent))
        
        if k >= all_scores.numel():
            print("Would prune all scores, skipping")
            return
        
        print(f"Pruning {k} of {all_scores.numel()} ranks (bottom {prune_percent:.1%})")
        threshold = torch.kthvalue(all_scores, k).values
        
        # Apply pruning
        with torch.no_grad():
            pruned_count = 0
            for block_idx, block in enumerate(self.middle_layers):
                for name, module in block.named_modules():
                    if isinstance(module, AdaptiveLoraConv1D):
                        # All layers handled uniformly
                        full_name = f"block_{block_idx}_{name}"
                        if full_name in self.sensitivity_score_dict:
                            mask = self.sensitivity_score_dict[full_name] <= threshold
                            pruned_this_module = mask.sum().item()
                            module.lora_scaler[mask] = 0
                            self.finally_mask_dict[full_name] = mask
                            pruned_count += pruned_this_module
            
            print(f"Pruned {pruned_count} ranks in total")

    def debug_lora_layers(self):
        """Print debugging information about all LoRA layers"""
        print("\n=== LoRA Layer Debug Information ===")
        for block_idx, block in enumerate(self.middle_layers):
            for name, module in block.named_modules():
                if isinstance(module, AdaptiveLoraConv1D):
                    full_name = f"block_{block_idx}_{name}"
                    print(f"Layer: {full_name} ({module.layer_type})")  # Changed from layer_name to layer_type
                    print(f"  Input dim: {module.in_features}, Output dim: {module.out_features}")
                    print(f"  LoRA A: {module.lora_a.shape}, LoRA B: {module.lora_b.shape}")
                    print(f"  Active ranks: {(module.lora_scaler != 0).sum().item()} / {len(module.lora_scaler)}")
        print("=====================================\n")

server_model = ServerModel(model, split1, split2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
server_model = ServerModel(model, split1, split2).to(device)

app = FastAPI()

class InputData(BaseModel):
    hidden_states: list

class TrainingConfig(BaseModel):
    learning_rate: float = 2e-4
    batch_size: int = 4
    epochs: int = 3
    grad_accumulation_steps: int = 1

class OptimizationState:
    def __init__(self):
        self.optimizer = None
        self.step = 0
        self.epoch = 0
        self.metrics = {"loss": [], "pruning_stats": []}
        self.start_time = None

opt_state = OptimizationState()

# Add this variable at the top level, alongside opt_state
server_state = {
    "last_hidden_output": None,
    "requires_backward": False
}

@app.post("/start_training")
def start_training(config: TrainingConfig):
    """Initialize optimizer and training state"""
    global opt_state
    
    # Create optimizer for server-side trainable parameters
    trainable_params = [p for p in server_model.parameters() if p.requires_grad]
    opt_state.optimizer = optim.AdamW(trainable_params, lr=config.learning_rate)
    opt_state.step = 0
    opt_state.epoch = 0
    opt_state.metrics = {"loss": [], "pruning_stats": []}
    opt_state.start_time = time.time()
    
    print(f"Training started with learning_rate={config.learning_rate}")
    print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    return {"status": "Training initialized", 
            "trainable_params": sum(p.numel() for p in trainable_params)}

@app.post("/process")
def process(data: InputData):
    hidden_states = torch.tensor(data.hidden_states)
    result = server_model(hidden_states=hidden_states)
    return result

@app.post("/process_train")
def process_train(data: InputData):
    """Process hidden states during training and return gradients"""
    global opt_state, server_model, server_state
    
    hidden_states = torch.tensor(data.hidden_states, requires_grad=True).to(device)
    
    # Forward pass
    server_model.train()
    
    # Process through the middle layers
    output = hidden_states
    for i, layer in enumerate(server_model.middle_layers):
        output = layer(output)[0]
    
    # Store output and mark that we need backward pass
    server_state["last_hidden_output"] = output
    server_state["requires_backward"] = True
    
    # Track metrics for this step
    opt_state.step += 1
    
    # Return the processed hidden states for continuation on client
    return {"hidden_states": output.detach().tolist()}

@app.post("/backward")
def backward_pass(data: dict):
    """Process backward pass with gradients from client"""
    global opt_state, server_model, server_state
    
    # Extract the gradients sent from client
    grad_tensor = torch.tensor(data["gradients"]).to(device)
    loss_value = data["loss"]
    
    # Store loss for metrics
    opt_state.metrics["loss"].append(loss_value)
    
    # Check if we have a tensor waiting for backward
    if server_state["requires_backward"] and server_state["last_hidden_output"] is not None:
        if opt_state.optimizer is not None:
            opt_state.optimizer.zero_grad()
            
            # Get the stored output tensor from forward pass
            output_tensor = server_state["last_hidden_output"]
            
            # Apply backward pass with the client-provided gradients
            output_tensor.backward(gradient=grad_tensor)
            
            # Update parameters
            opt_state.optimizer.step()
            
            # Reset the state
            server_state["last_hidden_output"] = None
            server_state["requires_backward"] = False
    else:
        print("Error: No tensor available for backward pass")
    
    # Run pruning every 50 steps
    if opt_state.step % 50 == 0:
        print(f"Step {opt_state.step}: Running pruning...")
        server_model.prune_lora_scaler()
        
        # Save pruning stats
        stats = get_prune_stats()
        opt_state.metrics["pruning_stats"].append({
            "step": opt_state.step,
            "active_ranks": stats["total_active_ranks"],
            "potential_ranks": stats["potential_ranks"]
        })
    
    return {"status": "ok", "step": opt_state.step}

@app.post("/end_epoch")
def end_epoch(data: dict):
    """Handle end of epoch, save metrics"""
    global opt_state
    
    opt_state.epoch += 1
    
    # Calculate average loss for epoch
    avg_loss = sum(opt_state.metrics["loss"]) / len(opt_state.metrics["loss"])
    
    print(f"Epoch {opt_state.epoch} completed. Avg loss: {avg_loss:.4f}")
    
    # Save metrics at end of training
    if data.get("is_final", False):
        elapsed_time = time.time() - opt_state.start_time
        
        # Save metrics to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = f"metrics_{timestamp}.json"
        
        with open(metrics_file, "w") as f:
            json.dump({
                "training_time": elapsed_time,
                "epochs": opt_state.epoch,
                "steps": opt_state.step,
                "final_loss": avg_loss,
                "pruning_stats": opt_state.metrics["pruning_stats"]
            }, f, indent=2)
        
        print(f"Training completed in {elapsed_time:.2f}s. Metrics saved to {metrics_file}")
    
    # Reset loss tracking for next epoch
    opt_state.metrics["loss"] = []
    
    return {"status": "ok", "epoch": opt_state.epoch, "avg_loss": avg_loss}

@app.get("/prune_stats")
def get_prune_stats():
    stats = {
        "step": server_model.step_counter,
        "active_rank_counts": {}
    }
    
    total_layers = 0
    active_ranks = 0
    
    for block_idx, block in enumerate(server_model.middle_layers):
        for name, module in block.named_modules():
            if isinstance(module, AdaptiveLoraConv1D):
                # All layers handled uniformly
                total_layers += 1
                full_name = f"block_{block_idx}_{name}"
                nonzero_count = (module.lora_scaler != 0).sum().item()
                stats["active_rank_counts"][full_name] = nonzero_count
                active_ranks += nonzero_count
    
    stats["total_lora_layers"] = total_layers
    stats["total_active_ranks"] = active_ranks
    stats["potential_ranks"] = total_layers * server_model.lora_config.adaptive_lora_start_rank
    
    return stats

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    