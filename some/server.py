from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
import uvicorn
import transformers

from onlyDORA import Adaptive_Lora_Linear

# Load GPT-2 model and configuration
config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)

split1 = 2  
split2 = 6  

class AdaptiveLoraConfig:
    def __init__(self):
        self.lora_dropout = 0.1
        self.lora_alpha = 32
        self.adaptive_lora_start_rank = 8
        self.adaptive_lora_eps = 1e-6
        self.adaptive_lora_sensitivity_beta = 0.9

class AdaptiveLoraConv1D(nn.Module):
    def __init__(self, config, conv_module):
        super().__init__()
        self.config = config
        self.conv = conv_module
        self.conv.weight.requires_grad = False
        if hasattr(self.conv, 'bias') and self.conv.bias is not None:
            self.conv.bias.requires_grad = False

        self.in_features = self.conv.weight.shape[0]  
        self.out_features = self.conv.weight.shape[1]  

        rank = config.adaptive_lora_start_rank
        self.lora_dropout = nn.Dropout(config.lora_dropout)
        self.lora_a = nn.Parameter(torch.zeros(rank, self.in_features))  
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank))  
        self.lora_scaler = nn.Parameter(torch.ones(rank, dtype=torch.float32))

        nn.init.kaiming_uniform_(self.lora_a)
        nn.init.zeros_(self.lora_b)
        self.lora_scaling = config.lora_alpha / rank

    def forward(self, x):
        original_output = self.conv(x)
        if x.numel() > 1_000_000:
            return original_output

        lora_input = self.lora_dropout(x)
        lora_input_reshaped = lora_input.reshape(-1, self.in_features)
        lora_output = lora_input_reshaped @ self.lora_a.t()
        lora_output = lora_output * self.lora_scaler.unsqueeze(0)
        lora_output = lora_output @ self.lora_b.t()
        lora_output = lora_output.reshape(*x.shape[:-1], self.out_features)
        lora_output = lora_output * self.lora_scaling
        return original_output + lora_output

class ServerModel(torch.nn.Module):
    def __init__(self, model, split1, split2):
        super().__init__()
        self.config = model.config
        self.middle_layers = model.transformer.h[split1:split2]
        self.sensitivity_score_dict = {}
        self.finally_mask_dict = {}
        self.step_counter = 0
        self.max_steps = 2000
        self.lora_config = AdaptiveLoraConfig()
        self.apply_adaptive_lora()

    def apply_adaptive_lora(self):
        for block in self.middle_layers:
            # Convert named_modules to a list to avoid modifying the dict during iteration.
            modules_to_replace = list(block.named_modules())
            for name, module in modules_to_replace:
                if isinstance(module, transformers.pytorch_utils.Conv1D):
                    setattr(block, name, AdaptiveLoraConv1D(self.lora_config, module))

    def forward(self, hidden_states):
        for layer in self.middle_layers:
            hidden_states = layer(hidden_states)[0]
        self.step_counter += 1
        if self.step_counter % 50 == 0:
            self.prune_lora_scaler()
        return {"hidden_states": hidden_states.tolist()}

    def prune_lora_scaler(self):
        with torch.no_grad():
            for block in self.middle_layers:
                for name, module in block.named_modules():
                    if isinstance(module, AdaptiveLoraConv1D):
                        mask = module.lora_scaler < 0.05
                        module.lora_scaler[mask] = 0

server_model = ServerModel(model, split1, split2)

app = FastAPI()

class InputData(BaseModel):
    hidden_states: list

@app.post("/process")
def process(data: InputData):
    hidden_states = torch.tensor(data.hidden_states)
    result = server_model(hidden_states)
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
