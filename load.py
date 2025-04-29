import os
import torch
import requests
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from util import split_gpt2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./server_model")
    parser.add_argument("--server_url", type=str, default="http://localhost:8000")
    args = parser.parse_args()
    
    # First, load the server model
    r = requests.post(f"{args.server_url}/load_model", json={"path": args.model_path})
    print("Server model loaded:", r.json())
    
    # Now load client models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create new model instances
    full_model = AutoModelForCausalLM.from_pretrained("gpt2")
    head_model, _, tail_model = split_gpt2(full_model, head_layers=2, tail_layers=2)
    
    # Apply LoRA config
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        bias="none", use_dora=True, task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj"]
    )
    
    head_model = get_peft_model(head_model, lora_cfg).to(device)
    tail_model = get_peft_model(tail_model, lora_cfg).to(device)
    
    # Load saved weights
    head_path = os.path.join(args.model_path, "head_model.pt")
    tail_path = os.path.join(args.model_path, "tail_model.pt")
    
    if os.path.exists(head_path) and os.path.exists(tail_path):
        head_model.load_state_dict(torch.load(head_path))
        tail_model.load_state_dict(torch.load(tail_path))
        print(f"Client models loaded from {args.model_path}")
    else:
        print(f"Warning: Could not find client models at {args.model_path}")
    
    print("All models loaded successfully!")

if __name__ == "__main__":
    main()
