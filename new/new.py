# ==============================================================================
#
# Full End-to-End Pipeline for Training and Evaluating a
# U-Shaped Split-DoRA GPT-2 Model on the E2E Refined NLG Dataset
#
# Author: Gemini
# Date: July 19, 2025
#
# Description:
# This script merges two advanced concepts:
#   1. A U-shaped split architecture for GPT-2 (ClientHead, Server, ClientTail)
#      with a custom training loop, dual optimizers, and weight tying.
#   2. A robust data pipeline for the high-fidelity E2E Refined Dataset,
#      which is loaded from the official release's JSON files.
#
# It applies Weight-Decomposed Low-Rank Adaptation (DoRA) to all model parts
# and includes a bespoke beam search algorithm for generation with the split model,
# along with full checkpointing and evaluation capabilities.
#
# ==============================================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
import argparse
import json
import subprocess
import re
from tqdm import tqdm
import warnings
import pandas as pd

# Suppress a specific warning from the PEFT library if it occurs
warnings.filterwarnings("ignore", message=".*Could not find the quantized model in the `peft` library.*")

# Import necessary libraries
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    from datasets import Dataset, DatasetDict, load_dataset
except ImportError as e:
    print(f"Error: A required library is not installed. Please run 'pip install transformers peft datasets pandas torch tqdm'. Details: {e}")
    exit(1)

# ==============================================================================
# SECTION 1: MODEL ARCHITECTURE DEFINITION
# Defines the ClientHead, Server, and ClientTail modules for the split model.
# ==============================================================================

class ClientHead(nn.Module):
    """The first part of the split GPT-2 model, executed on the client."""
    def __init__(self, gpt2_model, split_point_1):
        super().__init__()
        self.config = gpt2_model.config
        # Deepcopy layers for the client head
        self.transformer = nn.ModuleDict({
            'wte': copy.deepcopy(gpt2_model.transformer.wte),
            'wpe': copy.deepcopy(gpt2_model.transformer.wpe),
            'drop': copy.deepcopy(gpt2_model.transformer.drop),
            'h': nn.ModuleList([copy.deepcopy(layer) for layer in gpt2_model.transformer.h[:split_point_1]])
        })

    def _prepare_attention_mask(self, attention_mask, input_shape, device, dtype):
        """Prepares the attention mask for the GPT-2 model."""
        if attention_mask is None:
            return torch.ones(input_shape, device=device)
        # Create a 4D attention mask from a 2D mask
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        else:
            raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape})")
        
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def forward(self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs):
        final_use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.transformer.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        device = input_ids.device
        
        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = self._prepare_attention_mask(attention_mask, (input_ids.shape[0], input_ids.shape[1] + past_length), device, self.transformer.wte.weight.dtype)

        inputs_embeds = self.transformer.wte(input_ids)
        position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0)
        position_embeds = self.transformer.wpe(position_ids)
        
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.transformer.drop(hidden_states)

        presents = [] if final_use_cache else None
        for i, (block, layer_past) in enumerate(zip(self.transformer.h, past_key_values)):
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                use_cache=final_use_cache
            )
            hidden_states = outputs[0]
            if final_use_cache:
                # Safely handle the output tuple which may or may not contain past_key_values
                present = outputs[1] if len(outputs) > 1 else None
                presents.append(present)

        return hidden_states, tuple(presents) if final_use_cache else None
        
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """
        A required method for PEFT compatibility with CausalLM tasks.
        This formats inputs for the `generate` method.
        """
        if past_key_values:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": kwargs.get("attention_mask"),
            "use_cache": kwargs.get("use_cache", True),
        }

class Server(nn.Module):
    """The middle part of the split GPT-2 model, executed on the server."""
    def __init__(self, gpt2_model, split_point_1, split_point_2):
        super().__init__()
        self.config = gpt2_model.config
        # Deepcopy layers for the server
        self.h = nn.ModuleList([copy.deepcopy(layer) for layer in gpt2_model.transformer.h[split_point_1:split_point_2]])

    def _prepare_attention_mask(self, attention_mask, input_shape, device, dtype):
        """Prepares the attention mask for the GPT-2 model."""
        if attention_mask is None:
            return torch.ones(input_shape, device=device)
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        else:
            raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape})")
        
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def forward(self, inputs_embeds, past_key_values=None, attention_mask=None, use_cache=None, **kwargs):
        final_use_cache = use_cache if use_cache is not None else self.config.use_cache
        hidden_states = inputs_embeds
        
        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))
        
        past_length = past_key_values[0][0].size(-2) if past_key_values[0] is not None else 0
        device = hidden_states.device
        
        if attention_mask is not None:
            attention_mask = self._prepare_attention_mask(attention_mask, (hidden_states.shape[0], hidden_states.shape[1] + past_length), device, hidden_states.dtype)
            
        presents = [] if final_use_cache else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                use_cache=final_use_cache
            )
            hidden_states = outputs[0]
            if final_use_cache:
                # Safely handle the output tuple which may or may not contain past_key_values
                present = outputs[1] if len(outputs) > 1 else None
                presents.append(present)
            
        return hidden_states, tuple(presents) if final_use_cache else None

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """A dummy method for PEFT compatibility. It's not called in our custom loop."""
        return kwargs

class ClientTail(nn.Module):
    """The final part of the split GPT-2 model, executed on the client."""
    def __init__(self, gpt2_model, split_point_2):
        super().__init__()
        self.config = gpt2_model.config
        # Deepcopy layers for the client tail
        self.h = nn.ModuleList([copy.deepcopy(layer) for layer in gpt2_model.transformer.h[split_point_2:]])
        self.ln_f = copy.deepcopy(gpt2_model.transformer.ln_f)
        self.lm_head = copy.deepcopy(gpt2_model.lm_head)

    def _prepare_attention_mask(self, attention_mask, input_shape, device, dtype):
        """Prepares the attention mask for the GPT-2 model."""
        if attention_mask is None:
            return torch.ones(input_shape, device=device)
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        else:
            raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape})")
        
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def forward(self, inputs_embeds, past_key_values=None, attention_mask=None, use_cache=None, **kwargs):
        final_use_cache = use_cache if use_cache is not None else self.config.use_cache
        hidden_states = inputs_embeds
        
        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))
        
        past_length = past_key_values[0][0].size(-2) if past_key_values[0] is not None else 0
        device = hidden_states.device
        
        if attention_mask is not None:
            attention_mask = self._prepare_attention_mask(attention_mask, (hidden_states.shape[0], hidden_states.shape[1] + past_length), device, hidden_states.dtype)

        presents = [] if final_use_cache else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                use_cache=final_use_cache
            )
            hidden_states = outputs[0]
            if final_use_cache:
                # Safely handle the output tuple which may or may not contain past_key_values
                present = outputs[1] if len(outputs) > 1 else None
                presents.append(present)
            
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits, tuple(presents) if final_use_cache else None

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """A dummy method for PEFT compatibility. It's not called in our custom loop."""
        return kwargs

# ==============================================================================
# SECTION 2: DATA PREPARATION FOR E2E REFINED DATASET
# This section is updated to load the official release's JSON files.
# ==============================================================================

def linearize_mr(mr_object):
    """
    Converts an MR object (dictionary) into a linearized string.
    This is robust to the JSON format of the E2E Refined Dataset release.
    Example: {'name': 'The Vaults', 'eatType': 'pub'} -> 'name: The Vaults | eatType: pub'
    """
    # The MR from the JSON files is already a dictionary.
    if not isinstance(mr_object, dict):
        # Fallback for safety, in case the input is not a dictionary.
        print(f"Warning: linearize_mr expected a dictionary, but got {type(mr_object)}. Returning empty string.")
        return ""
        
    linearized = []
    for key, value in mr_object.items():
        # Ensure key and value are strings and stripped of whitespace
        key_str = str(key).strip()
        value_str = str(value).strip()
        if value_str: # Only add if the value is not empty
            linearized.append(f"{key_str}: {value_str}")
            
    # Sorting provides a canonical representation of the MR, which is good practice
    return " | ".join(sorted(linearized))

def prepare_data(data_dir):
    """
    Loads, parses, and prepares the E2E Refined Dataset from JSON files.
    Returns a Hugging Face DatasetDict.
    """
    print(f"Loading E2E Refined Dataset from: {data_dir}")
    # Use the filenames you specified: e2e_train.json, etc.
    train_file = os.path.join(data_dir, "e2e_train.json")
    valid_file = os.path.join(data_dir, "e2e_valid.json")
    test_file = os.path.join(data_dir, "e2e_test.json")

    # Check for dataset files and provide instructions if they are missing
    if not all(os.path.exists(f) for f in [train_file, valid_file, test_file]):
        print("="*80)
        print("ERROR: Dataset files not found.")
        print(f"Please ensure you have downloaded and unzipped the release from the 'KSKTYM/E2E-refined-dataset' repository.")
        print(f"The resulting 'e2e_train.json', 'e2e_valid.json', and 'e2e_test.json' files should be in the directory specified by --data_dir ('{data_dir}').")
        print("="*80)
        exit(1)

    # Load data directly from JSON files using the datasets library
    raw_datasets = load_dataset('json', data_files={'train': train_file, 'validation': valid_file, 'test': test_file})
    
    print("\n--- Raw Datasets Info ---")
    print(raw_datasets)
    return raw_datasets

def preprocess_function(examples, tokenizer, max_length):
    """Tokenizes and formats the E2E dataset for training the causal LM."""
    # The reference text is in the 'txt' column, not 'ref'
    inputs = [linearize_mr(mr) for mr in examples['mr']]
    targets = [str(txt) for txt in examples['txt']]
    
    # Format for causal LM: input_mr <eos> target_ref <eos>
    model_inputs = tokenizer(
        [inp + tokenizer.eos_token + tar + tokenizer.eos_token for inp, tar in zip(inputs, targets)],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    
    # Create labels by cloning input_ids
    labels = torch.tensor(model_inputs["input_ids"]).clone()
    
    # Mask out the input part of the labels so loss is only calculated on the target text.
    input_only_tokens = tokenizer(
        [inp + tokenizer.eos_token for inp in inputs],
        max_length=max_length,
        padding=False,
        truncation=True
    )
    
    for i in range(len(labels)):
        input_len = len(input_only_tokens['input_ids'][i])
        labels[i, :input_len] = -100
        
    # Also mask out padding tokens in the labels
    labels[labels == tokenizer.pad_token_id] = -100

    model_inputs["labels"] = labels.tolist()
    return model_inputs

# ==============================================================================
# SECTION 3: GENERATION AND EVALUATION
# Custom beam search and a wrapper for the official E2E evaluation script.
# ==============================================================================

def beam_search_generate(models, tokenizer, input_ids, max_new_tokens, beam_width=5):
    """Custom beam search generation for the 3-part split model."""
    client_head, server, client_tail = models
    client_head.eval()
    server.eval()
    client_tail.eval()
    device = input_ids.device
    eos_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        prompt_attention_mask = torch.ones_like(input_ids)
        head_out, head_past = client_head(input_ids, attention_mask=prompt_attention_mask, use_cache=True)
        server_out, server_past = server(inputs_embeds=head_out, past_key_values=head_past, attention_mask=prompt_attention_mask, use_cache=True)
        logits, tail_past = client_tail(inputs_embeds=server_out, past_key_values=server_past, attention_mask=prompt_attention_mask, use_cache=True)

        next_token_logits = logits[:, -1, :]
        log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
        top_log_probs, top_indices = torch.topk(log_probs, beam_width, dim=-1)

        beams = []
        for i in range(beam_width):
            token_id = top_indices[:, i].unsqueeze(-1)
            log_prob = top_log_probs[:, i]
            beams.append({
                "sequence": torch.cat([input_ids, token_id], dim=-1),
                "log_prob": log_prob,
                "head_past": head_past, "server_past": server_past, "tail_past": tail_past,
                "finished": token_id.item() == eos_token_id
            })

        for _ in range(max_new_tokens - 1):
            new_beams = []
            any_beam_active = False
            for beam in beams:
                if beam["finished"]:
                    new_beams.append(beam)
                    continue
                
                any_beam_active = True
                last_token = beam["sequence"][:, -1].unsqueeze(-1)
                full_sequence_attention_mask = torch.ones_like(beam["sequence"])

                head_out, new_head_past = client_head(last_token, past_key_values=beam["head_past"], attention_mask=full_sequence_attention_mask, use_cache=True)
                server_out, new_server_past = server(inputs_embeds=head_out, past_key_values=beam["server_past"], attention_mask=full_sequence_attention_mask, use_cache=True)
                logits, new_tail_past = client_tail(inputs_embeds=server_out, past_key_values=beam["tail_past"], attention_mask=full_sequence_attention_mask, use_cache=True)

                next_token_logits = logits[:, -1, :]
                log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
                top_log_probs, top_indices = torch.topk(log_probs, beam_width, dim=-1)

                for i in range(beam_width):
                    token_id = top_indices[:, i].unsqueeze(-1)
                    new_log_prob = beam["log_prob"] + top_log_probs[:, i]
                    new_sequence = torch.cat([beam["sequence"], token_id], dim=-1)
                    new_beams.append({
                        "sequence": new_sequence, "log_prob": new_log_prob,
                        "head_past": new_head_past, "server_past": new_server_past, "tail_past": new_tail_past,
                        "finished": token_id.item() == eos_token_id
                    })
            
            if not any_beam_active:
                break
            
            beams = sorted(new_beams, key=lambda x: x["log_prob"].item(), reverse=True)[:beam_width]

        best_beam = sorted(beams, key=lambda x: x["log_prob"].item(), reverse=True)[0]
        return best_beam["sequence"]

def run_evaluation(models, tokenizer, test_dataset, args):
    """Generates predictions and runs the official E2E evaluation script."""
    print("\nRunning evaluation on the test set...")
    
    client_head, server, client_tail = models

    # Merge adapters into the base model for stable, predictable inference
    print("Merging DoRA adapters into the base model for evaluation...")
    client_head_merged = client_head.merge_and_unload()
    server_merged = server.merge_and_unload()
    client_tail_merged = client_tail.merge_and_unload()
    merged_models = [client_head_merged, server_merged, client_tail_merged]

    ref_map = {}
    for item in test_dataset:
        mr = linearize_mr(item['mr'])
        if not mr: continue
        if mr not in ref_map:
            ref_map[mr] = []
        # The reference text is in the 'txt' column
        ref_map[mr].append(str(item['txt']))
    
    ref_file_path = os.path.join(args.output_dir, "eval_references.txt")
    with open(ref_file_path, "w", encoding="utf-8") as f:
        for mr in ref_map:
            f.write("\n".join(ref_map[mr]))
            f.write("\n\n")

    pred_file_path = os.path.join(args.output_dir, "eval_predictions.txt")
    with open(pred_file_path, "w", encoding="utf-8") as f:
        for mr in tqdm(ref_map.keys(), desc="Generating Predictions"):
            input_text = mr + tokenizer.eos_token
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(args.device)
            
            max_gen_len = args.max_seq_length - input_ids.shape[1]
            if max_gen_len <= 0:
                generated_text = ""
            else:
                output_ids = beam_search_generate(
                    merged_models, tokenizer, input_ids, max_new_tokens=max_gen_len, beam_width=args.beam_width
                )
                generated_text = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
            
            f.write(generated_text.strip() + "\n")

    print(f"Executing official E2E metrics script: {args.e2e_metrics_path}")
    command = [
        "python", args.e2e_metrics_path,
        ref_file_path,
        pred_file_path
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        print(f"Error: The evaluation script was not found at '{args.e2e_metrics_path}'. Please check the path.")
        return None
    except subprocess.CalledProcessError as e:
        print("Error running the evaluation script. It might have missing dependencies (Java, Perl).")
        print("Command:", " ".join(e.cmd))
        print("Stderr:", e.stderr)
        return None

    output = result.stdout
    print("\n--- Evaluation Script Output ---")
    print(output)
    print("------------------------------")
    
    scores = {}
    metrics_to_find = ["BLEU", "NIST", "METEOR", "ROUGE_L", "CIDEr"]
    for metric in metrics_to_find:
        match = re.search(rf"{metric}:\s*([\d.]+)", output)
        if match:
            scores[metric] = float(match.group(1))
            
    return scores

# ==============================================================================
# SECTION 4: CHECKPOINTING
# Functions to save and load model adapters and optimizer states.
# ==============================================================================

def save_checkpoint(models, optimizers, epoch, args):
    """Saves the PEFT adapters and optimizer states."""
    client_head, server, client_tail = models
    client_optimizer, server_optimizer = optimizers
    
    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Saving checkpoint for epoch {epoch} to {checkpoint_dir}...")
    
    client_head.save_pretrained(os.path.join(checkpoint_dir, "client_head_dora"))
    server.save_pretrained(os.path.join(checkpoint_dir, "server_dora"))
    client_tail.save_pretrained(os.path.join(checkpoint_dir, "client_tail_dora"))
    
    torch.save(client_optimizer.state_dict(), os.path.join(checkpoint_dir, "client_optimizer.pt"))
    torch.save(server_optimizer.state_dict(), os.path.join(checkpoint_dir, "server_optimizer.pt"))
    
    print(f"Checkpoint saved successfully.")

def load_checkpoint(base_models, optimizers, checkpoint_dir):
    """Loads PEFT adapters and optimizer states into base models and optimizers."""
    client_head_base, server_base, client_tail_base = base_models
    client_optimizer, server_optimizer = optimizers

    print(f"Loading checkpoint from {checkpoint_dir}...")
    
    client_head = PeftModel.from_pretrained(client_head_base, os.path.join(checkpoint_dir, "client_head_dora"))
    server = PeftModel.from_pretrained(server_base, os.path.join(checkpoint_dir, "server_dora"))
    client_tail = PeftModel.from_pretrained(client_tail_base, os.path.join(checkpoint_dir, "client_tail_dora"))
    
    client_optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "client_optimizer.pt")))
    server_optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "server_optimizer.pt")))
    
    print(f"Checkpoint loaded successfully.")
    return [client_head, server, client_tail], [client_optimizer, server_optimizer]

# ==============================================================================
# SECTION 5: MAIN TRAINING FUNCTION
# The main orchestrator for the entire pipeline.
# ==============================================================================

def main(args):
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load and Prepare E2E Refined Dataset
    raw_datasets = prepare_data(args.data_dir)
    
    # --- EVALUATION-ONLY MODE ---
    if args.eval_only:
        if not args.checkpoint_path:
            raise ValueError("--checkpoint_path must be provided when using --eval_only")

        print(f"--- Running in Evaluation-Only Mode ---")
        print(f"Loading model from: {args.checkpoint_path}")

        # Initialize base models
        base_model = GPT2LMHeadModel.from_pretrained(args.model_name)
        split_points = [int(p.strip()) for p in args.split_points.split(',')]
        client_head_base = ClientHead(base_model, split_points[0])
        server_base = Server(base_model, split_points[0], split_points[1])
        client_tail_base = ClientTail(base_model, split_points[1])
        
        # Load the trained adapters
        client_head = PeftModel.from_pretrained(client_head_base, os.path.join(args.checkpoint_path, "client_head_dora"))
        server = PeftModel.from_pretrained(server_base, os.path.join(args.checkpoint_path, "server_dora"))
        client_tail = PeftModel.from_pretrained(client_tail_base, os.path.join(args.checkpoint_path, "client_tail_dora"))
        
        # Move to device
        client_head.to(device)
        server.to(device)
        client_tail.to(device)
        
        final_models = [client_head, server, client_tail]
        scores = run_evaluation(final_models, tokenizer, raw_datasets["test"], args)
        if scores:
            print("\n--- Final Evaluation Scores ---")
            print(json.dumps(scores, indent=2))
            with open(os.path.join(args.output_dir, "final_scores.json"), "w") as f:
                json.dump(scores, f, indent=2)
            print(f"Final scores saved to {os.path.join(args.output_dir, 'final_scores.json')}")
        else:
            print("Evaluation failed. Please check logs for errors.")
        return # Exit after evaluation

    # --- TRAINING MODE ---
    tokenized_datasets = raw_datasets.map(
        lambda x: preprocess_function(x, tokenizer, args.max_seq_length),
        batched=True,
        num_proc=4,
        remove_columns=raw_datasets["train"].column_names
    )
    tokenized_datasets.set_format("torch")
    
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=args.batch_size)
    
    # Model Initialization
    print("Initializing U-shaped split model...")
    base_model = GPT2LMHeadModel.from_pretrained(args.model_name)
    split_points = [int(p.strip()) for p in args.split_points.split(',')]
    
    client_head_base = ClientHead(base_model, split_points[0])
    server_base = Server(base_model, split_points[0], split_points[1])
    client_tail_base = ClientTail(base_model, split_points[1])
    
    # Apply DoRA Adapters
    print("Applying DoRA adapters to all model parts...")
    dora_config = LoraConfig(
        r=args.dora_rank, 
        lora_alpha=args.lora_alpha, 
        lora_dropout=0.05,
        use_dora=True,
        target_modules=["c_attn", "c_proj", "c_fc"], 
        task_type=TaskType.CAUSAL_LM,
    )
    client_head = get_peft_model(client_head_base, dora_config)
    server = get_peft_model(server_base, dora_config)
    client_tail = get_peft_model(client_tail_base, dora_config)

    print("\n--- Trainable Parameters ---")
    client_head.print_trainable_parameters()
    server.print_trainable_parameters()
    client_tail.print_trainable_parameters()
    print("--------------------------\n")

    # CRITICAL STEP: Enforce Weight Tying
    client_tail.base_model.model.lm_head.weight = client_head.base_model.model.transformer.wte.weight
    print("Weight tying between client_head embedding and client_tail lm_head enforced.")

    client_head.to(device)
    server.to(device)
    client_tail.to(device)

    # Optimizers
    client_params = list(client_head.parameters()) + list(client_tail.parameters())
    client_optimizer = optim.AdamW(filter(lambda p: p.requires_grad, client_params), lr=args.learning_rate)
    server_optimizer = optim.AdamW(filter(lambda p: p.requires_grad, server.parameters()), lr=args.learning_rate)
    
    loss_fn = nn.CrossEntropyLoss()

    # Initialize Gradient Scaler for stable mixed-precision training
    scaler = torch.cuda.amp.GradScaler(enabled=(args.device == "cuda"))

    # Load from checkpoint if specified
    start_epoch = 0
    if args.resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
        models, optimizers = load_checkpoint(
            [client_head_base, server_base, client_tail_base],
            [client_optimizer, server_optimizer],
            args.resume_from_checkpoint
        )
        client_head, server, client_tail = models
        client_optimizer, server_optimizer = optimizers
        client_head.to(device)
        server.to(device)
        client_tail.to(device)
        start_epoch = int(re.search(r'epoch-(\d+)', args.resume_from_checkpoint).group(1))

    # Training Loop
    print("Starting training...")
    for epoch in range(start_epoch, args.num_epochs):
        client_head.train()
        server.train()
        client_tail.train()
        
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}", unit="batch")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            client_optimizer.zero_grad()
            server_optimizer.zero_grad()

            # Use autocast for mixed precision
            with torch.amp.autocast(device_type=args.device, enabled=(args.device == "cuda")):
                head_output, _ = client_head(input_ids, attention_mask=attention_mask, use_cache=False)
                server_output, _ = server(inputs_embeds=head_output, attention_mask=attention_mask, use_cache=False)
                logits, _ = client_tail(inputs_embeds=server_output, attention_mask=attention_mask, use_cache=False)

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # Scale the loss and perform backward pass
            scaler.scale(loss).backward()

            # Unscale gradients before clipping
            scaler.unscale_(client_optimizer)
            scaler.unscale_(server_optimizer)
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(client_params, 1.0)
            torch.nn.utils.clip_grad_norm_(server.parameters(), 1.0)

            # Optimizer step
            scaler.step(client_optimizer)
            scaler.step(server_optimizer)

            # Update the scale for next iteration
            scaler.update()

            if not torch.isnan(loss):
                total_loss += loss.item()
            progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})
            
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} finished. Average Training Loss: {avg_loss:.4f}")

        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                [client_head, server, client_tail],
                [client_optimizer, server_optimizer],
                epoch + 1,
                args
            )

    # Final Evaluation
    final_models = [client_head, server, client_tail]
    scores = run_evaluation(final_models, tokenizer, raw_datasets["test"], args)
    if scores:
        print("\n--- Final Evaluation Scores ---")
        print(json.dumps(scores, indent=2))
        with open(os.path.join(args.output_dir, "final_scores.json"), "w") as f:
            json.dump(scores, f, indent=2)
        print(f"Final scores saved to {os.path.join(args.output_dir, 'final_scores.json')}")
    else:
        print("Evaluation failed. Please check logs for errors.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Split-DoRA GPT-2 model on the E2E Refined NLG dataset.")
    
    # Path and Device Arguments
    parser.add_argument("--model_name", type=str, default="gpt2", help="Base GPT-2 model from Hugging Face.")
    parser.add_argument("--data_dir", type=str, default="./e2e-refined-dataset/", help="Directory containing train.json, valid.json, and test.json from the E2E Refined Dataset release.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and evaluation results.")
    parser.add_argument("--e2e_metrics_path", type=str, required=True, help="Path to the official 'measure_scores.py' script.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on ('cuda' or 'cpu').")
    
    # Training Hyperparameters
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Peak learning rate for the AdamW optimizer.")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size per device.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Total number of training epochs.")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum sequence length for tokenization.")
    
    # Model & DoRA Hyperparameters
    parser.add_argument("--split_points", type=str, default="3,9", help="Comma-separated layer indices to split the model at (e.g., '3,9' for gpt2-small).")
    parser.add_argument("--dora_rank", type=int, default=16, help="Rank 'r' for the DoRA adapters.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha scaling parameter for LoRA/DoRA.")
    
    # Generation Hyperparameters
    parser.add_argument("--beam_width", type=int, default=5, help="Beam width for beam search generation during evaluation.")
    
    # Checkpointing & Evaluation Arguments
    parser.add_argument("--save_interval", type=int, default=1, help="Save a checkpoint every N epochs.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint directory to resume training from.")
    parser.add_argument("--eval_only", action="store_true", help="If set, skip training and only run evaluation on a trained model.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the trained model checkpoint to use for evaluation-only mode.")

    args = parser.parse_args()
    main(args)
