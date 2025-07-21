import torch
import torch.nn as nn
import pandas as pd
import json
import argparse
from torch.utils.data import Dataset, DataLoader
# --- FIX: AdamW is now imported from torch.optim ---
from torch.optim import AdamW 
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
from peft import get_peft_model, LoraConfig, TaskType

# --- 1. Data Preparation Class for JSON (unchanged) ---
class E2EJsonDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length):
        # Load data from the JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Special tokens to separate MR from the reference text
        self.mr_start_token = '<MR>'
        self.ref_start_token = '<REF>'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # --- The Core Change: Processing JSON MR ---
        # Linearize the MR from the dictionary format into a string
        # e.g., {"name": "The Eagle"} -> "name[The Eagle]"
        mr_dict = item['mr']['value']
        mr_parts = []
        for key, value in mr_dict.items():
            # Only include fields that have a value
            if value and str(value).strip():
                mr_parts.append(f"{key}[{value}]")
        mr = ", ".join(mr_parts)

        # The reference text is in the 'txt' field
        ref = item['txt']

        # Combine MR and REF into a single sequence for the causal LM
        # Format: <MR> MR_TEXT <REF> REF_TEXT <|endoftext|>
        formatted_text = (f"{self.mr_start_token} {mr} "
                          f"{self.ref_start_token} {ref} "
                          f"{self.tokenizer.eos_token}")

        tokenized = self.tokenizer(formatted_text,
                                   max_length=self.max_length,
                                   padding="max_length",
                                   truncation=True,
                                   return_tensors="pt")

        input_ids = tokenized['input_ids'].squeeze()
        attention_mask = tokenized['attention_mask'].squeeze()

        # Create labels for loss calculation
        labels = input_ids.clone()

        # Find the start of the reference text to apply the loss mask
        ref_start_token_id = self.tokenizer.convert_tokens_to_ids(self.ref_start_token)
        ref_start_indices = (labels == ref_start_token_id).nonzero(as_tuple=True)[0]

        if len(ref_start_indices) > 0:
            # Mask out the MR part by setting its labels to -100
            # The loss will only be calculated on the reference text.
            mask_end_index = ref_start_indices[0]
            labels[:mask_end_index] = -100
        else:
            # If the ref start token is not found, mask everything
            labels[:] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# --- 2. U-Shaped Split Architecture Components ---
class HeadModel(nn.Module):
    """First stage: Embeddings + first 4 transformer blocks"""
    def __init__(self, base_model):
        super().__init__()
        self.wte = base_model.transformer.wte
        self.wpe = base_model.transformer.wpe
        self.drop = base_model.transformer.drop
        # First 4 transformer blocks (0-3)
        self.h = nn.ModuleList([base_model.transformer.h[i] for i in range(4)])

    def forward(self, input_ids, attention_mask=None):
        # Token embeddings + position embeddings
        inputs_embeds = self.wte(input_ids)
        position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=input_ids.device)
        position_embeds = self.wpe(position_ids)
        hidden_states = self.drop(inputs_embeds + position_embeds)

        # Pass through first 4 transformer blocks
        for block in self.h:
            hidden_states = block(hidden_states, attention_mask=attention_mask)[0]

        return hidden_states, attention_mask

class ServerModel(nn.Module):
    """Middle stage: transformer blocks 4-7"""
    def __init__(self, base_model):
        super().__init__()
        # Middle 4 transformer blocks (4-7)
        self.h = nn.ModuleList([base_model.transformer.h[i] for i in range(4, 8)])

    def forward(self, hidden_states, attention_mask=None):
        # Pass through middle 4 transformer blocks
        for block in self.h:
            hidden_states = block(hidden_states, attention_mask=attention_mask)[0]

        return hidden_states, attention_mask

class TailModel(nn.Module):
    """Final stage: last 4 transformer blocks + LM head"""
    def __init__(self, base_model):
        super().__init__()
        # Last 4 transformer blocks (8-11)
        self.h = nn.ModuleList([base_model.transformer.h[i] for i in range(8, 12)])
        self.ln_f = base_model.transformer.ln_f
        self.lm_head = base_model.lm_head

    def forward(self, hidden_states, attention_mask=None, labels=None):
        # Pass through last 4 transformer blocks
        for block in self.h:
            hidden_states = block(hidden_states, attention_mask=attention_mask)[0]

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        # Language modeling head
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {"loss": loss, "logits": lm_logits}

class UShaped_GPT2_Model(nn.Module):
    """Complete U-shaped model pipeline"""
    def __init__(self, base_model):
        super().__init__()
        self.head = HeadModel(base_model)
        self.server = ServerModel(base_model)
        self.tail = TailModel(base_model)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Stage 1: Head processing
        hidden_states, attention_mask = self.head(input_ids, attention_mask)

        # Stage 2: Server processing  
        hidden_states, attention_mask = self.server(hidden_states, attention_mask)

        # Stage 3: Tail processing with loss calculation
        output = self.tail(hidden_states, attention_mask, labels)

        return output

    def generate(self, input_ids, attention_mask=None, **kwargs):
        """Generation method for inference"""
        self.eval()
        with torch.no_grad():
            max_new_tokens = kwargs.get("max_new_tokens", 25)
            temperature = kwargs.get("temperature", 0.8)
            top_p = kwargs.get("top_p", 0.9)
            repetition_penalty = kwargs.get("repetition_penalty", 1.3)
            pad_token_id = kwargs.get("pad_token_id", input_ids.shape[-1])
            eos_token_id = kwargs.get("eos_token_id", input_ids.shape[-1])

            generated = input_ids.clone()

            for step in range(max_new_tokens):
                # Forward pass through the U-shaped pipeline
                hidden_states, current_attention_mask = self.head(generated, attention_mask)
                hidden_states, current_attention_mask = self.server(hidden_states, current_attention_mask) 
                output = self.tail(hidden_states, current_attention_mask)

                # Get next token logits
                next_token_logits = output["logits"][:, -1, :] / temperature

                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(generated.shape[0]):
                        for token_id in set(generated[i].tolist()):
                            next_token_logits[i, token_id] /= repetition_penalty

                # Apply top-p sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Check for EOS token
                if next_token.item() == eos_token_id:
                    break

                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=-1)

                # Update attention mask
                if attention_mask is not None:
                    attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), 
                                                                          device=attention_mask.device)], dim=-1)

        self.train()
        return generated

# --- 3. Model and Tokenizer Setup (modified) ---
def setup_model_and_tokenizer(model_name):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    base_model = GPT2LMHeadModel.from_pretrained(model_name)

    # Add special tokens for structuring the input
    special_tokens_dict = {
        'bos_token': '<|endoftext|>',
        'eos_token': '<|endoftext|>',
        'pad_token': '<|pad|>',
        'additional_special_tokens': ['<MR>', '<REF>']
    }
    tokenizer.add_special_tokens(special_tokens_dict)

    # Resize model embeddings to accommodate the new tokens
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.pad_token_id = tokenizer.pad_token_id

    # Create U-shaped model
    u_shaped_model = UShaped_GPT2_Model(base_model)

    return u_shaped_model, tokenizer

# --- 4. Apply DoRA PEFT to U-shaped model ---
def apply_dora_peft(model):
    """Apply DoRA PEFT to each stage of the U-shaped model"""
    # Configure PEFT for head
    head_peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"],
        use_dora=True
    )
    model.head = get_peft_model(model.head, head_peft_config)

    # Configure PEFT for server
    server_peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"],
        use_dora=True
    )
    model.server = get_peft_model(model.server, server_peft_config)

    # Configure PEFT for tail
    tail_peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"],
        use_dora=True
    )
    model.tail = get_peft_model(model.tail, tail_peft_config)

    return model

# --- 5. Generation Function for Sanity Checks (modified) ---
def generate_sanity_check(model, tokenizer, device, mr_string="name[NAME], eatType[restaurant], food[Italian]"):
    model.eval()
    mr_start_token = '<MR>'
    ref_start_token = '<REF>'
    prompt = f"{mr_start_token} {mr_string} {ref_start_token}"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    generation_config = {
        "max_new_tokens": 25,  # Fixed for E2E dataset
        "temperature": 0.8,
        "top_p": 0.9,
        "repetition_penalty": 1.3,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }

    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask'], 
            **generation_config
        )

    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

    try:
        cleaned_text = generated_text.split(ref_start_token)[-1].strip()
    except IndexError:
        cleaned_text = generated_text 

    print("\n=== Sanity Generation ===")
    print(f"Sanity MR: {mr_string}")
    print(f"Sanity PRED: {cleaned_text}\n")
    model.train()

# --- 6. Main Training Function (modified) ---
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup U-shaped model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.model_name)

    # Apply DoRA PEFT to all stages
    model = apply_dora_peft(model)
    model.to(device)

    # Print trainable parameters for each stage
    print("=== Trainable Parameters ===")
    print("Head stage:")
    model.head.print_trainable_parameters()
    print("Server stage:")
    model.server.print_trainable_parameters()
    print("Tail stage:")
    model.tail.print_trainable_parameters()

    train_dataset = E2EJsonDataset(json_file=args.train_file, tokenizer=tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizer for all trainable parameters across all stages
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps)

    model.train()
    for epoch in range(args.num_epochs):
        print(f"--- Epoch {epoch+1}/{args.num_epochs} ---")
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for i, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            model.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]

            # Check for overfitting
            if loss.item() < 0.1:
                print(f"\nWarning: Loss {loss.item():.4f} indicates potential overfitting. Consider early stopping.")

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'loss': loss.item()})

            if (i + 1) % args.sanity_check_steps == 0:
                generate_sanity_check(model, tokenizer, device)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Save each stage separately
    model.head.save_pretrained(os.path.join(args.output_dir, "head_stage"))
    model.server.save_pretrained(os.path.join(args.output_dir, "server_stage"))
    model.tail.save_pretrained(os.path.join(args.output_dir, "tail_stage"))
    tokenizer.save_pretrained(args.output_dir)

    print(f"\nTraining complete. U-shaped model stages saved to {args.output_dir}")

# --- 7. Entry Point and Argument Parsing (unchanged) ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune a U-shaped GPT-2 model on E2E NLG from JSON data.")

    # File paths
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training JSON file.")
    parser.add_argument("--dev_file", type=str, required=True, help="Path to the validation JSON file.")
    parser.add_argument("--output_dir", type=str, default="./e2e_u_shaped_gpt2_dora", help="Directory to save the fine-tuned model.")

    # Model and training hyperparameters
    parser.add_argument("--model_name", type=str, default="gpt2-medium", help="Name of the pre-trained model to use.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length for the tokenizer.")
    parser.add_argument("--sanity_check_steps", type=int, default=500, help="Perform a sanity check every N steps.")

    args = parser.parse_args()

    # Check if data files exist
    if not os.path.exists(args.train_file) or not os.path.exists(args.dev_file):
        print(f"Error: Make sure '{args.train_file}' and '{args.dev_file}' are present.")
    else:
        main(args)
