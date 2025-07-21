import torch
import pandas as pd
import json
import argparse
from torch.utils.data import Dataset, DataLoader
# --- FIX: AdamW is now imported from torch.optim ---
from torch.optim import AdamW 
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import os

# --- 1. Data Preparation Class for JSON ---
# This class loads data from the specified JSON format, linearizes the MR,
# and prepares it for the causal language model with loss masking.
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

# --- 2. Model and Tokenizer Setup ---
def setup_model_and_tokenizer(model_name):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Add special tokens for structuring the input
    special_tokens_dict = {
        'bos_token': '<|endoftext|>',
        'eos_token': '<|endoftext|>',
        'pad_token': '<|pad|>',
        'additional_special_tokens': ['<MR>', '<REF>']
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    
    # Resize model embeddings to accommodate the new tokens
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer

# --- 3. Generation Function for Sanity Checks ---
def generate_sanity_check(model, tokenizer, device, mr_string="name[NAME], eatType[restaurant], food[Italian]"):
    model.eval()
    mr_start_token = '<MR>'
    ref_start_token = '<REF>'
    prompt = f"{mr_start_token} {mr_string} {ref_start_token}"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    generation_config = {
        "max_new_tokens": 100, "do_sample": True, "temperature": 0.7,
        "top_p": 0.92, "top_k": 50, "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 2, "pad_token_id": tokenizer.eos_token_id,
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

# --- 4. Main Training Function ---
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, tokenizer = setup_model_and_tokenizer(args.model_name)
    model.to(device)

    train_dataset = E2EJsonDataset(json_file=args.train_file, tokenizer=tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
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
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'loss': loss.item()})
            
            if (i + 1) % args.sanity_check_steps == 0:
                generate_sanity_check(model, tokenizer, device)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nTraining complete. Model saved to {args.output_dir}")

# --- 5. Entry Point and Argument Parsing ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune a GPT-2 model on E2E NLG from JSON data.")
    
    # File paths
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training JSON file.")
    parser.add_argument("--dev_file", type=str, required=True, help="Path to the validation JSON file.")
    parser.add_argument("--output_dir", type=str, default="./e2e_gpt2_json_finetuned", help="Directory to save the fine-tuned model.")
    
    # Model and training hyperparameters
    parser.add_argument("--model_name", type=str, default="gpt2-medium", help="Name of the pre-trained model to use.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length for the tokenizer.")
    parser.add_argument("--sanity_check_steps", type=int, default=500, help="Perform a sanity check every N steps.")

    args = parser.parse_args()
    
    # Check if data files exist
    if not os.path.exists(args.train_file) or not os.path.exists(args.dev_file):
        print(f"Error: Make sure '{args.train_file}' and '{args.dev_file}' are present.")
    else:
        main(args)
