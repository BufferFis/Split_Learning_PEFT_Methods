import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os

# --- 1. Configuration ---
MODEL_NAME = 'gpt2-medium'
# Ensure these files are in the same directory as the script
TRAIN_FILE = 'trainset.csv'
DEV_FILE = 'devset.csv' 
OUTPUT_DIR = './e2e_gpt2_medium_finetuned'
NUM_EPOCHS = 3
BATCH_SIZE = 4 # Adjust based on your GPU memory
LEARNING_RATE = 3e-5
MAX_LENGTH = 256 # Max sequence length for tokenizer

# --- 2. Corrected Data Preparation Class ---
# This class formats the data correctly for a causal LM, including loss masking.
class E2EDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Using special tokens to clearly separate MR from the reference text
        self.mr_start_token = '<MR>'
        self.ref_start_token = '<REF>'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        mr = row['mr']
        ref = row['ref']

        # The core fix: Combine MR and REF into a single sequence for the causal LM
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
        # We use the tokenized version of the ref_start_token to find its ID
        ref_start_token_id = self.tokenizer.convert_tokens_to_ids(self.ref_start_token)
        
        # Find all occurrences of the ref_start_token_id
        ref_start_indices = (labels == ref_start_token_id).nonzero(as_tuple=True)[0]

        if len(ref_start_indices) > 0:
            # The crucial step: Mask out the MR part by setting its labels to -100
            # The loss will only be calculated on the reference text.
            mask_end_index = ref_start_indices[0]
            labels[:mask_end_index] = -100
        else:
            # If the ref start token is not found (e.g., due to truncation), mask everything
            labels[:] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# --- 3. Model and Tokenizer Setup with Special Tokens ---
def setup_model_and_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

    # Define and add special tokens. This is critical for the model to understand structure.
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

# --- 4. Robust Generation Function for Sanity Checks ---
# This function uses advanced decoding to prevent repetitive and nonsensical output.
def generate_sanity_check(model, tokenizer, device, mr_string="name[NAME], eatType[restaurant], food[Italian]"):
    model.eval() # Set model to evaluation mode for inference
    mr_start_token = '<MR>'
    ref_start_token = '<REF>'
    prompt = f"{mr_start_token} {mr_string} {ref_start_token}"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # The fix for generation: Use sampling and penalties instead of greedy search
    generation_config = {
        "max_new_tokens": 100,          # Limit the length of the generated text
        "do_sample": True,              # Enable stochastic sampling
        "temperature": 0.7,             # Sharpen the distribution to reduce randomness
        "top_p": 0.92,                  # Use nucleus sampling for dynamic vocabulary
        "top_k": 50,                    # Can be used alongside top_p
        "repetition_penalty": 1.2,      # Penalize words that have already been said
        "no_repeat_ngram_size": 2,      # Prevent 2-grams from repeating
        "pad_token_id": tokenizer.eos_token_id, # Important for clean generation
        "eos_token_id": tokenizer.eos_token_id
    }

    with torch.no_grad():
        # Generate returns a tensor of shape (batch_size, sequence_length)
        output_sequences = model.generate(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask'], 
            **generation_config
        )
    
    # Decode the first sequence in the batch
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    
    # Clean up the output to only show the generated part after the <REF> token
    try:
        # Split the text at the reference token and take the last part
        cleaned_text = generated_text.split(ref_start_token)[-1].strip()
    except IndexError:
        # Fallback if the ref_start_token is not in the generated text
        cleaned_text = generated_text 

    print("\n=== Sanity Generation ===")
    print(f"Sanity MR: {mr_string}")
    print(f"Sanity PRED: {cleaned_text}\n")
    model.train() # Set model back to training mode

# --- 5. Main Training Function ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, tokenizer = setup_model_and_tokenizer()
    model.to(device)

    train_dataset = E2EDataset(csv_file=TRAIN_FILE, tokenizer=tokenizer, max_length=MAX_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # The fix for training stability: AdamW optimizer and a learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps)

    model.train()
    for epoch in range(NUM_EPOCHS):
        print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for i, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            model.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Standard backpropagation
            loss.backward()
            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'loss': loss.item()})
            
            # Perform a sanity check every 500 steps
            if (i + 1) % 500 == 0:
                generate_sanity_check(model, tokenizer, device)

    # Save the final fine-tuned model and tokenizer
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nTraining complete. Model saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    # Make sure you have trainset.csv and devset.csv in the same folder as this script
    if not os.path.exists(TRAIN_FILE) or not os.path.exists(DEV_FILE):
        print(f"Error: Make sure '{TRAIN_FILE}' and '{DEV_FILE}' are present in the current directory.")
    else:
        main()
