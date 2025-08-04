#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import json
import logging
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import pandas as pd
import evaluate
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download necessary NLTK packages
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set up metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")

class E2EDataset(Dataset):
    """E2E NLG Challenge dataset."""
    
    def __init__(self, file_path, tokenizer, max_length=512):
        """
        Args:
            file_path (str): Path to the E2E dataset file (CSV format)
            tokenizer: Tokenizer for the model
            max_length (int): Maximum sequence length
        """
        logger.info(f"Loading dataset from {file_path}")
        self.df = pd.read_csv(file_path, delimiter=",")
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Process the data
        self.examples = self._process_data()
        logger.info(f"Loaded {len(self.examples)} examples")
        
    def _process_data(self):
        examples = []
        for _, row in self.df.iterrows():
            mr = row['mr']
            ref = row['ref']
            # Format: "MR: [mr] REF: [ref]"
            text = f"MR: {mr} REF: {ref}"
            encodings = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            examples.append({
                "input_ids": encodings["input_ids"][0],
                "attention_mask": encodings["attention_mask"][0],
                "labels": encodings["input_ids"][0].clone(),
            })
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 with DoRA on E2E dataset")
    # Dataset paths now default to the expected repository structure
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default=".", 
        help="Path to the E2E dataset directory"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./results", 
        help="Output directory for models and results"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="gpt2",
        help="Path to pre-trained model or model identifier"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training and evaluation"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="Rank for LoRA adapters"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="Alpha parameter for LoRA"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="Dropout probability for LoRA layers"
    )
    return parser.parse_args()

def set_random_seeds(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_peft_model(model):
    """Configure and return a PEFT model with DoRA."""
    # Define LoRA configuration with DoRA enabled
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["c_attn", "c_proj", "c_fc"],
        use_dora=True,  # Enable DoRA
    )
    
    # Prepare model for k-bit training if needed
    model = prepare_model_for_kbit_training(model)
    
    # Get PEFT model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model

def train(args, model, tokenizer, train_dataloader, valid_dataloader):
    """Train the model and evaluate on validation set."""
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)
    
    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    logger.info("***** Running training *****")
    
    global_step = 0
    tr_loss = 0.0
    model.train()
    
    # Store losses for plotting
    losses = []
    perplexities = []
    
    for epoch in range(args.epochs):
        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0.0
        
        for step, batch in enumerate(epoch_iterator):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            
            tr_loss += loss.item()
            epoch_loss += loss.item()
            global_step += 1
            
            # Update progress bar
            epoch_iterator.set_postfix(loss=loss.item())
            
            # Record loss every 10 steps
            if global_step % 10 == 0:
                avg_loss = tr_loss / global_step
                perplexity = math.exp(avg_loss)
                losses.append({"step": global_step, "loss": avg_loss})
                perplexities.append({"step": global_step, "perplexity": perplexity})
                
        # Evaluate after each epoch
        eval_results = evaluate_model(args, model, tokenizer, valid_dataloader)
        
        # Log epoch results
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1} - Average Loss: {avg_epoch_loss:.4f}, Perplexity: {math.exp(avg_epoch_loss):.4f}")
        logger.info(f"Evaluation results: {eval_results}")
    
    # Save losses and perplexities for plotting
    with open(os.path.join(args.output_dir, "losses.json"), "w") as f:
        json.dump(losses, f)
    
    with open(os.path.join(args.output_dir, "perplexities.json"), "w") as f:
        json.dump(perplexities, f)
    
    return global_step, tr_loss / global_step

def evaluate_model(args, model, tokenizer, eval_dataloader):
    """Evaluate the model on the evaluation dataloader."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    eval_loss = 0.0
    nb_eval_steps = 0
    
    all_predictions = []
    all_references = []
    
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass for evaluation
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            
            loss = outputs.loss
            eval_loss += loss.item()
            nb_eval_steps += 1
            
            # Generate predictions
            for i in range(len(input_ids)):
                # Extract the MR part to use as prompt
                text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                mr_part = text.split("REF:")[0].strip()
                
                # Generate completion based on MR
                prompt = f"{mr_part} REF:"
                prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                
                generated_ids = model.generate(
                    prompt_ids,
                    max_length=args.max_length,
                    num_beams=5,
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                )
                
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                reference_text = tokenizer.decode(labels[i], skip_special_tokens=True)
                
                # Extract only the generated reference part
                if "REF:" in generated_text:
                    generated_text = generated_text.split("REF:")[1].strip()
                
                # Extract only the reference part from the reference text
                if "REF:" in reference_text:
                    reference_text = reference_text.split("REF:")[1].strip()
                
                all_predictions.append(generated_text)
                all_references.append([reference_text])
    
    perplexity = math.exp(eval_loss / nb_eval_steps) if eval_loss > 0 else float("inf")
    
    # Calculate metrics
    # BLEU
    tokenized_predictions = [word_tokenize(pred.lower()) for pred in all_predictions]
    tokenized_references = [[word_tokenize(ref.lower()) for ref in refs] for refs in all_references]
    bleu_score = corpus_bleu(tokenized_references, tokenized_predictions)
    
    # ROUGE
    rouge_result = rouge.compute(predictions=all_predictions, references=[r[0] for r in all_references])
    
    # METEOR
    meteor_result = meteor.compute(predictions=all_predictions, references=[r[0] for r in all_references])
    
    results = {
        "perplexity": perplexity,
        "bleu": bleu_score,
        "rouge1": rouge_result["rouge1"],
        "rouge2": rouge_result["rouge2"],
        "rougeL": rouge_result["rougeL"],
        "meteor": meteor_result["meteor"],
    }
    
    return results

def test_model(args, model, tokenizer, test_dataloader):
    """Test the model on the test set."""
    logger.info("***** Running testing *****")
    results = evaluate_model(args, model, tokenizer, test_dataloader)
    logger.info(f"Test results: {results}")
    
    # Save results to file
    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump(results, f)
    
    return results

def find_dataset_files(data_dir):
    """Find the E2E dataset files in the repository."""
    logger.info(f"Looking for dataset files in {data_dir}")
    
    # First, look for the exact standard filenames
    train_path = os.path.join(data_dir, "trainset.csv")
    valid_path = os.path.join(data_dir, "devset.csv")
    test_path = os.path.join(data_dir, "testset_w_refs.csv")
    
    # If not found, look for the files in the data directory
    if not os.path.exists(train_path):
        for subdir in ["data", "e2e-dataset"]:
            potential_path = os.path.join(data_dir, subdir, "trainset.csv")
            if os.path.exists(potential_path):
                train_path = potential_path
                valid_path = os.path.join(os.path.dirname(potential_path), "devset.csv")
                test_path = os.path.join(os.path.dirname(potential_path), "testset_w_refs.csv")
                break
    
    # If testset_w_refs.csv doesn't exist, try testset.csv
    if not os.path.exists(test_path):
        alternative_test = test_path.replace("testset_w_refs.csv", "testset.csv")
        if os.path.exists(alternative_test):
            test_path = alternative_test
    
    # Verify files exist
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Cannot find training set file at {train_path}")
    if not os.path.exists(valid_path):
        raise FileNotFoundError(f"Cannot find validation set file at {valid_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Cannot find test set file at {test_path}")
    
    logger.info(f"Found dataset files:")
    logger.info(f"Train: {train_path}")
    logger.info(f"Valid: {valid_path}")
    logger.info(f"Test: {test_path}")
    
    return train_path, valid_path, test_path

def main():
    global args
    args = parse_args()
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Find dataset files
    train_file, valid_file, test_file = find_dataset_files(args.data_dir)
    
    # Load tokenizer and model
    logger.info(f"Loading model: {args.model_name_or_path}")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    model = setup_peft_model(model)
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = E2EDataset(train_file, tokenizer, max_length=args.max_length)
    valid_dataset = E2EDataset(valid_file, tokenizer, max_length=args.max_length)
    test_dataset = E2EDataset(test_file, tokenizer, max_length=args.max_length)
    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    logger.info(f"Number of training examples: {len(train_dataset)}")
    logger.info(f"Number of validation examples: {len(valid_dataset)}")
    logger.info(f"Number of test examples: {len(test_dataset)}")
    
    # Train model
    logger.info("Starting training...")
    global_step, train_loss = train(args, model, tokenizer, train_dataloader, valid_dataloader)
    logger.info(f"Training complete. Global step: {global_step}, Average loss: {train_loss}")
    
    # Save trained model
    model.save_pretrained(os.path.join(args.output_dir, "model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))
    
    # Test model
    test_results = test_model(args, model, tokenizer, test_dataloader)
    
    return test_results

if __name__ == "__main__":
    main()