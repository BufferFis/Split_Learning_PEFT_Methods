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
from torch.cuda.amp import autocast, GradScaler  # Added for mixed precision
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm.auto import tqdm
import pandas as pd
from datasets import load_dataset
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
    
    def __init__(self, hf_dataset, split, tokenizer, max_length=512):
        """
        Args:
            hf_dataset: Hugging Face dataset object
            split: Dataset split to use ('train', 'validation', 'test')
            tokenizer: Tokenizer for the model
            max_length (int): Maximum sequence length
        """
        logger.info(f"Preparing {split} dataset")
        self.data = hf_dataset[split]
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Store MR to references mapping for evaluation
        self.mr_to_refs = defaultdict(list)
        for item in self.data:
            mr = item['meaning_representation']
            ref = item['human_reference']
            self.mr_to_refs[mr].append(ref)
        
        # Process the data
        self.examples = self._process_data()
        logger.info(f"Loaded {len(self.examples)} examples from {split} split")
        logger.info(f"Found {len(self.mr_to_refs)} unique MRs with an average of {sum(len(refs) for refs in self.mr_to_refs.values())/len(self.mr_to_refs):.1f} references each")
        
    def _process_data(self):
        examples = []
        for item in self.data:
            mr = item['meaning_representation']
            ref = item['human_reference']
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
                "mr": mr,  # Store MR for reference
            })
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
    
    def get_unique_mrs(self, num_samples=5):
        """Get a sample of unique MRs for sanity checks"""
        unique_mrs = list(self.mr_to_refs.keys())
        if num_samples > 0 and num_samples < len(unique_mrs):
            return random.sample(unique_mrs, num_samples)
        return unique_mrs[:5]  # Default to first 5 if no sampling needed

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 with DoRA on E2E dataset")
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
    parser.add_argument(
        "--sanity_check_steps", 
        type=int, 
        default=200, 
        help="Number of steps between sanity checks"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision training",
        default=True  # Enable by default since user has 30GB VRAM
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

def sanity_check(model, tokenizer, mrs, device, max_length=256):
    """Generate text from the model for a few MRs to see if it's learning."""
    logger.info("===== SANITY CHECK: GENERATING TEXT FOR SAMPLE MRs =====")
    
    # Save original model state
    training = model.training
    
    # Set model to eval mode for generation
    model.eval()
    
    generations = []
    
    with torch.no_grad():
        for i, mr in enumerate(mrs[:5]):  # Generate for up to 5 MRs
            prompt = f"MR: {mr} REF:"
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            # Generate text
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=3,
                no_repeat_ngram_size=2,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False  # Use greedy decoding initially for more stable outputs
            )
            
            # Decode the generated text
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Extract only the generated part after "REF:"
            if "REF:" in generated_text:
                generated_text = generated_text.split("REF:")[1].strip()
            
            logger.info(f"MR: {mr}")
            logger.info(f"Generated: {generated_text}")
            logger.info("-" * 40)
            
            generations.append({"mr": mr, "generated": generated_text})
    
    # Restore original model state
    if training:
        model.train()
    
    return generations

def train(args, model, tokenizer, train_dataloader, valid_dataloader, train_dataset):
    """Train the model and evaluate on validation set."""
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)
    
    # Get sample MRs for sanity checks
    sample_mrs = train_dataset.get_unique_mrs(5)
    
    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Set up mixed precision training
    scaler = GradScaler() if args.fp16 else None
    
    # Training loop
    logger.info("***** Running training *****")
    logger.info(f"Using FP16: {args.fp16}")
    
    global_step = 0
    tr_loss = 0.0
    model.train()
    
    # Store losses for plotting
    losses = []
    perplexities = []
    sanity_check_results = []
    
    for epoch in range(args.epochs):
        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0.0
        
        for step, batch in enumerate(epoch_iterator):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass with mixed precision if enabled
            if args.fp16:
                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        return_dict=True,
                    )
                    loss = outputs.loss
                    
                    # Scale loss if gradient accumulation is used
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                
                # Backward pass with scaled gradients
                scaler.scale(loss).backward()
            else:
                # Standard forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True,
                )
                loss = outputs.loss
                
                # Scale loss if gradient accumulation is used
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                # Standard backward pass
                loss.backward()
            
            tr_loss += loss.item() * args.gradient_accumulation_steps
            epoch_loss += loss.item() * args.gradient_accumulation_steps
            
            # Only update parameters after accumulating enough gradients
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # Unscale gradients for clipping
                    scaler.unscale_(optimizer)
                    
                    # Gradient clipping (optional but recommended with fp16)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    # Update with scaler
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard optimizer step
                    optimizer.step()
                
                scheduler.step()
                model.zero_grad()
                global_step += 1
                
                # Update progress bar
                epoch_iterator.set_postfix(loss=loss.item() * args.gradient_accumulation_steps)
                
                # Record loss every 10 steps
                if global_step % 10 == 0:
                    avg_loss = tr_loss / global_step
                    perplexity = math.exp(avg_loss)
                    losses.append({"step": global_step, "loss": avg_loss})
                    perplexities.append({"step": global_step, "perplexity": perplexity})
                
                # Run sanity check at regular intervals
                if global_step % args.sanity_check_steps == 0:
                    logger.info(f"\nRunning sanity check at step {global_step}...")
                    generations = sanity_check(model, tokenizer, sample_mrs, device, args.max_length)
                    
                    # Store results with step number
                    for gen in generations:
                        gen["step"] = global_step
                    sanity_check_results.extend(generations)
                    
                    # Return to training mode
                    model.train()
        
        # Evaluate after each epoch
        eval_results = evaluate_model(args, model, tokenizer, valid_dataloader)
        
        # Log epoch results
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1} - Average Loss: {avg_epoch_loss:.4f}, Perplexity: {math.exp(avg_epoch_loss):.4f}")
        logger.info(f"Evaluation results: {eval_results}")
        
        # Run sanity check after each epoch
        logger.info(f"\nRunning sanity check after epoch {epoch+1}...")
        generations = sanity_check(model, tokenizer, sample_mrs, device, args.max_length)
        
        # Store results with epoch number
        for gen in generations:
            gen["epoch"] = epoch + 1
            gen["step"] = global_step
        sanity_check_results.extend(generations)
        
        # Return to training mode
        model.train()
    
    # Save losses and perplexities for plotting
    with open(os.path.join(args.output_dir, "losses.json"), "w") as f:
        json.dump(losses, f)
    
    with open(os.path.join(args.output_dir, "perplexities.json"), "w") as f:
        json.dump(perplexities, f)
    
    # Save sanity check results
    with open(os.path.join(args.output_dir, "sanity_checks.json"), "w") as f:
        json.dump(sanity_check_results, f)
    
    return global_step, tr_loss / global_step

def evaluate_model(args, model, tokenizer, eval_dataloader):
    """Evaluate the model on the evaluation dataloader using multiple references."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    eval_loss = 0.0
    nb_eval_steps = 0
    
    # Group predictions by MR
    mr_to_predictions = {}
    mr_to_references = defaultdict(list)
    
    # First pass: calculate loss and collect references
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            mrs = batch.get("mr", None)
            
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
            
            # Extract references for each MR
            for i in range(len(input_ids)):
                text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                mr = mrs[i] if mrs is not None else text.split("REF:")[0].strip().replace("MR: ", "")
                
                reference_text = tokenizer.decode(labels[i], skip_special_tokens=True)
                if "REF:" in reference_text:
                    reference_text = reference_text.split("REF:")[1].strip()
                
                mr_to_references[mr].append(reference_text)
    
    # Second pass: generate predictions for unique MRs
    unique_mrs = list(mr_to_references.keys())
    logger.info(f"Generating predictions for {len(unique_mrs)} unique MRs...")
    
    for mr in tqdm(unique_mrs, desc="Generating"):
        # Generate completion based on MR
        prompt = f"MR: {mr} REF:"
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=args.max_length,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Extract only the generated reference part
        if "REF:" in generated_text:
            generated_text = generated_text.split("REF:")[1].strip()
        
        mr_to_predictions[mr] = generated_text
    
    # Calculate perplexity
    perplexity = math.exp(eval_loss / nb_eval_steps) if eval_loss > 0 else float("inf")
    
    # Prepare data for metrics calculation
    all_predictions = []
    all_references = []
    
    for mr in unique_mrs:
        if mr in mr_to_predictions:
            all_predictions.append(mr_to_predictions[mr])
            all_references.append(mr_to_references[mr])
    
    # Calculate metrics with multiple references
    # BLEU (handles multiple references correctly)
    tokenized_predictions = [word_tokenize(pred.lower()) for pred in all_predictions]
    tokenized_references = [[word_tokenize(ref.lower()) for ref in refs] for refs in all_references]
    bleu_score = corpus_bleu(tokenized_references, tokenized_predictions)
    
    # ROUGE - take maximum score across references for each prediction
    rouge_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    for i, pred in enumerate(all_predictions):
        refs = all_references[i]
        best_rouge = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
        
        for ref in refs:
            score = rouge.compute(predictions=[pred], references=[ref])
            for key in best_rouge:
                best_rouge[key] = max(best_rouge[key], score[key])
        
        for key in rouge_scores:
            rouge_scores[key] += best_rouge[key]
    
    # Average ROUGE scores
    for key in rouge_scores:
        rouge_scores[key] /= len(all_predictions) if all_predictions else 1
    
    # METEOR - similar approach (best match)
    meteor_score = 0
    for i, pred in enumerate(all_predictions):
        refs = all_references[i]
        best_meteor = 0
        
        for ref in refs:
            score = meteor.compute(predictions=[pred], references=[ref])
            best_meteor = max(best_meteor, score["meteor"])
        
        meteor_score += best_meteor
    
    meteor_score /= len(all_predictions) if all_predictions else 1
    
    results = {
        "perplexity": perplexity,
        "bleu": bleu_score,
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "meteor": meteor_score,
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

def main():
    global args
    args = parse_args()
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Optimize CUDA operations for faster training
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info(f"CUDA available: {torch.cuda.device_count()} devices")
        logger.info(f"CUDA current device: {torch.cuda.current_device()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        logger.info(f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    # Load dataset directly using Hugging Face datasets
    logger.info("Loading E2E NLG dataset...")
    try:
        dataset = load_dataset("e2e_nlg")
        logger.info("Dataset loaded successfully")
    except Exception as e:
        # Try with trust_remote_code flag if regular loading fails
        logger.info(f"Standard loading failed: {e}. Trying with trust_remote_code=True")
        dataset = load_dataset("e2e_nlg", trust_remote_code=True)
        logger.info("Dataset loaded successfully with trust_remote_code=True")
    
    # Load tokenizer and model
    logger.info(f"Loading model: {args.model_name_or_path}")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    # Properly set up pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    model = setup_peft_model(model)
    
    # Create datasets
    logger.info("Preparing datasets...")
    train_dataset = E2EDataset(dataset, 'train', tokenizer, max_length=args.max_length)
    valid_dataset = E2EDataset(dataset, 'validation', tokenizer, max_length=args.max_length)
    test_dataset = E2EDataset(dataset, 'test', tokenizer, max_length=args.max_length)
    
    # Create data loaders with multiple workers for faster data loading
    num_workers = 4 if torch.cuda.is_available() else 0
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    
    logger.info(f"Number of training examples: {len(train_dataset)}")
    logger.info(f"Number of validation examples: {len(valid_dataset)}")
    logger.info(f"Number of test examples: {len(test_dataset)}")
    
    # Train model
    logger.info("Starting training...")
    global_step, train_loss = train(args, model, tokenizer, train_dataloader, valid_dataloader, train_dataset)
    logger.info(f"Training complete. Global step: {global_step}, Average loss: {train_loss}")
    
    # Save trained model
    model.save_pretrained(os.path.join(args.output_dir, "model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))
    
    # Test model
    test_results = test_model(args, model, tokenizer, test_dataloader)
    
    return test_results

if __name__ == "__main__":
    main()