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

# Download necessary NLTK packages with SSL workaround
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except:
        logger.warning("Could not download punkt, but will try to continue anyway")

# Set up metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")

class E2EDataset(Dataset):
    """E2E NLG Challenge dataset."""
    
    def __init__(self, hf_dataset, split, tokenizer, max_length=256):
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
        default=False  # Enable by default since user has 30GB VRAM
    )
    # New arguments for checkpoint saving/loading
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every X steps"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from"
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

def save_checkpoint(args, model, tokenizer, optimizer, scheduler, epoch, global_step, tr_loss,
                   losses, perplexities, sanity_check_results, scaler=None):
    """Save model checkpoint and training state."""
    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    logger.info(f"Saving checkpoint to {checkpoint_dir}")
    
    # Save model
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    
    # Save optimizer and scheduler states
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
    
    # Save scaler state if using FP16
    if scaler is not None:
        torch.save(scaler.state_dict(), os.path.join(checkpoint_dir, "scaler.pt"))
    
    # Save training state
    training_state = {
        "epoch": epoch,
        "global_step": global_step,
        "tr_loss": tr_loss,
        "losses": losses,
        "perplexities": perplexities,
        "args": vars(args) # Convert args to dict for JSON serialization
    }
    
    with open(os.path.join(checkpoint_dir, "training_state.json"), "w") as f:
        json.dump(training_state, f)
    
    # Save sanity check results
    with open(os.path.join(checkpoint_dir, "sanity_checks.json"), "w") as f:
        json.dump(sanity_check_results, f)
    
    logger.info(f"Checkpoint saved at step {global_step}")
    return checkpoint_dir


def load_checkpoint(args, model, tokenizer, optimizer, scheduler, scaler=None):
    """Load model and training state from checkpoint while fixing optimizer state."""
    checkpoint_dir = args.resume_from_checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_dir}")
    
    # Load model weights (PEFT adapter)
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    logger.info("Model weights loaded successfully")
    
    # Load training state
    training_state = {}
    if os.path.exists(os.path.join(checkpoint_dir, "training_state.json")):
        with open(os.path.join(checkpoint_dir, "training_state.json"), "r") as f:
            training_state = json.load(f)
        logger.info("Training state loaded")
    
    # Get training progress
    epoch = training_state.get("epoch", 0)
    global_step = training_state.get("global_step", 0)
    tr_loss = training_state.get("tr_loss", 0.0)
    losses = training_state.get("losses", [])
    perplexities = training_state.get("perplexities", [])
    
    # Attempt to load optimizer with parameter matching
    if os.path.exists(os.path.join(checkpoint_dir, "optimizer.pt")):
        try:
            # First initialize a fresh optimizer with the correct parameter structure
            fresh_optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
            # Load the saved state
            checkpoint = torch.load(os.path.join(checkpoint_dir, "optimizer.pt"))
            # Map parameters by name instead of position to fix mismatch
            saved_groups = checkpoint['state']
            current_groups = fresh_optimizer.state_dict()['state']
            # Map saved state to current parameters where possible
            for k, v in current_groups.items():
                if k in saved_groups:
                    fresh_optimizer.state[k] = saved_groups[k]
            optimizer = fresh_optimizer
            logger.info("Optimizer state partially recovered")
        except Exception as e:
            logger.warning(f"Advanced optimizer recovery failed: {e}")
            logger.warning("Training will continue with a fresh optimizer")
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    else:
        logger.warning("No optimizer checkpoint found")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Load scheduler state
    if os.path.exists(os.path.join(checkpoint_dir, "scheduler.pt")):
        try:
            scheduler.load_state_dict(torch.load(os.path.join(checkpoint_dir, "scheduler.pt")))
            logger.info("Scheduler state loaded")
        except Exception as e:
            logger.warning(f"Failed to load scheduler: {e}")
    
    # Load scaler state if it exists
    if scaler is not None and os.path.exists(os.path.join(checkpoint_dir, "scaler.pt")):
        try:
            scaler.load_state_dict(torch.load(os.path.join(checkpoint_dir, "scaler.pt")))
            logger.info("Scaler state loaded")
        except Exception as e:
            logger.warning(f"Failed to load scaler: {e}")
            logger.warning("Training will continue with fresh scaler")
    
    # Load sanity check results
    sanity_check_results = []
    if os.path.exists(os.path.join(checkpoint_dir, "sanity_checks.json")):
        with open(os.path.join(checkpoint_dir, "sanity_checks.json"), "r") as f:
            sanity_check_results = json.load(f)
        logger.info("Sanity check results loaded")
    
    logger.info(f"Resumed from epoch {epoch}, global step {global_step}")
    return model, optimizer, scheduler, epoch, global_step, tr_loss, losses, perplexities, sanity_check_results


def train(args, model, tokenizer, train_dataloader, valid_dataloader, train_dataset, valid_dataset):
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
    
    # Create scaler BEFORE loading checkpoint
    # if args.fp16:
    #     from torch.amp import GradScaler
    #     scaler = GradScaler()
    #     logger.info("Created GradScaler for mixed precision training")
    # else:
    scaler = None
    
    # Training loop variables
    global_step = 0
    tr_loss = 0.0
    start_epoch = 0
    losses = []
    perplexities = []
    sanity_check_results = []
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        model, optimizer, scheduler, start_epoch, global_step, tr_loss, losses, perplexities, sanity_check_results = load_checkpoint(
            args, model, tokenizer, optimizer, scheduler, scaler
        )
    
    # Training loop
    logger.info("***** Running training *****")
    #logger.info(f"Using FP16: {args.fp16}")
    model.train()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0.0
        
        for step, batch in enumerate(epoch_iterator):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass with mixed precision if enabled
            # if args.fp16:
            #     from torch.amp import autocast
            #     with autocast(device_type='cuda'):
            #         outputs = model(
            #             input_ids=input_ids,
            #             attention_mask=attention_mask,
            #             labels=labels,
            #             return_dict=True,
            #         )
            #         loss = outputs.loss
                    
            #     # Scale loss if gradient accumulation is used
            #     if args.gradient_accumulation_steps > 1:
            #         loss = loss / args.gradient_accumulation_steps
                
            #     # Backward pass with scaled gradients
            #     scaler.scale(loss).backward()
            # else:
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
                # if args.fp16:
                #     # Check if gradients are finite before stepping
                #     scaler.unscale_(optimizer)
                    
                #     # Clip gradients if needed
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                #     # Step with scaler
                #     scaler.step(optimizer)
                #     scaler.update()
                # else:
                # Standard optimizer step
                optimizer.step()
                
                # Always update scheduler - it's robust to skipped steps
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
                
                # Save checkpoint at specified intervals
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_checkpoint(
                        args, model, tokenizer, optimizer, scheduler, epoch,
                        global_step, tr_loss, losses, perplexities, sanity_check_results, scaler
                    )
                
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
        
        # Log epoch results
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1} - Average Loss: {avg_epoch_loss:.4f}, Perplexity: {math.exp(avg_epoch_loss):.4f}")
        
        # Save checkpoint BEFORE evaluation
        logger.info(f"Saving checkpoint after epoch {epoch+1} (before evaluation)...")
        save_checkpoint(
            args, model, tokenizer, optimizer, scheduler, epoch+1,
            global_step, tr_loss, losses, perplexities, sanity_check_results, scaler
        )
        
        # Evaluate after saving checkpoint
        logger.info(f"Running evaluation for epoch {epoch+1}...")
        eval_results = evaluate_model(args, model, tokenizer, valid_dataloader, valid_dataset)
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
    
    # Save final results
    with open(os.path.join(args.output_dir, "losses.json"), "w") as f:
        json.dump(losses, f)
    with open(os.path.join(args.output_dir, "perplexities.json"), "w") as f:
        json.dump(perplexities, f)
    with open(os.path.join(args.output_dir, "sanity_checks.json"), "w") as f:
        json.dump(sanity_check_results, f)
    
    return global_step, tr_loss / global_step


def evaluate_model(args, model, tokenizer, eval_dataloader, eval_dataset):
    """Evaluate using official E2E NLG Challenge methodology (Python-only implementation)."""
    import subprocess
    import tempfile
    import os
    from pathlib import Path
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    eval_loss = 0.0
    nb_eval_steps = 0
    
    # Use the dataset's mr_to_refs mapping for true references
    mr_to_references = eval_dataset.mr_to_refs
    mr_to_predictions = {}
    
    # First pass: calculate loss only
    for batch in tqdm(eval_dataloader, desc="Calculating loss"):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            
            loss = outputs.loss
            eval_loss += loss.item()
            nb_eval_steps += 1
    
    # Second pass: generate predictions for unique MRs
    unique_mrs = list(mr_to_references.keys())
    logger.info(f"Generating predictions for {len(unique_mrs)} unique MRs...")
    
    for mr in tqdm(unique_mrs, desc="Generating"):
        prompt = f"MR: {mr} REF:"
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=args.max_length,
            num_beams=5,  # Standard E2E setting
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        if "REF:" in generated_text:
            generated_text = generated_text.split("REF:")[1].strip()
        
        mr_to_predictions[mr] = generated_text
    
    # Calculate perplexity
    perplexity = math.exp(eval_loss / nb_eval_steps) if eval_loss > 0 else float("inf")
    
    # === CREATE E2E-STYLE FILES ===
    # Create temporary files in E2E format for evaluation
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create predictions file (system output)
        pred_file = os.path.join(temp_dir, "predictions.txt")
        with open(pred_file, 'w', encoding='utf-8') as f:
            for mr in unique_mrs:
                if mr in mr_to_predictions:
                    f.write(f"{mr_to_predictions[mr]}\n")
        
        # Create references file (multi-reference format)
        ref_file = os.path.join(temp_dir, "references.txt")
        with open(ref_file, 'w', encoding='utf-8') as f:
            for mr in unique_mrs:
                if mr in mr_to_references:
                    # Write all references for this MR
                    for ref in mr_to_references[mr]:
                        f.write(f"{ref}\n")
                    # Add empty line to separate different MRs (E2E format)
                    if len(mr_to_references[mr]) > 1:
                        f.write("\n")
        
        # === PYTHON-ONLY E2E EVALUATION ===
        predictions = []
        references_list = []
        
        # Read predictions
        with open(pred_file, 'r', encoding='utf-8') as f:
            predictions = [line.strip() for line in f if line.strip()]
        
        # Read multi-references (E2E format)
        with open(ref_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            ref_groups = content.split('\n\n')  # Split by empty lines
            
            for group in ref_groups:
                refs = [line.strip() for line in group.split('\n') if line.strip()]
                if refs:
                    references_list.append(refs)
        
        # Ensure we have matching numbers
        min_len = min(len(predictions), len(references_list))
        predictions = predictions[:min_len]
        references_list = references_list[:min_len]
        
        logger.info(f"Evaluating {len(predictions)} predictions with E2E methodology")
        
        # === E2E OFFICIAL BLEU (corpus-level) ===
        tokenized_predictions = []
        tokenized_references = []
        
        for i, pred in enumerate(predictions):
            # Tokenize prediction (lowercase, as per E2E standard)
            pred_tokens = word_tokenize(pred.lower())
            tokenized_predictions.append(pred_tokens)
            
            # Tokenize all references for this prediction
            ref_tokens_list = []
            for ref in references_list[i]:
                ref_tokens = word_tokenize(ref.lower())
                ref_tokens_list.append(ref_tokens)
            tokenized_references.append(ref_tokens_list)
        
        # Calculate corpus-level BLEU (official E2E method)
        try:
            from nltk.translate.bleu_score import corpus_bleu
            bleu_score = corpus_bleu(tokenized_references, tokenized_predictions)
        except Exception as e:
            logger.warning(f"BLEU calculation failed: {e}")
            bleu_score = 0.0
        
        # === E2E OFFICIAL METEOR (sentence-level then averaged) ===
        meteor_scores = []
        
        for i, pred in enumerate(predictions):
            refs = references_list[i]
            
            # Calculate METEOR against all references, take maximum (E2E standard)
            best_meteor = 0.0
            for ref in refs:
                try:
                    score = meteor.compute(predictions=[pred], references=[ref])
                    meteor_score_val = score["meteor"]
                    best_meteor = max(best_meteor, meteor_score_val)
                except Exception as e:
                    continue
            
            meteor_scores.append(best_meteor)
        
        # Average METEOR scores (official E2E method)
        avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
        
        # === E2E OFFICIAL ROUGE-L (sentence-level then averaged) ===
        rouge_scores = []
        
        for i, pred in enumerate(predictions):
            refs = references_list[i]
            
            # Calculate ROUGE-L against all references, take maximum
            best_rouge = 0.0
            for ref in refs:
                try:
                    score = rouge.compute(predictions=[pred], references=[ref])
                    rouge_score_val = score["rougeL"]
                    best_rouge = max(best_rouge, rouge_score_val)
                except Exception as e:
                    continue
            
            rouge_scores.append(best_rouge)
        
        # Average ROUGE-L scores
        avg_rouge_l = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    
    # === RESULTS IN E2E FORMAT ===
    results = {
        "perplexity": perplexity,
        "bleu": bleu_score,
        "meteor": avg_meteor,
        "rouge_l": avg_rouge_l,
        "num_predictions": len(predictions),
        "num_unique_mrs": len(unique_mrs)
    }
    
    # === E2E-STYLE LOGGING ===
    logger.info("=" * 50)
    logger.info("E2E NLG CHALLENGE EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"BLEU:      {bleu_score:.4f}")
    logger.info(f"METEOR:    {avg_meteor:.4f}")
    logger.info(f"ROUGE-L:   {avg_rouge_l:.4f}")
    logger.info(f"Perplexity: {perplexity:.4f}")
    logger.info("=" * 50)
    logger.info("E2E Benchmark Comparison:")
    logger.info("TGen Baseline:  BLEU=0.6925, METEOR=0.4703, ROUGE-L=0.7257")
    logger.info("Challenge Winner: BLEU=0.7128, METEOR=0.4770, ROUGE-L=0.7378")
    logger.info("=" * 50)
    
    # Debug: Show a few examples
    logger.info("=== SAMPLE PREDICTIONS ===")
    for i in range(min(3, len(predictions))):
        logger.info(f"Prediction: {predictions[i]}")
        logger.info(f"References: {references_list[i]}")
        logger.info(f"METEOR: {meteor_scores[i]:.4f}, ROUGE-L: {rouge_scores[i]:.4f}")
        logger.info("-" * 40)
    
    return results



def test_model(args, model, tokenizer, test_dataloader, test_dataset):

    """Test the model on the test set."""
    logger.info("***** Running testing *****")
    results = evaluate_model(args, model, tokenizer, test_dataloader, test_dataset)
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
    
    # Check if we're resuming from checkpoint
    if args.resume_from_checkpoint:
        logger.info(f"Loading base model for fine-tuning from: {args.model_name_or_path}")
        model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
        # We'll set up PEFT and load weights in the load_checkpoint function
    else:
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
    global_step, train_loss = train(args, model, tokenizer, train_dataloader, valid_dataloader, train_dataset, valid_dataset)

    logger.info(f"Training complete. Global step: {global_step}, Average loss: {train_loss}")
    
    # Save trained model
    model.save_pretrained(os.path.join(args.output_dir, "model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))
    
    # Test model
    test_results = test_model(args, model, tokenizer, test_dataloader, test_dataset)
    
    return test_results

if __name__ == "__main__":
    main()