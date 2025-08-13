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
import sacrebleu
from sacremoses import MosesDetokenizer
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


import warnings
import logging

# At the top of your script, suppress this specific warning
warnings.filterwarnings("ignore", message=".*right-padding was detected.*")

# Or suppress transformers generation warnings
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)


def calculate_e2e_bleu(predictions, references_list):
    """Calculate BLEU using sacreBLEU (E2E standard)"""
    try:
        import sacrebleu
        from sacremoses import MosesDetokenizer

        # Detokenizer
        md = MosesDetokenizer(lang='en')

        # Detokenize predictions
        detok_preds = [md.detokenize(pred.split()) for pred in predictions]

        # Detokenize references while keeping all refs per MR
        detok_refs_per_mr = []
        for refs in references_list:
            detok_refs_per_mr.append([md.detokenize(r.split()) for r in refs])

        # Transpose: sacreBLEU expects list-of-lists where each inner list is
        # all refs of the same index across all hypotheses
        # Example: [[ref1_of_hyp1, ref1_of_hyp2, ...], [ref2_of_hyp1, ref2_of_hyp2, ...], ...]
        refs_transposed = list(zip(*detok_refs_per_mr))

        # Calculate BLEU (0–100 scale)
        bleu = sacrebleu.corpus_bleu(detok_preds, refs_transposed)
        print("BLEU (sacrebleu):", bleu.score)

        # If you still want 0–1 scale:
        bleu_fraction = bleu.score / 100.0
        print("BLEU (fraction):", bleu_fraction)

        
    except ImportError:
        logger.warning("sacreBLEU not available, falling back to NLTK")
        return corpus_bleu(references_list, predictions)

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
        
    import torch

    def _process_data(self):
        examples = []
        for item in self.data:
            mr = item['meaning_representation']
            ref = item['human_reference']

            prompt = f"MR: {mr} REF:"
            prompt_ids = self.tokenizer(prompt, truncation=True, max_length=self.max_length, return_tensors="pt")["input_ids"][0]
            ref_ids = self.tokenizer(" " + ref, truncation=True, max_length=self.max_length, return_tensors="pt")["input_ids"][0]
            # after you compute ref_ids
            eos_id = self.tokenizer.eos_token_id
            # append eos if it won't exceed max_length (we'll also truncate later)
            ref_ids = torch.cat([ref_ids, torch.tensor([eos_id], dtype=torch.long)])

            # Build combined input (prompt + ref)
            combined = torch.cat([prompt_ids, ref_ids])

            # If combined longer than max_length, keep the last max_length tokens (preserve ref)
            if combined.size(0) > self.max_length:
                combined = combined[-self.max_length:]

            # Create pad-filled tensors and LEFT-PAD tokens (so padding is on the left)
            input_ids = torch.full((self.max_length,), self.tokenizer.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((self.max_length,), dtype=torch.long)
            offset = self.max_length - combined.size(0)
            input_ids[offset:] = combined
            attention_mask[offset:] = 1

            # Labels: -100 for prompt tokens, actual token ids for reference tokens
            labels = torch.full((self.max_length,), -100, dtype=torch.long)
            prompt_len = prompt_ids.size(0)
            start = offset + prompt_len
            end = min(offset + prompt_len + ref_ids.size(0), self.max_length)
            if start < end:
                labels[start:end] = input_ids[start:end]

            examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "mr": mr,
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
        "--coverage_loss_weight",
        type=float,
        default=0.3,
        help="Weight for coverage loss (0.2-0.4 recommended)"
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
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, padding_side="left")
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            # Generate text
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + 80,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id,
                length_penalty=1.0,
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


def prepare_for_generation(input_ids: torch.LongTensor,
                           attention_mask: torch.LongTensor,
                           pad_token_id: int = None):
    """
    Trim pad tokens while preserving the original padding side.
    If the batch was left-padded (padding on the left, tokens at right),
    the returned tensors will also be left-padded (tokens at right).
    If the batch was right-padded (padding on the right, tokens at left),
    the returned tensors will also be right-padded (tokens at left).

    Returns:
        new_input_ids, new_attention_mask  # shapes (B, max_nonpad_len)
    """
    device = input_ids.device
    b, s = input_ids.size()

    # detect padding side from attention_mask
    first_col_nonpad = int(attention_mask[:, 0].sum().item())
    last_col_nonpad = int(attention_mask[:, -1].sum().item())

    if first_col_nonpad == 0 and last_col_nonpad > 0:
        padding_side = "left"   # original data is LEFT-PADDED (tokens at right)
    elif last_col_nonpad == 0 and first_col_nonpad > 0:
        padding_side = "right"  # original data is RIGHT-PADDED (tokens at left)
    else:
        # ambiguous -> assume 'right' (common), but this is unlikely for your dataset
        padding_side = "right"

    nonpad_lens = attention_mask.sum(dim=1).long().tolist()
    max_len = max(nonpad_lens) if max(nonpad_lens) > 0 else 1

    # determine pad token fallback
    if pad_token_id is None:
        try:
            pad_val = int(input_ids.new_tensor([0]).item())
        except Exception:
            pad_val = 0
    else:
        pad_val = int(pad_token_id)

    # create output tensors of shape (B, max_len)
    new_input_ids = torch.full((b, max_len), pad_val, dtype=input_ids.dtype, device=device)
    new_attention_mask = torch.zeros((b, max_len), dtype=attention_mask.dtype, device=device)

    for i, l in enumerate(nonpad_lens):
        if l == 0:
            continue
        if padding_side == "right":
            # original has tokens at left -> keep them at left in the trimmed tensor
            new_input_ids[i, :l] = input_ids[i, :l]
            new_attention_mask[i, :l] = 1
        else:
            # original has tokens at right (LEFT-PAD) -> keep tokens at right:
            # place tokens in the LAST `l` positions of the trimmed tensor
            start = max_len - l
            new_input_ids[i, start:] = input_ids[i, -l:]
            new_attention_mask[i, start:] = 1

    return new_input_ids, new_attention_mask





def train(args, model, tokenizer, train_dataloader, valid_dataloader, train_dataset, valid_dataset):
    """Train the model and evaluate on validation set."""
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)

    # open a per-step loss log file
    loss_log_path = os.path.join(args.output_dir, "losses_per_step.csv")
    # If file doesn't exist, write header
    if not os.path.exists(loss_log_path):
        with open(loss_log_path, "w") as f:
            f.write("global_step,epoch,step_in_epoch,loss\n")

    
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
            total_loss = outputs.loss

            # if global_step % 20 == 0:
            #     try:
            #         with torch.no_grad():
            #             # Get MRs for this batch
            #             batch_mrs = batch["mr"]
                        
            #             prompts = [f"MR: {mr} REF:" for mr in batch_mrs]
            #             prompt_inputs = tokenizer(
            #                 prompts, 
            #                 return_tensors="pt", 
            #                 padding=True, 
            #                 padding_side="left",
            #                 truncation=True
            #             ).to(device)
                        
            #             # Use prepare_for_generation on the prompt inputs
            #             trim_ids, trim_mask = prepare_for_generation(
            #                 prompt_inputs["input_ids"], 
            #                 prompt_inputs["attention_mask"], 
            #                 pad_token_id=tokenizer.pad_token_id
            #             )

            #             gen_outputs = model.generate(
            #                 input_ids=trim_ids,
            #                 attention_mask=trim_mask,
            #                 max_length=trim_ids.size(1) + 25,
            #                 num_beams=3,
            #                 early_stopping=True,
            #                 pad_token_id=tokenizer.eos_token_id,
            #                 do_sample=False
            #             )
                        
            #             # Decode generated texts
            #             generated_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in gen_outputs]
                    
            #         # Calculate coverage loss (outside no_grad context)
            #         cv_loss = coverage_loss_function(generated_texts, batch_mrs, weight=args.coverage_loss_weight)
                    
            #         # Combine losses
            #         total_loss = loss + cv_loss
                    
            #         # Log coverage loss for monitoring
            #         if global_step % 100 == 0:
            #             logger.info(f"Step {global_step} - LM Loss: {loss:.4f}, Coverage Loss: {cv_loss:.4f}")
                        
            #     except Exception as e:
            #         logger.warning(f"Coverage loss computation failed: {e}. Using LM loss only.")
            #         total_loss = loss
            # else:
            #     total_loss = loss

            
            # Scale loss if gradient accumulation is used
            if args.gradient_accumulation_steps > 1:
                total_loss = total_loss / args.gradient_accumulation_steps
            
            # Standard backward pass
            total_loss.backward()
            
            tr_loss += total_loss.item() * args.gradient_accumulation_steps
            epoch_loss += total_loss.item() * args.gradient_accumulation_steps
            
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Always update scheduler - it's robust to skipped steps
                scheduler.step()
                model.zero_grad()
                global_step += 1

                step_loss = total_loss.item() * args.gradient_accumulation_steps

                # append to CSV
                with open(loss_log_path, "a") as f:
                    f.write(f"{global_step},{epoch+1},{step},{step_loss:.6f}\n")
                
                # Update progress bar
                epoch_iterator.set_postfix(loss=total_loss.item() * args.gradient_accumulation_steps)
                
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

from collections import Counter
import re

def is_complete_sentence(text):
    """Check if text ends with proper punctuation"""
    return text.strip().endswith(('.', '?', '!'))

def extract_mr_slots(mr):
    """Extract actual slots and values from MR string"""
    slots = {}
    # Parse MR format: name[Blue Spice], eatType[coffee shop], etc.
    pattern = r'(\w+)\[([^\]]+)\]'
    matches = re.findall(pattern, mr)
    for slot_name, slot_value in matches:
        slots[slot_name.lower()] = slot_value.lower()
    return slots

def calculate_slot_coverage(generated_text, mr):
    """Calculate how well the generated text covers MR slots"""
    mr_slots = extract_mr_slots(mr)
    generated_lower = generated_text.lower()
    
    coverage_score = 0
    total_slots = len(mr_slots)
    
    for slot_name, slot_value in mr_slots.items():
        # Check if slot value appears in generated text
        if slot_value in generated_lower:
            coverage_score += 1
        # Partial credit for slot names
        elif slot_name in generated_lower:
            coverage_score += 0.5
    
    return coverage_score / total_slots if total_slots > 0 else 0

def coverage_loss_function(generated_texts, mrs, weight=0.3):
    """
    Compute coverage loss penalizing missing slots in generated texts.
    Fixed version that maintains gradients properly.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not generated_texts or not mrs:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    losses = []
    for gen_text, mr in zip(generated_texts, mrs):
        # Extract only the generated part after "REF:"
        if "REF:" in gen_text:
            gen_text = gen_text.split("REF:")[1].strip()
        
        slot_coverage = calculate_slot_coverage(gen_text, mr)
        coverage_penalty = 1.0 - slot_coverage  # Penalize missing slots
        losses.append(coverage_penalty)
    
    # FIXED: Create tensor that requires gradients
    if losses:
        losses_tensor = torch.tensor(losses, dtype=torch.float32, device=device, requires_grad=True)
        return losses_tensor.mean() * weight
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)


def evaluate_model(args, model, tokenizer, eval_dataloader, eval_dataset):
    """Evaluate using E2E Python implementation with beam reranking -- preserving all debug functionality."""
    import sys
    import re
    sys.path.append('./e2e-metrics')
    
    try:
        from metrics.pymteval import BLEUScore, NISTScore
    except ImportError:
        logger.error("Could not import E2E metrics. Make sure e2e-metrics repo is cloned.")
        return {"bleu": 0.0, "nist": 0.0, "meteor": 0.0, "num_predictions": 0}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Beam reranking helper functions
    def extract_mr_slots(mr):
        """Extract actual slots and values from MR string"""
        slots = {}
        pattern = r'(\w+)\[([^\]]+)\]'
        matches = re.findall(pattern, mr)
        for slot_name, slot_value in matches:
            slots[slot_name.lower()] = slot_value.lower()
        return slots

    def calculate_slot_coverage(generated_text, mr):
        """Calculate how well the generated text covers MR slots"""
        mr_slots = extract_mr_slots(mr)
        generated_lower = generated_text.lower()
        
        coverage_score = 0
        total_slots = len(mr_slots)
        
        for slot_name, slot_value in mr_slots.items():
            if slot_value in generated_lower:
                coverage_score += 1
            elif slot_name in generated_lower:
                coverage_score += 0.5
        
        return coverage_score / total_slots if total_slots > 0 else 0

    def generate_with_beam_reranking(input_ids, attention_mask, mr):
        """Generate text using beam search and rerank by slot coverage"""
        
        # Generate multiple candidates
        with torch.no_grad():
            use_amp = torch.cuda.is_available() and getattr(args, "fp16", False)
            gen_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_length": input_ids.shape[1] + 25,      # Reduced from 25
                    "num_beams": 8,
                    "num_return_sequences": 5,
                    "early_stopping": True,
                    "no_repeat_ngram_size": 4,
                    "repetition_penalty": 1.25,                  # Reduced from 1.25
                    "length_penalty": 0.8,                      
                    "pad_token_id": tokenizer.eos_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
            if use_amp:
                with autocast():
                    outputs = model.generate(**gen_kwargs)
            else:
                outputs = model.generate(**gen_kwargs)
        
        # Decode candidates
        candidates = []
        for output in outputs:
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            if "REF:" in decoded:
                generated_part = decoded.split("REF:")[1].strip()
                candidates.append(generated_part)
        
        if not candidates:
            return ""
        

        # Rerank by slot coverage and length
        scored_candidates = []
        for candidate in candidates:
            coverage_score = calculate_slot_coverage(candidate, mr)
            length_factor = len(candidate.split()) / 15.0  # Target 15 words
            length_score = 1.0 if length_factor <= 1.0 else 1.0 / length_factor
            completeness_score = 1.0 if is_complete_sentence(candidate) else 0.1  # Heavy penalty for incomplete
            
            total_score = coverage_score * 0.5 + length_score * 0.2 + completeness_score * 0.3
            scored_candidates.append((candidate, total_score))
        
        # Return best candidate
        best_candidate = max(scored_candidates, key=lambda x: x[1])[0]
        return best_candidate

    # MR -> refs mapping (unchanged)
    mr_to_references = eval_dataset.mr_to_refs
    all_mrs = list(mr_to_references.keys())
    predictions = []
    references_list = []

    logger.info("Generating predictions for E2E evaluation with beam reranking...")

    # Heuristics tuned for A4000 (16GB) & safety - reduced due to multiple candidates
    if getattr(args, "fp16", False) and torch.cuda.is_available():
        eval_batch_size = 6   # Reduced due to multiple candidates generation
        num_beams = 8
    else:
        eval_batch_size = 4   # Reduced due to multiple candidates generation
        num_beams = 8

    # global generation defaults (tweakable)
    max_new_tokens_cap = 120      # max tokens to generate beyond the prompt
    no_repeat_ngram_size = 0
    repetition_penalty = 1.0
    length_penalty = 1.0
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.eos_token_id

    logger.info(f"Eval batch size: {eval_batch_size}, num_beams: {num_beams}, max_new_tokens_cap: {max_new_tokens_cap}, fp16: {getattr(args, 'fp16', False)}")
    logger.info("Using beam reranking with slot coverage scoring...")

    from torch.cuda.amp import autocast

    for start in tqdm(range(0, len(all_mrs), eval_batch_size), desc="Generating"):
        mrs_batch = all_mrs[start:start + eval_batch_size]
        prompts = [f"MR: {mr} REF:" for mr in mrs_batch]

        # Left padding already set earlier (tokenizer.padding_side = "left")
        inputs = tokenizer(prompts, return_tensors="pt", padding=True,padding_side="left" ,truncation=True).to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Generate with beam reranking for each MR individually
        for i, mr in enumerate(mrs_batch):
            single_input = input_ids[i:i+1]
            single_mask = attention_mask[i:i+1]
            
            best_generation = generate_with_beam_reranking(single_input, single_mask, mr)
            
            predictions.append(best_generation)
            references_list.append(mr_to_references[mr])

        # Your existing debug functionality - PRESERVED EXACTLY
        if len(predictions) >= 10:
            logger.info("=== DEBUGGING GENERATION LENGTH ===")
            sample_lengths = [len(pred[0].split()) if isinstance(pred, tuple) else len(pred.split()) for pred in predictions[:10]]
            ref_lengths = [len(refs[0].split()) for refs in references_list[:10]]

            logger.info(f"Generated lengths: {sample_lengths} (avg: {sum(sample_lengths)/len(sample_lengths):.1f})")
            logger.info(f"Reference lengths: {ref_lengths} (avg: {sum(ref_lengths)/len(ref_lengths):.1f})")
            
            logger.info("=== SAMPLE PREDICTIONS vs REFERENCES ===")
            for i in range(5):
                logger.info(f"MR: {all_mrs[i]}")
                logger.info(f"Generated ({len(predictions[i].split())} tokens): {predictions[i]}")
                logger.info(f"Reference ({len(references_list[i][0].split())} tokens): {references_list[i]}")
                logger.info("-" * 50)

    # Your existing metrics calculation - PRESERVED EXACTLY
    bleu_scorer = BLEUScore()
    for pred, refs in zip(predictions, references_list):
        bleu_scorer.append(pred, refs)
    bleu_score = bleu_scorer.score()

    nist_scorer = NISTScore()
    for pred, refs in zip(predictions, references_list):
        nist_scorer.append(pred, refs)
    nist_score = nist_scorer.score()

    # Calculate METEOR (best-of-refs per MR)
    meteor_scores = []
    for pred, refs in zip(predictions, references_list):
        best_meteor = max(meteor.compute(predictions=[pred], references=[ref])["meteor"]
                         for ref in refs)
        meteor_scores.append(best_meteor)
    meteor_score = sum(meteor_scores) / len(meteor_scores)

    results = {
        "bleu": bleu_score,
        "nist": nist_score,
        "meteor": meteor_score,
        "num_predictions": len(predictions)
    }

    # Your existing results output - PRESERVED EXACTLY
    logger.info("=" * 60)
    logger.info("E2E PYTHON EVALUATION RESULTS (WITH BEAM RERANKING)")
    logger.info("=" * 60)
    logger.info(f"BLEU:    {bleu_score:.4f}")
    logger.info(f"NIST:    {nist_score:.4f}")
    logger.info(f"METEOR:  {meteor_score:.4f}")
    logger.info("=" * 60)

    return results


def parse_e2e_output(output_str):
    """Parse the output from official E2E evaluation script"""
    scores = {}
    
    for line in output_str.strip().split('\n'):
        line = line.strip()
        if ':' in line:
            metric, score_str = line.split(':', 1)
            metric = metric.strip().lower()
            try:
                score = float(score_str.strip())
                scores[metric] = score
            except ValueError:
                continue
    
    return scores


def fallback_evaluation(predictions, references_list):
    """Fallback evaluation using our Python implementation"""
    logger.info("Using fallback evaluation...")
    
    # Calculate METEOR
    meteor_scores = []
    for pred, refs in zip(predictions, references_list):
        best_meteor = max(meteor.compute(predictions=[pred], references=[ref])["meteor"] 
                         for ref in refs)
        meteor_scores.append(best_meteor)
    meteor_score = sum(meteor_scores) / len(meteor_scores)
    
    # Calculate BLEU using NLTK corpus_bleu (closest to E2E)
    from nltk.translate.bleu_score import corpus_bleu
    
    # Tokenize for BLEU
    tokenized_preds = [pred.lower().split() for pred in predictions]
    tokenized_refs = [[ref.lower().split() for ref in refs] for refs in references_list]
    
    bleu_score = corpus_bleu(tokenized_refs, tokenized_preds)
    
    return {
        "bleu": bleu_score,
        "meteor": meteor_score,
        "num_predictions": len(predictions)
    }



    
 


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
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path, padding_side="left",      # set at init (recommended)
                                    truncation_side="left")
    #tokenizer.padding_side = "left"
    

    # Properly set up pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = args.max_length
    # Check if we're resuming from checkpoint
    # Verify the setting
    logger.info(f"Tokenizer padding side: {tokenizer.padding_side}")
    logger.info(f"Tokenizer pad token: {tokenizer.pad_token}")

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