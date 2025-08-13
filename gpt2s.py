#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple GPT-2 fine-tuning on the E2E NLG dataset using Hugging Face datasets:
- Loads dataset via load_dataset("e2e_nlg") (falls back to "GEM/e2e_nlg")
- GPT-2 + PEFT LoRA (optional DoRA)
- Mixed precision optional (--fp16)
- Per-step JSON logging of loss and perplexity (metrics.jsonl)
- Periodic sanity generation on a random MR
- Validation perplexity during training
- Final beam-search decoding on test and multi-reference BLEU/METEOR/ROUGE-L
- Checkpointing for full training resume: adapters + tokenizer + optimizer + scheduler + GradScaler (fp16) + RNG state
- Progress bar with ETA per epoch (tqdm)

Resuming:
  Use --resume_from PATH pointing to a checkpoint directory created by this script (e.g., outputs/ckpt_step_00001000).
"""

import os
import re
import math
import json
import time
import random
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
)

from tqdm.auto import tqdm

import nltk
from nltk.translate import bleu_score
from nltk.translate.meteor_score import meteor_score


# -------------------------
# Small helpers
# -------------------------

_WS_RE = re.compile(r"\s+")

def normalize_ws(s: str) -> str:
    return _WS_RE.sub(" ", str(s).strip())

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_nltk():
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4', quiet=True)


# -------------------------
# Load E2E from HF datasets
# -------------------------

def load_e2e_hf(dataset_name: str = "e2e_nlg"):
    """
    Returns (train, validation, test) as datasets.Dataset splits.
    Tries dataset_name, falls back to "GEM/e2e_nlg".
    """
    try:
        raw = load_dataset(dataset_name)
    except Exception:
        raw = load_dataset("GEM/e2e_nlg")

    train = raw["train"]
    if "validation" in raw:
        val = raw["validation"]
    elif "dev" in raw:
        val = raw["dev"]
    else:
        val = raw["train"].select(range(min(500, len(raw["train"]))))
    test = raw["test"] if "test" in raw else val
    return train, val, test

def get_mr_and_refs(example) -> Tuple[str, List[str]]:
    """
    Extract MR string and list of reference strings from a dataset example.
    Supports common field variants across E2E dataset mirrors.
    """
    mr = example.get("meaning_representation") or example.get("mr") or example.get("source") or ""
    refs = example.get("references") or example.get("human_references") or example.get("human_reference") or example.get("ref") or example.get("reference")
    if isinstance(refs, list):
        ref_list = [normalize_ws(r) for r in refs if isinstance(r, str)]
    elif isinstance(refs, str):
        ref_list = [normalize_ws(refs)]
    else:
        ref_list = []
    return normalize_ws(mr), ref_list

def flatten_pairs(split_ds) -> List[Tuple[str, str]]:
    """
    Build (MR, ref) training pairs, one example per reference.
    """
    pairs = []
    for ex in split_ds:
        mr, refs = get_mr_and_refs(ex)
        if not refs:
            continue
        for r in refs:
            pairs.append((mr, r))
    return pairs

def group_refs(split_ds) -> Dict[str, List[str]]:
    """
    MR -> [ref1, ref2, ...] (multi-reference map).
    """
    mp: Dict[str, List[str]] = {}
    for ex in split_ds:
        mr, refs = get_mr_and_refs(ex)
        if not refs:
            continue
        mp.setdefault(mr, []).extend(refs)
    return mp


# -------------------------
# Dataset + Collator
# -------------------------

def build_prompt(mr: str, delimiter: str = " =>") -> str:
    return f"{mr}{delimiter}"

class E2ETrainDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], tokenizer, max_source_len: int, max_target_len: int, delimiter: str = " =>"):
        self.pairs = pairs
        self.tok = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.delimiter = delimiter

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        mr, ref = self.pairs[idx]
        prompt = build_prompt(mr, self.delimiter)
        prompt_ids = self.tok(prompt, add_special_tokens=False)["input_ids"][: self.max_source_len]
        target_ids = self.tok(ref, add_special_tokens=False)["input_ids"][: self.max_target_len]

        input_ids = prompt_ids + target_ids + [self.tok.eos_token_id]
        labels = [-100] * len(prompt_ids) + target_ids + [self.tok.eos_token_id]
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

class PadCollator:
    def __init__(self, pad_token_id: int):
        self.pad = pad_token_id

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        def pad1d(t, fill):
            if len(t) == max_len:
                return t
            pad_len = max_len - len(t)
            return torch.cat([t, torch.full((pad_len,), fill, dtype=t.dtype)])
        input_ids = torch.stack([pad1d(x["input_ids"], self.pad) for x in batch], dim=0)
        attention_mask = torch.stack([pad1d(x["attention_mask"], 0) for x in batch], dim=0)
        labels = torch.stack([pad1d(x["labels"], -100) for x in batch], dim=0)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# -------------------------
# PEFT (LoRA/DoRA)
# -------------------------

def build_lora_config(r=8, alpha=16, dropout=0.05, use_dora=False) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["c_attn", "c_proj", "c_fc"],
        use_dora=use_dora,
        bias="none",
    )


# -------------------------
# Metrics (simple multi-ref)
# -------------------------

def tok_simple(s: str) -> List[str]:
    s = normalize_ws(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return [t for t in s.split() if t]

def rouge_l_f(hyp_tokens: List[str], ref_tokens: List[str]) -> float:
    m, n = len(hyp_tokens), len(ref_tokens)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if hyp_tokens[i] == ref_tokens[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    lcs = dp[m][n]
    prec = lcs / m if m else 0.0
    rec  = lcs / n if n else 0.0
    if prec + rec == 0:
        return 0.0
    beta2 = 1.2**2
    return (1 + beta2) * prec * rec / (rec + beta2 * prec)

def compute_bleu_multi(hyps: List[str], refs_list: List[List[str]]) -> float:
    refs_tok = [[tok_simple(r) for r in refs] for refs in refs_list]
    hyps_tok = [tok_simple(h) for h in hyps]
    weights = (0.25, 0.25, 0.25, 0.25)
    smoothie = bleu_score.SmoothingFunction().method1
    return bleu_score.corpus_bleu(refs_tok, hyps_tok, weights=weights, smoothing_function=smoothie) * 100.0

def compute_meteor_multi(hyps: List[str], refs_list: List[List[str]]) -> float:
    ensure_nltk()
    vals = []
    for h, refs in zip(hyps, refs_list):
        try:
            vals.append(meteor_score(refs, h))
        except Exception:
            vals.append(0.0)
    return float(np.mean(vals) * 100.0)

def compute_rouge_l_multi(hyps: List[str], refs_list: List[List[str]]) -> float:
    vals = []
    for h, refs in zip(hyps, refs_list):
        ht = tok_simple(h)
        best = 0.0
        for r in refs:
            best = max(best, rouge_l_f(ht, tok_simple(r)))
        vals.append(best)
    return float(np.mean(vals) * 100.0)


# -------------------------
# Generation and PPL eval
# -------------------------

def generate_for_mrs(model, tokenizer, mrs: List[str], device, num_beams=10, max_new_tokens=120, delimiter=" =>", no_repeat_ngram_size=3) -> List[str]:
    model.eval()
    outs = []
    with torch.no_grad():
        for mr in mrs:
            prompt = f"{mr}{delimiter}"
            enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            out = model.generate(
                **enc,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                early_stopping=True,
                do_sample=False,
                no_repeat_ngram_size=no_repeat_ngram_size,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            gen_ids = out[0, enc["input_ids"].shape[1]:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            outs.append(normalize_ws(text))
    model.train()
    return outs

def eval_perplexity(model: nn.Module, loader: DataLoader, device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            losses.append(float(out.loss.item()))
    model.train()
    mean_loss = float(np.mean(losses)) if losses else float("inf")
    return math.exp(min(20.0, mean_loss))


# -------------------------
# Checkpointing (resume-ready)
# -------------------------

def checkpoint_dir(output_dir: str, global_step: int) -> str:
    return os.path.join(output_dir, f"ckpt_step_{global_step:08d}")

def save_checkpoint(model: PeftModel,
                    tokenizer,
                    optimizer,
                    scheduler,
                    scaler: torch.cuda.amp.GradScaler,
                    output_dir: str,
                    global_step: int,
                    epoch: int,
                    base_model_name: str):
    ckpt = checkpoint_dir(output_dir, global_step)
    os.makedirs(ckpt, exist_ok=True)
    # Save adapters + tokenizer
    model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    # Save training state (optimizer/scheduler/scaler + RNG)
    state = {
        "global_step": int(global_step),
        "epoch": int(epoch),
        "base_model_name": base_model_name,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    torch.save(state, os.path.join(ckpt, "training_state.pt"))
    # Write a small latest marker
    with open(os.path.join(output_dir, "latest_checkpoint.txt"), "w", encoding="utf-8") as f:
        f.write(ckpt + "\n")
    print(f"[checkpoint] saved: {ckpt}")
    return ckpt

def load_checkpoint(resume_dir: str, device):
    # Load training state
    state_fp = os.path.join(resume_dir, "training_state.pt")
    if not os.path.exists(state_fp):
        raise FileNotFoundError(f"training_state.pt not found in {resume_dir}")
    state = torch.load(state_fp, map_location="cpu")
    base_model_name = state["base_model_name"]

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(resume_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Base model + adapters
    base_cfg = AutoConfig.from_pretrained(base_model_name)
    base = AutoModelForCausalLM.from_pretrained(base_model_name, config=base_cfg)
    base.resize_token_embeddings(len(tokenizer))
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, resume_dir).to(device)

    return model, tokenizer, state

def latest_checkpoint_dir(output_dir: str) -> Optional[str]:
    marker = os.path.join(output_dir, "latest_checkpoint.txt")
    if os.path.exists(marker):
        with open(marker, "r", encoding="utf-8") as f:
            path = f.readline().strip()
        if path and os.path.isdir(path):
            return path
    # Fallback: find by pattern
    candidates = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("ckpt_step_")]
    candidates = [d for d in candidates if os.path.isdir(d)]
    if not candidates:
        return None
    return sorted(candidates)[-1]


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="gpt2", choices=["gpt2", "gpt2-medium"])
    ap.add_argument("--output_dir", type=str, default="./outputs")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_epochs", type=int, default=3)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--num_beams", type=int, default=10)
    ap.add_argument("--gen_max_new_tokens", type=int, default=100)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=3)
    ap.add_argument("--sample_every_steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--use_dora", action="store_true", help="Enable DoRA in LoRA (requires recent peft).")
    ap.add_argument("--metrics_jsonl", type=str, default="metrics.jsonl")
    ap.add_argument("--dataset_name", type=str, default="e2e_nlg", help='HF dataset id (default "e2e_nlg"; falls back to "GEM/e2e_nlg")')
    ap.add_argument("--resume_from", type=str, default="", help="Path to a previous ckpt_step_xxxxxxxx directory to resume from.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}")

    # Data from HF
    train_split, val_split, test_split = load_e2e_hf(args.dataset_name)
    train_pairs = flatten_pairs(train_split)
    val_pairs = flatten_pairs(val_split)
    val_refs = group_refs(val_split)
    test_refs_grouped = group_refs(test_split) if len(group_refs(test_split)) > 0 else val_refs
    val_mrs = list(val_refs.keys())

    # Tokenizer/model (load or build)
    global_step = 0
    start_epoch = 0

    if args.resume_from:
        # Build dummy optimizer/scheduler to load state after model is ready
        base_model_name = None  # will be filled from state
        # Load model/tokenizer and state
        model, tokenizer, state = load_checkpoint(args.resume_from, device)
        base_model_name = state.get("base_model_name", args.model_name)
        global_step = int(state.get("global_step", 0))
        start_epoch = int(state.get("epoch", 0))
        print(f"[resume] from {args.resume_from} (step={global_step}, epoch={start_epoch})")
    else:
        base_model_name = args.model_name
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad; use EOS
        base_cfg = AutoConfig.from_pretrained(base_model_name)
        base = AutoModelForCausalLM.from_pretrained(base_model_name, config=base_cfg)
        base.resize_token_embeddings(len(tokenizer))
        base.config.pad_token_id = tokenizer.pad_token_id
        base.config.use_cache = False
        lora_cfg = build_lora_config(use_dora=args.use_dora)
        model = get_peft_model(base, lora_cfg).to(device)
        model.print_trainable_parameters()

    # Dataloaders
    collator = PadCollator(pad_token_id=tokenizer.pad_token_id)
    train_ds = E2ETrainDataset(train_pairs, tokenizer, max_source_len=128, max_target_len=128)
    val_ds   = E2ETrainDataset(val_pairs, tokenizer, max_source_len=128, max_target_len=128)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=2, pin_memory=True)

    # Optimizer/scheduler/scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = (len(train_loader) // max(1, args.gradient_accumulation_steps)) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16 and device.type == "cuda"))

    # Resume optimizer/scheduler/scaler + RNG if requested
    if args.resume_from:
        _, _, state = load_checkpoint(args.resume_from, device)
        if state.get("optimizer"):
            optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        if scaler and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        rng = state.get("rng_state", {})
        try:
            if "python" in rng and rng["python"] is not None:
                random.setstate(rng["python"])
            if "numpy" in rng and rng["numpy"] is not None:
                np.random.set_state(rng["numpy"])
            if "torch" in rng and rng["torch"] is not None:
                torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and "cuda" in rng and rng["cuda"] is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
        except Exception as e:
            print(f"[resume] RNG restore skipped: {e}")

    # Metrics JSONL
    metrics_fp = os.path.join(args.output_dir, args.metrics_jsonl)
    mfile = open(metrics_fp, "a", encoding="utf-8")
    def log_jsonl(obj: Dict):
        obj = dict(obj)
        obj["ts"] = time.time()
        mfile.write(json.dumps(obj, ensure_ascii=False) + "\n")
        mfile.flush()

    # Training
    print("[train] starting...")
    running_loss = 0.0
    try:
        for epoch in range(start_epoch, args.num_epochs):
            # Progress bar over optimizer steps in this epoch
            steps_this_epoch = len(train_loader) // max(1, args.gradient_accumulation_steps)
            pbar = tqdm(total=steps_this_epoch, desc=f"Epoch {epoch+1}/{args.num_epochs}", unit="step", leave=False)

            for step, batch in enumerate(train_loader):
                model.train()
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=(args.fp16 and device.type == "cuda")):
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = out.loss / max(1, args.gradient_accumulation_steps)

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Optimizer step when accumulation boundary hit
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    global_step += 1
                    train_loss = float(loss.item() * max(1, args.gradient_accumulation_steps))
                    train_ppl = math.exp(min(20.0, train_loss))
                    running_loss += train_loss
                    lr = float(scheduler.get_last_lr()[0])

                    # JSON per-step
                    log_jsonl({"phase": "train", "epoch": epoch+1, "step": global_step, "loss": train_loss, "ppl": train_ppl, "lr": lr})

                    # Progress bar + optional postfix
                    try:
                        pbar.update(1)
                        pbar.set_postfix({"loss": f"{train_loss:.4f}", "ppl": f"{train_ppl:.2f}"})
                    except Exception:
                        pass

                    # Console log
                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        avg = running_loss / args.logging_steps
                        avg_ppl = math.exp(min(20.0, avg))
                        print(f"[epoch {epoch+1}/{args.num_epochs}] step {global_step} - loss {avg:.4f} - ppl {avg_ppl:.2f}")
                        running_loss = 0.0

                    # Sanity sample
                    if args.sample_every_steps > 0 and global_step % args.sample_every_steps == 0 and val_mrs:
                        try:
                            mr = random.choice(val_mrs)
                            hyp = generate_for_mrs(model, tokenizer, [mr], device, num_beams=min(5, args.num_beams), max_new_tokens=min(60, args.gen_max_new_tokens))[0]
                            print(f"[sanity] MR: {mr}\n[snty ] H:  {hyp}\n")
                            log_jsonl({"phase": "sanity", "epoch": epoch+1, "step": global_step, "mr": mr, "hyp": hyp})
                        except Exception as e:
                            print(f"[sanity] failed: {e}")

                    # Validation
                    if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                        val_ppl = eval_perplexity(model, val_loader, device)
                        val_loss = min(20.0, math.log(max(val_ppl, 1e-8)))
                        print(f"[val] step {global_step} - ppl {val_ppl:.2f} (loss≈{val_loss:.4f})")
                        log_jsonl({"phase": "val", "epoch": epoch+1, "step": global_step, "loss": float(val_loss), "ppl": float(val_ppl)})

                    # Checkpoint save
                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        save_checkpoint(model, tokenizer, optimizer, scheduler, scaler, args.output_dir, global_step, epoch, base_model_name)

            # End of epoch checkpoint
            try:
                pbar.close()
            except Exception:
                pass
            save_checkpoint(model, tokenizer, optimizer, scheduler, scaler, args.output_dir, global_step, epoch, base_model_name)
            log_jsonl({"phase": "epoch_end", "epoch": epoch+1, "step": global_step})

    finally:
        try:
            mfile.close()
        except Exception:
            pass

    # -------------------------
    # Final eval on test set
    # -------------------------
    print("[eval] decoding test...")
    # Unique MRs preserving order
    seen = set()
    test_mrs = []
    for ex in test_split:
        mr, _ = get_mr_and_refs(ex)
        if mr not in seen:
            seen.add(mr)
            test_mrs.append(mr)

    hyps = generate_for_mrs(model, tokenizer, test_mrs, device, num_beams=args.num_beams, max_new_tokens=args.gen_max_new_tokens, no_repeat_ngram_size=args.no_repeat_ngram_size)
    refs_grouped = test_refs_grouped
    refs_list = [refs_grouped.get(mr, [""]) for mr in test_mrs]

    print("[eval] computing BLEU/METEOR/ROUGE-L ...")
    bleu = compute_bleu_multi(hyps, refs_list)
    meteor = compute_meteor_multi(hyps, refs_list)
    rouge_l = compute_rouge_l_multi(hyps, refs_list)
    scores = {"BLEU": bleu, "METEOR": meteor, "ROUGE_L": rouge_l}
    print(json.dumps(scores, indent=2))

    # Save generations and scores
    gens_fp = os.path.join(args.output_dir, "test_generations.jsonl")
    with open(gens_fp, "w", encoding="utf-8") as f:
        for mr, h in zip(test_mrs, hyps):
            f.write(json.dumps({"mr": mr, "hyp": h}, ensure_ascii=False) + "\n")
    with open(os.path.join(args.output_dir, "test_scores.json"), "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    print(f"[done] wrote {gens_fp} and test_scores.json")


if __name__ == "__main__":
    main()