#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train GPT-2 or GPT-2-medium on the E2E NLG dataset with:
- Mixed precision (FP16)
- PEFT (LoRA with DoRA)
- Linear LR decay scheduler with warmup
- Robust checkpointing (adapters, optimizer, scheduler, GradScaler, metadata)
- Periodic validation and early stopping to prevent overfitting
- Beam search decoding (10 beams)
- Evaluation with BLEU, NIST, METEOR, ROUGE-L, CIDEr using a pure-Python path:
  * Try to import/install the official E2E pure-Python metrics
  * Fallback to pure-Python implementations (BLEU, NIST, ROUGE-L, METEOR via NLTK, CIDEr)

Run:
  python train_e2e_gpt2.py --model_name gpt2 --output_dir ./outputs

Install deps (example):
  pip install torch transformers peft pandas numpy nltk requests

"""

import os
import re
import csv
import io
import sys
import gc
import math
import json
import time
import copy
import glob
import types
import random
import shutil
import string
import zipfile
import logging
import argparse
import itertools
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import requests

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    set_seed as hf_set_seed,
)

# PEFT: DoRA via LoRA config with use_dora=True
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
    PeftConfig,
)

import nltk
from nltk.translate import bleu_score
from nltk.translate.nist_score import sentence_nist
from nltk.translate.meteor_score import meteor_score


# -------------------------
# Utilities and constants
# -------------------------

E2E_DEFAULT_URLS = {
    "train": "https://raw.githubusercontent.com/tuetschek/e2e-dataset/master/e2e-dataset/trainset.csv",
    "dev":   "https://raw.githubusercontent.com/tuetschek/e2e-dataset/master/e2e-dataset/devset.csv",
    "test":  "https://raw.githubusercontent.com/tuetschek/e2e-dataset/master/e2e-dataset/testset.csv",
}

# Tokenizer-safe whitespace normalization
_WS_RE = re.compile(r"\s+")

def normalize_ws(s: str) -> str:
    return _WS_RE.sub(" ", s.strip())

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)

def ensure_nltk_resources():
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4', quiet=True)
    # For tokenizers (optional, but harmless)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)

def download_file(url: str, dest_path: str, timeout: int = 60):
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(resp.content)


# ---------------------------------
# Data handling for E2E CSV format
# ---------------------------------

def auto_find_csv(data_dir: str, split_keyword: str) -> Optional[str]:
    """
    Try to find a CSV file in data_dir whose name contains split_keyword (e.g., 'train', 'dev', 'test').
    """
    candidates = glob.glob(os.path.join(data_dir, "*.csv"))
    for c in candidates:
        name = os.path.basename(c).lower()
        if split_keyword in name:
            return c
    return None

def load_e2e_csvs(data_dir: Optional[str], work_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train/dev/test CSVs. If data_dir is None, download from default URLs into work_dir.
    The function expects columns: 'mr' and 'ref' for train/dev. Test may or may not include 'ref'.
    """
    safe_mkdir(work_dir)
    if data_dir and os.path.isdir(data_dir):
        train_fp = auto_find_csv(data_dir, "train")
        dev_fp = auto_find_csv(data_dir, "dev")
        test_fp = auto_find_csv(data_dir, "test")
        if not train_fp or not dev_fp or not test_fp:
            raise FileNotFoundError("Could not locate train/dev/test CSV files in provided data_dir.")
        train_df = pd.read_csv(train_fp)
        dev_df = pd.read_csv(dev_fp)
        test_df = pd.read_csv(test_fp)
    else:
        # Download
        train_fp = os.path.join(work_dir, "trainset.csv")
        dev_fp   = os.path.join(work_dir, "devset.csv")
        test_fp  = os.path.join(work_dir, "testset.csv")
        if not os.path.exists(train_fp):
            download_file(E2E_DEFAULT_URLS["train"], train_fp)
        if not os.path.exists(dev_fp):
            download_file(E2E_DEFAULT_URLS["dev"], dev_fp)
        if not os.path.exists(test_fp):
            download_file(E2E_DEFAULT_URLS["test"], test_fp)
        train_df = pd.read_csv(train_fp)
        dev_df   = pd.read_csv(dev_fp)
        test_df  = pd.read_csv(test_fp)

    # Normalize columns
    req_cols = ["mr", "ref"]
    for df, name in [(train_df, "train"), (dev_df, "dev")]:
        if not all(c in df.columns for c in req_cols):
            raise ValueError(f"{name} CSV must contain columns {req_cols}. Found: {list(df.columns)}")

    # Test may lack refs historically; we try to proceed if present, fallback to dev for eval if absent
    if "mr" not in test_df.columns:
        raise ValueError("test CSV must contain 'mr' column.")
    return train_df, dev_df, test_df

def group_references_by_mr(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Build a mapping from MR string to list of references.
    """
    grouped: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        mr = normalize_ws(str(row["mr"]))
        ref = normalize_ws(str(row["ref"])) if "ref" in row and not (pd.isna(row["ref"])) else None
        if ref is None:
            # skip rows without ref
            continue
        grouped.setdefault(mr, []).append(ref)
    return grouped

def build_training_pairs(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    Create (mr, ref) pairs for training. Each row is one training sample.
    """
    pairs: List[Tuple[str, str]] = []
    for _, row in df.iterrows():
        mr = normalize_ws(str(row["mr"]))
        ref = normalize_ws(str(row["ref"]))
        pairs.append((mr, ref))
    return pairs


# ---------------------------
# Dataset and Collator
# ---------------------------

def build_prompt_from_mr(mr: str, delimiter: str = " =>") -> str:
    return f"{mr}{delimiter}"

class E2ETrainDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        tokenizer,
        max_source_len: int,
        max_target_len: int,
        delimiter: str = " =>",
    ):
        self.examples = pairs
        self.tok = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.delimiter = delimiter

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        mr, ref = self.examples[idx]
        prompt = build_prompt_from_mr(mr, self.delimiter)
        # Tokenize separately to know prompt length
        prompt_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tok(ref, add_special_tokens=False)["input_ids"]
        # Truncate prompt and target while preserving prompt+target within model max length
        prompt_ids = prompt_ids[: self.max_source_len]
        target_ids = target_ids[: self.max_target_len]
        input_ids = prompt_ids + target_ids + [self.tok.eos_token_id]
        attention_mask = [1] * len(input_ids)

        labels = [-100] * len(prompt_ids) + target_ids + [self.tok.eos_token_id]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt_len": len(prompt_ids),
        }

@dataclass
class PadCollator:
    pad_token_id: int

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attention_mask, labels = [], [], []
        for x in batch:
            pad_len = max_len - len(x["input_ids"])
            input_ids.append(
                torch.cat([x["input_ids"], torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
            )
            attention_mask.append(
                torch.cat([x["attention_mask"], torch.zeros((pad_len,), dtype=torch.long)])
            )
            # Pad labels with -100
            labels.append(
                torch.cat([x["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
            )
        return {
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
            "labels": torch.stack(labels, dim=0),
        }


# ------------------------------------
# PEFT: LoRA with DoRA configuration
# ------------------------------------

def build_lora_dora_config() -> LoraConfig:
    # Target GPT-2 Conv1D modules commonly adapted: attention and MLP projections
    # "c_attn", "c_proj", "c_fc" are module name substrings in GPT-2 blocks.
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj", "c_fc"],
        use_dora=True,
        bias="lora_only",
    )


# ------------------------------------
# Checkpointing (robust, resumable)
# ------------------------------------

def save_checkpoint(
    peft_model: PeftModel,
    tokenizer,
    optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    output_dir: str,
    step: int,
    epoch: int,
    best_val_loss: float,
    patience_count: int,
    base_model_name: str,
):
    ckpt_dir = os.path.join(output_dir, f"checkpoint-step{step:08d}")
    safe_mkdir(ckpt_dir)

    # Save PEFT adapters (contains DoRA weights/config)
    peft_model.save_pretrained(ckpt_dir)
    # Save tokenizer for reproducibility
    tokenizer.save_pretrained(ckpt_dir)

    # Save training state
    training_state = {
        "step": step,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "patience_count": patience_count,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "base_model_name": base_model_name,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().tolist(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }
    torch.save(training_state, os.path.join(ckpt_dir, "training_state.pt"))
    return ckpt_dir

def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not os.path.isdir(output_dir):
        return None
    ckpts = sorted(glob.glob(os.path.join(output_dir, "checkpoint-step*")))
    return ckpts[-1] if ckpts else None

def load_checkpoint(
    base_model_name: str,
    device: torch.device,
    resume_path: str,
):
    """
    Load a base model, then load PEFT adapters from resume_path, plus optimizer/scheduler/scaler state.
    Returns: peft_model, tokenizer, training_state
    """
    tokenizer = AutoTokenizer.from_pretrained(resume_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(base_model_name, config=config)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False  # important for training with gradient checkpointing

    peft_model = PeftModel.from_pretrained(model, resume_path)
    peft_model.to(device)

    # Load training state
    state_fp = os.path.join(resume_path, "training_state.pt")
    if not os.path.exists(state_fp):
        raise FileNotFoundError(f"training_state.pt not found in {resume_path}")
    training_state = torch.load(state_fp, map_location="cpu")

    return peft_model, tokenizer, training_state


# ------------------------------------
# Official E2E metrics (pure Python)
# ------------------------------------

def try_import_official_e2e_metrics():
    """
    Try to import official E2E metrics (pure Python) from the public repository.
    We attempt:
      - import e2e_metrics if already installed
      - pip install directly from GitHub (pure Python) and import
    Returns a callable evaluate_fn(references: List[List[str]], hypotheses: List[str]) -> Dict[str, float]
    or None if not available.
    """
    # 1) Try a pre-installed package-style import
    for modname in ["e2e_metrics", "e2e_metrics.metrics", "metrics"]:
        try:
            mod = __import__(modname, fromlist=['*'])
            # heuristic: look for a function that returns dict of metrics
            if hasattr(mod, "evaluate"):
                return getattr(mod, "evaluate")
            if hasattr(mod, "calc_scores"):
                return getattr(mod, "calc_scores")
        except Exception:
            pass

    # 2) Attempt to pip install from GitHub in pure Python
    try:
        import subprocess, sys as _sys
        subprocess.check_call([_sys.executable, "-m", "pip", "install", "--quiet",
                               "git+https://github.com/tuetschek/e2e-metrics"])
        # retry import
        for modname in ["e2e_metrics", "e2e_metrics.metrics", "metrics"]:
            try:
                mod = __import__(modname, fromlist=['*'])
                if hasattr(mod, "evaluate"):
                    return getattr(mod, "evaluate")
                if hasattr(mod, "calc_scores"):
                    return getattr(mod, "calc_scores")
            except Exception:
                pass
    except Exception:
        pass

    return None


# ------------------------------------
# Fallback pure-Python metrics
# ------------------------------------

def tokenize_for_metrics(s: str) -> List[str]:
    # Simple tokenizer: lowercase, split on whitespace and punctuation boundaries
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return [t for t in s.split() if t]

def corpus_bleu_nltk(references: List[List[str]], hypotheses: List[str], max_n: int = 4) -> float:
    refs_tok = [[tokenize_for_metrics(r) for r in refs] for refs in references]
    hyps_tok = [tokenize_for_metrics(h) for h in hypotheses]
    weights = tuple([1.0/max_n]*max_n)
    # Smoothing method
    smoothie = bleu_score.SmoothingFunction().method1
    return bleu_score.corpus_bleu(refs_tok, hyps_tok, weights=weights, smoothing_function=smoothie) * 100.0

def corpus_nist_nltk(references: List[List[str]], hypotheses: List[str], n: int = 5) -> float:
    """
    Average sentence-level NIST score (scaled to 100)
    """
    refs_tok = [[tokenize_for_metrics(r) for r in refs] for refs in references]
    hyps_tok = [tokenize_for_metrics(h) for h in hypotheses]
    scores = []
    for refs, hyp in zip(refs_tok, hyps_tok):
        try:
            scores.append(sentence_nist(refs, hyp, n=n))
        except ZeroDivisionError:
            scores.append(0.0)
        except ValueError:
            scores.append(0.0)
    return float(np.mean(scores) * 100.0)

def rouge_l_score(hyp_tokens: List[str], ref_tokens: List[str]) -> Tuple[float, float, float]:
    """
    ROUGE-L F-measure computation via LCS.
    Returns (precision, recall, f1)
    """
    # LCS length
    m, n = len(hyp_tokens), len(ref_tokens)
    if m == 0 or n == 0:
        return 0.0, 0.0, 0.0
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if hyp_tokens[i] == ref_tokens[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    lcs = dp[m][n]
    prec = lcs / max(m, 1)
    rec = lcs / max(n, 1)
    if prec + rec == 0:
        f1 = 0.0
    else:
        beta = (1.2)**2  # common in ROUGE-L F-measure
        f1 = ((1 + beta) * prec * rec) / (rec + beta * prec)
    return prec, rec, f1

def corpus_rouge_l(references: List[List[str]], hypotheses: List[str]) -> float:
    scores = []
    for refs, hyp in zip(references, hypotheses):
        hyp_tok = tokenize_for_metrics(hyp)
        best_f1 = 0.0
        for r in refs:
            ref_tok = tokenize_for_metrics(r)
            _, _, f1 = rouge_l_score(hyp_tok, ref_tok)
            if f1 > best_f1:
                best_f1 = f1
        scores.append(best_f1)
    return float(np.mean(scores) * 100.0)

def corpus_meteor_nltk(references: List[List[str]], hypotheses: List[str]) -> float:
    ensure_nltk_resources()
    scores = []
    for refs, hyp in zip(references, hypotheses):
        # NLTK's meteor_score expects list of reference strings and one hypothesis string
        try:
            scores.append(meteor_score(refs, hyp))
        except Exception:
            scores.append(0.0)
    return float(np.mean(scores) * 100.0)

def extract_ngrams(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    counts: Dict[Tuple[str, ...], int] = {}
    if len(tokens) < n:
        return counts
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i:i+n])
        counts[ng] = counts.get(ng, 0) + 1
    return counts

def compute_cider(
    references: List[List[str]],
    hypotheses: List[str],
    n: int = 4,
    sigma: float = 6.0
) -> float:
    """
    CIDEr-D-like implementation.
    Steps:
      - Build DF over all reference sentences.
      - For each sentence, build TF vectors for n=1..4 grams.
      - Compute idf = log((N + eps) / (df + eps)).
      - Compute cosine similarity between hyp and mean(refs) vectors across n, apply Gaussian length penalty, average, then scale by 10.
    """
    eps = 1e-12
    # Build DF across all refs
    all_ref_tokens = [tokenize_for_metrics(r) for refs in references for r in refs]
    df: List[Dict[Tuple[str, ...], int]] = [dict() for _ in range(n)]
    for r_tokens in all_ref_tokens:
        for k in range(1, n+1):
            seen = set(extract_ngrams(r_tokens, k).keys())
            for ng in seen:
                df[k-1][ng] = df[k-1].get(ng, 0) + 1
    N_docs = len(all_ref_tokens) + eps

    def tf_vec(tokens: List[str], k: int) -> Dict[Tuple[str, ...], float]:
        counts = extract_ngrams(tokens, k)
        total = sum(counts.values()) + eps
        return {ng: c / total for ng, c in counts.items()}

    def idf(ng: Tuple[str, ...], k: int) -> float:
        return math.log((N_docs) / (df[k-1].get(ng, 0) + eps))

    def cider_for_pair(h_tokens: List[str], refs_tokens: List[List[str]]) -> float:
        scores_n = []
        for k in range(1, n+1):
            h_tf = tf_vec(h_tokens, k)
            ref_tfs = [tf_vec(r, k) for r in refs_tokens]
            # Average ref tf
            # Build union keys
            keys = set(h_tf.keys())
            for tf in ref_tfs:
                keys |= set(tf.keys())
            if not keys:
                scores_n.append(0.0)
                continue
            # Weighted vectors with idf
            h_vec = []
            r_vec = []
            for ng in keys:
                w = idf(ng, k)
                h_val = h_tf.get(ng, 0.0) * w
                r_val = sum(tf.get(ng, 0.0) for tf in ref_tfs) / max(len(ref_tfs), 1) * w
                h_vec.append(h_val)
                r_vec.append(r_val)
            # Cosine similarity
            h_norm = math.sqrt(sum(v*v for v in h_vec)) + eps
            r_norm = math.sqrt(sum(v*v for v in r_vec)) + eps
            dot = sum(hv*rv for hv, rv in zip(h_vec, r_vec))
            cos = dot / (h_norm * r_norm)
            scores_n.append(cos)
        # Gaussian length penalty
        ref_lens = [len(r) for r in refs_tokens if len(r) > 0]
        if len(ref_lens) == 0:
            gp = 1.0
        else:
            ref_len = np.mean(ref_lens)
            diff = len(h_tokens) - ref_len
            gp = math.exp(-(diff*diff) / (2 * sigma * sigma))
        return float(np.mean(scores_n) * gp * 10.0)

    scores = []
    for refs, hyp in zip(references, hypotheses):
        h_tokens = tokenize_for_metrics(hyp)
        refs_tokens = [tokenize_for_metrics(r) for r in refs]
        scores.append(cider_for_pair(h_tokens, refs_tokens))
    return float(np.mean(scores))


def evaluate_with_metrics(
    grouped_refs: Dict[str, List[str]],
    mr_list: List[str],
    generations: List[str],
) -> Dict[str, float]:
    """
    Evaluate with official E2E metrics if available (pure Python), otherwise fallback to in-script metrics.
    Input:
      - grouped_refs: dict MR -> list of refs
      - mr_list: list of MRs corresponding to generated hypotheses
      - generations: list of strings (hypotheses)
    Output: dict with BLEU, NIST, METEOR, ROUGE_L, CIDEr
    """
    # Align references for the given MR order
    references = [grouped_refs[mr] if mr in grouped_refs else [""] for mr in mr_list]
    hypotheses = generations

    # Try official evaluator first
    evaluate_fn = try_import_official_e2e_metrics()
    if evaluate_fn is not None:
        try:
            scores = evaluate_fn(references=references, hypotheses=hypotheses)
            # Normalize keys if needed
            norm_scores = {}
            for k, v in scores.items():
                key = k.upper().replace("-", "_").replace(" ", "_")
                norm_scores[key] = float(v)
            # Ensure all requested keys exist; if not, compute fallback for missing ones
            required = ["BLEU", "NIST", "METEOR", "ROUGE_L", "CIDEr", "CIDER"]
            have = set(norm_scores.keys())
            # Canonical CIDEr key
            if "CIDER" in have and "CIDEr" not in have:
                norm_scores["CIDEr"] = norm_scores["CIDER"]
            # Add missing via fallback if necessary
            if "BLEU" not in have:
                norm_scores["BLEU"] = corpus_bleu_nltk(references, hypotheses)
            if "NIST" not in have:
                norm_scores["NIST"] = corpus_nist_nltk(references, hypotheses)
            if "METEOR" not in have:
                norm_scores["METEOR"] = corpus_meteor_nltk(references, hypotheses)
            if "ROUGE_L" not in have:
                norm_scores["ROUGE_L"] = corpus_rouge_l(references, hypotheses)
            if "CIDEr" not in have:
                norm_scores["CIDEr"] = compute_cider(references, hypotheses)
            return {
                "BLEU": float(norm_scores["BLEU"]),
                "NIST": float(norm_scores["NIST"]),
                "METEOR": float(norm_scores["METEOR"]),
                "ROUGE_L": float(norm_scores["ROUGE_L"]),
                "CIDEr": float(norm_scores["CIDEr"]),
            }
        except Exception:
            # Fall back to local
            pass

    # Fallback local implementation
    return {
        "BLEU": corpus_bleu_nltk(references, hypotheses),
        "NIST": corpus_nist_nltk(references, hypotheses),
        "METEOR": corpus_meteor_nltk(references, hypotheses),
        "ROUGE_L": corpus_rouge_l(references, hypotheses),
        "CIDEr": compute_cider(references, hypotheses),
    }


# ---------------------------
# Training and evaluation
# ---------------------------

def evaluate_perplexity(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss.detach().float()
            losses.append(loss.item())
    model.train()
    mean_loss = float(np.mean(losses)) if losses else float("inf")
    ppl = math.exp(min(20.0, mean_loss))
    return ppl

def generate_for_eval(
    model: nn.Module,
    tokenizer,
    mrs: List[str],
    device: torch.device,
    num_beams: int = 10,
    max_new_tokens: int = 120,
    delimiter: str = " =>",
    no_repeat_ngram_size: int = 3,
) -> List[str]:
    model.eval()
    generations = []
    for mr in mrs:
        prompt = build_prompt_from_mr(mr, delimiter)
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        prompt_len = input_ids.shape[1]
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                early_stopping=True,
                do_sample=False,
                no_repeat_ngram_size=no_repeat_ngram_size,
                length_penalty=1.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        # Take only the newly generated tokens after the prompt
        new_tokens = gen_ids[0][prompt_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        # Basic cleanup
        text = normalize_ws(text)
        generations.append(text)
    model.train()
    return generations


def main():
    parser = argparse.ArgumentParser()
    # Model/tokenizer
    parser.add_argument("--model_name", type=str, default="gpt2", choices=["gpt2", "gpt2-medium"], help="Pretrained GPT-2 variant.")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Where to save checkpoints and final artifacts.")
    parser.add_argument("--data_dir", type=str, default="", help="Directory containing E2E CSVs; if empty, auto-download.")
    # Data parameters
    parser.add_argument("--max_source_len", type=int, default=128)
    parser.add_argument("--max_target_len", type=int, default=128)
    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience on validation loss.")
    parser.add_argument("--resume_from", type=str, default="", help="Path to a checkpoint directory to resume from.")
    parser.add_argument("--sample_every_steps", type=int, default=100, help="Run a quick sanity generation every N optimizer steps.")
    # Generation
    parser.add_argument("--num_beams", type=int, default=10)
    parser.add_argument("--gen_max_new_tokens", type=int, default=120)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    # Mixed precision
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mixed-precision training.")
    args = parser.parse_args()

    safe_mkdir(args.output_dir)
    set_all_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    work_dir = os.path.join(args.output_dir, "data_cache")
    train_df, dev_df, test_df = load_e2e_csvs(args.data_dir if args.data_dir else None, work_dir)

    # Build mappings and datasets
    train_pairs = build_training_pairs(train_df)
    dev_pairs = build_training_pairs(dev_df)

    # References grouped for validation/test
    dev_refs_grouped = group_references_by_mr(dev_df)
    test_has_refs = "ref" in test_df.columns and not test_df["ref"].isna().all()
    test_refs_grouped = group_references_by_mr(test_df) if test_has_refs else None

    # Precompute a list of dev MRs for quick sanity generations
    dev_mr_list = list(dev_refs_grouped.keys())

    # Tokenizer and model
    if args.resume_from:
        # Resume path contains tokenizer + adapters; base model determined from state
        # Need training state to know base model
        _, _, training_state = load_checkpoint(args.model_name, device, args.resume_from)
        base_model_name = training_state.get("base_model_name", args.model_name)
        peft_model, tokenizer, _ = load_checkpoint(base_model_name, device, args.resume_from)
        model = peft_model
        print(f"Resumed PEFT model from {args.resume_from} (base: {base_model_name})")
    else:
        base_model_name = args.model_name
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        config = AutoConfig.from_pretrained(base_model_name)
        model = AutoModelForCausalLM.from_pretrained(base_model_name, config=config)
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False  # important for training with gradient checkpointing and PEFT

        # Apply PEFT (LoRA with DoRA)
        lora_config = build_lora_dora_config()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        model.to(device)

    # Dataloaders
    train_dataset = E2ETrainDataset(
        train_pairs, tokenizer, args.max_source_len, args.max_target_len
    )
    dev_dataset = E2ETrainDataset(
        dev_pairs, tokenizer, args.max_source_len, args.max_target_len
    )

    collator = PadCollator(pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collator,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collator,
    )

    # Optimizer and scheduler
    # Only optimize adapter params (PEFT wraps model such that .parameters() returns trainables)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_training_steps = (len(train_loader) // max(1, args.gradient_accumulation_steps)) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_training_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16 and device.type == "cuda"))

    # Resume optimizer/scheduler/scaler if requested
    global_step = 0
    start_epoch = 0
    best_val_loss = float("inf")
    patience_count = 0

    if args.resume_from:
        _, _, training_state = load_checkpoint(args.model_name, device, args.resume_from)
        if training_state.get("optimizer"):
            optimizer.load_state_dict(training_state["optimizer"])
        if training_state.get("scheduler"):
            scheduler.load_state_dict(training_state["scheduler"])
        if scaler and training_state.get("scaler"):
            scaler.load_state_dict(training_state["scaler"])
        global_step = training_state.get("step", 0)
        start_epoch = training_state.get("epoch", 0)
        best_val_loss = training_state.get("best_val_loss", float("inf"))
        patience_count = training_state.get("patience_count", 0)
        print(f"Resumed training state: step={global_step}, epoch={start_epoch}, best_val_loss={best_val_loss}")

    # Training loop
    model.train()
    print("Starting training...")
    running_loss = 0.0

    for epoch in range(start_epoch, args.num_epochs):
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(args.fp16 and device.type == "cuda")):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / max(1, args.gradient_accumulation_steps)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

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

                running_loss += loss.item() * max(1, args.gradient_accumulation_steps)

                # Logging (loss and perplexity). Set --logging_steps 1 to log every step.
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    avg_loss = running_loss / args.logging_steps
                    ppl = math.exp(min(20.0, avg_loss))
                    print(f"[epoch {epoch+1}/{args.num_epochs}] step {global_step} - loss: {avg_loss:.4f} - ppl: {ppl:.2f}")
                    running_loss = 0.0

                # Periodic validation to monitor overfitting
                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    val_ppl = evaluate_perplexity(model, dev_loader, device)
                    # Convert PPL to loss approximation
                    val_loss = min(20.0, math.log(max(val_ppl, 1e-8)))
                    print(f"Validation - step {global_step} - ppl: {val_ppl:.2f} (loss≈{val_loss:.4f})")
                    if val_loss + 1e-6 < best_val_loss:
                        best_val_loss = val_loss
                        patience_count = 0
                    else:
                        patience_count += 1
                        print(f"Validation loss did not improve. Patience {patience_count}/{args.patience}")
                        if patience_count >= args.patience:
                            print("Early stopping triggered.")
                            # Save final checkpoint before stopping
                            save_checkpoint(
                                model, tokenizer, optimizer, scheduler, scaler,
                                args.output_dir, global_step, epoch, best_val_loss, patience_count,
                                base_model_name
                            )
                            # Proceed to evaluation and exit
                            raise StopIteration

                # NEW: quick sanity generation every N steps
                if args.sample_every_steps > 0 and global_step % args.sample_every_steps == 0:
                    try:
                        sanity_mr = random.choice(dev_mr_list) if dev_mr_list else None
                        if sanity_mr:
                            model.eval()
                            hyp = generate_for_eval(
                                model,
                                tokenizer,
                                [sanity_mr],
                                device,
                                num_beams=min(5, args.num_beams),
                                max_new_tokens=min(60, args.gen_max_new_tokens),
                                delimiter=" =>",
                                no_repeat_ngram_size=args.no_repeat_ngram_size,
                            )[0]
                            refs = dev_refs_grouped.get(sanity_mr, [])[:3]
                            print(f"[sanity] step {global_step}")
                            print(f"[sanity] MR  : {sanity_mr}")
                            for i, r in enumerate(refs, 1):
                                print(f"[sanity] ref{i}: {r}")
                            print(f"[sanity] hyp : {hyp}")
                    except Exception as e:
                        print(f"[sanity] generation failed at step {global_step}: {e}")
                    finally:
                        model.train()

                # Periodic checkpointing
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    ckpt_dir = save_checkpoint(
                        model, tokenizer, optimizer, scheduler, scaler,
                        args.output_dir, global_step, epoch, best_val_loss, patience_count,
                        base_model_name
                    )
                    print(f"Saved checkpoint: {ckpt_dir}")

        # End of epoch: save checkpoint
        ckpt_dir = save_checkpoint(
            model, tokenizer, optimizer, scheduler, scaler,
            args.output_dir, global_step, epoch, best_val_loss, patience_count,
            base_model_name
        )
        print(f"Saved end-of-epoch checkpoint: {ckpt_dir}")

    # End training

    # Evaluation on test set using beam search
    print("Decoding on test set with beam search...")
    # Prepare MR lists for test (unique MRs preserving order)
    test_mrs = []
    seen = set()
    for _, row in test_df.iterrows():
        mr = normalize_ws(str(row["mr"]))
        if mr not in seen:
            seen.add(mr)
            test_mrs.append(mr)

    generations = generate_for_eval(
        model,
        tokenizer,
        test_mrs,
        device,
        num_beams=args.num_beams,
        max_new_tokens=args.gen_max_new_tokens,
        delimiter=" =>",
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    # Determine references set to use for evaluation
    if test_has_refs and test_refs_grouped and len(test_refs_grouped) > 0:
        refs_grouped = test_refs_grouped
        print("Using test references for evaluation.")
    else:
        refs_grouped = dev_refs_grouped
        print("Test references missing; using validation (dev) references for evaluation as a fallback.")

    scores = evaluate_with_metrics(refs_grouped, test_mrs, generations)

    print("Final evaluation scores:")
    print(json.dumps(scores, indent=2, sort_keys=True))

    # Save generations and scores
    gen_fp = os.path.join(args.output_dir, "test_generations.jsonl")
    with open(gen_fp, "w", encoding="utf-8") as f:
        for mr, hyp in zip(test_mrs, generations):
            f.write(json.dumps({"mr": mr, "hyp": hyp}, ensure_ascii=False) + "\n")
    print(f"Wrote generations: {gen_fp}")

    scores_fp = os.path.join(args.output_dir, "test_scores.json")
    with open(scores_fp, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    print(f"Wrote scores: {scores_fp}")


if __name__ == "__main__":
    try:
        main()
    except StopIteration:
        # Early stopping short-circuit; still run final evaluation on latest checkpoint
        # Try to resume from latest checkpoint and evaluate
        parser = argparse.ArgumentParser()
        parser.add_argument("--output_dir", type=str, default="./outputs")
        parser.add_argument("--model_name", type=str, default="gpt2")
        parser.add_argument("--num_beams", type=int, default=10)
        parser.add_argument("--gen_max_new_tokens", type=int, default=120)
        parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
        parser.add_argument("--data_dir", type=str, default="")
        args, _ = parser.parse_known_args()

        latest_ckpt = find_latest_checkpoint(args.output_dir)
        if not latest_ckpt:
            print("No checkpoint found to evaluate after early stopping.")
            sys.exit(0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        peft_model, tokenizer, training_state = load_checkpoint(args.model_name, device, latest_ckpt)
        model = peft_model
        model.eval()

        # Reload data to evaluate
        work_dir = os.path.join(args.output_dir, "data_cache")
        train_df, dev_df, test_df = load_e2e_csvs(args.data_dir if args.data_dir else None, work_dir)
        dev_refs_grouped = group_references_by_mr(dev_df)
        test_has_refs = "ref" in test_df.columns and not test_df["ref"].isna().all()
        test_refs_grouped = group_references_by_mr(test_df) if test_has_refs else None

        # Prepare MR list
        test_mrs = []
        seen = set()
        for _, row in test_df.iterrows():
            mr = normalize_ws(str(row["mr"]))
            if mr not in seen:
                seen.add(mr)
                test_mrs.append(mr)

        generations = generate_for_eval(
            model,
            tokenizer,
            test_mrs,
            device,
            num_beams=args.num_beams,
            max_new_tokens=args.gen_max_new_tokens,
            delimiter=" =>",
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )

        refs_grouped = test_refs_grouped if (test_has_refs and test_refs_grouped) else dev_refs_grouped
        scores = evaluate_with_metrics(refs_grouped, test_mrs, generations)
        print("Final evaluation scores (after early stopping):")
        print(json.dumps(scores, indent=2, sort_keys=True))

        gen_fp = os.path.join(args.output_dir, "test_generations.jsonl")
        with open(gen_fp, "w", encoding="utf-8") as f:
            for mr, hyp in zip(test_mrs, generations):
                f.write(json.dumps({"mr": mr, "hyp": hyp}, ensure_ascii=False) + "\n")
        scores_fp = os.path.join(args.output_dir, "test_scores.json")
        with open(scores_fp, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)
        print(f"Wrote generations: {gen_fp}")
        print(f"Wrote scores: {scores_fp}")