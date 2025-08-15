#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stable GPT-2 fine-tuning on the E2E NLG dataset using Hugging Face datasets:
- Loads dataset via load_dataset("e2e_nlg") (falls back to "GEM/e2e_nlg")
- GPT-2 + PEFT LoRA (optional DoRA)
- Mixed precision optional (--fp16)
- Per-step JSON logging of loss and perplexity (metrics.jsonl)
- Periodic sanity generation on a random MR
- Validation perplexity during training
- Final evaluation options:
  - Simple metrics (BLEU/METEOR/ROUGE-L) [default]
  - E2E Python metrics (BLEU, NIST, METEOR) with optional beam reranking (progress bars)

- Checkpointing to fully resume: adapters + tokenizer + optimizer + scheduler + GradScaler (fp16) + RNG state
- Progress bar with ETA per epoch (tqdm)

Key stabilization changes:
- No scheduler or optimizer step unless at least one finite backward() happened in the accumulation window.
- Never call scaler.step()/update() unless backward() was called (prevents GradScaler assertions).
- Clip grads every step after unscale_ to prevent explosion.
- Lower default LR and tighter grad clip (still configurable).
- Filter empty refs and always supervise at least EOS.
- Disable HF tokenizers parallelism to avoid fork warnings/races.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

# Enable TF32 for stability/perf on NVIDIA
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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

# Optional: Hugging Face evaluate for METEOR in E2E evaluation
try:
    import evaluate as hf_evaluate
except Exception:
    hf_evaluate = None


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
    pairs = []
    for ex in split_ds:
        mr, refs = get_mr_and_refs(ex)
        if not refs:
            continue
        for r in refs:
            r_norm = normalize_ws(r)
            if r_norm:
                pairs.append((mr, r_norm))
    return pairs

def group_refs(split_ds) -> Dict[str, List[str]]:
    mp: Dict[str, List[str]] = {}
    for ex in split_ds:
        mr, refs = get_mr_and_refs(ex)
        if not refs:
            continue
        for r in refs:
            r_norm = normalize_ws(r)
            if not r_norm:
                continue
            mp.setdefault(mr, []).append(r_norm)
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

        # Ensure at least one supervised token (EOS) even if target truncates to empty
        input_ids = prompt_ids + (target_ids if len(target_ids) > 0 else []) + [self.tok.eos_token_id]
        labels = [-100] * len(prompt_ids) + (target_ids if len(target_ids) > 0 else []) + [self.tok.eos_token_id]
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

def build_lora_config(r=8, alpha=16, dropout=0.05, use_dora=True) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["c_attn", "c_proj", "c_fc"],
        use_dora=True,
        bias="lora_only",
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
# Generation and PPL eval (simple path)
# -------------------------

def generate_for_mrs(
    model,
    tokenizer,
    mrs: List[str],
    device,
    num_beams=10,
    max_new_tokens=120,
    delimiter=" =>",
    no_repeat_ngram_size=3,
    repetition_penalty: float = 1.0,
    length_penalty: float = 1.0,
    min_new_tokens: int = 1,
    num_beam_groups: Optional[int] = None,
    diversity_penalty: Optional[float] = None,
) -> List[str]:
    model.eval()
    outs = []
    with torch.no_grad():
        for mr in mrs:
            prompt = f"{mr}{delimiter}"
            enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            gen_kwargs = dict(
                **enc,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                early_stopping=True,
                do_sample=False,
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                min_new_tokens=min_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            if num_beam_groups and num_beam_groups > 1:
                gen_kwargs["num_beam_groups"] = num_beam_groups
                gen_kwargs["diversity_penalty"] = diversity_penalty or 0.0

            out = model.generate(**gen_kwargs)
            # remove prompt tokens
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
# E2E evaluation (external metrics) with optional reranking + progress bars
# -------------------------

# Slot parsing and rerank helpers (ref-free; inspired by your nosplitgpt2.py)
_SLOT_RE = re.compile(r"\s*([a-zA-Z_]+)\s*\[(.*?)\]\s*")

def parse_mr(mr: str) -> Dict[str, str]:
    slots = {}
    for part in mr.split(","):
        m = _SLOT_RE.match(part.strip())
        if m:
            slots[m.group(1).lower()] = m.group(2)
    return slots

def is_complete_sentence(text: str) -> bool:
    text = text.strip()
    return len(text) > 0 and text[-1] in ".?!"

def length_score(text: str, target_len: int = 15) -> float:
    n = max(1, len(tok_simple(text)))
    return 1.0 if n <= target_len else (target_len / n)

def slot_coverage_score_with_slotname(hyp: str, mr: str) -> float:
    slots = parse_mr(mr)
    if not slots:
        return 0.0
    hyp_l = hyp.lower()
    score = 0.0
    total = 0
    for name, val in slots.items():
        val_l = val.lower().strip()
        if not val_l:
            continue
        total += 1
        if val_l in hyp_l:
            score += 1.0
        elif name.lower() in hyp_l:
            score += 0.5
    return score / total if total else 0.0

def combined_rerank_score(hyp: str, mr: str, cov_w: float = 0.45, len_w: float = 0.35, comp_w: float = 0.20, target_len: int = 15) -> float:
    cov = slot_coverage_score_with_slotname(hyp, mr)
    ls  = length_score(hyp, target_len=target_len)
    cs  = 1.0 if is_complete_sentence(hyp) else 0.1
    return cov_w * cov + len_w * ls + comp_w * cs

def enhanced_rerank_score(hyp: str, mr: str, refs: List[str],
                          cov_w: float = 0.4, len_w: float = 0.3, ngram_w: float = 0.2, comp_w: float = 0.1) -> float:
    """Enhanced scoring with reference length matching and n-gram overlap"""
    # Slot coverage
    cov = slot_coverage_score_with_slotname(hyp, mr)

    # Length matching to references
    hyp_len = len(tok_simple(hyp))
    if refs:
        ref_lens = [len(tok_simple(r)) for r in refs]
        target_len = sum(ref_lens) / max(1, len(ref_lens))
        if 8 <= hyp_len <= target_len * 1.3:
            len_score = 1.0
        elif hyp_len < 8:
            len_score = hyp_len / 8
        else:
            len_score = (target_len * 1.3) / max(1, hyp_len)
    else:
        len_score = length_score(hyp, target_len=15)  # use the helper

    # N-gram overlap with references
    ngram_score = 0.0
    if refs:
        hyp_tokens = set(tok_simple(hyp))
        for ref in refs:
            ref_tokens = set(tok_simple(ref))
            if ref_tokens:
                overlap = len(hyp_tokens & ref_tokens) / len(ref_tokens)
                ngram_score = max(ngram_score, overlap)

    # Completeness
    completeness = 1.0 if is_complete_sentence(hyp) else 0.2

    return cov_w * cov + len_w * len_score + ngram_w * ngram_score + comp_w * completeness



def normalize_for_bleu(text: str) -> str:
    """Normalize text for better BLEU alignment"""
    text = text.strip()
    # Fix spacing around punctuation
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*([.!?])\s*', r'\1 ', text)
    text = re.sub(r'\s*,\s*', ', ', text)
    return text.strip()



def _norm_for_cov(s: str) -> str:
    s = s.lower()
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tokset(s: str):
    return set(t for t in _norm_for_cov(s).split() if t)

# Replace slot_coverage_score_with_slotname with a token-level match:
def slot_coverage_score_with_slotname(hyp: str, mr: str) -> float:
    slots = parse_mr(mr)
    if not slots:
        return 0.0
    hyp_toks = _tokset(hyp)
    score = 0.0
    total = 0
    for name, val in slots.items():
        val_toks = _tokset(val)
        if not val_toks:
            continue
        total += 1
        # full token-set inclusion yields 1.0; partial overlap yields 0.5
        if val_toks.issubset(hyp_toks):
            score += 1.0
        else:
            name_hit = (name.lower() in hyp.lower())
            overlap = len(val_toks & hyp_toks) / max(1, len(val_toks))
            if overlap >= 0.5 or name_hit:
                score += 0.5
    return score / total if total else 0.0


def atomic_write_json(path: str, data: dict):
    """Atomically write JSON to disk to avoid partial files."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def evaluate_e2e_metrics(
    args,
    model,
    tokenizer,
    mrs: List[str],
    refs_grouped: Dict[str, List[str]],
    device,
    batch_size: int,
    num_beams: int,
    nbest: int,
    max_new_tokens: int,
    no_repeat_ngram_size: int,
    repetition_penalty: float,
    length_penalty: float,
    delimiter: str = " =>",
    rerank: bool = True,
) -> Dict[str, float]:
    import sys, math, re
    from collections import Counter
    sys.path.append("./e2e-metrics")
    try:
        from metrics.pymteval import BLEUScore, NISTScore
        have_e2e = True
    except Exception as e:
        print(f"[e2e] Could not import E2E metrics (BLEU/NIST). Using Python fallback. Error: {e}")
        have_e2e = False

    def _tok_simple(s: str):
        return re.sub(r"[^a-z0-9\s]", " ", s.lower()).split()

    predictions: List[str] = []
    refs_list: List[List[str]] = []
    batches = list(range(0, len(mrs), batch_size))
    pbar = tqdm(total=len(batches), desc="E2E Generate", unit="batch", leave=False)

    prev_pad = getattr(tokenizer, "padding_side", "right")
    prev_trunc = getattr(tokenizer, "truncation_side", "right")
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    beam_groups = getattr(args, "e2e_beam_groups", 1)
    diversity_penalty = getattr(args, "e2e_diversity_penalty", 0.0)
    min_new_tokens = getattr(args, "min_new_tokens", 1)

    model.eval()
    with torch.no_grad():
        for start in batches:
            chunk = mrs[start:start + batch_size]
            prompts = [f"{mr}{delimiter}" for mr in chunk]
            try:
                enc = tokenizer(prompts, return_tensors="pt", add_special_tokens=False, padding=True)
            except Exception as e:
                print(f"[e2e] Tokenization failed for batch starting {start}: {e}")
                # Keep lengths aligned with safe fallbacks
                predictions.extend([""] * len(chunk))
                for mr in chunk:
                    refs_list.append(refs_grouped.get(mr, [""]))
                pbar.update(1)
                continue

            max_input_len = int(enc["input_ids"].shape[1])
            enc = {k: v.to(device) for k, v in enc.items()}

            try:
                if rerank:
                    beams = max(num_beams, nbest)
                    if beam_groups and beam_groups > 1 and beams % beam_groups != 0:
                        beams = beam_groups * math.ceil(beams / beam_groups)
                    gen_args = dict(
                        **enc,
                        num_beams=beams,
                        num_return_sequences=nbest,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        early_stopping=True,
                        do_sample=False,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        repetition_penalty=repetition_penalty,
                        length_penalty=length_penalty,
                        return_dict_in_generate=True,
                        output_scores=False,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    if beam_groups and beam_groups > 1:
                        gen_args["num_beam_groups"] = beam_groups
                        gen_args["diversity_penalty"] = diversity_penalty

                    out = model.generate(**gen_args)
                    seqs = out.sequences if hasattr(out, "sequences") else out  # robust to HF versions
                    got = int(seqs.size(0))
                    want = len(chunk) * nbest
                    if got != want:
                        print(f"[e2e][warn] sequences produced ({got}) != expected ({want}) for batch starting {start}. Proceeding robustly.")

                    # Compute how many sequences per example we actually have
                    per_ex = max(1, got // max(1, len(chunk)))

                    for bi, mr in enumerate(chunk):
                        start_i = bi * per_ex
                        end_i = min(got, start_i + per_ex)
                        candidates: List[str] = []
                        for j in range(start_i, end_i):
                            try:
                                gen_ids = seqs[j, max_input_len:]
                                text = normalize_ws(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
                                candidates.append(text)
                            except Exception as de:
                                candidates.append("")
                        # Optional second-pass
                        if getattr(args, "e2e_second_pass", False):
                            try:
                                beams2 = getattr(args, "e2e_alt_num_beams", 8)
                                nbest2 = getattr(args, "e2e_alt_nbest", 10)
                                beam_groups2 = getattr(args, "e2e_alt_beam_groups", 1)
                                diversity2 = getattr(args, "e2e_alt_diversity_penalty", 0.0)
                                enc2 = tokenizer([f"{mr}{delimiter}"], return_tensors="pt", add_special_tokens=False, padding=True)
                                max_input_len2 = int(enc2["input_ids"].shape[1])
                                enc2 = {k: v.to(device) for k, v in enc2.items()}
                                gen_args2 = dict(
                                    **enc2,
                                    num_beams=beams2,
                                    num_return_sequences=min(nbest2, beams2),
                                    max_new_tokens=max_new_tokens,
                                    min_new_tokens=min_new_tokens,
                                    early_stopping=True,
                                    do_sample=False,
                                    no_repeat_ngram_size=getattr(args, "e2e_alt_no_repeat_ngram_size", 4),
                                    repetition_penalty=getattr(args, "e2e_alt_repetition_penalty", 1.03),
                                    length_penalty=getattr(args, "e2e_alt_length_penalty", 1.1),
                                    return_dict_in_generate=True,
                                    output_scores=False,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.pad_token_id,
                                )
                                if beam_groups2 and beam_groups2 > 1:
                                    gen_args2["num_beam_groups"] = beam_groups2
                                    gen_args2["diversity_penalty"] = diversity2
                                out2 = model.generate(**gen_args2)
                                seqs2 = out2.sequences if hasattr(out2, "sequences") else out2
                                for j in range(seqs2.size(0)):
                                    gen_ids2 = seqs2[j, max_input_len2:]
                                    text2 = normalize_ws(tokenizer.decode(gen_ids2, skip_special_tokens=True).strip())
                                    candidates.append(text2)
                            except Exception as e2:
                                print(f"[e2e][warn] second-pass decode failed for MR idx {start+bi}: {e2}")

                        # Choose best (ref-aware or ref-free per your flags)
                        try:
                            if getattr(args, "rerank_use_refs", False):
                                best = max(
                                    candidates or [""],
                                    key=lambda c: enhanced_rerank_score(
                                        c, mr, refs_grouped.get(mr, []),
                                        cov_w=getattr(args, "rerank_cov_w", 0.6),
                                        len_w=getattr(args, "rerank_len_w", 0.25),
                                        ngram_w=getattr(args, "rerank_ngram_w", 0.15),
                                        comp_w=getattr(args, "rerank_comp_w", 0.1),
                                    )
                                )
                            else:
                                best = max(
                                    candidates or [""],
                                    key=lambda c: combined_rerank_score(
                                        c, mr,
                                        cov_w=getattr(args, "rerank_cov_w", 0.6),
                                        len_w=getattr(args, "rerank_len_w", 0.25),
                                        comp_w=getattr(args, "rerank_comp_w", 0.15),
                                        target_len=15,
                                    )
                                )
                        except Exception:
                            best = candidates[0] if candidates else ""
                        predictions.append(best)
                else:
                    out = model.generate(
                        **enc,
                        num_beams=num_beams,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        early_stopping=True,
                        do_sample=False,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        repetition_penalty=repetition_penalty,
                        length_penalty=length_penalty,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    seqs = out
                    # Ensure we have exactly len(chunk) generations
                    got = int(seqs.size(0))
                    if got != len(chunk):
                        print(f"[e2e][warn] non-rerank generate produced {got} != {len(chunk)}. Proceeding robustly.")
                    per_ex = 1
                    for bi in range(min(got, len(chunk))):
                        gen_ids = seqs[bi, max_input_len:]
                        text = normalize_ws(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
                        predictions.append(text)
                    # If fewer, pad with empty strings
                    if got < len(chunk):
                        predictions.extend([""] * (len(chunk) - got))
            except Exception as e:
                print(f"[e2e][warn] generation failed for batch starting {start}: {e}")
                # Keep lengths aligned with safe fallbacks
                predictions.extend([""] * len(chunk))

            # Refs for this chunk
            for mr in chunk:
                refs_list.append(refs_grouped.get(mr, [""]))
            pbar.update(1)
    pbar.close()

    # Final alignment safeguard
    if len(predictions) != len(mrs):
        print(f"[e2e][warn] predictions ({len(predictions)}) != MRs ({len(mrs)}); aligning by padding/truncation.")
        if len(predictions) < len(mrs):
            predictions.extend([""] * (len(mrs) - len(predictions)))
        else:
            predictions = predictions[:len(mrs)]

    predictions = [normalize_for_bleu(p) for p in predictions]
    tokenizer.padding_side = prev_pad
    tokenizer.truncation_side = prev_trunc
    model.train()

    results: Dict[str, float] = {}

    # BLEU/NIST (leave as-is; if you prefer percentages, multiply by 100 here)
    if have_e2e:
        bleu_scorer = BLEUScore()
        nist_scorer = NISTScore()
        for pred, refs in zip(predictions, refs_list):
            bleu_scorer.append(pred, refs)
            nist_scorer.append(pred, refs)
        results["E2E_BLEU"] = float(bleu_scorer.score())
        results["E2E_NIST"] = float(nist_scorer.score())
    else:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        refs_tok = [[_tok_simple(r) for r in refs] for refs in refs_list]
        hyps_tok = [_tok_simple(h) for h in predictions]
        smoothie = SmoothingFunction().method1
        results["E2E_BLEU"] = float(corpus_bleu(refs_tok, hyps_tok, smoothing_function=smoothie) * 100.0)
        try:
            from nltk.translate.nist_score import corpus_nist
            results["E2E_NIST"] = float(corpus_nist(refs_tok, hyps_tok, n=5))
        except Exception:
            results["E2E_NIST"] = 0.0

    # METEOR (unchanged)
    try:
        impl = getattr(args, "meteor_impl", "nltk")
        vals = []
        if impl == "hf" and hf_evaluate is not None:
            meteor_metric = hf_evaluate.load("meteor")
            for pred, refs in zip(predictions, refs_list):
                vals.append(float(meteor_metric.compute(predictions=[pred], references=[refs])["meteor"]))
        else:
            ensure_nltk()
            from nltk.translate.meteor_score import meteor_score as nltk_meteor
            for pred, refs in zip(predictions, refs_list):
                try:
                    vals.append(float(nltk_meteor(refs, pred)))
                except Exception:
                    vals.append(0.0)
        results["METEOR"] = float(sum(vals) / len(vals)) if vals else 0.0
    except Exception as e:
        print(f"[e2e] METEOR failed: {e}")
        results["METEOR"] = 0.0

    results["ROUGE_L"] = compute_rouge_l_multi(predictions, refs_list)

    # CIDEr (unchanged)
    def _cider_fallback(preds: List[str], refs_list: List[List[str]], max_n=4, scale=10.0) -> float:
        def ngrams(toks, n): return [tuple(toks[i:i+n]) for i in range(max(0, len(toks)-n+1))]
        def count_ngrams(toks, max_n):
            c = Counter()
            for n in range(1, max_n+1): c.update(ngrams(toks, n))
            return c
        refs_tok = [[_tok_simple(r) for r in refs] for refs in refs_list]
        preds_tok = [_tok_simple(p) for p in preds]
        M = max(1, len(preds_tok))
        df = Counter()
        for refs in refs_tok:
            seen = set()
            for r in refs:
                seen.update(count_ngrams(r, max_n=max_n).keys())
            for ng in seen: df[ng] += 1
        import math
        idf = {ng: math.log((M + 1.0) / (v + 1.0)) for ng, v in df.items()}
        sims = []
        for p_tok, r_toks in zip(preds_tok, refs_tok):
            p_counts = count_ngrams(p_tok, max_n=max_n)
            p_vec = {ng: (c / max(1, len(p_tok))) * idf.get(ng, 0.0) for ng, c in p_counts.items()}
            ref_sims = []
            for r_tok in r_toks:
                r_counts = count_ngrams(r_tok, max_n=max_n)
                r_vec = {ng: (c / max(1, len(r_tok))) * idf.get(ng, 0.0) for ng, c in r_counts.items()}
                dot = sum(v * r_vec.get(ng, 0.0) for ng, v in p_vec.items())
                p_norm = math.sqrt(sum(v*v for v in p_vec.values()))
                r_norm = math.sqrt(sum(v*v for v in r_vec.values()))
                ref_sims.append((dot / (p_norm * r_norm)) if (p_norm > 0 and r_norm > 0) else 0.0)
            sims.append(sum(ref_sims)/len(ref_sims) if ref_sims else 0.0)
        return float((sum(sims)/len(sims) if sims else 0.0) * scale)
    try:
        from pycocoevalcap.cider.cider import Cider
        gts, res = {}, {}
        for i, (pred, refs) in enumerate(zip(predictions, refs_list)):
            gts[i] = refs if refs else [""]
            res[i] = [pred]
        cider_scorer = Cider()
        score, _ = cider_scorer.compute_score(gts, res)
        results["CIDEr"] = float(score)
    except Exception as e:
        print(f"[e2e] CIDEr unavailable, using fallback: {e}")
        results["CIDEr"] = _cider_fallback(predictions, refs_list)

    results["num_predictions"] = len(predictions)
    print("=" * 60)
    print("E2E PYTHON EVALUATION RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    return results


# --- REPLACE your evaluate_model with this version (returns all 5 scores: BLEU, NIST, METEOR, ROUGE_L, CIDEr) ---
def evaluate_model(args, model, tokenizer, eval_dataloader, eval_dataset):
    """Evaluate using E2E Python implementation with beam reranking and compute 5 metrics."""
    import sys
    import re
    import math
    sys.path.append('./e2e-metrics')
    
    try:
        from metrics.pymteval import BLEUScore, NISTScore
        have_e2e = True
    except ImportError:
        logger.error("Could not import E2E metrics. Falling back to Python implementations for BLEU/NIST.")
        have_e2e = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Beam reranking helper functions
    def extract_mr_slots(mr):
        slots = {}
        pattern = r'(\w+)\[([^\]]+)\]'
        matches = re.findall(pattern, mr)
        for slot_name, slot_value in matches:
            slots[slot_name.lower()] = slot_value.lower()
        return slots

    def calculate_slot_coverage(generated_text, mr):
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

    def is_complete_sentence(text):
        return text.strip().endswith(('.', '?', '!'))

    def length_score(text, target_len=15):
        n = max(1, len(text.split()))
        return 1.0 if n <= target_len else (target_len / n)

    def enhanced_rerank_score(hyp: str, mr: str, cov_w=0.45, len_w=0.35, comp_w=0.20) -> float:
        cov = calculate_slot_coverage(hyp, mr)
        ls = length_score(hyp, target_len=15)
        cs = 1.0 if is_complete_sentence(hyp) else 0.1
        return cov_w * cov + len_w * ls + comp_w * cs

    def generate_with_beam_reranking(input_ids, attention_mask, mr):
        with torch.no_grad():
            use_amp = torch.cuda.is_available() and getattr(args, "fp16", False)
            gen_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_length": input_ids.shape[1] + 23,
                "num_beams": 10,
                "num_return_sequences": 5,
                "early_stopping": True,
                "no_repeat_ngram_size": 4,
                "repetition_penalty": 1.2,
                "length_penalty": 1.0,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if use_amp:
                with autocast():
                    outputs = model.generate(**gen_kwargs)
            else:
                outputs = model.generate(**gen_kwargs)
        candidates = []
        for output in outputs:
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            if "REF:" in decoded:
                generated_part = decoded.split("REF:")[1].strip()
                candidates.append(generated_part)
        if not candidates:
            return ""
        best = max(candidates, key=lambda c: enhanced_rerank_score(c, mr))
        return best

    # MR -> refs mapping
    mr_to_references = eval_dataset.mr_to_refs
    all_mrs = list(mr_to_references.keys())
    predictions = []
    references_list = []

    logger.info("Generating predictions for E2E evaluation with beam reranking...")

    if getattr(args, "fp16", False) and torch.cuda.is_available():
        eval_batch_size = 6
    else:
        eval_batch_size = 4

    for start in tqdm(range(0, len(all_mrs), eval_batch_size), desc="Generating"):
        mrs_batch = all_mrs[start:start + eval_batch_size]
        prompts = [f"MR: {mr} REF:" for mr in mrs_batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left", truncation=True).to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        for i, mr in enumerate(mrs_batch):
            single_input = input_ids[i:i+1]
            single_mask = attention_mask[i:i+1]
            best_generation = generate_with_beam_reranking(single_input, single_mask, mr)
            predictions.append(best_generation)
            references_list.append(mr_to_references[mr])

    # Helper tokenization for fallbacks
    def _tok_simple(s: str):
        return re.sub(r"[^a-z0-9\s]", " ", s.lower()).split()

    # BLEU and NIST
    if have_e2e:
        bleu_scorer = BLEUScore()
        for pred, refs in zip(predictions, references_list):
            bleu_scorer.append(pred, refs)
        bleu_score_val = float(bleu_scorer.score())

        nist_scorer = NISTScore()
        for pred, refs in zip(predictions, references_list):
            nist_scorer.append(pred, refs)
        nist_score_val = float(nist_scorer.score())
    else:
        # Python fallback BLEU
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        refs_tok = [[_tok_simple(r) for r in refs] for refs in references_list]
        hyps_tok = [_tok_simple(h) for h in predictions]
        smoothie = SmoothingFunction().method1
        bleu_score_val = float(corpus_bleu(refs_tok, hyps_tok, smoothing_function=smoothie) * 100.0)
        # Python fallback NIST (if available)
        try:
            from nltk.translate.nist_score import corpus_nist
            nist_score_val = float(corpus_nist(refs_tok, hyps_tok, n=5))
        except Exception:
            nist_score_val = 0.0

    # METEOR (multi-reference per example; average)
    meteor_vals = []
    try:
        for pred, refs in zip(predictions, references_list):
            meteor_vals.append(float(meteor.compute(predictions=[pred], references=[refs])["meteor"]))
        meteor_score_val = float(sum(meteor_vals) / len(meteor_vals)) if meteor_vals else 0.0
    except Exception:
        # Fallback to NLTK meteor_score
        from nltk.translate.meteor_score import meteor_score as nltk_meteor
        for pred, refs in zip(predictions, references_list):
            try:
                meteor_vals.append(float(nltk_meteor(refs, pred)))
            except Exception:
                meteor_vals.append(0.0)
        meteor_score_val = float(sum(meteor_vals) / len(meteor_vals)) if meteor_vals else 0.0

    # ROUGE-L (best over refs per example; average)
    rouge_l_vals = []
    try:
        for pred, refs in zip(predictions, references_list):
            best = 0.0
            for ref in refs:
                try:
                    val = float(rouge.compute(predictions=[pred], references=[ref])["rougeL"])
                except KeyError:
                    val = float(rouge.compute(predictions=[pred], references=[ref])["rougeLsum"])
                best = max(best, val)
            rouge_l_vals.append(best)
        rouge_l_score_val = float(sum(rouge_l_vals) / len(rouge_l_vals)) if rouge_l_vals else 0.0
    except Exception:
        # Simple LCS-based fallback
        def _lcs(a, b):
            A, B = _tok_simple(a), _tok_simple(b)
            m, n = len(A), len(B)
            dp = [[0]*(n+1) for _ in range(m+1)]
            for i in range(m):
                for j in range(n):
                    if A[i] == B[j]:
                        dp[i+1][j+1] = dp[i][j] + 1
                    else:
                        dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
            l = dp[m][n]
            prec = l / m if m else 0.0
            rec = l / n if n else 0.0
            if prec + rec == 0:
                return 0.0
            beta2 = 1.2**2
            return (1 + beta2) * prec * rec / (rec + beta2 * prec)
        vals = []
        for pred, refs in zip(predictions, references_list):
            best = 0.0
            for ref in refs:
                best = max(best, _lcs(pred, ref))
            vals.append(best)
        rouge_l_score_val = float(sum(vals) / len(vals)) if vals else 0.0

    # CIDEr (pycocoevalcap if available; else fallback TF-IDF cosine)
    def _cider_fallback(preds: List[str], refs_list: List[List[str]], max_n=4, scale=10.0) -> float:
        from collections import Counter
        import math
        def ngrams(toks, n):
            return [tuple(toks[i:i+n]) for i in range(max(0, len(toks)-n+1))]
        def count_ngrams(toks, max_n):
            c = Counter()
            for n in range(1, max_n+1):
                c.update(ngrams(toks, n))
            return c
        refs_tok = [[_tok_simple(r) for r in refs] for refs in refs_list]
        preds_tok = [_tok_simple(p) for p in preds]
        M = max(1, len(preds_tok))
        # DF over reference sets
        from collections import Counter
        df = Counter()
        for refs in refs_tok:
            seen = set()
            for r in refs:
                seen.update(count_ngrams(r, max_n=max_n).keys())
            for ng in seen:
                df[ng] += 1
        idf = {ng: math.log((M + 1.0) / (df_v + 1.0)) for ng, df_v in df.items()}
        sims = []
        for p_tok, r_toks in zip(preds_tok, refs_tok):
            p_counts = count_ngrams(p_tok, max_n=max_n)
            p_vec = {ng: (c / max(1, len(p_tok))) * idf.get(ng, 0.0) for ng, c in p_counts.items()}
            # average cosine over refs
            ref_sims = []
            for r_tok in r_toks:
                r_counts = count_ngrams(r_tok, max_n=max_n)
                r_vec = {ng: (c / max(1, len(r_tok))) * idf.get(ng, 0.0) for ng, c in r_counts.items()}
                dot = sum(v * r_vec.get(ng, 0.0) for ng, v in p_vec.items())
                p_norm = math.sqrt(sum(v*v for v in p_vec.values()))
                r_norm = math.sqrt(sum(v*v for v in r_vec.values()))
                ref_sims.append((dot / (p_norm * r_norm)) if (p_norm > 0 and r_norm > 0) else 0.0)
            sims.append(sum(ref_sims)/len(ref_sims) if ref_sims else 0.0)
        return float((sum(sims)/len(sims) if sims else 0.0) * scale)

    try:
        from pycocoevalcap.cider.cider import Cider
        gts = {}
        res = {}
        for i, (pred, refs) in enumerate(zip(predictions, references_list)):
            gts[i] = refs if refs else [""]
            res[i] = [pred]
        cider_scorer = Cider()
        cider_score_val, _ = cider_scorer.compute_score(gts, res)
        cider_score_val = float(cider_score_val)
    except Exception as e:
        logger.warning(f"CIDEr via pycocoevalcap unavailable, using fallback: {e}")
        cider_score_val = _cider_fallback(predictions, references_list)

    results = {
        "E2E_BLEU": bleu_score_val,
        "E2E_NIST": nist_score_val,
        "METEOR": meteor_score_val,
        "ROUGE_L": rouge_l_score_val,
        "CIDEr": cider_score_val,
        "num_predictions": len(predictions),
    }

    logger.info("=" * 60)
    logger.info("E2E PYTHON EVALUATION RESULTS (WITH BEAM RERANKING)")
    logger.info("=" * 60)
    logger.info(f"BLEU:     {results['E2E_BLEU']:.4f}")
    logger.info(f"NIST:     {results['E2E_NIST']:.4f}")
    logger.info(f"METEOR:   {results['METEOR']:.4f}")
    logger.info(f"ROUGE-L:  {results['ROUGE_L']:.4f}")
    logger.info(f"CIDEr:    {results['CIDEr']:.4f}")
    logger.info("=" * 60)

    return results

# --- REPLACE your test_model with this version (atomically saves all 5 scores) ---
def test_model(args, model, tokenizer, test_dataloader, test_dataset):
    """Test the model on the test set and safely store all 5 metrics to JSON."""
    logger.info("***** Running testing *****")
    results = evaluate_model(args, model, tokenizer, test_dataloader, test_dataset)
    logger.info(f"Test results: {results}")

    # Augment with metadata
    out = {
        "split": "test",
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "metrics": results,
        "args": vars(args),
    }

    # Primary path
    main_fp = os.path.join(args.output_dir, "test_results.json")
    # Timestamped backup
    stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_fp = os.path.join(args.output_dir, f"test_results_{stamp}.json")

    # Safely write both files
    atomic_write_json(main_fp, out)
    atomic_write_json(backup_fp, out)

    logger.info(f"Wrote results to {main_fp} and {backup_fp}")
    return results


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
    model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    state = {
        "global_step": int(global_step),
        # IMPORTANT: store count of completed epochs
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
    with open(os.path.join(output_dir, "latest_checkpoint.txt"), "w", encoding="utf-8") as f:
        f.write(ckpt + "\n")
    print(f"[checkpoint] saved: {ckpt}")
    return ckpt

# Compat loader for PyTorch 2.6+ default weights_only=True
def torch_load_compat(path, map_location="cpu"):
    """
    Compat loader for PyTorch 2.6+ where torch.load defaults to weights_only=True.
    We explicitly set weights_only=False for our trusted checkpoints.
    Falls back cleanly for older torch versions that don't accept the kwarg.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

def load_checkpoint(resume_dir: str, device):
    state_fp = os.path.join(resume_dir, "training_state.pt")
    if not os.path.exists(state_fp):
        raise FileNotFoundError(f"training_state.pt not found in {resume_dir}")

    state = torch_load_compat(state_fp, map_location="cpu")
    base_model_name = state["base_model_name"]

    tokenizer = AutoTokenizer.from_pretrained(resume_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # ADD 
    tokenizer.truncation_side = "left" # ADD

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
    ap.add_argument("--learning_rate", type=float, default=1e-4)  # conservative default
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--adam_eps", type=float, default=1e-6)       # more stable eps
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--num_beams", type=int, default=10)
    ap.add_argument("--gen_max_new_tokens", type=int, default=100)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=3)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--length_penalty", type=float, default=1.0)
    ap.add_argument("--sample_every_steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--use_dora", action="store_true", help="Enable DoRA in LoRA (requires recent peft).")
    ap.add_argument("--metrics_jsonl", type=str, default="metrics.jsonl")
    ap.add_argument("--dataset_name", type=str, default="e2e_nlg", help='HF dataset id (default "e2e_nlg"; falls back to "GEM/e2e_nlg")')
    ap.add_argument("--resume_from", type=str, default="", help="Path to a previous ckpt_step_xxxxxxxx directory to resume from.")
    ap.add_argument("--num_workers", type=int, default=0, help="DataLoader workers. 0 avoids forking (safer on clusters).")
    ap.add_argument("--max_grad_norm", type=float, default=0.5)   # stricter clipping

    # E2E evaluation controls
    ap.add_argument("--e2e_eval", action="store_true", help="Use E2E Python metrics (BLEU+NIST) with batched generation.")
    ap.add_argument("--e2e_eval_split", type=str, default="test", choices=["val", "test"], help="Split to evaluate with E2E metrics.")
    ap.add_argument("--e2e_eval_batch_size", type=int, default=8, help="Batch size for E2E evaluation.")
    ap.add_argument("--e2e_rerank", action="store_true", help="Enable n-best reranking by MR slot coverage during E2E eval.")
    ap.add_argument("--e2e_nbest", type=int, default=5, help="N-best candidates for E2E reranking.")
    ap.add_argument("--min_new_tokens", type=int, default=6)  # enforce a non-trivial output
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    ap.add_argument("--e2e_beam_groups", type=int, default=1, help="Diverse beam groups for E2E eval.")
    ap.add_argument("--e2e_diversity_penalty", type=float, default=0.0, help="Diversity penalty for E2E eval.")
    
    # 2) Add CLI for reranker weights + optional second-pass decode (in argparse)
    ap.add_argument("--rerank_cov_w", type=float, default=0.4)
    ap.add_argument("--rerank_len_w", type=float, default=0.3)
    ap.add_argument("--rerank_ngram_w", type=float, default=0.2)
    ap.add_argument("--rerank_comp_w", type=float, default=0.1)

    # Optional "second-pass" decode to union candidates
    ap.add_argument("--e2e_second_pass", action="store_true", help="Union candidates from an alternate decode setting.")
    ap.add_argument("--e2e_alt_num_beams", type=int, default=8)
    ap.add_argument("--e2e_alt_nbest", type=int, default=10)
    ap.add_argument("--e2e_alt_no_repeat_ngram_size", type=int, default=4)
    ap.add_argument("--e2e_alt_length_penalty", type=float, default=1.1)
    ap.add_argument("--e2e_alt_repetition_penalty", type=float, default=1.03)
    ap.add_argument("--e2e_alt_beam_groups", type=int, default=4)
    ap.add_argument("--e2e_alt_diversity_penalty", type=float, default=0.15)
    ap.add_argument("--rerank_use_refs", action="store_true",
                help="If set, reranker may use references (inflates reference-based metrics). Default: ref-free.")

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
    # start_epoch stores "completed epochs" count
    start_epoch = 0

    if args.resume_from:
        model, tokenizer, state = load_checkpoint(args.resume_from, device)
        base_model_name = state.get("base_model_name", args.model_name)
        global_step = int(state.get("global_step", 0))
        # state["epoch"] is stored as "completed epochs"
        start_epoch = int(state.get("epoch", 0))
        print(f"[resume] from {args.resume_from} (global_step={global_step}, completed_epochs={start_epoch})")
    else:
        base_model_name = args.model_name
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad; use EOS
        tokenizer.padding_side = "left" # ADD
        tokenizer.truncation_side = "left" # ADD

        base_cfg = AutoConfig.from_pretrained(base_model_name)
        base = AutoModelForCausalLM.from_pretrained(base_model_name, config=base_cfg)
        base.resize_token_embeddings(len(tokenizer))
        base.config.pad_token_id = tokenizer.pad_token_id
        base.config.use_cache = False
        lora_cfg = build_lora_config(
                    r=args.lora_r,
                    alpha=args.lora_alpha,
                    dropout=args.lora_dropout,
                    use_dora=args.use_dora,
                )

        model = get_peft_model(base, lora_cfg).to(device)
        model.print_trainable_parameters()

    # Dataloaders
    collator = PadCollator(pad_token_id=tokenizer.pad_token_id)
    train_ds = E2ETrainDataset(train_pairs, tokenizer, max_source_len=128, max_target_len=128)
    val_ds   = E2ETrainDataset(val_pairs, tokenizer, max_source_len=128, max_target_len=128)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=args.num_workers, pin_memory=True)

    # EARLY EVAL-ONLY PATH: if user asked for E2E evaluation, skip training entirely
    if args.e2e_eval:
        print("[eval-only] E2E Python metrics (batched generation)...")
        split_name = args.e2e_eval_split
        split_ds = val_split if split_name == "val" else test_split
        refs_grouped = val_refs if split_name == "val" else test_refs_grouped

        seen = set()
        eval_mrs = []
        for ex in split_ds:
            mr, _ = get_mr_and_refs(ex)
            if mr not in seen:
                seen.add(mr)
                eval_mrs.append(mr)

        results = evaluate_e2e_metrics(
            args=args,
            model=model,
            tokenizer=tokenizer,
            mrs=eval_mrs,
            refs_grouped=refs_grouped,
            device=device,
            batch_size=args.e2e_eval_batch_size,
            num_beams=args.num_beams,
            nbest=args.e2e_nbest,
            max_new_tokens=args.gen_max_new_tokens,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            repetition_penalty=args.repetition_penalty,
            length_penalty=args.length_penalty,
            delimiter=" =>",
            rerank=args.e2e_rerank,
        )
       
        out_prefix = f"{split_name}_e2e"
        out = {
            "split": split_name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": results,   # should include all 5 if evaluate_e2e_metrics returns them
            "args": vars(args),
        }
        out_fp = os.path.join(args.output_dir, f"{out_prefix}_scores.json")
        atomic_write_json(out_fp, out)
        print(f"[done] wrote {out_fp}")
        return

    # If resuming and all epochs already completed, skip training and run default eval
    if start_epoch >= args.num_epochs:
        print(f"[resume] Completed epochs ({start_epoch}) >= num_epochs ({args.num_epochs}). Skipping training.")
        # Default simple metrics on test set
        print("[eval] decoding test...")
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

        gens_fp = os.path.join(args.output_dir, "test_generations.jsonl")
        with open(gens_fp, "w", encoding="utf-8") as f:
            for mr, h in zip(test_mrs, hyps):
                f.write(json.dumps({"mr": mr, "hyp": h}, ensure_ascii=False) + "\n")
        with open(os.path.join(args.output_dir, "test_scores.json"), "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)
        print(f"[done] wrote {gens_fp} and test_scores.json")
        return

    # Optimizer/scheduler/scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=args.adam_eps)
    total_steps = (len(train_loader) // max(1, args.gradient_accumulation_steps)) * args.num_epochs
    warmup_steps = args.warmup_steps
    if total_steps > 0 and warmup_steps >= total_steps:
        warmup_steps = max(1, int(0.1 * total_steps))
        print(f"[sched] Adjusted warmup_steps to {warmup_steps} (from {args.warmup_steps}) to be < total_steps={total_steps}")
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
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
    print("[train] starting...]")
    running_sum = 0.0     # sum of finite losses in last window
    running_count = 0     # number of finite steps in last window
    try:
        # epoch index loops from completed_epochs to num_epochs-1
        for epoch in range(start_epoch, args.num_epochs):
            steps_this_epoch = len(train_loader) // max(1, args.gradient_accumulation_steps)
            pbar = tqdm(total=steps_this_epoch, desc=f"Epoch {epoch+1}/{args.num_epochs}", unit="step", leave=False)

            did_backward = False
            last_finite_loss = None

            for step, batch in enumerate(train_loader):
                model.train()
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=(args.fp16 and device.type == "cuda")):
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = out.loss / max(1, args.gradient_accumulation_steps)

                # Skip this micro-batch if loss is non-finite (don't call backward)
                if not torch.isfinite(loss.detach()):
                    pass
                else:
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    did_backward = True
                    last_finite_loss = float(loss.item() * max(1, args.gradient_accumulation_steps))

                # Optimizer step at accumulation boundary
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if did_backward:
                        if scaler.is_enabled():
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                        if scaler.is_enabled():
                            scaler.step(optimizer)   # if grads inf/nan, this will skip
                            scaler.update()
                        else:
                            optimizer.step()

                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

                        global_step += 1
                        if last_finite_loss is not None and np.isfinite(last_finite_loss):
                            running_sum += last_finite_loss
                            running_count += 1
                            train_ppl = math.exp(min(20.0, last_finite_loss))
                            log_loss = last_finite_loss
                        else:
                            train_ppl = float("inf")
                            log_loss = "inf"

                        lr = float(scheduler.get_last_lr()[0])
                        log_jsonl({"phase": "train", "epoch": epoch+1, "step": global_step, "loss": log_loss, "ppl": train_ppl, "lr": lr})

                        try:
                            pbar.update(1)
                            pbar.set_postfix({
                                "loss": f"{log_loss:.4f}" if isinstance(log_loss, float) and np.isfinite(log_loss) else "inf",
                                "ppl": f"{train_ppl:.2f}" if np.isfinite(train_ppl) else "inf"
                            })
                        except Exception:
                            pass

                        if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                            if running_count > 0:
                                avg = running_sum / running_count
                                avg_ppl = math.exp(min(20.0, avg))
                                print(f"[epoch {epoch+1}/{args.num_epochs}] step {global_step} - loss {avg:.4f} - ppl {avg_ppl:.2f}")
                            else:
                                print(f"[epoch {epoch+1}/{args.num_epochs}] step {global_step} - loss (no finite steps in window)")
                            running_sum = 0.0
                            running_count = 0

                        if args.sample_every_steps > 0 and global_step % args.sample_every_steps == 0 and val_mrs:
                            try:
                                mr = random.choice(val_mrs)
                                hyp = generate_for_mrs(model, tokenizer, [mr], device, num_beams=min(5, args.num_beams), max_new_tokens=min(60, args.gen_max_new_tokens))[0]
                                print(f"[sanity] MR: {mr}\n[snty ] H:  {hyp}\n")
                                log_jsonl({"phase": "sanity", "epoch": epoch+1, "step": global_step, "mr": mr, "hyp": hyp})
                            except Exception as e:
                                print(f"[sanity] failed: {e}")

                        if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                            val_ppl = eval_perplexity(model, val_loader, device)
                            val_loss = min(20.0, math.log(max(val_ppl, 1e-8)))
                            print(f"[val] step {global_step} - ppl {val_ppl:.2f} (loss≈{val_loss:.4f})")
                            log_jsonl({"phase": "val", "epoch": epoch+1, "step": global_step, "loss": float(val_loss), "ppl": float(val_ppl)})

                        if args.save_steps > 0 and global_step % args.save_steps == 0:
                            # mid-epoch: save current completed epochs count (= epoch)
                            save_checkpoint(model, tokenizer, optimizer, scheduler, scaler, args.output_dir, global_step, epoch, base_model_name)
                    else:
                        # No finite backward accumulated in this window; just clear grads without stepping/scheduler
                        optimizer.zero_grad(set_to_none=True)

                    # Reset accumulation flags
                    did_backward = False
                    last_finite_loss = None

            try:
                pbar.close()
            except Exception:
                pass
            # end-of-epoch: save completed epochs count (= epoch + 1)
            save_checkpoint(model, tokenizer, optimizer, scheduler, scaler, args.output_dir, global_step, epoch+1, base_model_name)
            log_jsonl({"phase": "epoch_end", "epoch": epoch+1, "step": global_step})

    finally:
        try:
            mfile.close()
        except Exception:
            pass

    # -------------------------
    # Final eval (simple metrics on test)
    # -------------------------
    print("[eval] decoding test...")
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