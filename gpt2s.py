#!/usr/bin/env python3
"""
(Updated) Full training + evaluation script for fine-tuning GPT-2 on the E2E NLG dataset
Changes from original:
 - small sanity-check generation after wrapping model with PEFT (prints a generated sample)
 - set model.config.pad_token_id after setting tokenizer.pad_token
 - save tokenizer files and adapter config when checkpointing adapter
 - gentle handling/diagnostic message if pycocoevalcap is unavailable
 - comments to explain the pad-token caveat
"""

# ---------------------------
# Requirements (install before running)
# ---------------------------
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install transformers datasets accelerate peft sentencepiece tqdm
# pip install nltk rouge-score
# pip install pycocoevalcap        # optional but recommended for CIDEr
# pip install git+https://github.com/NVlabs/DoRA.git  # optional if you want NVlabs DoRA implementation
#
# ---------------------------

import os
import math
import json
import time
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)

# PEFT imports
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training

# Mixed precision tools
from torch.cuda.amp import GradScaler, autocast

# Metrics (pure-Python where possible)
import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import sentence_nist
from rouge_score import rouge_scorer

# pycocoevalcap for CIDEr (Python implementation) - optional
try:
    from pycocoevalcap.cider.cider import Cider
    _HAS_PYCOCO = True
except Exception as e:
    print("[warn] pycocoevalcap import failed -- CIDEr will not be available. Install pycocoevalcap for CIDEr. Error:", e)
    _HAS_PYCOCO = False

# Ensure required NLTK downloads
nltk.download('wordnet', quiet=True)   # used by meteor
nltk.download('punkt', quiet=True)     # sentence tokenizer sometimes helpful

# ---------------------------
# Configurable arguments
# ---------------------------
@dataclass
class TRAIN_ARGS:
    model_name: str = "gpt2-medium"       # "gpt2" or "gpt2-medium"
    output_dir: str = "./checkpoints"
    adapter_dir: str = "./checkpoints/peft_adapter"
    dataset_name: str = "tuetschek/e2e_nlg"  # Hugging Face dataset id
    max_length: int = 128
    train_batch_size: int = 4
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    lr: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_steps: int = 200
    save_every_steps: int = 2000
    seed: int = 42
    fp16: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    patience: int = 2   # early stopping patience on validation loss
    num_beams: int = 10
    max_gen_len: int = 80
    use_dora: bool = True   # enable DoRA in PEFT LoraConfig
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None   # set below if None

# set target modules for GPT-2
if TRAIN_ARGS.target_modules is None:
    TRAIN_ARGS.target_modules = ["c_attn", "c_proj"]


# ---------------------------
# Utilities & dataset handling
# ---------------------------
class E2EDataset(Dataset):
    """Wrap HuggingFace E2E dataset into PyTorch Dataset ready for LM training.

    We create sequences like: "<MR>  <SEP>  <REFERENCE>"
    and set labels to -100 for MR portion so LM loss is computed only on the target.
    """
    def __init__(self, hf_dataset, tokenizer: AutoTokenizer, split: str = "train", max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        if isinstance(hf_dataset, dict) and split in hf_dataset:
            ds = hf_dataset[split]
        else:
            ds = hf_dataset

        for ex in ds:
            mr = ex.get("meaning_representation") or ex.get("mr") or ex.get("MR") or ex.get("source")
            ref = ex.get("human_reference") or ex.get("ref") or ex.get("references") or ex.get("reference")
            if isinstance(ref, list):
                if len(ref) == 0:
                    continue
                ref_text = ref[0]
            else:
                ref_text = ref
            mr_text = " ; ".join([part.strip() for part in str(mr).split(",")]) if isinstance(mr, str) else str(mr)
            seq = f"MR: {mr_text} ||| REF: {ref_text}"
            tok = tokenizer(seq, truncation=True, max_length=self.max_length, padding=False)
            input_ids = tok["input_ids"]
            attention_mask = tok["attention_mask"]
            # compute split index where the REF starts so we can mask MR portion in labels
            ref_prefix_tok = tokenizer(" REF: ", add_special_tokens=False)["input_ids"]
            ref_start_idx = 0
            seq_ids = input_ids
            for i in range(len(seq_ids) - len(ref_prefix_tok) + 1):
                if seq_ids[i:i+len(ref_prefix_tok)] == ref_prefix_tok:
                    ref_start_idx = i + len(ref_prefix_tok)
                    break
            labels = input_ids.copy()
            for i in range(0, ref_start_idx):
                labels[i] = -100
            self.examples.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "mr_text": mr_text, "ref_text": ref_text})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def collate_fn(batch: List[Dict], tokenizer: AutoTokenizer):
    input_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in batch]
    labels = [torch.tensor(example["labels"], dtype=torch.long) for example in batch]
    attention_mask = [torch.tensor(example["attention_mask"], dtype=torch.long) for example in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ---------------------------
# Checkpointing helpers
# ---------------------------
def save_checkpoint(state: dict, checkpoint_dir: str, step: int):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
    torch.save(state, path)
    print(f"[checkpoint] saved {path}")


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, scaler=None, device="cuda"):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    print(f"[checkpoint] loading {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    epoch = ckpt.get("epoch", 0)
    step = ckpt.get("step", 0)
    return epoch, step


# ---------------------------
# Metric computations (pure Python)
# ---------------------------
def compute_cider(hypotheses: List[str], references_list: List[List[str]]) -> float:
    if not _HAS_PYCOCO:
        print("[warn] CIDEr unavailable: pycocoevalcap not installed.")
        return 0.0
    gts = {}
    res = {}
    for i, (hyps, refs) in enumerate(zip(hypotheses, references_list)):
        gts[i] = refs
        res[i] = [hyps]
    cider = Cider()
    score, scores = cider.compute_score(gts, res)
    return float(score)


def compute_rouge_l(hypotheses: List[str], references_list: List[List[str]]) -> float:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    f_scores = []
    for hyp, refs in zip(hypotheses, references_list):
        best_f = 0.0
        for r in refs:
            sc = scorer.score(r, hyp)
            f = sc["rougeL"].fmeasure
            if f > best_f:
                best_f = f
        f_scores.append(best_f)
    return float(np.mean(f_scores))


def compute_bleu(hypotheses: List[str], references_list: List[List[str]]) -> float:
    tokenized_refs = [[nltk.word_tokenize(r.lower()) for r in refs] for refs in references_list]
    tokenized_hyps = [nltk.word_tokenize(h.lower()) for h in hypotheses]
    bleu_score = corpus_bleu(tokenized_refs, tokenized_hyps)
    return float(bleu_score)


def compute_meteor(hypotheses: List[str], references_list: List[List[str]]) -> float:
    # Use NLTK's multi-reference meteor by passing refs list directly (preferred).
    scores = []
    for hyp, refs in zip(hypotheses, references_list):
        try:
            sc = meteor_score(refs, hyp)
        except Exception:
            # fallback to best-of per-ref if something goes wrong
            best = 0.0
            for r in refs:
                sc2 = meteor_score([r], hyp)
                if sc2 > best:
                    best = sc2
            sc = best
        scores.append(sc)
    return float(np.mean(scores))


def compute_nist(hypotheses: List[str], references_list: List[List[str]]) -> float:
    scores = []
    for hyp, refs in zip(hypotheses, references_list):
        tok_hyp = nltk.word_tokenize(hyp.lower())
        tok_refs = [nltk.word_tokenize(r.lower()) for r in refs]
        try:
            sc = sentence_nist(tok_refs, tok_hyp, n=5)
        except Exception:
            sc = 0.0
        scores.append(sc)
    return float(np.mean(scores))


def compute_all_metrics(hypotheses: List[str], references_list: List[List[str]]) -> Dict[str, float]:
    cider = compute_cider(hypotheses, references_list)
    rouge_l = compute_rouge_l(hypotheses, references_list)
    bleu = compute_bleu(hypotheses, references_list)
    meteor = compute_meteor(hypotheses, references_list)
    nist = compute_nist(hypotheses, references_list)
    return {"CIDEr": cider, "ROUGE-L": rouge_l, "BLEU": bleu, "METEOR": meteor, "NIST": nist}


# ---------------------------
# Training & evaluation flow
# ---------------------------
def train_and_evaluate(args: TRAIN_ARGS):
    # reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    # load tokenizer and model
    print(f"[init] loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # If pad token is missing, set it. NOTE: setting pad_token == eos_token is common for GPT-2,
    # but it has the caveat that the model may learn not to output the eos token if it's used as pad.
    # We keep pad_token == eos_token to allow batching/padding; generation uses eos_token_id explicitly.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # ensure model.config has pad_token_id set later after loading model

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)
    model.resize_token_embeddings(len(tokenizer))  # in case we set pad token
    # make sure model.config.pad_token_id is set for generation/configs
    model.config.pad_token_id = tokenizer.pad_token_id

    model.to(args.device)

    # Apply PEFT with DoRA via LoraConfig (peft exposes a DoRA toggle)
    print("[peft] preparing PEFT DoRA config")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        enabled=True,
        use_dora=args.use_dora,  # enable DoRA
    )

    # Wrap model with PEFT
    print("[peft] wrapping model with get_peft_model()")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # helpful logging

    # ---------------------------
    # SANITY CHECK: quick generate on a toy MR to ensure tokenizer+model generate pipeline works
    # (This is done *before* training; the adapter is freshly created so generation will be close to base behavior)
    # ---------------------------
    try:
        model.eval()
        sample_mr = "name[Blue Spice], food[Indian], area[riverside], familyFriendly[yes]"
        prefix = f"MR: {' ; '.join([s.strip() for s in sample_mr.split(',')])} ||| REF:"
        tokens = tokenizer(prefix, return_tensors="pt").to(args.device)
        with torch.no_grad():
            # small, fast beam for sanity
            gen = model.generate(
                **tokens,
                max_length=50,
                num_beams=3,
                early_stopping=True,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(gen[0], skip_special_tokens=True)
        if gen_text.startswith(prefix):
            sanity_out = gen_text[len(prefix):].strip()
        else:
            sanity_out = gen_text
        print("[sanity check] sample MR:", sample_mr)
        print("[sanity check] model generated:", sanity_out)
    except Exception as e:
        print("[sanity check] generation failed:", e)
    finally:
        model.train()
    # ---------------------------

    # Prepare datasets
    print(f"[data] loading dataset {args.dataset_name} from HuggingFace datasets")
    raw = load_dataset(args.dataset_name)
    train_ds = E2EDataset(raw, tokenizer, split="train", max_length=args.max_length)
    val_raw = raw["validation"] if "validation" in raw else (raw["dev"] if "dev" in raw else raw.get("test", raw["train"][:200]))
    val_ds = E2EDataset(val_raw, tokenizer, split=None, max_length=args.max_length)
    test_raw = raw["test"] if "test" in raw else val_raw

    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer))

    # optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, (len(train_loader) // args.gradient_accumulation_steps) * args.num_epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    # mixed precision scaler
    scaler = GradScaler(enabled=args.fp16 and (args.device.startswith("cuda")))

    # trainer state
    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0

    # optionally resume from the latest checkpoint
    latest_ckpt = None
    ckpt_dir = args.output_dir
    if os.path.isdir(ckpt_dir):
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.startswith("checkpoint_step_") and f.endswith(".pt")]
        if ckpt_files:
            ckpt_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            latest_ckpt = os.path.join(ckpt_dir, ckpt_files[-1])
    if latest_ckpt:
        try:
            start_epoch, global_step = load_checkpoint(latest_ckpt, model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, device=args.device)
            print(f"[resume] resumed from checkpoint {latest_ckpt} at epoch {start_epoch}, global_step {global_step}")
        except Exception as e:
            print(f"[resume] failed to load checkpoint {latest_ckpt}: {e}")

    model.train()

    for epoch in range(1, args.num_epochs + 1):
        epoch_start = time.time()
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for step, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = batch["labels"].to(args.device)

            with autocast(enabled=args.fp16 and (args.device.startswith("cuda"))):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                train_loss = float(loss.item() * args.gradient_accumulation_steps)
                epoch_loss += train_loss
                ppl = math.exp(train_loss) if train_loss < 20 else float("inf")
                progress.set_postfix({"step": global_step, "train_loss": f"{train_loss:.4f}", "ppl": f"{ppl:.2f}"})

                # periodic checkpointing
                if global_step % args.save_every_steps == 0:
                    state = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "epoch": epoch,
                        "step": global_step,
                    }
                    save_checkpoint(state, args.output_dir, global_step)
                    # save PEFT adapter separately and tokenizer
                    os.makedirs(args.adapter_dir, exist_ok=True)
                    model.save_pretrained(args.adapter_dir)  # adapter config & weights
                    tokenizer.save_pretrained(args.adapter_dir)  # helpful to have tokenizer there too
                    print(f"[peft save] adapter + tokenizer saved to {args.adapter_dir}")

        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / max(1, len(train_loader) // args.gradient_accumulation_steps)
        print(f"[epoch {epoch}] avg_train_loss={avg_epoch_loss:.4f} epoch_time={epoch_time:.1f}s")

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(args.device)
                attention_mask = batch["attention_mask"].to(args.device)
                labels = batch["labels"].to(args.device)
                with autocast(enabled=args.fp16 and (args.device.startswith("cuda"))):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    val_losses.append(float(outputs.loss.item()))
        mean_val_loss = float(np.mean(val_losses)) if len(val_losses) > 0 else float("inf")
        mean_val_ppl = math.exp(mean_val_loss) if mean_val_loss < 20 else float("inf")
        print(f"[validation] val_loss={mean_val_loss:.4f}, val_ppl={mean_val_ppl:.2f}")

        # early stopping logic based on validation loss
        if mean_val_loss < best_val_loss - 1e-4:
            best_val_loss = mean_val_loss
            patience_counter = 0
            best_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,
                "step": global_step,
            }
            save_checkpoint(best_state, args.output_dir, global_step)
            model.save_pretrained(args.adapter_dir)
            tokenizer.save_pretrained(args.adapter_dir)
            print(f"[best] new best val loss; saved adapter+tokenizer to {args.adapter_dir}")
        else:
            patience_counter += 1
            print(f"[earlystop] patience {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print("[earlystop] stopping training due to validation loss increase")
                break
        model.train()

    # Training complete: finalize save
    final_state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "step": global_step,
    }
    save_checkpoint(final_state, args.output_dir, global_step)
    model.save_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.adapter_dir)
    print("[train] training finished and model+adapter+tokenizer saved")

    # ---------------------------
    # Evaluation on test set (generation + metrics)
    # ---------------------------
    print("[eval] Generation on test set with beam search")
    test_split_data = raw["test"] if "test" in raw else raw["validation"]
    mrs = []
    refs_list = []
    for ex in test_split_data:
        mr = ex.get("meaning_representation") or ex.get("mr") or ex.get("source")
        refs = ex.get("human_reference") or ex.get("references") or ex.get("ref") or ex.get("reference")
        if isinstance(refs, list):
            ref_texts = refs
        elif isinstance(refs, str):
            ref_texts = [refs]
        else:
            ref_texts = [str(refs)]
        mrs.append(mr)
        refs_list.append(ref_texts)

    hypotheses = []
    model.eval()
    for mr in tqdm(mrs, desc="gen"):
        mr_text = " ; ".join([part.strip() for part in str(mr).split(",")]) if isinstance(mr, str) else str(mr)
        prefix = f"MR: {mr_text} ||| REF:"
        tokens = tokenizer(prefix, return_tensors="pt").to(args.device)
        gen_tokens = model.generate(
            **tokens,
            max_length=args.max_gen_len,
            num_beams=args.num_beams,
            early_stopping=True,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        if gen_text.startswith(prefix):
            hyp = gen_text[len(prefix):].strip()
        else:
            if "REF:" in gen_text:
                hyp = gen_text.split("REF:", 1)[1].strip()
            else:
                hyp = gen_text.strip()
        hypotheses.append(hyp)

    print("[eval] computing metrics (CIDEr, ROUGE-L, BLEU, METEOR, NIST) -- this may take a minute")
    scores = compute_all_metrics(hypotheses, refs_list)
    print("[RESULTS]")
    for k, v in scores.items():
        print(f"{k}: {v:.6f}")

    return scores


# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on E2E NLG with PEFT DoRA and evaluate with pure-Python metrics.")
    parser.add_argument("--model_name", default=TRAIN_ARGS.model_name)
    parser.add_argument("--output_dir", default=TRAIN_ARGS.output_dir)
    parser.add_argument("--adapter_dir", default=TRAIN_ARGS.adapter_dir)
    parser.add_argument("--dataset_name", default=TRAIN_ARGS.dataset_name)
    parser.add_argument("--epochs", type=int, default=TRAIN_ARGS.num_epochs)
    parser.add_argument("--batch_size", type=int, default=TRAIN_ARGS.train_batch_size)
    parser.add_argument("--eval_batch_size", type=int, default=TRAIN_ARGS.eval_batch_size)
    parser.add_argument("--max_length", type=int, default=TRAIN_ARGS.max_length)
    parser.add_argument("--save_every_steps", type=int, default=TRAIN_ARGS.save_every_steps)
    args_ns = parser.parse_args()

    ta = TRAIN_ARGS
    ta.model_name = args_ns.model_name
    ta.output_dir = args_ns.output_dir
    ta.adapter_dir = args_ns.adapter_dir
    ta.dataset_name = args_ns.dataset_name
    ta.num_epochs = args_ns.epochs
    ta.train_batch_size = args_ns.batch_size
    ta.eval_batch_size = args_ns.eval_batch_size
    ta.max_length = args_ns.max_length
    ta.save_every_steps = args_ns.save_every_steps

    train_and_evaluate(ta)
