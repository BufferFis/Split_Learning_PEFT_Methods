# Full new.py with U-shape SplitGPT2 + DoRA + Python E2E Eval using args and JSON input

import random
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
import os
import json
import argparse
import sys
import subprocess

# ============ SplitGPT2 U-Shape Setup ============
class SplitGPT2_UShape(nn.Module):
    def __init__(self, model_name="gpt2", peft_config=None):
        super().__init__()
        full_model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        full_model.resize_token_embeddings(len(self.tokenizer))

        self.client_head = nn.Sequential(*full_model.transformer.h[:4])
        self.server = nn.Sequential(*full_model.transformer.h[4:8])
        self.client_tail = nn.Sequential(*full_model.transformer.h[8:])
        self.wte = full_model.transformer.wte
        self.wpe = full_model.transformer.wpe
        self.ln_f = full_model.transformer.ln_f
        self.lm_head = full_model.lm_head
        self.drop = nn.Dropout(0.1)

        if peft_config:
            base = GPT2LMHeadModel.from_pretrained(model_name)
            base = get_peft_model(base, peft_config)
            self.server = base.transformer.h[4:8]

    def forward(self, input_ids, attention_mask=None, labels=None):
        device = input_ids.device
        inputs_embeds = self.wte(input_ids) + self.wpe(torch.arange(input_ids.size(1), device=device))
        hidden = self.drop(inputs_embeds)
        for layer in self.client_head: hidden = layer(hidden)[0]
        for layer in self.server: hidden = layer(hidden)[0]
        for layer in self.client_tail: hidden = layer(hidden)[0]
        hidden = self.ln_f(hidden)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {'loss': loss, 'logits': logits}

    def generate(self, input_ids, **gen_kwargs):
        from transformers import PreTrainedModel
        class Wrapper(PreTrainedModel):
            def __init__(self, module, config):
                super().__init__(config)
                self.module = module
                self.config = config
                self.transformer = nn.Module()
                self.lm_head = lambda x: module.lm_head(module.ln_f(x))
            def forward(self, input_ids, **kwargs):
                return self.module(input_ids, **kwargs)
        wrapped = Wrapper(self, GPT2LMHeadModel.from_pretrained("gpt2").config).cuda()
        return wrapped.generate(input_ids, **gen_kwargs)

# ============ Dataset ============
def linearize_mr_dict(mr_dict):
    kv = mr_dict.copy()
    for key in ["name", "area", "near"]:
        if key in kv: kv[key] = key.upper()
    keys = list(kv.keys())
    random.shuffle(keys)
    return " ".join([f"{k}=[{kv[k]}]" for k in keys if kv[k]])

def load_json_dataset(path):
    with open(path, "r") as f:
        data = json.load(f)
    inputs, targets = [], []
    for ex in data:
        mr = ex["mr"]["value"]
        txts = [ex["txt"], ex.get("txt_lex", ex["txt"])]
        inputs.append(mr)
        targets.append(txts)
    return {"inputs": inputs, "targets": targets}

def preprocess(batch, tokenizer):
    inputs, labels = [], []
    for mr, targets in zip(batch['inputs'], batch['targets']):
        target = random.choice(targets).strip()
        mr_lin = linearize_mr_dict(mr)
        prompt = mr_lin + tokenizer.eos_token + target + tokenizer.eos_token
        enc = tokenizer(prompt, add_special_tokens=False)
        input_ids = enc['input_ids']
        eos = tokenizer.eos_token_id
        sep_idx = input_ids.index(eos) + 1 if eos in input_ids else len(input_ids)
        label_ids = [-100]*sep_idx + input_ids[sep_idx:]
        inputs.append(input_ids)
        labels.append(label_ids)
    enc = tokenizer.pad({"input_ids": inputs, "labels": labels}, padding=True, return_tensors="pt")
    return enc

# ============ Train ============
def train(args):
    peft_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, task_type="CAUSAL_LM", fan_in_fan_out=True, use_dora=True)
    model = SplitGPT2_UShape("gpt2", peft_config=peft_cfg).cuda()
    tokenizer = model.tokenizer
    raw_data = load_json_dataset(args.train_path)
    processed = preprocess(raw_data, tokenizer)
    loader = DataLoader([{k: v[i:i+1] for k,v in processed.items()} for i in range(len(processed['input_ids']))], batch_size=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    for epoch in range(3):
        for step, batch in enumerate(loader):
            for k in batch: batch[k] = batch[k].cuda()
            out = model(**batch)
            out['loss'].backward()
            optimizer.step(); optimizer.zero_grad()
            if step % 100 == 0:
                print(f"Epoch {epoch} Step {step} Loss: {out['loss'].item():.4f}")
    torch.save(model.state_dict(), args.save_path)

# ============ Eval ============
def evaluate(args):
    peft_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, task_type="CAUSAL_LM", fan_in_fan_out=True, use_dora=True)
    model = SplitGPT2_UShape("gpt2", peft_config=peft_cfg).cuda()
    tokenizer = model.tokenizer
    model.load_state_dict(torch.load(args.save_path))
    model.eval()

    val_data = load_json_dataset(args.val_path)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    pred_path = os.path.join(out_dir, "valid.pred.txt")
    ref_path = os.path.join(out_dir, "valid.refs.txt")

    with open(pred_path, "w") as pf, open(ref_path, "w") as rf:
        for mr, refs in zip(val_data["inputs"], val_data["targets"]):
            mr_lin = linearize_mr_dict(mr)
            input_ids = tokenizer(mr_lin + tokenizer.eos_token, return_tensors="pt").input_ids.cuda()
            out_ids = model.generate(input_ids, max_new_tokens=60, num_beams=5, do_sample=True, top_k=50, top_p=0.95)[0]
            pred = tokenizer.decode(out_ids, skip_special_tokens=True)
            pf.write(pred.strip() + "\n")
            rf.write("|||" + "|||".join(ref.strip() for ref in refs) + "\n")

    print("Running official evaluation script...")
    subprocess.run(["python", args.eval_script, "-p", pred_path, "-r", ref_path, "-o", out_dir])

# ============ Entry Point ============
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, help="Path to train JSON")
    parser.add_argument("--val_path", type=str, help="Path to val JSON")
    parser.add_argument("--save_path", type=str, default="e2e_model.pt")
    parser.add_argument("--out_dir", type=str, default="e2e_outputs")
    parser.add_argument("--eval_script", type=str, help="Path to official measure.py script")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train")
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)
