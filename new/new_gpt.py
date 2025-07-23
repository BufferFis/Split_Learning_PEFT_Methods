# Full new.py with U-shape SplitGPT2 + DoRA + Python E2E Eval using args and JSON input

import random
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import os
import json
import argparse
import sys
import subprocess

# ============ SmoothCrossEntropyLoss ============
class SmoothCELoss(nn.Module):
    def __init__(self, eps=0.1):
        super().__init__()
        self.eps = eps
    def forward(self, logits, labels):
        log_preds = torch.log_softmax(logits, dim=-1)
        vocab_size = logits.size(-1)
        safe_labels = labels.clamp(min=0, max=vocab_size - 1)
        loss = -log_preds.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
        smooth_loss = -log_preds.mean(dim=-1)
        mask = labels != -100
        loss = loss * mask + smooth_loss * self.eps
        return loss.sum() / mask.sum()

# ============ SplitGPT2 U-Shape Setup ============
class SplitGPT2_UShape(nn.Module):
    def __init__(self, model_name="gpt2", peft_config=None):
        super().__init__()
        full_model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        full_model.resize_token_embeddings(len(self.tokenizer))

        # More balanced split with fewer layers on client side
        self.client_head = nn.Sequential(*full_model.transformer.h[:2])
        self.server = nn.Sequential(*full_model.transformer.h[2:10])
        self.client_tail = nn.Sequential(*full_model.transformer.h[10:])
        
        self.wte = full_model.transformer.wte
        self.wpe = full_model.transformer.wpe
        self.ln_f = full_model.transformer.ln_f
        self.lm_head = full_model.lm_head
        self.drop = nn.Dropout(0.1)
        self.config = full_model.config

        if peft_config:
            base = GPT2LMHeadModel.from_pretrained(model_name)
            base = get_peft_model(base, peft_config)
            self.server = nn.Sequential(*[base.transformer.h[i] for i in range(2, 10)])

    def forward(self, input_ids, attention_mask=None, labels=None):
        device = input_ids.device
        position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        inputs_embeds = self.wte(input_ids) + self.wpe(position_ids)
        hidden = self.drop(inputs_embeds)
        
        # Forward through client head
        for layer in self.client_head:
            outputs = layer(hidden, attention_mask=attention_mask)
            hidden = outputs[0]
        
        # Forward through server
        for layer in self.server:
            outputs = layer(hidden, attention_mask=attention_mask)
            hidden = outputs[0]
        
        # Forward through client tail
        for layer in self.client_tail:
            outputs = layer(hidden, attention_mask=attention_mask)
            hidden = outputs[0]
        
        hidden = self.ln_f(hidden)
        logits = self.lm_head(hidden)
        
        loss = None
        if labels is not None:
            loss_fct = SmoothCELoss(eps=0.1)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        return {'logits': logits, 'loss': loss}

    def generate(self, input_ids, attention_mask=None, **gen_kwargs):
        # Create a complete model for generation
        config = GPT2LMHeadModel.from_pretrained("gpt2").config
        full = GPT2LMHeadModel(config=config).to(input_ids.device)
        full.transformer.wte = self.wte
        full.transformer.wpe = self.wpe
        full.transformer.ln_f = self.ln_f
        full.transformer.h = nn.ModuleList(list(self.client_head) + list(self.server) + list(self.client_tail))
        full.lm_head = self.lm_head
        full.config = self.config
        full.eval()
        
        return full.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=60
            do_sample=True,
            top_k=50,           # Increased from 30
            top_p=0.92,         # Increased from 0.85
            temperature=0.8,    
            repetition_penalty=1.2,  # Reduced from 1.5
            pad_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs
        )

# ============ Dataset ============
def linearize_mr_dict(mr_dict):
    kv = mr_dict.copy()
    for key in ["name", "area", "near"]:
        if key in kv:
            kv[key] = key.upper()
    keys = list(kv.keys())
    random.shuffle(keys)
    return " ".join([f"{k}=[{kv[k]}]" for k in keys if kv[k]])

def load_json_dataset(path):
    with open(path, "r") as f:
        data = json.load(f)
    inputs, targets = [], []
    for ex in data:
        mr = ex["mr"]["value"] if isinstance(ex["mr"], dict) else ex["mr"]
        txts = [ex["txt"]]
        if "txt_lex" in ex and ex["txt_lex"] != ex["txt"]:
            txts.append(ex["txt_lex"])
        inputs.append(mr)
        targets.append(txts)
    return {"inputs": inputs, "targets": targets}

def preprocess(batch, tokenizer):
    inputs, labels = [], []
    for mr, targets in zip(batch['inputs'], batch['targets']):
        target = random.choice(targets).strip()
        mr_lin = linearize_mr_dict(mr)
        prompt = mr_lin + tokenizer.eos_token
        full_text = prompt + target + tokenizer.eos_token
        enc_input = tokenizer(full_text, add_special_tokens=False, truncation=False, padding=False)['input_ids']
        sep_len = len(tokenizer(prompt, add_special_tokens=False)['input_ids'])
        label_ids = [-100]*sep_len + enc_input[sep_len:]
        inputs.append(torch.tensor(enc_input))
        labels.append(torch.tensor(label_ids))
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_ids": inputs, "labels": labels}

# ============ Train ============
def train(args):
    # Improved DoRA configuration
    peft_cfg = LoraConfig(
        r=8,               
        lora_alpha=32,      # Increased from 16
        lora_dropout=0.05,  # Decreased from 0.1
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj", "c_fc"],
        fan_in_fan_out=True,
        use_dora=True
    )
    
    model = SplitGPT2_UShape("gpt2", peft_config=peft_cfg).cuda()
    tokenizer = model.tokenizer
    raw_data = load_json_dataset(args.train_path)
    processed = preprocess(raw_data, tokenizer)
    dataset = list(zip(processed['input_ids'], processed['labels']))
    
    # Increase batch size and add validation split
    train_size = int(0.9 * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    
    # Improved batch size and learning parameters
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Improved optimizer settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    
    # Better learning rate scheduler with longer warmup
    total_steps = len(train_loader) * 10  # Training for 10 epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

    best_val_loss = float('inf')
    patience, max_patience = 0, 3
    
    for epoch in range(1, 11):  # Increased epochs from 4 to 10
        model.train()
        train_losses = []
        
        for step, (input_ids, labels) in enumerate(train_loader):
            batch = {"input_ids": input_ids.cuda(), "labels": labels.cuda(), 
                     "attention_mask": (input_ids != tokenizer.pad_token_id).cuda()}
            out = model(**batch)
            loss = out.get('loss', None)
            if loss is None:  # Fallback if loss not returned from forward
                loss = SmoothCELoss()(out['logits'], batch['labels'])
                
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            clip_grad_norm_(model.parameters(), 1.0)  # Increased from 0.5
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            train_losses.append(loss.item())
            
            if step % 50 == 0:  # Reduced logging frequency
                print(f"Epoch {epoch} Step {step}/{len(train_loader)} Loss: {loss.item():.4f}")
                
        # Validation phase
        model.eval()
        val_losses = []
        with torch.no_grad():
            for input_ids, labels in val_loader:
                batch = {"input_ids": input_ids.cuda(), "labels": labels.cuda(),
                         "attention_mask": (input_ids != tokenizer.pad_token_id).cuda()}
                out = model(**batch)
                val_loss = out.get('loss', None)
                if val_loss is None:
                    val_loss = SmoothCELoss()(out['logits'], batch['labels'])
                val_losses.append(val_loss.item())
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Sample generation for monitoring
        print("\n=== Generation Examples ===")
        for _ in range(3):
            rand_idx = random.randint(0, len(raw_data['inputs']) - 1)
            sample_input = linearize_mr_dict(raw_data['inputs'][rand_idx]) + tokenizer.eos_token
            input_tensor = tokenizer(sample_input, return_tensors="pt").to(model.lm_head.weight.device)
            
            # Add attention mask
            input_tensor['attention_mask'] = torch.ones_like(input_tensor['input_ids'])
            
            gen_ids = model.generate(
                input_tensor['input_ids'],
                attention_mask=input_tensor['attention_mask']
            )[0]
            
            print("MR:", sample_input)
            print("PRED:", tokenizer.decode(gen_ids[input_tensor['input_ids'].shape[1]:], skip_special_tokens=True))
            print("REF:", random.choice(raw_data['targets'][rand_idx]))
            print("-" * 50)
        
        # Save best model and early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 0
            print(f"New best model with val loss: {best_val_loss:.4f}")
            torch.save(model.state_dict(), args.save_path)
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break

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
        for idx, (mr, refs) in enumerate(zip(val_data['inputs'], val_data['targets'])):
            mr_lin = linearize_mr_dict(mr)
            input = tokenizer(mr_lin + tokenizer.eos_token, return_tensors="pt", padding=True)
            input_ids = input["input_ids"].cuda()
            attention_mask = input["attention_mask"].cuda()
            out_ids = model.generate(
                input_ids,
                attention_mask=attention_mask
            )[0]
            generated = out_ids[input_ids.shape[1]:]
            pred = tokenizer.decode(generated, skip_special_tokens=True)
            pf.write(pred.strip() + "\n")
            rf.write("|||" + "|||".join(ref.strip() for ref in refs) + "\n")
            if idx % 100 == 0:
                print(f"Decoded {idx} samples...")
                print("MR:", mr_lin)
                print("PRED:", pred)
                print("REF:", refs[0])

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
