#!/usr/bin/env python
"""
Example script to:
  - Read GLUE CoLA CSV files (train/validation/test).
  - Select among 'full_parameter' (full finetuning), 'fixed_lora', or 'DORA' (adaptive LoRA).
  - Train and evaluate the chosen method.
  - Benchmark them by comparing metrics and trainable parameters.

Usage:
  python run_glue.py --task cola --method full_parameter
  python run_glue.py --task cola --method fixed_lora
  python run_glue.py --task cola --method DORA
"""

import argparse
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
import numpy as np
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl


############################################
# 1. Model Wrappers (full_parameter, Fixed LoRA, DORA)
############################################

class Finetune_Config:
    """
    Minimal config object. You can expand as needed.
    """
    def __init__(self):
        # For fixed LoRA
        self.fixed_lora_rank = 4
        self.lora_dropout = 0.1
        self.lora_alpha = 32
        # For adaptive LoRA (now renamed to DORA)
        self.adaptive_lora_start_rank = 8
        self.adaptive_lora_start_prune_step_ratio = 0.2
        self.adaptive_lora_end_prune_step_ratio = 0.1
        self.adaptive_lora_prune_interval_step = 100
        self.adaptive_lora_end_avg_rank = 2
        self.adaptive_lora_eps = 1e-6
        self.adaptive_lora_sensitivity_beta = 0.9
        # Logging or saving path
        self.log_path = Path("./logs")


class Model_Wrapper_Base(nn.Module):
    def __init__(self, config: Finetune_Config, model: nn.Module):
        super().__init__()
        self.config = config
        self.model = model

    def forward(self, input_dict):
        # Expecting input_dict keys: input_ids, attention_mask, labels
        output = self.model(**input_dict)
        logits = output.logits  # standard classification
        loss = output.loss
        return logits, loss

    def regularization_loss(self, current_step: int) -> torch.Tensor:
        return torch.tensor(0, dtype=torch.float32, device=next(self.parameters()).device)

    def trainable_param(self):
        return (p for p in self.parameters() if p.requires_grad)

    def trainable_param_num(self) -> int:
        return sum(p.numel() for p in self.trainable_param())


############## full_parameter = Full Finetuning ##############
class Dora_Model_Wrapper(Model_Wrapper_Base):
    """
    Originally 'DORA' = Full Finetuning (no LoRA). Now used for full parameter training.
    """
    def __init__(self, config: Finetune_Config, model: nn.Module):
        super().__init__(config, model)
        # Everything is trainable by default, no changes needed.


############## Fixed LoRA ##############
class Fixed_Lora_Linear(nn.Module):
    def __init__(self, config: Finetune_Config, linear: nn.Linear):
        super().__init__()
        self.config = config
        self.linear = linear
        self.linear.weight.requires_grad = False
        if linear.bias is not None:
            linear.bias.requires_grad = False

        rank = config.fixed_lora_rank
        self.lora_dropout = nn.Dropout(config.lora_dropout)
        self.lora_a_linear = nn.Linear(linear.in_features, rank, bias=False)
        self.lora_b_linear = nn.Linear(rank, linear.out_features, bias=False)
        self.lora_scaling = config.lora_alpha / rank

        nn.init.kaiming_uniform_(self.lora_a_linear.weight)
        nn.init.zeros_(self.lora_b_linear.weight)

    def forward(self, x):
        # Original
        hidden = self.linear(x)
        # LoRA
        lora_hidden = self.lora_dropout(x)
        lora_hidden = self.lora_a_linear(lora_hidden)
        lora_hidden = self.lora_b_linear(lora_hidden) * self.lora_scaling
        return hidden + lora_hidden


class Fixed_Lora_Model_Wrapper(Model_Wrapper_Base):
    def __init__(self, config: Finetune_Config, model: nn.Module):
        super().__init__(config, model)
        self.apply_fixed_lora()

    def apply_fixed_lora(self):
        # Replace certain nn.Linear modules with Fixed_Lora_Linear
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Example: apply LoRA to *all* linear layers
                lora_linear = Fixed_Lora_Linear(self.config, module)
                # We need to set it on the parent module
                parent_module = self.get_parent_module(name)
                setattr(parent_module, name.split('.')[-1], lora_linear)

        # Set LoRA modules trainable, freeze everything else
        for param_name, param in self.model.named_parameters():
            if "lora_a_linear" in param_name or "lora_b_linear" in param_name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def get_parent_module(self, full_name):
        """
        Given "encoder.layer.0.attention.self.query" returns the parent module
        so we can reassign the last attribute.
        """
        names = full_name.split('.')
        obj = self.model
        for n in names[:-1]:
            obj = getattr(obj, n)
        return obj

    def trainable_param_num(self) -> int:
        count = 0
        for module in self.model.modules():
            if isinstance(module, Fixed_Lora_Linear):
                count += module.lora_a_linear.weight.numel()
                count += module.lora_b_linear.weight.numel()
        return count


############## DORA (Adaptive LoRA with optional pruning) ##############
class Adaptive_Lora_Linear(nn.Module):
    def __init__(self, config: Finetune_Config, linear: nn.Linear):
        super().__init__()
        self.config = config
        self.linear = linear
        # Freeze original weight and bias
        self.linear.weight.requires_grad = False
        if linear.bias is not None:
            linear.bias.requires_grad = False

        # Use DORA’s decomposition: r' single-rank components
        self.rank = config.adaptive_lora_start_rank
        self.lora_dropout = nn.Dropout(config.lora_dropout)
        # Instead of two Linear layers, we directly learn parameters for each rank-1 component:
        self.A = nn.Parameter(torch.empty(linear.in_features, self.rank))
        self.B = nn.Parameter(torch.empty(self.rank, linear.out_features))
        # c holds a scalar per component for pruning (initialized to 1)
        self.c = nn.Parameter(torch.ones(self.rank, dtype=torch.float32))
        # Scaling as in LoRA (using the initial rank)
        self.lora_scaling = config.lora_alpha / self.rank

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
    
    def forward(self, x):
        # Compute the original frozen output
        hidden = self.linear(x)
        dropped = self.lora_dropout(x)
        # Compute x @ A: (batch, rank)
        xA = torch.matmul(dropped, self.A)
        # Multiply each rank channel by its scalar c (elementwise multiplication)
        xA_scaled = xA * self.c
        # Project back via B and apply scaling – this is equivalent to summing over rank-1 updates
        lora_update = torch.matmul(xA_scaled, self.B) * self.lora_scaling
        return hidden + lora_update

    def importance_scores(self):
        """
        Compute an importance score for each single-rank component.
        Here we use: score_i = |c_i| * ||A_i|| * ||B_i||,
        which is proportional to the Frobenius norm of the rank-1 update.
        """
        scores = []
        for i in range(self.rank):
            a_i = self.A[:, i]
            b_i = self.B[i, :]
            score = torch.abs(self.c[i]) * torch.norm(a_i, p=2) * torch.norm(b_i, p=2)
            scores.append(score)
        return torch.stack(scores)

class Adaptive_Lora_Model_Wrapper(Model_Wrapper_Base):
    def __init__(self, config: Finetune_Config, model: nn.Module, max_step: int):
        super().__init__(config, model)
        self.max_step = max_step
        self.sensitivity_score_dict = {}
        self.finally_mask_dict = {}
        self.apply_adaptive_lora()

    def apply_adaptive_lora(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                lora_linear = Adaptive_Lora_Linear(self.config, module)
                parent_module = self.get_parent_module(name)
                setattr(parent_module, name.split('.')[-1], lora_linear)

        # Set LoRA modules trainable, freeze everything else
        for param_name, param in self.model.named_parameters():
            if "lora" in param_name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def get_parent_module(self, full_name):
        names = full_name.split('.')
        obj = self.model
        for n in names[:-1]:
            obj = getattr(obj, n)
        return obj

    def trainable_param_num(self) -> int:
        count = 0
        for module in self.model.modules():
            if isinstance(module, Adaptive_Lora_Linear):
                count += module.lora_a_linear.weight.numel()
                count += module.lora_scaler.numel()
                count += module.lora_b_linear.weight.numel()
        return count

    def regularization_loss(self, current_step: int) -> torch.Tensor:
        # Only active if within the pruning window
        current_prune_step, max_prune_step = self.get_prune_step(current_step)
        if current_prune_step > max_prune_step:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        # Compute average variance of LoRA weights
        sum_var = 0.0
        count = 0
        for module in self.model.modules():
            if isinstance(module, Adaptive_Lora_Linear):
                a_var = module.lora_a_linear.weight.var(dim=1).sum()
                b_var = module.lora_b_linear.weight.var(dim=0).sum()
                sum_var += (a_var + b_var)
                count += 2 * self.config.adaptive_lora_start_rank
        if count == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        mean_var = sum_var / count
        return mean_var

    def get_prune_step(self, current_step: int):
        c = int(current_step - self.max_step * self.config.adaptive_lora_start_prune_step_ratio)
        m = int(self.max_step * (1 - self.config.adaptive_lora_start_prune_step_ratio - self.config.adaptive_lora_end_prune_step_ratio))
        return c, m
    
    def update_importance_score(self):
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if isinstance(module, Adaptive_Lora_Linear):
                    score = module.importance_scores()  # Tensor of shape (rank,)
                    if name not in self.sensitivity_score_dict:
                        self.sensitivity_score_dict[name] = score.clone()
                        self.finally_mask_dict[name] = torch.zeros_like(score, dtype=torch.bool)
                    else:
                        beta = self.config.adaptive_lora_sensitivity_beta
                        self.sensitivity_score_dict[name] = beta * self.sensitivity_score_dict[name] + (1 - beta) * score

    def prune_lora_scaler(self, current_step: int) -> bool:
        # Compute pruning window parameters
        current_prune_step, max_prune_step = self.get_prune_step(current_step)
        if current_prune_step > max_prune_step:
            with torch.no_grad():
                for name, module in self.model.named_modules():
                    if isinstance(module, Adaptive_Lora_Linear):
                        mask = self.finally_mask_dict.get(name, torch.zeros_like(module.c, dtype=torch.bool))
                        module.c[mask] = 0
            return False

        self.update_importance_score()
        if current_prune_step < 0:
            return False
        if current_prune_step % self.config.adaptive_lora_prune_interval_step != 0 and current_prune_step != max_prune_step:
            return False

        with torch.no_grad():
            score_dict = {n: self.sensitivity_score_dict[n] for n in self.sensitivity_score_dict}
            all_score = torch.cat(list(score_dict.values()))
            start_rank = self.config.adaptive_lora_start_rank
            end_rank = self.config.adaptive_lora_end_avg_rank
            prune_rank_rate = ((start_rank - end_rank) / start_rank) * ((current_prune_step / max_prune_step) ** 3)
            prune_rank_num = int(all_score.numel() * prune_rank_rate)
            if prune_rank_num < 1:
                prune_rank_num = 1
            threshold = torch.kthvalue(all_score, prune_rank_num).values
            for name, module in self.model.named_modules():
                if isinstance(module, Adaptive_Lora_Linear):
                    mask = score_dict[name] <= threshold
                    module.c[mask] = 0
                    if current_prune_step == max_prune_step:
                        self.finally_mask_dict[name] = mask.clone()
        return (current_prune_step == max_prune_step)

    def pruned_param_num(self) -> int:
        # Count only un-pruned (non-zero) scalers
        total = 0
        for module in self.model.modules():
            if isinstance(module, Adaptive_Lora_Linear):
                active_r = (module.lora_scaler != 0).sum().item()
                # a_linear has shape [rank, in_features], b_linear has shape [out_features, rank]
                # so effectively the trainable param = active_r * in_features + active_r * out_features + active_r
                # (the last + active_r is for the lora_scaler itself)
                in_f = module.lora_a_linear.weight.shape[1]
                out_f = module.lora_b_linear.weight.shape[0]
                total += (active_r * in_f) + (active_r * out_f) + active_r
        return total


############################################
# 2. CSV-based GLUE Dataset
############################################
class GlueCsvDataset(Dataset):
    """
    Reads a GLUE-style CSV with columns: sentence, label
    For CoLA specifically: 'sentence' and 'label'
    Adjust as needed for other tasks (e.g. two-sentence tasks).
    """
    def __init__(self, csv_path, tokenizer, max_length=128):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = self.data["sentence"].tolist()
        self.labels = self.data["label"].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long)


############################################
# 3. Training & Evaluation
############################################

def evaluate(model_wrapper, dataloader, device):
    model_wrapper.eval()
    preds, golds = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            input_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
            logits, loss = model_wrapper(input_dict)
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(batch_preds)
            golds.extend(labels.cpu().numpy())
    model_wrapper.train()
    acc = accuracy_score(golds, preds)
    f1 = f1_score(golds, preds, average='weighted')
    return acc, f1

def train_and_benchmark(task_name, method, device):
    """
    Train on {task_name}_train.csv and {task_name}_validation.csv,
    then evaluate on {task_name}_test.csv.
    """

    # 1) Setup
    model_name = "roberta-base"  # or "bert-base-cased", etc.
    config_obj = Finetune_Config()

    # 2) Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3) Load CSV files
    train_file = f"{task_name}_train.csv"
    dev_file = f"{task_name}_validation.csv"
    test_file = f"{task_name}_test.csv"
    train_dataset = GlueCsvDataset(train_file, tokenizer)
    dev_dataset = GlueCsvDataset(dev_file, tokenizer)
    test_dataset = GlueCsvDataset(test_file, tokenizer)

    # 4) Create DataLoaders
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 5) Load base model
    model_config = AutoConfig.from_pretrained(model_name, num_labels=2)  # 2 for CoLA
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config)

    # 6) Wrap the model
    if method == "full_parameter":
        model_wrapper = Dora_Model_Wrapper(config_obj, base_model)
    elif method == "fixed_lora":
        model_wrapper = Fixed_Lora_Model_Wrapper(config_obj, base_model)
    elif method == "DORA":
        # For DORA (adaptive LoRA), suppose we define max_step = #batches * epochs (rough estimate)
        max_steps = len(train_loader) * 3
        model_wrapper = Adaptive_Lora_Model_Wrapper(config_obj, base_model, max_step=max_steps)
    else:
        raise ValueError("Unknown method. Use full_parameter, fixed_lora, or DORA.")

    model_wrapper.to(device)

    # 7) Setup optimizer
    lr = 2e-5
    optimizer = optim.AdamW(model_wrapper.trainable_param(), lr=lr)

    # 8) Training loop
    epochs = 3
    global_step = 0
    for epoch in range(epochs):
        model_wrapper.train()
        for batch in train_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            input_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
            logits, loss = model_wrapper(input_dict)
            reg_loss = model_wrapper.regularization_loss(global_step)
            total_loss = loss + reg_loss
            total_loss.backward()
            xm.optimizer_step(optimizer)
            xm.mark_step()  # Add this line

            # For DORA (adaptive LoRA), check pruning
            if method == "DORA":
                pruned_now = model_wrapper.prune_lora_scaler(global_step)
                if pruned_now:
                    print(f"[DORA] step={global_step} done.")

            if global_step % 50 == 0:
                print(f"[{method.upper()}] Epoch={epoch} Step={global_step} Loss={total_loss.item():.4f}")
            global_step += 1

        # Evaluate after each epoch
        dev_acc, dev_f1 = evaluate(model_wrapper, dev_loader, device)
        print(f"[{method.upper()}] Epoch={epoch} Dev Acc={dev_acc:.4f}, F1={dev_f1:.4f}")

    # 9) Final test evaluation
    test_acc, test_f1 = evaluate(model_wrapper, test_loader, device)
    print(f"[{method.upper()}] Test Acc={test_acc:.4f}, F1={test_f1:.4f}")

    # 10) Print some stats
    tparams = model_wrapper.trainable_param_num()
    print(f"[{method.upper()}] Trainable params: {tparams}")
    if method == "DORA":
        pruned = model_wrapper.pruned_param_num()
        print(f"[{method.upper()}] Pruned param count: {pruned}")

    # You can optionally save the model_wrapper state:
    torch.save(model_wrapper.state_dict(), f"{task_name}_{method}_model.pt")


############################################
# 4. Main entry point
############################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="cola",
                        help="Which GLUE task? e.g. cola, sst2, mrpc, etc.")
    parser.add_argument("--method", type=str, default="full_parameter",
                        choices=["full_parameter", "fixed_lora", "DORA"],
                        help="Which method: 'full_parameter' (full finetuning), 'fixed_lora', or 'DORA' (adaptive LoRA)")
    args = parser.parse_args()
    device = xm.xla_device()
    xm.set_rng_state(42)  # Set random seed for XLA operations


    print(f"Task: {args.task}, Method: {args.method}, Device: {device}")
    train_and_benchmark(args.task, args.method, device)


if __name__ == "__main__":
    main()
