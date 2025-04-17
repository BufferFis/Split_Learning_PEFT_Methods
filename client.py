import os
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from evaluate import load as load_metric
from tqdm import tqdm

from util import split_gpt2
from peft import LoraConfig, get_peft_model

# ----- ENV & DDP SETUP -----
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def setup_ddp():
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

# ----- Trainer Class -----
class SplitModelTrainer:
    def __init__(self, head_model, tail_model, tokenizer, server_url):
        self.head_model = head_model
        self.tail_model = tail_model
        self.tokenizer = tokenizer
        self.server_url = server_url
        self.device = head_model.device
        self.head_optimizer = optim.AdamW(
            [p for p in head_model.parameters() if p.requires_grad], lr=2e-4
        )
        self.tail_optimizer = optim.AdamW(
            [p for p in tail_model.parameters() if p.requires_grad], lr=2e-4
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def load_e2e_dataset(self):
        dataset = load_dataset("e2e_nlg", trust_remote_code=True)
        def preprocess(example):
            text = example["meaning_representation"] + " " + example["human_reference"]
            enc = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=128
            )
            return {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": enc["input_ids"],
                "human_reference": example["human_reference"]
            }
        train = dataset["train"].map(preprocess, remove_columns=dataset["train"].column_names)
        test  = dataset["test"].map(preprocess,  remove_columns=dataset["test"].column_names)
        return train, test

    def create_dataloader(self, ds, batch_size, shuffle=True, sampler=None):
        def collate_fn(batch):
            return {
                "input_ids": torch.tensor([b["input_ids"] for b in batch], dtype=torch.long),
                "attention_mask": torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long),
                "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.long),
                "human_reference": [b["human_reference"] for b in batch]
            }
        return DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(shuffle if sampler is None else False),
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )

    def train(self, dataloader, epochs):
        resp = requests.post(
            f"{self.server_url}/start_training",
            json={"learning_rate": 2e-4}
        )
        print("Server start_training:", resp.json())

        for epoch in range(epochs):
            self.head_model.train()
            self.tail_model.train()
            total_loss = 0.0
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
                input_ids = batch["input_ids"].to(self.device)
                attn_mask = batch["attention_mask"].to(self.device)
                labels    = batch["labels"].to(self.device)

                self.head_optimizer.zero_grad()
                self.tail_optimizer.zero_grad()

                # Head forward
                head_out = self.head_model(
                    input_ids=input_ids, attention_mask=attn_mask,
                    output_hidden_states=True
                )
                head_hid = head_out.hidden_states[-1]

                # Server forward_train
                payload = {"activations": head_hid.detach().cpu().tolist()}
                sr = requests.post(f"{self.server_url}/forward_train", json=payload)
                body_act = torch.tensor(sr.json()["body_activations"], device=self.device)
                body_act.requires_grad_()

                # Tail forward & loss
                tail_out = self.tail_model(inputs_embeds=body_act, attention_mask=attn_mask)
                logits   = tail_out.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = self.loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

                # Backward
                loss.backward()
                grad_output = body_act.grad.cpu().tolist()
                br = requests.post(
                    f"{self.server_url}/backward",
                    json={"grad_output": grad_output, "loss": loss.item()}
                )

                self.head_optimizer.step()
                self.tail_optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1} avg loss: {total_loss/len(dataloader):.4f}")
            is_final = (epoch == epochs-1)
            requests.post(f"{self.server_url}/end_epoch", json={"is_final": is_final})

    def generate(self, input_ids, attn_mask, max_length=128):
        with torch.no_grad():
            head_out = self.head_model(input_ids=input_ids, attention_mask=attn_mask)
            hid = head_out.hidden_states[-1]
            payload = {"activations": hid.cpu().tolist()}
            resp = requests.post(f"{self.server_url}/forward", json=payload)
            body_act = torch.tensor(resp.json()["body_activations"], device=self.device)
            tail_out = self.tail_model(inputs_embeds=body_act, attention_mask=attn_mask)
            logits = tail_out.logits

            gen_ids = input_ids.clone()
            for _ in range(max_length - gen_ids.size(1)):
                next_tok = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
                gen_ids = torch.cat([gen_ids, next_tok], dim=1)
                if next_tok.item() == self.tokenizer.eos_token_id:
                    break
                am = torch.ones_like(gen_ids)
                head_out = self.head_model(input_ids=gen_ids, attention_mask=am)
                hid = head_out.hidden_states[-1]
                payload = {"activations": hid.cpu().tolist()}
                resp = requests.post(f"{self.server_url}/forward", json=payload)
                body_act = torch.tensor(resp.json()["body_activations"], device=self.device)
                tail_out = self.tail_model(inputs_embeds=body_act, attention_mask=am)
                logits = tail_out.logits

            return self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    def evaluate(self, test_ds):
        self.head_model.eval()
        self.tail_model.eval()
        bleu   = load_metric("bleu")
        meteor = load_metric("meteor")
        rouge  = load_metric("rouge")

        preds, refs = [], []
        for ex in tqdm(test_ds, desc="Evaluating"):
            ids  = torch.tensor(ex["input_ids"]).unsqueeze(0).to(self.device)
            mask = torch.tensor(ex["attention_mask"]).unsqueeze(0).to(self.device)
            out  = self.generate(ids, mask)
            preds.append(out); refs.append([ex["human_reference"]])

        print("BLEU:", bleu.compute(predictions=preds, references=refs))
        print("METEOR:", meteor.compute(predictions=preds, references=refs))
        print("ROUGE:", rouge.compute(predictions=preds, references=[r[0] for r in refs]))

# ----- MAIN -----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:8000")
    args = parser.parse_args()

    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    # Load & split AFTER DDP setup
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    full_model = AutoModelForCausalLM.from_pretrained("gpt2")
    head_m, _, tail_m = split_gpt2(full_model, head_layers=2, tail_layers=2)

    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        bias="none", use_dora=True, task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj"]
    )
    head_m = get_peft_model(head_m, lora_cfg).to(device)
    tail_m = get_peft_model(tail_m, lora_cfg).to(device)

    head_ddp = DDP(head_m, device_ids=[local_rank])
    tail_ddp = DDP(tail_m, device_ids=[local_rank])

    trainer = SplitModelTrainer(head_ddp, tail_ddp, tokenizer, args.server_url)

    train_ds, test_ds = trainer.load_e2e_dataset()
    train_sampler = DistributedSampler(train_ds)
    train_dl = trainer.create_dataloader(
        train_ds, batch_size=args.batch_size, shuffle=False, sampler=train_sampler
    )

    if not args.eval_only:
        trainer.train(train_dl, epochs=args.epochs)
    if local_rank == 0:
        trainer.evaluate(test_ds)

    cleanup_ddp()

if __name__ == "__main__":
    main()