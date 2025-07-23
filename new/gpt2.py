import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2Config
from transformers import GPT2LMHeadModel
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import evaluate
from tqdm.auto import tqdm

# ---- Helper: MR to string ----
def mr_to_str(mr):
    """
    Convert meaning_representation to string prompt.
    Handles string like "slot[val], slot[val], ..." and fallback.
    """
    if isinstance(mr, str):
        parts = []
        for item in mr.split(','):
            item = item.strip()
            if '[' in item and item.endswith(']'):
                slot, val = item.split('[', 1)
                val = val[:-1]
                if val:
                    parts.append(f"{slot.strip()} {val}")
        return ' '.join(parts)
    return str(mr)

# ---- Preprocessing ----
dataset = load_dataset("tuetschek/e2e_nlg")
# Build raw validation list for evaluation
raw_val = dataset['validation']

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
max_length = 128

# ---- Preprocessing Function ----
# Converts each raw example into model inputs:
# 1) Parses the meaning representation string (e.g., "name[The Vaults], eatType[pub], ...")
#    into a prompt string: "name The Vaults eatType pub ..." via mr_to_str.
# 2) Appends GPT-2 EOS token as a separator between MR and target text.
# 3) Tokenizes the full sequence to fixed max_length.
# 4) Builds labels by masking out the MR tokens (set to -100) so that loss
#    is computed only on the target text portion.
# Example:
#   MR: "name[The Vaults], eatType[pub]"
#   Text: "The Vaults pub is great."
#   Sequence: "name The Vaults eatType pub <|eos|> The Vaults pub is great."
#   Labels: [-100, -100, -100, -100, -100, -100, id(The), id(Vaults), ...]
#
def preprocess_fn(ex):
    mr_str = mr_to_str(ex['meaning_representation'])
    text = ex['human_reference']
    seq = mr_str + tokenizer.eos_token + text
    enc = tokenizer(seq, truncation=True, padding='max_length', max_length=max_length)
    input_ids = enc.input_ids
    mr_ids = tokenizer(mr_str, add_special_tokens=False).input_ids
    mr_len = len(mr_ids)
    labels = [-100] * mr_len + input_ids[mr_len:]  
    return {"input_ids": input_ids, "attention_mask": enc.attention_mask, "labels": labels}

train_ds = dataset['train'].map(preprocess_fn, remove_columns=dataset['train'].column_names)
val_ds = dataset['validation'].map(preprocess_fn, remove_columns=dataset['validation'].column_names)
train_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
val_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

# Dataloaders
def collate(batch):
    return dict(
        input_ids=torch.stack([b['input_ids'] for b in batch]),
        attention_mask=torch.stack([b['attention_mask'] for b in batch]),
        labels=torch.stack([b['labels'] for b in batch])
    )
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_ds, batch_size=8, collate_fn=collate)

# ---- Model Split Definition ----
class Split3GPT2(nn.Module):
    def __init__(self, head_split=1, tail_split=1):
        super().__init__()
        base = GPT2LMHeadModel.from_pretrained("gpt2")
        self.config = base.config
        self.wte = base.transformer.wte
        self.wpe = base.transformer.wpe
        self.drop = base.transformer.drop
        num_blocks = len(base.transformer.h)
        self.head_blocks = nn.ModuleList(base.transformer.h[:head_split])
        self.middle_blocks = nn.ModuleList(base.transformer.h[head_split:num_blocks-tail_split])
        self.tail_blocks = nn.ModuleList(base.transformer.h[num_blocks-tail_split:])
        self.ln_f = base.transformer.ln_f
        self.lm_head = base.lm_head
        self.register_buffer("position_ids", torch.arange(max_length).unsqueeze(0))

    def forward(self, input_ids, attention_mask=None, labels=None):
        bsz, seq_len = input_ids.size()
        device = input_ids.device
        inputs_embeds = self.wte(input_ids) + self.wpe(self.position_ids[:, :seq_len])
        hidden = self.drop(inputs_embeds)

        if attention_mask is not None:
            attn = attention_mask.view(bsz, 1, 1, seq_len).to(device)
            attn = (1.0 - attn) * torch.finfo(hidden.dtype).min
        else:
            attn = None

        for blk in self.head_blocks:
            hidden = blk(hidden, attention_mask=attn)[0]
        for blk in self.middle_blocks:
            hidden = blk(hidden, attention_mask=attn)[0]
        for blk in self.tail_blocks:
            hidden = blk(hidden, attention_mask=attn)[0]

        hidden = self.ln_f(hidden)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = CrossEntropyLoss(ignore_index=-100)(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
        return (loss, logits) if loss is not None else logits

# Instantiate & apply DoRA
model = Split3GPT2(1,1)
peft_cfg = LoraConfig(r=4, lora_alpha=16, target_modules=["c_attn","c_proj"], use_dora=True)
model = get_peft_model(model, peft_cfg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---- Metrics ----
metric_bleu = evaluate.load('bleu')
metric_meteor = evaluate.load('meteor')
metric_rouge = evaluate.load('rouge')

def evaluate_model(model, tokenizer, raw_examples, max_len=100):
    model.eval(); preds, refs = [], []
    with torch.no_grad():
        for ex in raw_examples:
            mr_str = mr_to_str(ex['meaning_representation'])
            prompt = mr_str + tokenizer.eos_token
            inp = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
            out = model.generate(inp, max_length=max_len, pad_token_id=tokenizer.eos_token_id)
            gen = tokenizer.decode(out[0], skip_special_tokens=True)
            text = gen.split(tokenizer.eos_token,1)[-1].strip()
            preds.append(text); refs.append(ex['human_reference'])
    bleu = metric_bleu.compute(predictions=preds, references=[[r] for r in refs])
    meteor = metric_meteor.compute(predictions=preds, references=refs)
    rouge = metric_rouge.compute(predictions=preds, references=refs)
    return bleu, meteor, rouge

# ---- Training w/ Progress, Sanity & Eval ----
opt = AdamW(model.parameters(), lr=5e-5)
epochs = 3
sanity = raw_val.select(range(5))
for e in range(1, epochs+1):
    model.train(); tloss = 0.0
    for batch in tqdm(train_loader, desc=f"Train Epoch {e}"):
        inp = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        lbl = batch['labels'].to(device)
        loss,_ = model(inp, attention_mask=mask, labels=lbl)
        loss.backward(); opt.step(); opt.zero_grad()
        tloss += loss.item()
    avg_t = tloss/len(train_loader)

    # val loss
    model.eval(); vloss=0
    for batch in tqdm(val_loader, desc=f"Val Epoch {e}"):
        inp = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        lbl = batch['labels'].to(device)
        loss,_ = model(inp, attention_mask=mask, labels=lbl)
        vloss += loss.item()
    avg_v = vloss/len(val_loader)

    # Sanity
    print(f"\nEpoch {e} Sanity:")
    for ex in sanity:
        mr_s = mr_to_str(ex['meaning_representation'])
        inp = tokenizer(mr_s+tokenizer.eos_token, return_tensors='pt').input_ids.to(device)
        out = model.generate(inp, max_length=50, pad_token_id=tokenizer.eos_token_id)
        print(mr_s, "->", tokenizer.decode(out[0], skip_special_tokens=True).split(tokenizer.eos_token,1)[-1].strip())

    # Full eval
    bleu, meteor, rouge = evaluate_model(model, tokenizer, raw_val)
    print(f"Epoch {e}: TrainL={avg_t:.4f} ValL={avg_v:.4f} BLEU={bleu['bleu']:.4f} METEOR={meteor['meteor']:.4f} ROUGE-L={rouge['rougeL']:.4f}\n")

# Save
od = "./split3_gpt2_dora"; os.makedirs(od, exist_ok=True)
model.save_pretrained(od); tokenizer.save_pretrained(od)
