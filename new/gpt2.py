import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import evaluate
from tqdm.auto import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---- Helper: MR to string ----
def mr_to_str(mr):
    """
    Convert meaning_representation string into prompt.
    Format: "slot[val], slot[val]" -> "slot val slot val"
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
raw_val = dataset['validation']
# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
max_length = 128

# ---- Preprocessing Function ----
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
        self.drop = nn.Dropout(0.3)
        num_blocks = len(base.transformer.h)
        self.head_blocks = nn.ModuleList(base.transformer.h[:head_split])
        self.middle_blocks = nn.ModuleList(base.transformer.h[head_split:num_blocks-tail_split])
        self.tail_blocks = nn.ModuleList(base.transformer.h[num_blocks-tail_split:])
        self.ln_f = base.transformer.ln_f
        self.lm_head = base.lm_head
        self.register_buffer("position_ids", torch.arange(max_length).unsqueeze(0).to(device))

    def forward(self, input_ids, attention_mask=None, labels=None):
        bsz, seq_len = input_ids.size()
        pos_ids = self.position_ids[:, :seq_len].to(input_ids.device)
        inputs_embeds = self.wte(input_ids) + self.wpe(pos_ids)
        hidden = self.drop(inputs_embeds)
        attn = None
        if attention_mask is not None:
            attn = (1.0 - attention_mask.view(bsz,1,1,seq_len).to(hidden.device)) * torch.finfo(hidden.dtype).min
        for blk in self.head_blocks: hidden = blk(hidden, attention_mask=attn)[0]
        for blk in self.middle_blocks: hidden = blk(hidden, attention_mask=attn)[0]
        for blk in self.tail_blocks: hidden = blk(hidden, attention_mask=attn)[0]
        hidden = self.ln_f(hidden); logits = self.lm_head(hidden)
        loss=None
        if labels is not None:
            shift_logits=logits[..., :-1,:].contiguous(); shift_labels=labels[...,1:].contiguous()
            loss=CrossEntropyLoss(ignore_index=-100)(shift_logits.view(-1,shift_logits.size(-1)), shift_labels.view(-1))
        return (loss, logits) if loss is not None else logits

# Instantiate & apply DoRA
model = Split3GPT2(1,1)
peft_cfg = LoraConfig(r=4, lora_alpha=16, target_modules=["c_attn","c_proj"], use_dora=True)
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()

optimizer = AdamW(
    [p for n, p in model.named_parameters() if p.requires_grad],
    lr=1e-5,
    weight_decay=0.01
)

epochs = 3
total_steps = len(train_loader) * epochs
from transformers import get_linear_schedule_with_warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# ---- Generator (custom) ----
def generate_sequence(model, input_ids, max_len=50, ban_eos_steps=2, top_k=50):
    cur = input_ids
    steps = 0
    for _ in range(max_len - input_ids.size(1)):
        with torch.no_grad():
            out = model(cur, attention_mask=torch.ones_like(cur).to(device))
            logits = out[1] if isinstance(out, tuple) else out
            next_logits = logits[:, -1, :]
            if steps < ban_eos_steps:
                next_logits[:, tokenizer.eos_token_id] = -float('Inf')
            values, indices = torch.topk(next_logits, top_k)
            probs = torch.softmax(values, dim=-1)
            choice = torch.multinomial(probs, num_samples=1)
            next_id = indices.gather(-1, choice)
        cur = torch.cat([cur, next_id], dim=1)
        steps += 1
        if next_id.item() == tokenizer.eos_token_id and steps >= ban_eos_steps:
            break
    return cur

# ---- Metrics & Eval ----
metric_bleu=evaluate.load('bleu'); metric_meteor=evaluate.load('meteor'); metric_rouge=evaluate.load('rouge')
def evaluate_model(model, tokenizer, raw_examples, max_samples=None):
    preds, refs = [], []
    examples = raw_examples if max_samples is None else raw_examples.select(range(max_samples))
    for ex in tqdm(examples, desc="Metric Eval", leave=False):
        mr_s = mr_to_str(ex['meaning_representation'])
        inp = tokenizer(mr_s+tokenizer.eos_token, return_tensors='pt').input_ids.to(device)
        out = generate_sequence(model, inp, max_len=100)
        txt = tokenizer.decode(out[0], skip_special_tokens=True).split(tokenizer.eos_token,1)[-1].strip()
        preds.append(txt); refs.append(ex['human_reference'])
    bleu = metric_bleu.compute(predictions=preds, references=[[r] for r in refs])
    meteor = metric_meteor.compute(predictions=preds, references=refs)
    rouge = metric_rouge.compute(predictions=preds, references=refs)
    return bleu, meteor, rouge

# ---- Training w/ Progress & Eval ----
opt=AdamW(model.parameters(),lr=5e-5); epochs=3; sanity=raw_val.select(range(5))
for e in range(1,epochs+1):
    model.train(); tl=0.0
    for i,b in enumerate(tqdm(train_loader,desc=f"Epoch{e} Train")):
        inp=b['input_ids'].to(device); m=b['attention_mask'].to(device); lbl=b['labels'].to(device)
        loss,_=model(inp,attention_mask=m,labels=lbl); loss.backward(); opt.step(); opt.zero_grad()
        tl+=loss.item(); tqdm.write(f"Batch{i+1}/{len(train_loader)} loss={tl/(i+1):.4f}")
    vt=0.0
    for j,b in enumerate(tqdm(val_loader,desc=f"Epoch{e} Val")):
        inp=b['input_ids'].to(device); m=b['attention_mask'].to(device); lbl=b['labels'].to(device)
        loss,_=model(inp,attention_mask=m,labels=lbl); vt+=loss.item(); tqdm.write(f"ValBatch{j+1}/{len(val_loader)} loss={vt/(j+1):.4f}")
    print(f"\nEpoch{e} Sanity:")
    for ex in sanity:
        mr_s=mr_to_str(ex['meaning_representation']); inp=tokenizer(mr_s+tokenizer.eos_token, return_tensors='pt').input_ids.to(device)
        out=generate_sequence(model,inp,max_len=50)
        print(mr_s, "->", tokenizer.decode(out[0], skip_special_tokens=True).split(tokenizer.eos_token,1)[-1].strip())
    bleu, meteor, rouge = evaluate_model(model, tokenizer, raw_val)
    print(f"Epoch{e}: TrainL={tl/len(train_loader):.4f} ValL={vt/len(val_loader):.4f} "
          f"BLEU={bleu['bleu']:.4f} METEOR={meteor['meteor']:.4f} ROUGE-L={rouge['rougeL']:.4f}\n")
# Save
od="./split3_gpt2_dora";os.makedirs(od,exist_ok=True); model.save_pretrained(od);tokenizer.save_pretrained(od)
