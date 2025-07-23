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

# ---- Preprocessing ----
# Load E2E NLG dataset
dataset = load_dataset("tuetschek/e2e_nlg")

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
max_length = 128

# Preprocessing function: combine MR and text, mask MR in labels
def preprocess_fn(ex):
    mr = ex["meaning_representation"]
    text = ex["human_reference"]
    seq = mr + tokenizer.eos_token + text
    enc = tokenizer(seq, truncation=True, padding='max_length', max_length=max_length)
    input_ids = enc.input_ids
    mr_ids = tokenizer(mr, add_special_tokens=False).input_ids
    mr_len = len(mr_ids)
    labels = [-100] * mr_len + input_ids[mr_len:]
    return {"input_ids": input_ids, "attention_mask": enc.attention_mask, "labels": labels}

# Map and set format
train_ds = dataset['train'].map(preprocess_fn, remove_columns=dataset['train'].column_names)
val_ds = dataset['validation'].map(preprocess_fn, remove_columns=dataset['validation'].column_names)
train_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
val_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

# Dataloaders
def collate(batch):
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    labels = torch.stack([b['labels'] for b in batch])
    return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_ds, batch_size=8, collate_fn=collate)

# ---- Model Split Definition ----
class Split3GPT2(nn.Module):
    def __init__(self, head_split=1, tail_split=1):
        super().__init__()
        base = GPT2LMHeadModel.from_pretrained("gpt2")
        self.config = base.config
        # Embeddings
        self.wte = base.transformer.wte
        self.wpe = base.transformer.wpe
        self.drop = base.transformer.drop
        # Transformer blocks split
        num_blocks = len(base.transformer.h)
        self.head_blocks = nn.ModuleList(base.transformer.h[:head_split])
        self.middle_blocks = nn.ModuleList(base.transformer.h[head_split:num_blocks-tail_split])
        self.tail_blocks = nn.ModuleList(base.transformer.h[num_blocks-tail_split:])
        self.ln_f = base.transformer.ln_f
        # LM head
        self.lm_head = base.lm_head
        # Position IDs buffer
        self.register_buffer("position_ids", torch.arange(max_length).unsqueeze(0))

    def forward(self, input_ids, attention_mask=None, labels=None):
        bsz, seq_len = input_ids.size()
        device = input_ids.device
        # Embeddings
        inputs_embeds = self.wte(input_ids) + self.wpe(self.position_ids[:, :seq_len])
        hidden = self.drop(inputs_embeds)

        # Prepare 4D causal attention mask like client implementation
        if attention_mask is not None:
            # attention_mask: [bsz, seq_len] -> [bsz, 1, 1, seq_len]
            attn_mask = attention_mask.view(bsz, -1).to(device)
            attn_mask = attn_mask[:, None, None, :].to(dtype=hidden.dtype)
            attn_mask = (1.0 - attn_mask) * torch.finfo(hidden.dtype).min
        else:
            attn_mask = None

        # Pass through head blocks
        for block in self.head_blocks:
            hidden = block(hidden, attention_mask=attn_mask)[0]
        # Middle (server)
        for block in self.middle_blocks:
            hidden = block(hidden, attention_mask=attn_mask)[0]
        # Tail blocks
        for block in self.tail_blocks:
            hidden = block(hidden, attention_mask=attn_mask)[0]

        # Final layer norm and LM head
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

# Instantiate and apply DoRA
model = Split3GPT2(head_split=1, tail_split=1)
peft_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["c_attn", "c_proj"],
    use_dora=True
)
model = get_peft_model(model, peft_config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---- Metrics Setup ----
metric_bleu = evaluate.load('bleu')
metric_meteor = evaluate.load('meteor')
metric_rouge = evaluate.load('rouge')

# ---- Evaluation Function ----
def evaluate_model(model, tokenizer, examples, max_gen_len=100):
    model.eval()
    preds, refs = [], []
    with torch.no_grad():
        for ex in examples:
            mr = ex['meaning_representation']
            prompt = mr + tokenizer.eos_token
            input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
            out = model.generate(input_ids, max_length=max_gen_len, pad_token_id=tokenizer.eos_token_id)
            gen = tokenizer.decode(out[0], skip_special_tokens=True)
            text = gen.split(tokenizer.eos_token, 1)[-1].strip()
            preds.append(text)
            refs.append(ex['human_reference'])
    bleu = metric_bleu.compute(predictions=preds, references=[[r] for r in refs])
    meteor = metric_meteor.compute(predictions=preds, references=refs)
    rouge = metric_rouge.compute(predictions=preds, references=refs)
    return bleu, meteor, rouge

# ---- Training Loop with Sanity Check & Eval ----
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
sanity_samples = dataset['validation'][:5]
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    avg_train = total_loss / len(train_loader)

    # Mid-training sanity check
    print(f"Epoch {epoch+1} Sanity Check:")
    for ex in sanity_samples:
        prompt = ex['meaning_representation'] + tokenizer.eos_token
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        out = model.generate(input_ids, max_length=50, pad_token_id=tokenizer.eos_token_id)
        print("MR:", ex['meaning_representation'])
        print("Gen:", tokenizer.decode(out[0], skip_special_tokens=True).split(tokenizer.eos_token,1)[-1].strip())

    # Validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += loss.item()
    avg_val = val_loss / len(val_loader)

    # Full eval metrics
    bleu, meteor, rouge = evaluate_model(model, tokenizer, dataset['validation'])
    print(f"Epoch {epoch+1}: Train Loss={avg_train:.4f}, Val Loss={avg_val:.4f}")
    print(f" BLEU={bleu['bleu']:.4f}, METEOR={meteor['meteor']:.4f}, ROUGE-L={rouge['rougeL']:.4f}\n")

# ---- Save the fine-tuned model ----
out_dir = "./split3_gpt2_dora"
# create output directory
os.makedirs(out_dir, exist_ok=True)
model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)
