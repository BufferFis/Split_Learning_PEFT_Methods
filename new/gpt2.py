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

# Get colon token ID (it's a single token in GPT-2)
COLON_TOKEN_ID = tokenizer.encode(":", add_special_tokens=False)[0]
print(f"Colon token ID: {COLON_TOKEN_ID}")

# ---- FIXED Preprocessing Function with Debug ----
def preprocess_fn(ex):
    mr_str = mr_to_str(ex['meaning_representation'])
    text = ex['human_reference']
    
    # Use colon as delimiter (single token)
    seq = mr_str + ":" + text + tokenizer.eos_token
    
    enc = tokenizer(seq, truncation=True, padding='max_length', max_length=max_length)
    input_ids = enc.input_ids
    
    # Find colon position
    colon_positions = [i for i, token_id in enumerate(input_ids) if token_id == COLON_TOKEN_ID]
    
    if colon_positions:
        colon_pos = colon_positions[0]
        # Mask everything up to and including colon
        labels = [-100] * (colon_pos + 1) + input_ids[colon_pos + 1:]
    else:
        # Fallback: find by encoding separately
        mr_with_colon = mr_str + ":"
        mr_ids = tokenizer(mr_with_colon, add_special_tokens=False).input_ids
        mr_len = len(mr_ids)
        labels = [-100] * mr_len + input_ids[mr_len:]
    
    # Ensure labels length matches input_ids and handle padding properly
    labels = labels[:len(input_ids)]
    
    # Find the actual EOS token (not padding) and mask everything after it
    try:
        actual_eos_pos = input_ids.index(tokenizer.eos_token_id)
        # Keep labels up to EOS, then mask padding
        for i in range(actual_eos_pos + 1, len(labels)):
            labels[i] = -100
    except ValueError:
        # If no EOS found, the sequence was truncated - that's fine
        pass
    
    # DEBUG: Print first few examples to verify masking
    if hasattr(preprocess_fn, 'debug_count'):
        preprocess_fn.debug_count += 1
    else:
        preprocess_fn.debug_count = 1
        
    if preprocess_fn.debug_count <= 3:
        print(f"\n=== DEBUG EXAMPLE {preprocess_fn.debug_count} ===")
        print(f"MR: {mr_str}")
        print(f"Text: {text}")
        print(f"Full sequence: {seq}")
        print(f"Tokenized: {tokenizer.decode(input_ids[:50], skip_special_tokens=False)}")
        
        # Show what gets masked vs not masked
        masked_part = []
        unmasked_part = []
        for i, (token_id, label) in enumerate(zip(input_ids, labels)):
            if i >= len(input_ids):
                break
            token = tokenizer.decode([token_id])
            if label == -100:
                masked_part.append(token)
            else:
                unmasked_part.append(token)
        
        print(f"MASKED (not trained on): {''.join(masked_part)}")
        print(f"UNMASKED (trained on): {''.join(unmasked_part)}")
        print(f"Colon found at position: {colon_positions[0] if colon_positions else 'NOT FOUND'}")
        print("=" * 50)
    
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

# REDUCED batch size for more stable training
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_ds, batch_size=4, collate_fn=collate)

# ---- Model Split Definition ----
class Split3GPT2(nn.Module):
    def __init__(self, head_split=1, tail_split=1):
        super().__init__()
        base = GPT2LMHeadModel.from_pretrained("gpt2")
        self.config = base.config
        self.wte = base.transformer.wte
        self.wpe = base.transformer.wpe
        self.drop = nn.Dropout(0.1)  # REDUCED dropout
        num_blocks = len(base.transformer.h)
        self.head_blocks = nn.ModuleList(base.transformer.h[:head_split])
        self.middle_blocks = nn.ModuleList(base.transformer.h[head_split:num_blocks-tail_split])
        self.tail_blocks = nn.ModuleList(base.transformer.h[num_blocks-tail_split:])
        self.ln_f = base.transformer.ln_f
        self.lm_head = base.lm_head
        self.register_buffer("position_ids", torch.arange(max_length).unsqueeze(0))

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

# Instantiate model (no need to resize embeddings since colon is already in vocab)
model = Split3GPT2(1,1)

peft_cfg = LoraConfig(r=2, lora_alpha=8, target_modules=["c_attn","c_proj"], use_dora=True)  # REDUCED rank
model = get_peft_model(model, peft_cfg)
model = model.to(device)
model.print_trainable_parameters()

# MUCH lower learning rate
optimizer = AdamW(
    [p for n, p in model.named_parameters() if p.requires_grad],
    lr=1e-6,  # REDUCED from 1e-5
    weight_decay=0.01
)

epochs = 5  # More epochs with lower LR
total_steps = len(train_loader) * epochs
from transformers import get_linear_schedule_with_warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# ---- FIXED Generator with Stronger Anti-Repetition ----
def generate_sequence(model, input_ids, max_len=50, ban_eos_steps=5, top_k=40, temperature=1.0):
    model.eval()
    cur = input_ids.to(device)
    steps = 0
    
    for _ in range(max_len - input_ids.size(1)):
        with torch.no_grad():
            out = model(cur, attention_mask=torch.ones_like(cur).to(device))
            logits = out[1] if isinstance(out, tuple) else out
            next_logits = logits[:, -1, :] / temperature
            
            # Ban EOS for initial steps
            if steps < ban_eos_steps:
                next_logits[:, tokenizer.eos_token_id] = -float('Inf')
            
            # MUCH STRONGER repetition penalty
            if cur.size(1) >= 2:
                # Get more recent tokens for penalty
                lookback = min(8, cur.size(1))  # Look back up to 8 tokens
                recent_tokens = cur[0, -lookback:].tolist()
                
                # Count occurrences and apply stronger penalties
                token_counts = {}
                for token_id in recent_tokens:
                    token_counts[token_id] = token_counts.get(token_id, 0) + 1
                
                for token_id, count in token_counts.items():
                    if count > 1:  # If token appeared more than once
                        penalty = 5.0 * count  # Much stronger penalty
                        next_logits[:, token_id] -= penalty
            
            # Additional penalty for common problematic tokens
            problem_tokens = ["family", "friendly", "no", "yes", "the", "is", "a"]
            for word in problem_tokens:
                token_ids = tokenizer.encode(word, add_special_tokens=False)
                for token_id in token_ids:
                    if token_id in cur[0, -5:].tolist():  # If in recent 5 tokens
                        next_logits[:, token_id] -= 3.0
            
            # Use nucleus sampling instead of just top-k
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Nucleus sampling with p=0.9
            sorted_indices_to_remove = cumulative_probs > 0.9
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_logits[indices_to_remove] = -float('Inf')
            
            # Sample from remaining tokens
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        
        cur = torch.cat([cur, next_id], dim=1)
        steps += 1
        
        # Early stopping if we detect repetition
        if steps >= 3:
            last_3_tokens = cur[0, -3:].tolist()
            if len(set(last_3_tokens)) == 1:  # All same token
                break
                
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
        # Use colon as delimiter in generation prompt
        inp = tokenizer(mr_s + ":", return_tensors='pt').input_ids.to(device)
        out = generate_sequence(model, inp, max_len=80)
        
        # Split at colon and take the generated part
        full_text = tokenizer.decode(out[0], skip_special_tokens=False)
        if ":" in full_text:
            txt = full_text.split(":", 1)[-1].strip()
            txt = txt.replace(tokenizer.eos_token, "").strip()
        else:
            txt = tokenizer.decode(out[0], skip_special_tokens=True).split(mr_s, 1)[-1].strip()
        
        preds.append(txt); refs.append(ex['human_reference'])
    bleu = metric_bleu.compute(predictions=preds, references=[[r] for r in refs])
    meteor = metric_meteor.compute(predictions=preds, references=refs)
    rouge = metric_rouge.compute(predictions=preds, references=refs)
    return bleu, meteor, rouge

# ---- FIXED Training with Gradient Clipping ----
model.train()
for e in range(1, epochs + 1):
    model.train()
    tl = 0.0
    
    for i, b in enumerate(tqdm(train_loader, desc=f"Epoch{e} Train")):
        inp = b['input_ids'].to(device)
        m = b['attention_mask'].to(device) 
        lbl = b['labels'].to(device)
        
        loss, _ = model(inp, attention_mask=m, labels=lbl)
        loss.backward()
        
        # GRADIENT CLIPPING
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()  # Step scheduler
        optimizer.zero_grad()
        
        tl += loss.item()
        if (i + 1) % 50 == 0:  # Less frequent logging
            tqdm.write(f"Batch{i+1}/{len(train_loader)} loss={tl/(i+1):.4f}")
    
    # Validation
    model.eval()
    with torch.no_grad():
        vt = 0.0
        for j, b in enumerate(tqdm(val_loader, desc=f"Epoch{e} Val")):
            inp = b['input_ids'].to(device)
            m = b['attention_mask'].to(device)
            lbl = b['labels'].to(device)
            loss, _ = model(inp, attention_mask=m, labels=lbl)
            vt += loss.item()
    
    # Sanity check with colon delimiter
    print(f"\nEpoch{e} Sanity:")
    sanity = raw_val.select(range(3))  # Fewer examples
    for ex in sanity:
        mr_s = mr_to_str(ex['meaning_representation'])
        inp = tokenizer(mr_s + ":", return_tensors='pt').input_ids.to(device)
        out = generate_sequence(model, inp, max_len=50)
        
        full_text = tokenizer.decode(out[0], skip_special_tokens=False)
        if ":" in full_text:
            generated = full_text.split(":", 1)[-1].strip()
            generated = generated.replace(tokenizer.eos_token, "").strip()
        else:
            generated = "No colon found"
        
        print(f"MR: {mr_s}")
        print(f"Generated: {generated}")
        print(f"Reference: {ex['human_reference']}")
        print("-" * 50)
    
    # Only evaluate every 2 epochs to save time
    if e % 1 == 0:
        bleu, meteor, rouge = evaluate_model(model, tokenizer, raw_val.select(range(100)))
        print(f"Epoch{e}: TrainL={tl/len(train_loader):.4f} ValL={vt/len(val_loader):.4f} "
              f"BLEU={bleu['bleu']:.4f} METEOR={meteor['meteor']:.4f} ROUGE-L={rouge['rougeL']:.4f}\n")

# Save
od = "./split3_gpt2_dora_fixed"
os.makedirs(od, exist_ok=True)
model.save_pretrained(od)
tokenizer.save_pretrained(od)