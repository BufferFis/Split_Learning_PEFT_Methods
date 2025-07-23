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
import torch.nn.functional as F

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

# Get colon token ID (keeping your original approach)
COLON_TOKEN_ID = tokenizer.encode(":", add_special_tokens=False)[0]
print(f"Colon token ID: {COLON_TOKEN_ID}")

# ---- Your Original Preprocessing (keeping it as is since it works) ----
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

# Keep your original batch size and settings
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_ds, batch_size=4, collate_fn=collate)

# ---- FIXED Model Split with Proper State Passing ----
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
        
        # Register position embeddings buffer
        self.register_buffer("position_ids", torch.arange(max_length).unsqueeze(0))

    def forward(self, input_ids, attention_mask=None, labels=None, past_key_values=None, use_cache=False):
        bsz, seq_len = input_ids.size()
        
        # CRITICAL FIX: Handle past_key_values for generation
        if past_key_values is not None:
            # During generation, we only process the last token
            past_length = past_key_values[0][0].size(-2)  # Get past sequence length
            position_ids = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0)
        else:
            # During training, process full sequence
            position_ids = self.position_ids[:, :seq_len]
        
        # Embeddings
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = self.drop(inputs_embeds + position_embeds)
        
        # CRITICAL FIX: Proper attention mask handling for generation
        if attention_mask is not None:
            if past_key_values is not None:
                # During generation, extend attention mask
                past_length = past_key_values[0][0].size(-2)
                # Create extended attention mask
                batch_size = attention_mask.size(0)
                extended_attention_mask = torch.ones(batch_size, past_length + seq_len, device=attention_mask.device)
                extended_attention_mask[:, -seq_len:] = attention_mask
                attention_mask = extended_attention_mask
            
            # Convert to causal mask format
            attention_mask = attention_mask.view(bsz, 1, 1, -1)
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
        
        # Forward through blocks with proper past_key_values handling
        new_past_key_values = () if use_cache else None
        
        # Head blocks
        for i, block in enumerate(self.head_blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            outputs = block(hidden_states, attention_mask=attention_mask, past_key_value=past_kv, use_cache=use_cache)
            hidden_states = outputs[0]
            if use_cache:
                new_past_key_values += (outputs[1],)
        
        # Middle blocks  
        middle_start = len(self.head_blocks)
        for i, block in enumerate(self.middle_blocks):
            past_kv = past_key_values[middle_start + i] if past_key_values is not None else None
            outputs = block(hidden_states, attention_mask=attention_mask, past_key_value=past_kv, use_cache=use_cache)
            hidden_states = outputs[0]
            if use_cache:
                new_past_key_values += (outputs[1],)
        
        # Tail blocks
        tail_start = len(self.head_blocks) + len(self.middle_blocks)
        for i, block in enumerate(self.tail_blocks):
            past_kv = past_key_values[tail_start + i] if past_key_values is not None else None
            outputs = block(hidden_states, attention_mask=attention_mask, past_key_value=past_kv, use_cache=use_cache)
            hidden_states = outputs[0]
            if use_cache:
                new_past_key_values += (outputs[1],)
        
        # Final layer norm and projection
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate loss
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        if use_cache:
            return (loss, logits, new_past_key_values) if loss is not None else (logits, new_past_key_values)
        else:
            return (loss, logits) if loss is not None else logits

# Instantiate model with your original settings
model = Split3GPT2(1, 1)

# Your original PEFT config (r=2 works fine as you mentioned)
peft_cfg = LoraConfig(r=2, lora_alpha=8, target_modules=["c_attn","c_proj"], use_dora=True)
model = get_peft_model(model, peft_cfg)
model = model.to(device)
model.print_trainable_parameters()

# Your original optimizer settings
optimizer = AdamW(
    [p for n, p in model.named_parameters() if p.requires_grad],
    lr=4e-5,  # Keeping your original LR
    weight_decay=0.01
)

epochs = 5
total_steps = len(train_loader) * epochs
from transformers import get_linear_schedule_with_warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# ---- COMPLETELY FIXED Generator with Proper Cache Handling ----
def generate_sequence(model, input_ids, max_len=50, ban_eos_steps=5, temperature=1.0):
    """
    Fixed generation with proper attention mask and cache handling
    """
    model.eval()
    batch_size = input_ids.size(0)
    device = input_ids.device
    
    # Initialize
    generated = input_ids.clone()
    past_key_values = None
    
    for step in range(max_len - input_ids.size(1)):
        # CRITICAL FIX: Proper attention mask that grows with sequence
        current_length = generated.size(1)
        attention_mask = torch.ones(batch_size, current_length, device=device)
        
        with torch.no_grad():
            if past_key_values is None:
                # First step: process full sequence
                model_inputs = {
                    "input_ids": generated,
                    "attention_mask": attention_mask,
                    "use_cache": True
                }
            else:
                # Subsequent steps: only process last token
                model_inputs = {
                    "input_ids": generated[:, -1:],
                    "attention_mask": attention_mask[:, -1:],  # Only last token's mask
                    "past_key_values": past_key_values,
                    "use_cache": True
                }
            
            outputs = model(**model_inputs)
            logits, past_key_values = outputs[0], outputs[1]
            
            # Get next token logits
            next_logits = logits[:, -1, :] / temperature
            
            # Ban EOS for initial steps
            if step < ban_eos_steps:
                next_logits[:, tokenizer.eos_token_id] = -float('Inf')
            
            # Simple repetition penalty - check last few tokens
            if generated.size(1) >= 3:
                last_tokens = generated[0, -3:].tolist()
                for token_id in last_tokens:
                    next_logits[:, token_id] -= 2.0  # Moderate penalty
            
            # Sample next token
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Early stopping conditions
            if next_token.item() == tokenizer.eos_token_id and step >= ban_eos_steps:
                break
                
            # Stop if we detect simple repetition
            if step >= 2:
                last_3 = generated[0, -3:].tolist()
                if len(set(last_3)) == 1:  # All same token
                    break
    
    return generated

# ---- Your original metrics & eval (keeping as is) ----
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

# ---- Training Loop (keeping your original structure) ----
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
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step() 
        optimizer.zero_grad()
        
        tl += loss.item()
        if (i + 1) % 50 == 0:
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
    
    # Sanity check
    print(f"\nEpoch{e} Sanity:")
    sanity = raw_val.select(range(3))
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
    
    # Evaluate every 2 epochs
    if e % 2 == 0:
        bleu, meteor, rouge = evaluate_model(model, tokenizer, raw_val.select(range(100)))
        print(f"Epoch{e}: TrainL={tl/len(train_loader):.4f} ValL={vt/len(val_loader):.4f} "
              f"BLEU={bleu['bleu']:.4f} METEOR={meteor['meteor']:.4f} ROUGE-L={rouge['rougeL']:.4f}\n")

# Save
od = "./split3_gpt2_dora_fixed"
os.makedirs(od, exist_ok=True)
model.save_pretrained(od)
tokenizer.save_pretrained(od)