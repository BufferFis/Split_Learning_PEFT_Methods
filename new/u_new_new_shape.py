import torch
import torch.nn as nn
import pandas as pd
import json
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
import subprocess, sys
# --- Import GenerationMixin to address the warning ---
from transformers import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel
from transformers import DataCollatorWithPadding, GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
from peft import get_peft_model, LoraConfig, TaskType

# --- 1. Data Preparation Class for JSON (Largely Unchanged) ---
class E2EJsonDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        self.DELIM_TOKENS = self.tokenizer.encode(" <REF>", add_special_tokens=False)

        with open(json_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        for item in raw_data:
            if isinstance(item, dict) and 'mr' in item and 'txt' in item:
                mr_dict = item['mr']['value'] if 'value' in item['mr'] else item['mr']
                reference = item['txt']

                mr_parts = [f"{key.replace(' ', '')}[{value}]" for key, value in mr_dict.items() if value and str(value).strip()]

                if mr_parts:
                    self.data.append({
                        'meaning_representation': ", ".join(mr_parts),
                        'human_reference': reference
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        mr = item["meaning_representation"]
        ref = item["human_reference"]

        ids_mr = self.tokenizer.encode(f"<MR> {mr}", add_special_tokens=False)
        ids_ref = self.tokenizer.encode(ref, add_special_tokens=False)

        input_ids = ids_mr + self.DELIM_TOKENS + ids_ref
        input_ids = input_ids[:self.max_length]

        labels = [-100] * (len(ids_mr) + len(self.DELIM_TOKENS)) + ids_ref
        labels = labels[:self.max_length]

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# --- 2. U-Shaped Split Architecture Components (FIXED) ---
class HeadModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.config = base_model.config
        self.wte = base_model.transformer.wte
        self.wpe = base_model.transformer.wpe
        self.drop = base_model.transformer.drop
        self.h = nn.ModuleList([base_model.transformer.h[i] for i in range(4)])

    def forward(self, input_ids=None, past_key_values=None, attention_mask=None, use_cache=None, **kwargs):
        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        else:
            raise ValueError("You have to specify either input_ids")

        device = input_ids.device

        past_length = 0
        if past_key_values is not None and past_key_values[0] is not None:
            past_length = past_key_values[0][0].size(-2)

        position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = self.drop(inputs_embeds + position_embeds)

        if attention_mask is not None:
            attention_mask_4d = attention_mask.view(batch_size, 1, 1, -1)
            attention_mask_4d = attention_mask_4d.to(dtype=hidden_states.dtype)
            attention_mask_4d = (1.0 - attention_mask_4d) * torch.finfo(hidden_states.dtype).min

        presents = []
        for i, block in enumerate(self.h):
            layer_past = past_key_values[i] if past_key_values is not None else None
            # --- FIX: Pass use_cache flag down to the block ---
            outputs = block(hidden_states, layer_past=layer_past, attention_mask=attention_mask_4d, use_cache=use_cache)
            hidden_states = outputs[0]
            # --- FIX: Conditionally append cache only if it's returned ---
            if use_cache:
                presents.append(outputs[1])

        return hidden_states, tuple(presents) if use_cache else None

class ServerModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.config = base_model.config
        self.h = nn.ModuleList([base_model.transformer.h[i] for i in range(4, 8)])

    def forward(self, hidden_states, past_key_values=None, attention_mask=None, use_cache=None, **kwargs):
        batch_size = hidden_states.shape[0]
        if attention_mask is not None:
            attention_mask_4d = attention_mask.view(batch_size, 1, 1, -1)
            attention_mask_4d = attention_mask_4d.to(dtype=hidden_states.dtype)
            attention_mask_4d = (1.0 - attention_mask_4d) * torch.finfo(hidden_states.dtype).min

        presents = []
        for i, block in enumerate(self.h):
            layer_past = past_key_values[i] if past_key_values is not None else None
            # --- FIX: Pass use_cache flag down to the block ---
            outputs = block(hidden_states, layer_past=layer_past, attention_mask=attention_mask_4d, use_cache=use_cache)
            hidden_states = outputs[0]
            # --- FIX: Conditionally append cache only if it's returned ---
            if use_cache:
                presents.append(outputs[1])

        return hidden_states, tuple(presents) if use_cache else None

class TailModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.h = nn.ModuleList([base_model.transformer.h[i] for i in range(8, 12)])
        self.ln_f = base_model.transformer.ln_f
        self.lm_head = base_model.lm_head

    def forward(self, hidden_states, past_key_values=None, attention_mask=None, use_cache=None, **kwargs):
        batch_size = hidden_states.shape[0]
        if attention_mask is not None:
            attention_mask_4d = attention_mask.view(batch_size, 1, 1, -1)
            attention_mask_4d = attention_mask_4d.to(dtype=hidden_states.dtype)
            attention_mask_4d = (1.0 - attention_mask_4d) * torch.finfo(hidden_states.dtype).min

        presents = []
        for i, block in enumerate(self.h):
            layer_past = past_key_values[i] if past_key_values is not None else None
            # --- FIX: Pass use_cache flag down to the block ---
            outputs = block(hidden_states, layer_past=layer_past, attention_mask=attention_mask_4d, use_cache=use_cache)
            hidden_states = outputs[0]
            # --- FIX: Conditionally append cache only if it's returned ---
            if use_cache:
                presents.append(outputs[1])

        hidden_states = self.ln_f(hidden_states)
        lm_logits = self.lm_head(hidden_states)

        return lm_logits, tuple(presents) if use_cache else None


class UShaped_GPT2_Model(GPT2PreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        base_model = GPT2LMHeadModel(config)
        self.head = HeadModel(base_model)
        self.server = ServerModel(base_model)
        self.tail = TailModel(base_model)
        self.main_input_name = "input_ids"

    def get_input_embeddings(self):
        return self.head.wte

    def set_input_embeddings(self, new_embeddings):
        self.head.wte = new_embeddings

    def get_output_embeddings(self):
        return self.tail.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.tail.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "token_type_ids": token_type_ids,
        }

    def forward(self, input_ids, attention_mask=None, labels=None, past_key_values=None, use_cache=None, **kwargs):
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if past_key_values is None:
            past_key_values = [None] * self.config.num_hidden_layers

        head_past = tuple(past_key_values[i] for i in range(4))
        server_past = tuple(past_key_values[i] for i in range(4, 8))
        tail_past = tuple(past_key_values[i] for i in range(8, 12))

        hidden_states, head_present = self.head(input_ids=input_ids, past_key_values=head_past, attention_mask=attention_mask, use_cache=use_cache, **kwargs)
        hidden_states, server_present = self.server(hidden_states=hidden_states, past_key_values=server_past, attention_mask=attention_mask, use_cache=use_cache, **kwargs)
        logits, tail_present = self.tail(hidden_states=hidden_states, past_key_values=tail_past, attention_mask=attention_mask, use_cache=use_cache, **kwargs)

        past_key_values_present = None
        if use_cache:
            past_key_values_present = head_present + server_present + tail_present

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values_present,
        )

# --- 3. Model and Tokenizer Setup ---
def setup_model_and_tokenizer(model_name):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    special_tokens_dict = {
        'bos_token': '<|endoftext|>',
        'eos_token': '<|endoftext|>',
        'pad_token': '<|pad|>',
        'additional_special_tokens': ['<MR>', '<REF>']
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    config = GPT2LMHeadModel.from_pretrained(model_name).config
    config.pad_token_id = tokenizer.pad_token_id
    config.vocab_size = len(tokenizer)
    u_shaped_model = UShaped_GPT2_Model(config)
    u_shaped_model.resize_token_embeddings(len(tokenizer))
    return u_shaped_model, tokenizer

# --- 4. Apply DoRA PEFT to U-shaped model ---
def apply_dora_peft(model):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj", "c_fc"],
        use_dora=True
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

# --- Data Collator ---
class E2EDataCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tok = tokenizer
        self.mult = pad_to_multiple_of
    def _pad(self, seq, pad_id, max_len):
        return seq + [pad_id] * (max_len - len(seq))
    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        if self.mult:
            max_len = ((max_len + self.mult - 1) // self.mult) * self.mult
        input_ids, attn, labels = [], [], []
        for f in features:
            input_ids.append(self._pad(f["input_ids"], self.tok.pad_token_id, max_len))
            attn.append(self._pad(f["attention_mask"], 0, max_len))
            labels.append(self._pad(f["labels"], -100, max_len))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.float),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

# --- 5. Generation Function for Sanity Checks ---
def generate_sanity_check(model, tokenizer, device):
    model.eval()
    test_examples = [
        "name[The Wrestlers], eatType[pub], food[English], priceRange[more than £30], customer rating[high], area[city centre], familyFriendly[no], near[Café Sicilia]",
        "name[Alimentum], area[riverside], familyFriendly[yes], near[Burger King]"
    ]
    print("\n" + "="*20 + " GENERATION SANITY CHECK " + "="*20)
    for i, test_mr in enumerate(test_examples):
        print(f"\n--- Test {i+1} ---")
        print(f"Input MR: {test_mr}")
        input_text = f"<MR> {test_mr} <REF>"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=50,
                num_beams=5,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = output_sequences[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"Generated: {generated_text.strip()}")
    print("="*63 + "\n")
    model.train()

# --- 6. Main Training Function ---
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model, tokenizer = setup_model_and_tokenizer(args.model_name)
    model = apply_dora_peft(model)
    model.to(device)
    train_dataset = E2EJsonDataset(json_file=args.train_file, tokenizer=tokenizer, max_length=args.max_length)
    data_collator = E2EDataCollator(tokenizer, pad_to_multiple_of=8)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps)
    for epoch in range(args.num_epochs):
        print(f"--- Epoch {epoch+1}/{args.num_epochs} ---")
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for i, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            model.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            progress_bar.set_postfix({'loss': loss.item()})
            if (i + 1) % args.sanity_check_steps == 0:
                generate_sanity_check(model, tokenizer, device)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nTraining complete. Model saved to {args.output_dir}")

# --- 7. Entry Point and Argument Parsing ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune a U-shaped GPT-2 model on E2E NLG from JSON data.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training JSON file.")
    parser.add_argument("--dev_file", type=str, required=True, help="Path to the validation JSON file.")
    parser.add_argument("--output_dir", type=str, default="./e2e_u_shaped_gpt2_dora_fixed", help="Directory to save the fine-tuned model.")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Name of the pre-trained model to use.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length for the tokenizer.")
    parser.add_argument("--sanity_check_steps", type=int, default=500, help="Perform a sanity check every N steps.")
    args = parser.parse_args()
    if not os.path.exists(args.train_file) or not os.path.exists(args.dev_file):
        print(f"Error: Make sure '{args.train_file}' and '{args.dev_file}' are present.")
    else:
        main(args)