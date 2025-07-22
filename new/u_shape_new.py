import torch
import torch.nn as nn
import pandas as pd
import json
import argparse
from torch.utils.data import Dataset, DataLoader
# --- FIX: AdamW is now imported from torch.optim ---
from torch.optim import AdamW 
import torch.nn.functional as F

from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
from peft import get_peft_model, LoraConfig, TaskType

# --- 1. Data Preparation Class for JSON (unchanged) ---
class E2EJsonDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Define delimiter tokens
        self.DELIM_TOKENS = tokenizer.encode(" <REF>", add_special_tokens=False)
        print(f"🔍 Delimiter tokens: {self.DELIM_TOKENS}")
        print(f"🔍 Delimiter decoded: '{tokenizer.decode(self.DELIM_TOKENS)}'")
        
        # Verify special tokens exist
        print(f"🔍 Special tokens in vocab:")
        print(f"  <MR>: {tokenizer.encode('<MR>', add_special_tokens=False)}")
        print(f"  <REF>: {tokenizer.encode('<REF>', add_special_tokens=False)}")
        # Load E2E JSON data
        with open(json_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Process E2E dataset format correctly
        for item in raw_data:
            if isinstance(item, dict) and 'mr' in item and 'txt' in item:
                # Extract meaning representation from nested structure
                mr_dict = item['mr']['value'] if 'value' in item['mr'] else item['mr']
                reference = item['txt']  # Use actual reference text
                
                # Build MR string from non-empty attributes only
                mr_parts = []
                for key, value in mr_dict.items():
                    if value and str(value).strip():  # Only include non-empty values
                        # Clean up the key names for better formatting
                        clean_key = key.replace('customer rating', 'customerRating').replace(' ', '')
                        mr_parts.append(f"{clean_key}[{value}]")
                
                if mr_parts:  # Only process if we have valid MR
                    mr_string = ", ".join(mr_parts)
                    self.data.append({
                        'meaning_representation': mr_string,
                        'human_reference': reference
                    })
        
        print(f"✅ Loaded {len(self.data)} E2E examples")
        if len(self.data) > 0:
            print(f"📝 Sample MR: {self.data[0]['meaning_representation']}")
            print(f"📝 Sample Reference: {self.data[0]['human_reference']}")
    
    def __len__(self):
        return len(self.data)
    
    def preprocess(self, example, sequence_length=None):
        """Optimized preprocessing without excessive debug output"""
        SEQ_LEN = sequence_length if sequence_length is not None else self.max_length
        
        mr = example["meaning_representation"]
        ref = example["human_reference"]
        
        # Tokenize pieces
        ids_mr = self.tokenizer.encode(f"<MR> {mr}", add_special_tokens=False)
        ids_ref = self.tokenizer.encode(ref, add_special_tokens=False)
        ids_delim = self.DELIM_TOKENS
        
        # Build full sequence
        full_sequence = ids_mr + ids_delim + ids_ref
        
        # Truncate if necessary
        if len(full_sequence) > SEQ_LEN:
            input_ids = full_sequence[:SEQ_LEN]
        else:
            input_ids = full_sequence
        
        # Find delimiter and create labels
        delim_pos = None
        for i in range(len(input_ids) - len(ids_delim) + 1):
            if input_ids[i:i+len(ids_delim)] == ids_delim:
                delim_pos = i
                break
        
        if delim_pos is not None:
            mask_length = delim_pos + len(ids_delim)
            labels = [-100] * mask_length + input_ids[mask_length:]
        else:
            labels = [-100] * len(input_ids)
        
        attention_mask = [1] * len(input_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "human_reference": ref,
            "meaning_representation": mr,
        }


    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Use the corrected preprocessing function
        processed = self.preprocess(item, self.max_length)
        
        # Convert to tensors and pad if necessary
        input_ids = processed["input_ids"]
        attention_mask = processed["attention_mask"] 
        labels = processed["labels"]
        
        # Pad sequences to max_length
        while len(input_ids) < self.max_length:
            input_ids.append(self.tokenizer.pad_token_id)
            attention_mask.append(0)
            labels.append(-100)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long)
        }



# --- 2. U-Shaped Split Architecture Components ---
class HeadModel(nn.Module):
    """First stage: Embeddings + first 4 transformer blocks"""
    def __init__(self, base_model):
        super().__init__()
        self.config = base_model.config  # Store config for dimension checking
        self.wte = base_model.transformer.wte
        self.wpe = base_model.transformer.wpe
        self.drop = base_model.transformer.drop
        # First 4 transformer blocks (0-3)
        self.h = nn.ModuleList([base_model.transformer.h[i] for i in range(4)])
        
    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
            # Ensure correct position embeddings
            seq_length = inputs_embeds.size(1)
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=inputs_embeds.device)
            position_embeds = self.wpe(position_ids)
            hidden_states = self.drop(hidden_states + position_embeds)
        elif input_ids is not None:
            inputs_embeds = self.wte(input_ids)
            position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=input_ids.device)
            position_embeds = self.wpe(position_ids)
            hidden_states = self.drop(inputs_embeds + position_embeds)
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")
        
        # Debug: Print tensor shapes
        print(f"HeadModel input shape: {hidden_states.shape}")
        print(f"Expected hidden_size: {self.config.hidden_size}")
        
        # Verify dimension consistency
        if hidden_states.size(-1) != self.config.hidden_size:
            raise ValueError(f"Hidden states dimension {hidden_states.size(-1)} doesn't match "
                           f"expected hidden_size {self.config.hidden_size}")
        
        original_attention_mask = attention_mask
        # Fix attention mask handling
        # Transform to 4D for transformer blocks
        if attention_mask is not None:
            if attention_mask.dtype == torch.long:
                attention_mask = attention_mask.to(dtype=torch.float32)
            
            batch_size, seq_length = attention_mask.shape
            attention_mask_4d = attention_mask.view(batch_size, 1, 1, seq_length)
            attention_mask_4d = attention_mask_4d.to(dtype=hidden_states.dtype)
            attention_mask_4d = (1.0 - attention_mask_4d) * torch.finfo(hidden_states.dtype).min
        
        # Pass through transformer blocks with 4D mask
        for block in self.h:
            hidden_states = block(hidden_states, attention_mask=attention_mask_4d)[0]
        
        # Return original 2D attention mask for pipeline consistency
        return hidden_states, original_attention_mask




class ServerModel(nn.Module):
    """Middle stage: transformer blocks 4-7"""
    def __init__(self, base_model):
        super().__init__()
        self.config = base_model.config
        # Middle 4 transformer blocks (4-7)
        self.h = nn.ModuleList([base_model.transformer.h[i] for i in range(4, 8)])
        
    def forward(self, hidden_states=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        elif hidden_states is None:
            raise ValueError("You must specify either hidden_states or inputs_embeds")
        
        print(f"ServerModel input shape: {hidden_states.shape}")
        
        # Verify dimensions
        if hidden_states.size(-1) != self.config.hidden_size:
            raise ValueError(f"ServerModel: Hidden states dimension {hidden_states.size(-1)} "
                           f"doesn't match expected {self.config.hidden_size}")
        
        # CRITICAL FIX: Convert attention mask dtype
        if attention_mask is not None:
            # Convert from long to float if needed
            if attention_mask.dtype == torch.long:
                attention_mask = attention_mask.to(dtype=torch.float32)
            
            # If attention mask is 2D, convert to 4D for transformer blocks
            if attention_mask.dim() == 2:
                batch_size, seq_length = attention_mask.shape
                # Convert to 4D: [batch_size, 1, 1, seq_length]
                attention_mask = attention_mask.view(batch_size, 1, 1, seq_length)
                attention_mask = attention_mask.to(dtype=hidden_states.dtype)
                # Apply mask transformation (0 for attend, large negative for mask)
                attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
        
        # Pass through middle 4 transformer blocks with corrected attention mask
        for i, block in enumerate(self.h):
            print(f"ServerModel block {i} input shape: {hidden_states.shape}")
            hidden_states = block(hidden_states, attention_mask=attention_mask)[0]
            print(f"ServerModel block {i} output shape: {hidden_states.shape}")
        
        # Return 2D attention mask for consistency with pipeline
        if attention_mask is not None and attention_mask.dim() == 4:
            # Convert back to 2D for pipeline consistency
            original_attention_mask = attention_mask[:, 0, 0, :]  # Extract 2D from 4D
        else:
            original_attention_mask = attention_mask
            
        return hidden_states, original_attention_mask


class TailModel(nn.Module):
    """Final stage: last 4 transformer blocks + LM head"""
    def __init__(self, base_model):
        super().__init__()
        self.config = base_model.config
        self.h = nn.ModuleList([base_model.transformer.h[i] for i in range(8, 12)])
        self.ln_f = base_model.transformer.ln_f
        self.lm_head = base_model.lm_head
        self._base_model = base_model
        
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": input_ids, 
            "attention_mask": kwargs.get("attention_mask", None), 
            **kwargs
        }
        
    def get_input_embeddings(self):
        return self._base_model.transformer.wte
        
    def get_output_embeddings(self):
        return self.lm_head
        
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
        
    def forward(self, hidden_states=None, attention_mask=None, labels=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        elif hidden_states is None:
            raise ValueError("You must specify either hidden_states or inputs_embeds")
            
        print(f"TailModel input shape: {hidden_states.shape}")
        
        # Verify dimensions
        if hidden_states.size(-1) != self.config.hidden_size:
            raise ValueError(f"TailModel: Hidden states dimension {hidden_states.size(-1)} "
                           f"doesn't match expected {self.config.hidden_size}")
        
        # CRITICAL FIX: Convert attention mask dtype and shape
        if attention_mask is not None:
            print(f"TailModel attention mask input shape: {attention_mask.shape}")
            print(f"TailModel attention mask dtype: {attention_mask.dtype}")
            
            # Convert from long to float if needed
            if attention_mask.dtype == torch.long:
                attention_mask = attention_mask.to(dtype=torch.float32)
            
            # If attention mask is 2D, convert to 4D for transformer blocks
            if attention_mask.dim() == 2:
                batch_size, seq_length = attention_mask.shape
                # Convert to 4D: [batch_size, 1, 1, seq_length]
                attention_mask = attention_mask.view(batch_size, 1, 1, seq_length)
                attention_mask = attention_mask.to(dtype=hidden_states.dtype)
                # Apply mask transformation (0 for attend, large negative for mask)
                attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
                
            print(f"TailModel attention mask after conversion: {attention_mask.shape}")
        
        # Process through transformer blocks with corrected attention mask
        for i, block in enumerate(self.h):
            print(f"TailModel block {i} input shape: {hidden_states.shape}")
            hidden_states = block(hidden_states, attention_mask=attention_mask)[0]
            print(f"TailModel block {i} output shape: {hidden_states.shape}")
        
        # Final layer norm and output
        hidden_states = self.ln_f(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        return {"loss": loss, "logits": lm_logits}



class UShaped_GPT2_Model(nn.Module):
    """Complete U-shaped model pipeline"""
    def __init__(self, base_model, tokenizer=None):
        super().__init__()
        self.config = base_model.config
        self.head = HeadModel(base_model)
        self.server = ServerModel(base_model)
        self.tail = TailModel(base_model)
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask=None, labels=None):
        print(f"Input shape: {input_ids.shape}")
        
        # Stage 1: Head processing
        hidden_states, attention_mask = self.head(input_ids, attention_mask)
        print(f"After head: {hidden_states.shape}")
        
        # Stage 2: Server processing  
        hidden_states, attention_mask = self.server(hidden_states, attention_mask)
        print(f"After server: {hidden_states.shape}")
        
        # Stage 3: Tail processing - FIX: Use keyword arguments
        output = self.tail(
            hidden_states=hidden_states, 
            attention_mask=attention_mask, 
            labels=labels
        )
        print(f"After tail: output logits shape: {output['logits'].shape}")
        
        return output

    def generate(self, input_ids, attention_mask=None, max_new_tokens=25, temperature=0.8, 
             top_p=0.9, repetition_penalty=1.3, length_penalty=1.2, early_stopping=True,eos_token_id=None, **kwargs):
        self.eval()
        
        with torch.no_grad():
            batch_size, seq_length = input_ids.shape
            device = input_ids.device
            
            # Initialize sequences and attention masks (keep in 2D)
            generated_ids = input_ids.clone()
            current_attention_mask = attention_mask.clone() if attention_mask is not None else torch.ones_like(input_ids)
            
            for step in range(max_new_tokens):
                # Process through head and server (they will handle 4D conversion internally)
                hidden_states, _ = self.head(generated_ids, current_attention_mask)
                hidden_states, _ = self.server(hidden_states, current_attention_mask)
                
                # Process through tail
                output = self.tail(hidden_states=hidden_states, attention_mask=current_attention_mask)
                
                logits = output['logits']
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for previous_token in set(generated_ids[i].tolist()):
                            next_token_logits[i, previous_token] /= repetition_penalty
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    for i in range(batch_size):
                        indices_to_remove = sorted_indices[i, sorted_indices_to_remove[i]]
                        next_token_logits[i, indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
                
                # Append to sequences
                generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)
                
                # FIXED: Update attention mask in 2D format
                current_attention_mask = torch.cat([
                    current_attention_mask,  # 2D: [batch, seq_len]
                    torch.ones(batch_size, 1, device=device)  # 2D: [batch, 1]
                ], dim=-1)
                
                # Check for early stopping
                if early_stopping and eos_token_id is not None and (next_tokens == eos_token_id).any():
                    break
                    
            return generated_ids



# --- 3. Model and Tokenizer Setup (modified) ---
def setup_model_and_tokenizer(model_name):
    """Create U-shaped GPT-2 model with tokenizer"""
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    base_model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Print model config for debugging
    print(f"Model config: {base_model.config}")
    print(f"Hidden size: {base_model.config.hidden_size}")
    print(f"Number of layers: {base_model.config.num_hidden_layers}")

    special_tokens_dict = {
        'bos_token': '<|endoftext|>',
        'eos_token': '<|endoftext|>',
        'pad_token': '<|pad|>',
        'additional_special_tokens': ['<MR>', '<REF>']
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.pad_token_id = tokenizer.pad_token_id
    
    # ✅ Create U-shaped model instead of returning base model
    u_shaped_model = UShaped_GPT2_Model(base_model, tokenizer)
    
    return u_shaped_model, tokenizer




# --- 4. Apply DoRA PEFT to U-shaped model ---
def apply_dora_peft(model):
    """Apply DoRA PEFT to each stage of the U-shaped model"""
    
    # Unload any existing PEFT adapters first
    if hasattr(model.head, 'unload'):
        model.head.unload()
    if hasattr(model.server, 'unload'):
        model.server.unload()
    if hasattr(model.tail, 'unload'):
        model.tail.unload()
    
    # Configure PEFT for head
    head_peft_config = LoraConfig(
        task_type=None,  # Use None to avoid inputs_embeds issues
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"],
        use_dora=True
    )
    model.head = get_peft_model(model.head, head_peft_config)
    
    # Configure PEFT for server
    server_peft_config = LoraConfig(
        task_type=None,  # Use None to avoid inputs_embeds issues
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"],
        use_dora=True
    )
    model.server = get_peft_model(model.server, server_peft_config)
    
    # Configure PEFT for tail
    tail_peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"],
        use_dora=True
    )
    model.tail = get_peft_model(model.tail, tail_peft_config)
    
    return model


# --- 5. Generation Function for Sanity Checks (modified) ---
def generate_sanity_check(model, tokenizer, device):
    """Optimized generation with better parameters"""
    model.eval()
    
    test_examples = [
        "name[NAME], eatType[restaurant], food[Italian]",
        "name[Alimentum], area[riverside], familyFriendly[yes], near[Burger King]"
    ]
    
    print("\n🎯 E2E Generation Check")
    
    for i, test_mr in enumerate(test_examples):
        print(f"\n--- Test {i+1} ---")
        print(f"Input MR: {test_mr}")
        
        input_text = f"<MR> {test_mr} <REF>"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=20,              # Reduced for E2E length
                temperature=0.8,                # Slightly higher for creativity
                top_p=0.95,                    # Higher for more diverse vocabulary
                repetition_penalty=1.5,        # Reduced from 2.5
                no_repeat_ngram_size=3,        # Prevent 3-gram repetition
                early_stopping=True,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = output_sequences[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        print(f"Generated: {generated_text}")
        
        # Quality metrics
        words = generated_text.split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            print(f"Quality: {len(words)} words, {unique_ratio:.2f} uniqueness")
    
    model.train()



# --- 6. Main Training Function (modified) ---
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using model: {args.model_name}")
    
    # Verify the model name corresponds to expected dimensions
    model_configs = {
        'gpt2': 768,           # GPT-2 small
        'gpt2-medium': 1024,   # GPT-2 medium  
        'gpt2-large': 1280,    # GPT-2 large
        'gpt2-xl': 1600        # GPT-2 XL
    }
    expected_dim = model_configs.get(args.model_name)
    if expected_dim:
        print(f"Expected hidden dimension: {expected_dim}")
    # Setup U-shaped model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.model_name)

    # Apply DoRA PEFT to all stages
    model = apply_dora_peft(model)
    model.to(device)

    # Print trainable parameters for each stage
    print("=== Trainable Parameters ===")
    print("Head stage:")
    model.head.print_trainable_parameters()
    print("Server stage:")
    model.server.print_trainable_parameters()
    print("Tail stage:")
    model.tail.print_trainable_parameters()

    train_dataset = E2EJsonDataset(json_file=args.train_file, tokenizer=tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizer for all trainable parameters across all stages
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps)

    model.train()
    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(args.num_epochs):
        print(f"--- Epoch {epoch+1}/{args.num_epochs} ---")
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for i, batch in enumerate(progress_bar):
            # In your training loop, add after getting the batch:
            if i == 0 and epoch == 0:  # First batch of first epoch
                sample_input = batch['input_ids'][0]
                sample_labels = batch['labels'][0]
                sample_mask = batch['attention_mask'][0]
                
                print(f"\n🔍 TRAINING VERIFICATION:")
                print(f"Input tokens: {sample_input[:30].tolist()}")
                print(f"Label tokens: {sample_labels[:30].tolist()}")
                print(f"Input decoded: {tokenizer.decode(sample_input[sample_mask.bool()])}")
                
                # Count masked vs unmasked
                masked = (sample_labels == -100).sum().item()
                total = len(sample_labels)
                print(f"Masked: {masked}/{total} ({masked/total*100:.1f}%)")
                
                # Show what model is trained to predict
                target_tokens = sample_labels[sample_labels != -100]
                if len(target_tokens) > 0:
                    print(f"Training targets: {tokenizer.decode(target_tokens)}")

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            model.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]

            # Check for overfitting
            if loss.item() < 0.1:
                print(f"\nWarning: Loss {loss.item():.4f} indicates potential overfitting. Consider early stopping.")

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'loss': loss.item()})

            if (i + 1) % args.sanity_check_steps == 0:
                generate_sanity_check(model, tokenizer, device)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Save each stage separately
    model.head.save_pretrained(os.path.join(args.output_dir, "head_stage"))
    model.server.save_pretrained(os.path.join(args.output_dir, "server_stage"))
    model.tail.save_pretrained(os.path.join(args.output_dir, "tail_stage"))
    tokenizer.save_pretrained(args.output_dir)

    print(f"\nTraining complete. U-shaped model stages saved to {args.output_dir}")

# --- 7. Entry Point and Argument Parsing (unchanged) ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune a U-shaped GPT-2 model on E2E NLG from JSON data.")

    # File paths
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training JSON file.")
    parser.add_argument("--dev_file", type=str, required=True, help="Path to the validation JSON file.")
    parser.add_argument("--output_dir", type=str, default="./e2e_u_shaped_gpt2_dora", help="Directory to save the fine-tuned model.")

    # Model and training hyperparameters
    parser.add_argument("--model_name", type=str, default="gpt2-medium", help="Name of the pre-trained model to use.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length for the tokenizer.")
    parser.add_argument("--sanity_check_steps", type=int, default=500, help="Perform a sanity check every N steps.")

    args = parser.parse_args()

    # Check if data files exist
    if not os.path.exists(args.train_file) or not os.path.exists(args.dev_file):
        print(f"Error: Make sure '{args.train_file}' and '{args.dev_file}' are present.")
    else:
        main(args)
