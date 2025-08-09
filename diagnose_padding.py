# diagnose_padding_fixed.py
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer
from collections import defaultdict

MODEL = "gpt2"   # or your model name/path
MAX_LENGTH = 80  # keep small for quick checks
SAMPLES = 8

tokenizer = GPT2Tokenizer.from_pretrained(MODEL)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset("e2e_nlg")
data = ds["train"]

mr_to_refs = defaultdict(list)
for item in data:
    mr_to_refs[item["meaning_representation"]].append(item["human_reference"])

mrs = list(mr_to_refs.keys())[:SAMPLES]

for mr in mrs:
    prompt = f"MR: {mr} REF:"
    prompt_ids = tokenizer(prompt, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")["input_ids"][0]
    ref = mr_to_refs[mr][0]
    ref_ids = tokenizer(" " + ref, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")["input_ids"][0]
    combined = torch.cat([prompt_ids, ref_ids])

    # TRUNCATE to avoid shape mismatch (keep the END of the sequence)
    if combined.size(0) > MAX_LENGTH:
        combined = combined[-MAX_LENGTH:]

    # RIGHT-padded (simulate old buggy behaviour)
    input_ids_right = torch.full((MAX_LENGTH,), tokenizer.pad_token_id, dtype=torch.long)
    input_ids_right[:combined.size(0)] = combined
    attn_right = (input_ids_right != tokenizer.pad_token_id).long()

    # LEFT-padded (correct)
    input_ids_left = torch.full((MAX_LENGTH,), tokenizer.pad_token_id, dtype=torch.long)
    offset = MAX_LENGTH - combined.size(0)
    input_ids_left[offset:] = combined
    attn_left = (input_ids_left != tokenizer.pad_token_id).long()

    print("="*60)
    print("MR:", mr)
    print("Sample ref:", ref)
    print("--- RIGHT-padded (simulated) ---")
    print("input_ids_right (first 20):", input_ids_right[:20].tolist())
    print("attn_right (first 20):", attn_right[:20].tolist())
    print("decoded_right :", tokenizer.decode(input_ids_right[input_ids_right!=tokenizer.pad_token_id], clean_up_tokenization_spaces=False))
    print("--- LEFT-padded (correct) ---")
    print("input_ids_left (first 20):", input_ids_left[:20].tolist())
    print("attn_left (first 20):", attn_left[:20].tolist())
    print("offset (left pad):", offset)
    print("decoded_left :", tokenizer.decode(input_ids_left[input_ids_left!=tokenizer.pad_token_id], clean_up_tokenization_spaces=False))
    print("\n")
