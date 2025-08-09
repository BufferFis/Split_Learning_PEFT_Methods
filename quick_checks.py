# quick_checks.py
import torch
from nosplitgpt2 import E2EDataset  # ensure import path is correct
from datasets import load_dataset
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset("e2e_nlg")
ds_train = E2EDataset(ds, 'train', tokenizer, max_length=128)

# pick some examples
for i in range(5):
    ex = ds_train[i]
    input_ids = ex["input_ids"]
    attn = ex["attention_mask"]
    labels = ex["labels"]

    print("Example", i)
    print("input_ids dtype:", input_ids.dtype, "shape:", input_ids.shape)
    print("attention_mask sum:", attn.sum().item(), "first non-pad idx:",
          (input_ids != tokenizer.pad_token_id).nonzero(as_tuple=True)[0][0].item())
    lab_idxs = (labels != -100).nonzero(as_tuple=True)[0].tolist()
    print("labels count:", len(lab_idxs), "first/last label idx:", (lab_idxs[0], lab_idxs[-1]) if lab_idxs else None)
    # decode prompt and label text
    offset = (input_ids != tokenizer.pad_token_id).nonzero(as_tuple=True)[0][0].item()
    print("decoded non-pad tail:", tokenizer.decode(input_ids[offset:].tolist()))
    if lab_idxs:
        print("decoded labels portion:", tokenizer.decode(input_ids[lab_idxs[0]:lab_idxs[-1]+1].tolist()))
    print("-"*40)
