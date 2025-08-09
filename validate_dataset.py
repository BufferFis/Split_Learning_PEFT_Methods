# validate_dataset_examples.py
import torch
from nosplitgpt2 import E2EDataset  # assuming same folder & module name
from datasets import load_dataset
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset("e2e_nlg")
tokenizer_name = "gpt2"
train_dataset = E2EDataset(ds, 'train', tokenizer, max_length=128)

for i in range(5):
    ex = train_dataset[i]
    input_ids = ex["input_ids"]
    am = ex["attention_mask"]
    labels = ex["labels"]
    # find first non-pad index
    first_token_idx = (input_ids != tokenizer.pad_token_id).nonzero(as_tuple=True)[0][0].item()
    print("="*40)
    print("Example", i)
    print("first non-pad index (should be offset):", first_token_idx)
    decoded = tokenizer.decode(input_ids[first_token_idx:].tolist(), clean_up_tokenization_spaces=False)
    print("decoded (non-pad part):", decoded)
    # verify labels: find first label != -100
    labeled_idxs = (labels != -100).nonzero(as_tuple=True)[0].tolist()
    if labeled_idxs:
        print("labels cover indices (first, last):", labeled_idxs[0], labeled_idxs[-1])
        print("decoded labels portion:", tokenizer.decode(input_ids[labeled_idxs[0]:labeled_idxs[-1]+1].tolist(), clean_up_tokenization_spaces=False))
    else:
        print("No labels (unexpected).")
    print("attention_mask sum (seq length):", am.sum().item())
