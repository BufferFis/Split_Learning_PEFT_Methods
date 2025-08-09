# overfit_one_example.py
import torch, random
from nosplitgpt2 import E2EDataset, setup_peft_model
from datasets import load_dataset
from transformers import GPT2Tokenizer
from torch.optim import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset("e2e_nlg")
train_ds = E2EDataset(ds, 'train', tokenizer, max_length=128)

# pick single example index (or random)
idx = 10
ex = train_ds[idx]
input_ids = ex["input_ids"].unsqueeze(0).to(device)
attention_mask = ex["attention_mask"].unsqueeze(0).to(device)
labels = ex["labels"].unsqueeze(0).to(device)

# load base model (no resume), set up PEFT if you normally use it
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
# If you use PEFT adapters in normal runs, initialize them too:
# model = setup_peft_model(model)

model.train()
optimizer = AdamW(model.parameters(), lr=2e-4)  # higher lr for quick overfit
num_steps = 200

for step in range(1, num_steps+1):
    optimizer.zero_grad()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    if step % 10 == 0 or step == 1:
        print(f"step {step} loss: {loss.item():.6f}")

# Generate after training
model.eval()
with torch.no_grad():
    prompt = tokenizer.decode(input_ids[0][input_ids[0] != tokenizer.pad_token_id], skip_special_tokens=True)
    # keep only the prompt (before REF:) to generate
    if "REF:" in prompt:
        prompt = prompt.split("REF:")[0] + "REF:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    gen = model.generate(**inputs, max_new_tokens=80, num_beams=1, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
    print("=== Generated ===")
    print(decoded)
    print("=== Reference ===")
    # print the reference
    print(tokenizer.decode(input_ids[0][(labels[0] != -100).nonzero(as_tuple=True)[0]], skip_special_tokens=True))
