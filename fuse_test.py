from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, os

device     = "cuda"                       # A100
fused_dir  = "./fused_gpt2_splitlora"     # folder you saved earlier

tok   = AutoTokenizer.from_pretrained(fused_dir)
model = AutoModelForCausalLM.from_pretrained(fused_dir).to(device).eval()

mr      = "name[Blue Spice], eatType[coffee shop], area[city centre]"
prompt  = mr + " <|gen|> "
ids     = tok(prompt, return_tensors="pt").input_ids.to(device)

with torch.no_grad():
    out = model.generate(
            ids,
            max_new_tokens      = 64,
            num_beams           = 10,
            length_penalty      = 0.8,
            no_repeat_ngram_size= 4,
            repetition_penalty  = 1.0,
            early_stopping      = True,
            eos_token_id        = tok.eos_token_id,
            pad_token_id        = tok.pad_token_id)

print(tok.decode(out[0, ids.size(1):], skip_special_tokens=True))
