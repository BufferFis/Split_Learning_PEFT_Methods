# fuse_splitlora.py  –  run once
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel



device = "cuda"

def fuse_splitlora(split_root: str,
                   model_name: str = "gpt2",
                   out_dir: str   = "./fused_gpt2_splitlora"):

    tok   = AutoTokenizer.from_pretrained(model_name)
    tok.add_special_tokens(
        {"additional_special_tokens": ["<|gen|>"],
         "pad_token": "<|pad|>"})

    # 1️⃣  base model + resize for two new tokens
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.resize_token_embeddings(len(tok))

    # 2️⃣  wrap once with HEAD adapter
    model = PeftModel.from_pretrained(
                model,
                os.path.join(split_root, "head_model"),
                is_trainable=False)

    # 3️⃣  attach BODY and TAIL adapters (no re-wrapping)
    for name in ("body_model", "tail_model"):
        model.load_adapter(os.path.join(split_root, name),
                           adapter_name=name,
                           is_trainable=False)

    # 4️⃣  fuse LoRA + DoRA → plain GPT-2
    fused = model.merge_and_unload().eval()

    # 5️⃣  save
    os.makedirs(out_dir, exist_ok=True)
    fused.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print("✓ fused model written to", out_dir)

if __name__ == "__main__":
    fuse_splitlora("./splitlora_checkpoint")      # path with head/body/tail
