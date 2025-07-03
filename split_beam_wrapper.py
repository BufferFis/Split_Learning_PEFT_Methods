# split_beam_wrapper.py  ────────────────────────────────────────────────
import torch, torch.nn as nn
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutput
from typing import Optional

class SplitGPT2ForGeneration(PreTrainedModel, GenerationMixin):
    def __init__(self, tokenizer,
                 head_client, server, tail_client,
                 base_config):
        super().__init__(base_config)

        # --- 0.  dummy parameter so .device is defined -----------------
        self.dummy = nn.Parameter(torch.empty(0), requires_grad=False)

        # --- 1.  keep handles to the three split parts -----------------
        self.tokenizer   = tokenizer
        self.head_client = head_client
        self.server      = server
        self.tail_client = tail_client

        # --- 2.  make HF aware of the LM-head / embeddings -------------
        # 2-a  expose the lm_head itself
        self.lm_head = tail_client.tail_model.lm_head
        self.lm_head.weight = (
            head_client.head_model.base_model.wte.weight
        )
        # 2-b  tie vocab-size to the actual weight matrix
        vocab = self.lm_head.weight.size(0)
        self.config.vocab_size              = vocab                   
        self.generation_config.vocab_size   = vocab                   
        self.config.pad_token_id            = tokenizer.pad_token_id
        self.config.eos_token_id            = tokenizer.eos_token_id

    # GenerationMixin hook
    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kw):
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def can_generate(self):                       # required by ≥4.39
        return True

    # ---------- standard forward ---------------------------------------
    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask: Optional[torch.FloatTensor] = None,
                **ignored):
        with torch.no_grad():
            h = self.head_client.forward(input_ids, attention_mask)
            b = self.server.forward(h, attention_mask)  # Pass attention_mask
            logits = self.tail_client.forward(b, attention_mask)  # Pass attention_mask

        return CausalLMOutput(logits=logits)

    # optional: if you ever want to swap lm_head
    def get_output_embeddings(self):
        return self.lm_head                                           # NEW
