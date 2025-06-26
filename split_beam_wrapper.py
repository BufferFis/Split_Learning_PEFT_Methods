# split_beam_wrapper.py
import torch
import torch.nn as nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from typing import Optional

class SplitGPT2ForGeneration(PreTrainedModel, GenerationMixin):
    """
    Thin façade that turns (HeadClient, ServerModel, TailClient)
    into a regular Causal-LM usable by HF `generate()`.
    Nothing is fused – the three PEFT modules stay intact.
    """
    def __init__(self, tokenizer,
                 head_client, server, tail_client,
                 base_config):
        super().__init__(base_config)
        self.dummy = nn.Parameter(torch.empty(0), requires_grad=False)
        self.tokenizer = tokenizer
        self.head_client = head_client
        self.server      = server
        self.tail_client = tail_client
        # expose pad/eos so that generate() can build causal masks
        self.config.pad_token_id  = tokenizer.pad_token_id
        self.config.eos_token_id  = tokenizer.eos_token_id
        self.config.vocab_size    = tokenizer.vocab_size
        self.generation_config.vocab_size = tokenizer.vocab_size

    # -------- GenerationMixin hooks -----------------------------
    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      attention_mask=None,
                                      **kwargs):
        return {"input_ids": input_ids,
                "attention_mask": attention_mask}

    def can_generate(self):
        # required by very new HF versions
        return True

    # -------- standard forward pass -----------------------------
    def forward(self,
                input_ids:      torch.LongTensor,
                attention_mask: Optional[torch.FloatTensor] = None,
                **ignored):
        # no gradients during inference
        with torch.no_grad():
            head_acts = self.head_client.forward(
                input_ids, attention_mask)

            body_acts = self.server.forward(head_acts,
                                            attention_mask)

            logits    = self.tail_client.forward(
                body_acts, attention_mask)

        # GenerationMixin only needs `logits`
        return CausalLMOutput(logits=logits)
