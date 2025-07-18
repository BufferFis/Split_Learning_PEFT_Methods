# split_beam_wrapper.py  ──────────────────────────────────────────────
"""
Cache-aware wrapper that stitches the Head, Body and Tail parts of
a split GPT-2 back together and exposes a full `generate()` API.

Dependencies: torch ≥1.13, transformers ≥4.40
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from transformers import (
    PreTrainedModel,
    GenerationMixin
)

from transformers.modeling_outputs import CausalLMOutput 
class SplitGPT2ForGeneration(PreTrainedModel, GenerationMixin):
    """
    Bridges three separately-instantiated sub-models
    (HeadClient, ServerModel, TailClient) and presents them
    as a single causal-LM that can be trained or decoded
    with Hugging Face `generate`.
    """

    def __init__(
        self,
        tokenizer,
        head_client,
        server,
        tail_client,
        base_config,
        max_length: int = 256,
    ):
        super().__init__(base_config)

        # dummy parameter ⇒ lets `model.to(device)` move wrapper & cache
        self.register_parameter("dummy", nn.Parameter(torch.empty(0)))

        # references to the three split parts
        self.tokenizer = tokenizer
        self.head_client = head_client        # HeadModel + LoRA
        self.server = server                  # BodyModel + LoRA
        self.tail_client = tail_client        # TailModel + LoRA

        # expose lm_head so GenerationMixin can tie / resize embeddings
        self.lm_head = self.tail_client.tail_model.lm_head
        # weight tying (lm_head ↔ input embeddings)
        self.lm_head.weight = (
            self.head_client.head_model.base_model.wte.weight
        )

        # keep main generation attributes in the HF config
        self.config.vocab_size = self.lm_head.weight.size(0)
        self.config.pad_token_id = tokenizer.pad_token_id
        self.config.eos_token_id = tokenizer.eos_token_id
        self.config.max_length = max_length

    # ─────────────────────────────────────────────────────────
    # helper: one incremental forward pass, cache-aware
    # ─────────────────────────────────────────────────────────
    def _forward_step(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = True,
    ):
        """
        Parameters
        ----------
        input_ids : (B, T) tokens for the head part (usually last token)
        attention_mask : (B, S) cumulative mask (prompt + decoded)
        past_key_values : tuple(head, body, tail) from previous step
        use_cache : store & return new_past when True

        Returns
        -------
        logits : (B, T, V)
        new_past_key_values : same three-tuple structure or None
        """
        head_past, body_past, tail_past = (
            (None, None, None) if past_key_values is None else past_key_values
        )

        # ── HEAD ─────────────────────────────────────────────
        head_out = self.head_client.head_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=head_past,
            use_cache=use_cache,
        )

        # ── BODY ─────────────────────────────────────────────
        body_out = self.server.body_model(
            hidden_states=head_out.last_hidden_state,
            attention_mask=attention_mask,
            past_key_values=body_past,
            use_cache=use_cache,
        )

        # ── TAIL + lm_head ──────────────────────────────────
        tail_out = self.tail_client.tail_model(
            inputs_embeds=body_out.last_hidden_state,
            attention_mask=attention_mask,
            past_key_values=tail_past,
            use_cache=use_cache,
        )

        new_past = None
        if use_cache:
            new_past = (
                head_out.past_key_values,
                body_out.past_key_values,
                tail_out.past_key_values,
            )

        return tail_out.logits, new_past

    # ─────────────────────────────────────────────────────────
    # training / evaluation forward (no caching needed)
    # ─────────────────────────────────────────────────────────
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.BoolTensor] = None,
        **unused,
    ):
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id)

        # full-sequence pass (no cache)
        logits, _ = self._forward_step(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=False,
        )
        return CausalLMOutput(logits=logits)

    # ─────────────────────────────────────────────────────────
    # custom greedy / beam decoder using our cache
    # ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.BoolTensor] = None,
        max_new_tokens: int = 64,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        **kw,
    ):
        """Greedy decoding; easy to replace by your own beam."""
        eos_id = eos_token_id or self.config.eos_token_id
        pad_id = pad_token_id or self.config.pad_token_id

        if attention_mask is None:
            attention_mask = (input_ids != pad_id)

        # ── first pass: full prompt → build KV cache ────────
        logits, past = self._forward_step(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
        )

        # ── autoregressive loop ─────────────────────────────
        for _ in range(max_new_tokens):
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)  # greedy
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token, dtype=torch.bool)],
                dim=1,
            )

            if next_token.item() == eos_id:
                break

            # feed **only** the last token + cache
            logits, past = self._forward_step(
                input_ids=input_ids[:, -1:],      # (B, 1)
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )

        return input_ids

    # GenerationMixin hook (not used but keeps HF happy)
    def prepare_inputs_for_generation(self, input_ids, **kw):
        return {"input_ids": input_ids}

    def can_generate(self):  # transformers ≥4.39
        return True

    # optional: allow external code to fetch lm_head
    def get_output_embeddings(self):
        return self.lm_head
