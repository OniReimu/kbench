"""Decode-time Unlearning from Logit Difference (ULD).

ULD keeps the target model unchanged.  At each decode step this processor runs
the small assistant on the sequence generated so far and returns

    base_logits + weight * assistant_logits

matching ``ContrastLLM.forward`` in the ULD reference implementation.  A
negative weight therefore subtracts the forget-memorising assistant.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, LogitsProcessor


def combine_uld_logits(
    base_logits: torch.Tensor,
    assistant_logits: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    """Apply the reference ULD logit combination elementwise."""
    return base_logits + weight * assistant_logits


class ULDLogitsProcessor(LogitsProcessor):
    """Combine target and assistant logits on every generation step.

    The assistant KV cache is reused while the input extends the preceding
    prefix and reset automatically when a new prompt does not.  Its cache is
    only eight layers for the assistants used by this project.
    """

    def __init__(
        self,
        assistant_model: AutoModelForCausalLM,
        weight: float,
        top_logit_filter: float = 0.0,
    ) -> None:
        self.assistant_model = assistant_model
        self.weight = weight
        self.top_logit_filter = top_logit_filter
        self.calls = 0
        self._cached_input_ids: torch.Tensor | None = None
        self._past_key_values = None

    def _assistant_device(self) -> torch.device:
        return self.assistant_model.device

    def _relative_top_mask(self, base_logits: torch.Tensor) -> torch.Tensor:
        """Return the assistant mask used by ``ContrastLLM.relative_top_filter``."""
        normalized = base_logits.log_softmax(dim=-1)
        sorted_logits, _ = torch.sort(normalized, descending=True)
        min_tokens_to_keep = max(1, int(self.top_logit_filter * base_logits.shape[-1]))
        min_thresh = sorted_logits[..., min_tokens_to_keep - 1]
        probs_max = torch.max(normalized, dim=-1).values
        probs_thresh = torch.minimum(
            min_thresh,
            probs_max + np.log(self.top_logit_filter),
        )
        return normalized < probs_thresh.unsqueeze(-1)

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        assistant_input_ids = input_ids.to(self._assistant_device())
        cached_length = 0 if self._cached_input_ids is None else self._cached_input_ids.shape[1]
        extends_cached_prefix = (
            self._past_key_values is not None
            and cached_length < input_ids.shape[1]
            and torch.equal(
                self._cached_input_ids,
                input_ids[:, :cached_length].detach().cpu(),
            )
        )
        model_input_ids = (
            assistant_input_ids[:, cached_length:]
            if extends_cached_prefix
            else assistant_input_ids
        )
        past_key_values = self._past_key_values if extends_cached_prefix else None
        outputs = self.assistant_model(
            input_ids=model_input_ids,
            attention_mask=torch.ones_like(assistant_input_ids),
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        self._cached_input_ids = input_ids.detach().cpu().clone()
        self._past_key_values = outputs.past_key_values
        assistant_logits = outputs.logits[:, -1, :].to(
            device=scores.device,
            dtype=scores.dtype,
        )
        if assistant_logits.shape != scores.shape:
            raise ValueError(
                "ULD target/assistant next-token logits have different shapes: "
                f"{tuple(scores.shape)} != {tuple(assistant_logits.shape)}. "
                "The checkpoints must use the same tokenizer and vocabulary."
            )
        if self.top_logit_filter > 0.0:
            assistant_logits = assistant_logits.masked_fill(
                self._relative_top_mask(scores), 0.0
            )
        self.calls += 1
        return combine_uld_logits(scores, assistant_logits, self.weight)


def _parameter_gib(model: torch.nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3


def load_uld_logits_processor(
    base_model: AutoModelForCausalLM,
    assistant_path: Path,
    weight: float,
    top_logit_filter: float = 0.0,
    device: str = "auto",
    dtype: torch.dtype = torch.bfloat16,
) -> ULDLogitsProcessor:
    """Load the frozen assistant and report the two-model weight-memory floor."""
    assistant_model = AutoModelForCausalLM.from_pretrained(
        str(assistant_path), torch_dtype=dtype, device_map=device
    )
    assistant_model.eval()

    base_vocab = getattr(base_model.config, "vocab_size", None)
    assistant_vocab = getattr(assistant_model.config, "vocab_size", None)
    if base_vocab != assistant_vocab:
        raise ValueError(
            "ULD target and assistant vocabularies differ: "
            f"{base_vocab} != {assistant_vocab}"
        )

    base_gib = _parameter_gib(base_model)
    assistant_gib = _parameter_gib(assistant_model)
    print(
        "[uld] model-weight memory floor: "
        f"target={base_gib:.2f} GiB + assistant={assistant_gib:.2f} GiB "
        f"= {base_gib + assistant_gib:.2f} GiB; "
        "both KV caches and transient activations are additional"
    )
    return ULDLogitsProcessor(
        assistant_model=assistant_model,
        weight=weight,
        top_logit_filter=top_logit_filter,
    )
