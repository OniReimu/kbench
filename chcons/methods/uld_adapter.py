"""ULD adapter — Unlearning as Logit-Difference (Ji et al., NeurIPS'24).

Mechanism (decode-time, dual-model, NO weight-modified target):
  - the TARGET stays FROZEN (agent.model);
  - a small 8-layer ASSISTANT (copied from target's first 8 layers + LoRA,
    trained to MEMORIZE the forget content) is loaded alongside;
  - at every decode step the final next-token logits are
        final = base_logits + weight * assist_logits        (weight = -0.8)
    after a relative-top filter (top_logit_filter = 1e-2) that only contrasts
    positions where the base is confident. Subtracting the assistant suppresses
    the leaked forget answer while leaving retain ~untouched.

This is ULD's `ContrastLLM` decode (uld/model/contrastllm.py + gen_util.py).
The upstream `ContrastLLM.generate()` is hardwired to transformers-4.38
generation internals (`_get_logits_processor`, `greedy_search`,
`_update_model_kwargs_for_generation`) that no longer exist in the chcons
eval venv (transformers 5.7). So instead of importing that fragile generate(),
we reimplement the SAME per-step logit-difference math (gen_util.py lines
332-348, byte-faithful) on a stable greedy KV-cached loop, and wire it in by
replacing `agent.model.generate` with a ContrastLLM-backed callable that
matches the exact call contract the agent uses (both `_generate_block` in the
ReAct loop and `elicit_summary`):

    output_ids = agent.model.generate(
        **inputs,                 # input_ids, attention_mask
        max_new_tokens=...,
        do_sample=False, temperature=1.0, top_p=1.0,
        pad_token_id=...,
        logits_processor=LogitsProcessorList([...]) or None,
    )
    new = output_ids[0, input_len:]          # agent slices off the prompt

setup() installs the monkeypatch; teardown() restores the original bound
method. This is the ONLY thing that changes generation, and ONLY when
--unlearn uld. The default/other-method generation path is untouched.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from chcons.methods import UnlearnIntervention

# ULD eval-time hyperparameters (bashes/tofu/uld_train.sh eval block:
#   model_mode.weight=-0.8  model_mode.top_logit_filter=1e-2)
_DEFAULT_WEIGHT = -0.8
_DEFAULT_TOP_LOGIT_FILTER = 1e-2


def _relative_top_filter(scores: torch.FloatTensor, relative_top: float):
    """Byte-faithful port of ContrastGenerationMixin.relative_top_filter
    (uld/model/gen_util.py L208-222). Returns (scores, mask) where `mask`
    marks positions BELOW the relative-top threshold (i.e. positions to drop).
    """
    min_tokens_to_keep = int(relative_top * scores.shape[-1])
    scores_normalized = scores.log_softmax(dim=-1)
    sorted_logits, _ = torch.sort(scores_normalized, descending=True)
    min_thresh = sorted_logits[..., min_tokens_to_keep - 1]
    probs_max = torch.max(scores_normalized, dim=-1).values
    probs_thresh = probs_max + np.log(relative_top)
    probs_thresh = torch.min(min_thresh, probs_thresh)
    probs_thresh = probs_thresh.unsqueeze(-1)
    mask = scores_normalized < probs_thresh
    return scores, mask


class ULDIntervention(UnlearnIntervention):
    """ULD (Ji NeurIPS'24): decode-time logit-difference vs a memorized assistant."""

    @classmethod
    def name(cls) -> str:
        return "uld"

    def __init__(
        self,
        assistant_path: str | None = None,
        weight: float = _DEFAULT_WEIGHT,
        top_logit_filter: float = _DEFAULT_TOP_LOGIT_FILTER,
    ):
        # Assistant path: explicit arg > env ULD_ASSISTANT_PATH.
        self.assistant_path = assistant_path or os.environ.get("ULD_ASSISTANT_PATH")
        self.weight = weight
        self.top_logit_filter = top_logit_filter
        self._assist = None
        self._orig_generate = None  # saved bound method for restore

    # ---- lifecycle ----------------------------------------------------------

    def setup(self, agent, lora_path, forget_ids, facts_path) -> None:
        if not self.assistant_path:
            raise SystemExit(
                "[uld] no assistant path: set ULD_ASSISTANT_PATH=<uld_assistant_seedN> "
                "(the trained 8-layer assistant) before running --unlearn uld."
            )
        ap = Path(self.assistant_path)
        if not ap.exists():
            raise SystemExit(f"[uld] assistant path does not exist: {ap}")

        base = agent.model
        device = base.device
        dtype = next(base.parameters()).dtype
        print(f"[uld] loading 8-layer assistant from {ap} (device={device}, dtype={dtype})")
        self._assist = AutoModelForCausalLM.from_pretrained(
            str(ap), torch_dtype=dtype, attn_implementation="eager"
        ).to(device)
        self._assist.eval()

        # sanity: vocab must match so token-ids align between base and assistant
        bv = base.config.vocab_size
        av = self._assist.config.vocab_size
        if bv != av:
            raise SystemExit(
                f"[uld] vocab mismatch: base={bv} assistant={av} — assistant must "
                f"be copied from the same target (same tokenizer/vocab)."
            )
        print(f"[uld] setup OK: weight={self.weight} top_logit_filter={self.top_logit_filter} "
              f"vocab={bv} assist_layers={self._assist.config.num_hidden_layers}")

        # Install the ContrastLLM-backed generate. Save the original bound method
        # so teardown() can restore the unmodified generation path exactly.
        self._orig_generate = base.generate
        base.generate = self._make_contrast_generate(base, self._assist)
        print("[uld] monkeypatched agent.model.generate -> ContrastLLM logit-difference decode")

    def teardown(self) -> None:
        # Restore the original generate so no state leaks beyond this run.
        # (The eval process is single-method, but restore defensively.)
        if self._orig_generate is not None:
            # the saved object is the original bound method
            self._orig_generate.__self__.generate = self._orig_generate
            print("[uld] restored original agent.model.generate")
        self._orig_generate = None
        self._assist = None

    # before/after/per-query lifecycle: ULD is a GLOBAL decode contrast (no
    # per-query hooks, no prompt-length-dependent masks) — all no-ops.

    def summary_dict(self) -> dict:
        return {
            "method": "uld",
            "assistant_path": str(self.assistant_path),
            "weight": self.weight,
            "top_logit_filter": self.top_logit_filter,
            "decode": "logit_difference_greedy_kvcache",
        }

    # ---- the ContrastLLM decode --------------------------------------------

    def _make_contrast_generate(self, base, assist):
        """Build a drop-in `generate` matching the agent's call contract.

        Greedy, batch_size=1 (the agent always generates one prompt at a time),
        KV-cached for both models. Reimplements ULD's per-step logit difference
        (gen_util.py L332-348) under modern transformers.
        """
        weight = self.weight
        top_filter = self.top_logit_filter

        @torch.no_grad()
        def contrast_generate(
            input_ids=None,
            attention_mask=None,
            max_new_tokens=256,
            do_sample=False,           # accepted + ignored (ULD eval is greedy)
            temperature=1.0,           # accepted + ignored
            top_p=1.0,                 # accepted + ignored
            pad_token_id=None,
            eos_token_id=None,
            logits_processor=None,     # applied to the FINAL contrasted logits
            **_ignored,
        ):
            device = base.device
            if input_ids is None:
                raise ValueError("[uld] contrast_generate requires input_ids")
            input_ids = input_ids.to(device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            else:
                attention_mask = torch.ones_like(input_ids)

            eos_ids = eos_token_id
            if eos_ids is None:
                eos_ids = base.generation_config.eos_token_id
            if eos_ids is None:
                eos_ids = []
            if isinstance(eos_ids, int):
                eos_ids = [eos_ids]
            eos_set = set(int(e) for e in eos_ids)
            if pad_token_id is None:
                pad_token_id = base.generation_config.pad_token_id
                if pad_token_id is None and eos_set:
                    pad_token_id = next(iter(eos_set))

            generated = input_ids
            full_attn = attention_mask

            # Prefill both models over the full prompt to seed KV caches.
            base_out = base(
                input_ids=generated, attention_mask=full_attn,
                use_cache=True, return_dict=True,
            )
            assist_out = assist(
                input_ids=generated, attention_mask=full_attn,
                use_cache=True, return_dict=True,
            )
            base_past = base_out.past_key_values
            assist_past = assist_out.past_key_values
            base_last = base_out.logits[:, -1, :]
            assist_last = assist_out.logits[:, -1, :].to(base_last.device)

            for _ in range(max_new_tokens):
                # ---- ULD per-step logit difference (gen_util.py L332-348) ----
                if top_filter > 0.0:
                    next_logits, mask = _relative_top_filter(base_last, top_filter)
                    assist_masked = assist_last.clone()
                    assist_masked[mask] = 0
                    contrasted = next_logits + weight * assist_masked
                    contrasted[mask] = -1e3
                else:
                    next_logits = base_last.log_softmax(dim=-1)
                    al = assist_last.log_softmax(dim=-1)
                    contrasted = next_logits + weight * al
                    mask = None

                # Optional logits_processor (e.g. STaR/noise) on contrasted logits.
                if logits_processor is not None:
                    contrasted = logits_processor(generated, contrasted)

                # argmax — within the kept (~mask) region when filtering.
                if mask is not None:
                    next_token = torch.argmax(contrasted * (~mask), dim=-1)
                else:
                    next_token = torch.argmax(contrasted, dim=-1)
                next_token = next_token.view(-1)  # [1]

                generated = torch.cat([generated, next_token[:, None]], dim=-1)
                full_attn = torch.cat(
                    [full_attn, torch.ones_like(next_token[:, None])], dim=-1
                )

                if int(next_token.item()) in eos_set:
                    break

                # Incremental step: feed only the new token + grown cache.
                step_ids = next_token[:, None]
                base_out = base(
                    input_ids=step_ids, attention_mask=full_attn,
                    past_key_values=base_past, use_cache=True, return_dict=True,
                )
                assist_ids = step_ids.to(assist.device)
                assist_out = assist(
                    input_ids=assist_ids, attention_mask=full_attn.to(assist.device),
                    past_key_values=assist_past, use_cache=True, return_dict=True,
                )
                base_past = base_out.past_key_values
                assist_past = assist_out.past_key_values
                base_last = base_out.logits[:, -1, :]
                assist_last = assist_out.logits[:, -1, :].to(base_last.device)

            return generated

        return contrast_generate
