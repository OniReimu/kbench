"""RepE adapter — Representation Engineering (Zou et al. NeurIPS 2023).

Multi-layer linear concept removal via contrast-direction subtraction. For each
target layer, compute the contrast direction `v = mean(X_forget) - mean(X_retain)`,
normalize to unit vector v̂, and at inference subtract its projection from hidden
states: `h' = h - α * (h·v̂) * v̂`. This is a simpler, rank-1 sibling of LEACE
(no whitening, no closed-form least-squares).

Differences from LEACE (Belrose 2023):
- LEACE: closed-form `Σ_xx^{1/2} · U · U^T · Σ_xx^{-1/2}` (whitening-based,
  preserves marginals under the M-Mahalanobis norm).
- RepE: simple mean-difference unit-vector projection (no whitening). More
  aggressive in geometry but easier to interpret and faster to fit.

The K-test question for RepE is: does subtracting a SINGLE LINEAR DIRECTION
from representations propagate through all channels (REFUTING K) or only
suppress the channel that depends most directly on that direction (SUPPORTING K)?

Multi-layer: list of decoder layer indices. Each layer gets its own direction
computed from its own X_forget / X_retain means.

Implementation matches the post-fix LEACE adapter discipline:
- D7 split (disjoint fit/eval pools, A.6 protocol)
- ReAct chat-templated fit prompts (matches inference distribution)
- All token positions captured (matches hook's full-sequence apply)
- Streaming mean computation (memory-bounded)
- `add_special_tokens=False` (matches `agent._generate_block`)
"""

from __future__ import annotations

from pathlib import Path

import torch

from chcons.methods import UnlearnIntervention


class RepEIntervention(UnlearnIntervention):
    """RepE (Zou NeurIPS'23): multi-layer mean-difference linear projection."""

    @classmethod
    def name(cls) -> str:
        return "repe"

    def __init__(
        self,
        target_layer_indices: list[int] | None = None,
        alpha: float = 1.0,
        n_fit_samples: int = 200,
    ):
        # Default: single mid-layer (16) for direct parity with LEACE default.
        # Multi-layer config: pass e.g. [8, 16, 24] for 3-layer intervention.
        self.target_layer_indices = list(target_layer_indices) if target_layer_indices else [16]
        self.alpha = alpha
        self.n_fit_samples = n_fit_samples
        self._directions: dict[int, torch.Tensor] = {}  # layer_idx → unit vector [D]
        self._hook_handles: list = []
        self._target_layers: list = []

    def setup(self, agent, lora_path, forget_ids, facts_path):
        from chcons.pii import read_jsonl, QUERY_TEMPLATES, load_split_ids

        # D7 split (protocol A.6) — fit on adapter pool, eval on disjoint pool.
        all_recs = read_jsonl(facts_path)
        forget_adapter_ids = load_split_ids("forget", "adapter")
        retain_adapter_ids = load_split_ids("retain", "adapter")
        forget_recs = [r for r in all_recs if r.id in forget_adapter_ids]
        retain_recs = [r for r in all_recs if r.id in retain_adapter_ids]
        if not forget_recs or not retain_recs:
            raise RuntimeError(
                "RepE: empty forget_adapter or retain_adapter pool. Check "
                "data/pii_facts/forget_ids_adapter.txt + retain_ids_adapter.txt."
            )
        print(
            f"[repe] D7 split: fit on {len(forget_recs)} forget_adapter + "
            f"{len(retain_recs)} retain_adapter records (disjoint from eval pool)"
        )

        rng = torch.Generator().manual_seed(0)
        f_idx = torch.randperm(len(forget_recs), generator=rng)[: self.n_fit_samples].tolist()
        r_idx = torch.randperm(len(retain_recs), generator=rng)[: self.n_fit_samples].tolist()
        forget_examples = self._build_query_examples(
            [forget_recs[i] for i in f_idx], QUERY_TEMPLATES, agent=agent
        )
        retain_examples = self._build_query_examples(
            [retain_recs[i] for i in r_idx], QUERY_TEMPLATES, agent=agent
        )
        print(
            f"[repe] fit data: {len(forget_examples)} forget + "
            f"{len(retain_examples)} retain prompts (ReAct-format)"
        )

        # Locate target decoder layers; validate bounds.
        layers = self._get_decoder_layers(agent.model)
        for idx in self.target_layer_indices:
            if not (-len(layers) <= idx < len(layers)):
                raise ValueError(
                    f"RepE target_layer_idx={idx} out of range "
                    f"for {len(layers)}-layer model (valid: [-{len(layers)}, {len(layers) - 1}])."
                )
        resolved_indices = [
            idx if idx >= 0 else len(layers) + idx for idx in self.target_layer_indices
        ]
        self._target_layers = [layers[i] for i in resolved_indices]
        cfg = agent.model.config
        hidden_dim = cfg.hidden_size if hasattr(cfg, "hidden_size") else cfg.text_config.hidden_size
        print(
            f"[repe] target layers: {resolved_indices} (of {len(layers)}), "
            f"hidden_dim={hidden_dim}, alpha={self.alpha}"
        )

        # Stream mean computation per layer.
        # For each layer, accumulate (sum_x, count) over forget prompts and over retain prompts.
        forget_sums: dict[int, torch.Tensor] = {}
        retain_sums: dict[int, torch.Tensor] = {}
        forget_counts: dict[int, int] = {}
        retain_counts: dict[int, int] = {}
        for idx, layer in zip(resolved_indices, self._target_layers):
            device = next(layer.parameters()).device
            forget_sums[idx] = torch.zeros(hidden_dim, dtype=torch.float32, device=device)
            retain_sums[idx] = torch.zeros(hidden_dim, dtype=torch.float32, device=device)
            forget_counts[idx] = 0
            retain_counts[idx] = 0

        # Single forward pass per prompt; install temporary capture hooks on ALL target layers.
        def make_capture(idx_, layer_, target_sums, target_counts):
            def _capture(_m, _inp, output):
                h = output[0] if isinstance(output, tuple) else output
                # h shape [batch=1, seq_len, hidden] → sum over seq dim → [hidden]
                h32 = h[0].float()
                target_sums[idx_] += h32.sum(dim=0)
                target_counts[idx_] += h32.shape[0]
            return _capture

        # Forget pass
        capture_hooks = []
        for idx, layer in zip(resolved_indices, self._target_layers):
            h = layer.register_forward_hook(make_capture(idx, layer, forget_sums, forget_counts))
            capture_hooks.append(h)
        try:
            agent.model.eval()
            with torch.no_grad():
                for i, prompt in enumerate(forget_examples):
                    ids = agent.tokenizer(
                        prompt, return_tensors="pt", add_special_tokens=False
                    ).to(agent.model.device)
                    _ = agent.model(input_ids=ids["input_ids"], attention_mask=ids["attention_mask"])
                    if (i + 1) % 50 == 0:
                        torch.cuda.empty_cache()
        finally:
            for h in capture_hooks:
                h.remove()

        # Retain pass (different target dicts)
        capture_hooks = []
        for idx, layer in zip(resolved_indices, self._target_layers):
            h = layer.register_forward_hook(make_capture(idx, layer, retain_sums, retain_counts))
            capture_hooks.append(h)
        try:
            with torch.no_grad():
                for i, prompt in enumerate(retain_examples):
                    ids = agent.tokenizer(
                        prompt, return_tensors="pt", add_special_tokens=False
                    ).to(agent.model.device)
                    _ = agent.model(input_ids=ids["input_ids"], attention_mask=ids["attention_mask"])
                    if (i + 1) % 50 == 0:
                        torch.cuda.empty_cache()
        finally:
            for h in capture_hooks:
                h.remove()

        # Compute direction per layer = mean(forget) - mean(retain), unit-normalized
        for idx in resolved_indices:
            mean_f = forget_sums[idx] / max(forget_counts[idx], 1)
            mean_r = retain_sums[idx] / max(retain_counts[idx], 1)
            v = mean_f - mean_r
            norm = v.norm()
            if norm < 1e-8:
                print(f"[repe] WARN: layer {idx} direction near-zero (norm={norm:.2e}); skip")
                continue
            v_unit = v / norm
            self._directions[idx] = v_unit
            print(
                f"[repe] layer {idx}: forget_tokens={forget_counts[idx]} "
                f"retain_tokens={retain_counts[idx]} ||v||={norm:.4f}"
            )

        # Install persistent erase hooks (full-sequence projection subtraction).
        alpha = self.alpha
        for idx, layer in zip(resolved_indices, self._target_layers):
            if idx not in self._directions:
                continue
            v_unit = self._directions[idx]
            layer_dtype = next(layer.parameters()).dtype

            def make_hook(v_unit_, layer_dtype_):
                def _erase_hook(_m, _inp, output):
                    if isinstance(output, tuple):
                        h = output[0]
                    else:
                        h = output
                    h32 = h.float()
                    # Per-token projection coefficient: h @ v̂ → [B, seq, 1]
                    coef = (h32 @ v_unit_).unsqueeze(-1)
                    erased = (h32 - alpha * coef * v_unit_).to(layer_dtype_)
                    if isinstance(output, tuple):
                        return (erased,) + output[1:]
                    return erased
                return _erase_hook

            handle = layer.register_forward_hook(make_hook(v_unit, layer_dtype))
            self._hook_handles.append(handle)
            print(f"[repe] hook installed on layer {idx} (full-sequence subtract α·proj)")

    def teardown(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []
        self._directions = {}
        self._target_layers = []

    def summary_dict(self) -> dict:
        return {
            "method": "repe",
            "target_layer_indices": self.target_layer_indices,
            "alpha": self.alpha,
            "n_fit_samples": self.n_fit_samples,
        }

    # ---- shared helpers (parallel to LEACE adapter) ----

    @staticmethod
    def _get_decoder_layers(model):
        m = model
        for attr in ("base_model", "model", "model"):
            if hasattr(m, attr):
                inner = getattr(m, attr)
                if hasattr(inner, "layers"):
                    return inner.layers
                m = inner
        if hasattr(model, "get_decoder"):
            dec = model.get_decoder()
            if hasattr(dec, "layers"):
                return dec.layers
        raise RuntimeError("Could not locate decoder layers in model")

    @staticmethod
    def _build_query_examples(records, query_templates, agent=None) -> list[str]:
        """Build ReAct chat-templated prompts (matching inference distribution).

        Same shape as the post-fix LEACE adapter — see leace_adapter.py for
        the rationale (fit/apply distribution match is critical).
        """
        if agent is not None:
            from chcons.agent import (
                REACT_SYSTEM_TEMPLATE,
                REACT_SYSTEM_TEMPLATE_OPTIONAL_TOOLS,
                REACT_USER,
                _build_tools_section,
            )
            template = (
                REACT_SYSTEM_TEMPLATE_OPTIONAL_TOOLS
                if agent.allow_direct_answer
                else REACT_SYSTEM_TEMPLATE
            )
            system = (
                agent.secure_prefix
                + template.format(
                    max_iters=agent.max_iters,
                    tools_section=_build_tools_section(agent.available_tools),
                )
                + (("\n\n" + agent.incontext_pii_block) if agent.incontext_pii_block else "")
            )
        out: list[str] = []
        for r in records:
            d = r.to_dict()
            for field in ("date_of_birth", "address", "occupation", "employer"):
                if not d.get(field):
                    continue
                question = query_templates[field].format(name=r.name)
                if agent is None:
                    out.append(f"Q: {question}\nA:")
                else:
                    user = REACT_USER.format(question=question)
                    try:
                        messages = [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ]
                        prompt = agent.tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    except Exception as e:
                        if "system" not in str(e).lower():
                            raise
                        messages = [{"role": "user", "content": f"{system}\n\n{user}"}]
                        prompt = agent.tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    out.append(prompt)
                break
        return out
