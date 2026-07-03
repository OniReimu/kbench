"""DEPN adapter — closed-form weight intervention via privacy-neuron editing.

Faithful port of Wu et al. (EMNLP 2023, flamewei123/DEPN) integrated-gradients
algorithm to Llama causal LM + LoRA.

Original DEPN is BERT-MLM-only:
  - 1_calculate_attribution.py uses `model(... tmp_score=batch_weights ...)` where
    `tmp_score` is a custom argument added to BertForMaskedLM in custom_bert.py
    that SUBSTITUTES the FFN intermediate activation during forward.
  - Loss target is the gold token at [MASK] position (single-position MLM).
  - Integrated gradients are batched over `batch_size * num_batch` interpolation
    points (default 200) per (layer, sample).

Our port to Llama-3.1-8B + LoRA:
  - SAME core IG algorithm (scaled_input, ig = mean(grad at alpha*x) * x).
  - Substitution mechanism: PyTorch forward_pre_hook on each mlp.down_proj
    that returns the scaled-replacement tensor (no model monkey-patch needed).
  - Loss target: completion tokens only (causal LM next-token CE), masked via
    labels[:, :n_prompt] = -100.
  - Interpolation: n_steps=10 (vs DEPN's 200) to fit our compute budget. Each
    step does one forward+backward through ALL 32 layers simultaneously by
    installing all replacement hooks at once.
  - Helpers `scaled_input` and `convert_to_triplet_ig` are reproduced from
    external/depn/src/1_calculate_attribution.py with explicit attribution.

Editing step:
  - LoRA targets `down_proj` in our setup → zero column `neuron_idx` of
    `lora_A.weight` (and optionally `base_layer.weight` if `edit_lora_only=False`).
  - Default `edit_lora_only=True` because in our K-test setup PII is
    LoRA-injected, not pre-trained — editing base unnecessarily destroys
    instruction-following capability without affecting PII.

Lifecycle: pre-eval (one-shot), no per-query work.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import numpy as np

from chcons.methods import UnlearnIntervention, require_external

_DEPN_ROOT = Path(__file__).resolve().parents[3] / "external" / "depn"
if str(_DEPN_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_DEPN_ROOT / "src"))


# === Reproduced verbatim from external/depn/src/1_calculate_attribution.py ===
# Original: Wu et al. EMNLP'23, flamewei123/DEPN. Functions copied (not imported)
# because the source script does `from custom_bert import BertForMaskedLM` at
# module load, which fails outside BERT context. We keep the original logic
# byte-for-byte to preserve the IG algorithm's identity.

def scaled_input(emb, batch_size, num_batch):
    # emb: (1, ffn_size)
    baseline = torch.zeros_like(emb)  # (1, ffn_size)
    num_points = batch_size * num_batch
    step = (emb - baseline) / num_points  # (1, ffn_size)
    res = torch.cat(
        [torch.add(baseline, step * i) for i in range(num_points)], dim=0
    )  # (num_points, ffn_size)
    return res, step[0]


def convert_to_triplet_ig(ig_list):
    ig_triplet = []
    ig = np.array(ig_list)  # 12, 3072
    max_ig = ig.max()
    for i in range(ig.shape[0]):
        for j in range(ig.shape[1]):
            if ig[i][j] >= max_ig * 0.1:
                ig_triplet.append([i, j, ig[i][j]])
    return ig_triplet


# === End reproduced from DEPN ===


class DEPNIntervention(UnlearnIntervention):
    """DEPN (Wu EMNLP'23): faithful Integrated Gradients on FFN intermediate,
    edit top-K neurons via column zeroing of (lora_A and optionally base) weight."""

    @classmethod
    def name(cls) -> str:
        return "depn"

    def __init__(
        self,
        n_attribution_samples: int = 30,      # per-layer IG cost ~ N × 32 × steps; smaller N keeps it tractable
        n_ig_steps: int = 5,                  # IG interpolation points (DEPN uses 200; 5 enough at our setup)
        top_k_per_layer: int = 30,
        target_module: str = "down_proj",
        edit_lora_only: bool = True,
    ):
        self.n_attribution_samples = n_attribution_samples
        self.n_ig_steps = n_ig_steps
        self.top_k_per_layer = top_k_per_layer
        self.target_module = target_module
        self.edit_lora_only = edit_lora_only
        self._original_rows: dict[tuple, torch.Tensor] = {}
        self._zeroed_neurons: list[tuple[int, int]] = []

    def setup(self, agent, lora_path, forget_ids, facts_path):
        require_external("depn", _DEPN_ROOT)
        from chcons.pii import read_jsonl, QUERY_TEMPLATES

        forget = [r for r in read_jsonl(facts_path) if r.id in forget_ids]
        if not forget:
            raise RuntimeError(
                f"DEPN: empty forget sample. forget_ids has {len(forget_ids)} entries "
                f"but none match records in {facts_path}."
            )
        rng = torch.Generator().manual_seed(0)
        sample_idx = torch.randperm(len(forget), generator=rng)[: self.n_attribution_samples].tolist()
        sampled = [forget[i] for i in sample_idx]
        print(f"[depn] attribution sample: {len(sampled)} forget records (of {len(forget)})")

        layers = self._get_decoder_layers(agent.model)
        n_layers = len(layers)
        target_modules = []
        for i, layer in enumerate(layers):
            mod = self._get_attr_chain(layer, f"mlp.{self.target_module}")
            target_modules.append((i, mod))
        intermediate_size = self._down_proj_intermediate_dim(target_modules[0][1])
        print(f"[depn] target: {n_layers} layers × mlp.{self.target_module}, "
              f"intermediate_size={intermediate_size}, IG steps={self.n_ig_steps}")

        scores = torch.zeros(n_layers, intermediate_size, device=agent.model.device)
        agent.model.eval()
        for rec_i, rec in enumerate(sampled):
            d = rec.to_dict()
            for field in ("date_of_birth", "address", "occupation", "employer"):
                if not d.get(field):
                    continue
                question = QUERY_TEMPLATES[field].format(name=rec.name)
                completion = str(d[field])
                self._accumulate_ig(agent, target_modules, question, completion, scores)
            if (rec_i + 1) % 10 == 0:
                print(f"[depn] IG attribution: {rec_i+1}/{len(sampled)} done")

        # Top-K per layer
        all_scores = []
        for layer_idx in range(scores.shape[0]):
            top_vals, top_idx = torch.topk(scores[layer_idx], self.top_k_per_layer)
            for v, ni in zip(top_vals.tolist(), top_idx.tolist()):
                self._zeroed_neurons.append((layer_idx, ni))
                all_scores.append(v)
        print(f"[depn] picked top-{self.top_k_per_layer}/layer × {n_layers} layers "
              f"= {len(self._zeroed_neurons)} privacy neurons; "
              f"score range [{min(all_scores):.3g}, {max(all_scores):.3g}]")

        # Edit: zero columns
        n_base, n_lora = 0, 0
        with torch.no_grad():
            for layer_idx, neuron_idx in self._zeroed_neurons:
                _, mod = target_modules[layer_idx]
                if not self.edit_lora_only:
                    base = mod.base_layer if hasattr(mod, "base_layer") else mod
                    self._original_rows[(layer_idx, neuron_idx, "base")] = base.weight.data[:, neuron_idx].clone()
                    base.weight.data[:, neuron_idx] = 0
                    n_base += 1
                if hasattr(mod, "lora_A"):
                    for adapter_name, lora_a_mod in mod.lora_A.items():
                        self._original_rows[(layer_idx, neuron_idx, f"loraA:{adapter_name}")] = (
                            lora_a_mod.weight.data[:, neuron_idx].clone()
                        )
                        lora_a_mod.weight.data[:, neuron_idx] = 0
                        n_lora += 1
        print(f"[depn] zeroed {n_base} base + {n_lora} lora_A columns "
              f"in mlp.{self.target_module} (edit_lora_only={self.edit_lora_only})")

    def teardown(self) -> None:
        self._original_rows.clear()
        self._zeroed_neurons.clear()

    def summary_dict(self) -> dict:
        return {
            "method": "depn",
            "algorithm": "integrated_gradients",
            "n_attribution_samples": self.n_attribution_samples,
            "n_ig_steps": self.n_ig_steps,
            "top_k_per_layer": self.top_k_per_layer,
            "target_module": f"mlp.{self.target_module}",
            "edit_lora_only": self.edit_lora_only,
            "n_zeroed": len(self._zeroed_neurons),
        }

    # ---- internal helpers ----

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
    def _get_attr_chain(obj, dotted: str):
        for a in dotted.split("."):
            obj = getattr(obj, a)
        return obj

    @staticmethod
    def _down_proj_intermediate_dim(mod):
        """Return intermediate dim; works for both raw nn.Linear and PEFT-wrapped."""
        weight = (mod.base_layer.weight if hasattr(mod, "base_layer") else mod.weight)
        return weight.shape[1]

    def _accumulate_ig(self, agent, target_modules, question: str, completion: str,
                       scores: torch.Tensor) -> None:
        """Faithful integrated gradients for one (question, completion) pair.

        Algorithm (DEPN paper, eq. 3, faithful per-layer):
          IG_i(x_li) = x_li * (1/N) * sum_{k=1..N} d/dx_li F(x with alpha_k * x_li
                       substituted at layer li ONLY)

        Codex review caught a bug in the previous implementation that substituted
        all layers' activations in the same forward pass: the gradient at layer
        li was then computed in a context where downstream layers also had their
        activations clamped, violating the IG semantics. Fixed here by per-layer
        IG (outer loop over li, inner over alpha steps).

        Codex also caught a tokenizer-boundary bug (BPE may merge across "A:"
        and the first answer token, making n_prompt unreliable). Fixed by
        tokenizing prompt and answer separately + concatenating manually.

        Cost: n_layers * n_ig_steps forward+backward passes per (record, field).
        """
        # --- Tokenize prompt and answer separately for deterministic boundary ---
        # `add_special_tokens=True` for prompt → BOS prepended; False for answer
        # so we don't insert another BOS mid-sequence.
        prompt_part = agent.tokenizer(
            f"Q: {question}\nA:", return_tensors="pt", add_special_tokens=True
        )
        ans_part = agent.tokenizer(
            f" {completion}", return_tensors="pt", add_special_tokens=False
        )
        device = agent.model.device
        input_ids = torch.cat(
            [prompt_part["input_ids"], ans_part["input_ids"]], dim=1
        ).to(device)
        attention_mask = torch.cat(
            [prompt_part["attention_mask"], ans_part["attention_mask"]], dim=1
        ).to(device)
        n_prompt = prompt_part["input_ids"].shape[1]
        seq_len = input_ids.shape[1]
        # comp_slice indexes the completion tokens within the [seq_len-1] post-shift view:
        # labels[:, :n_prompt] = -100, so loss is over labels[n_prompt:seq_len], which after
        # `shift_labels = labels[..., 1:]` lives at positions [n_prompt-1 : seq_len-1] in the
        # shifted axis. Same indices apply to layer activations (pre-shift) for the IG values
        # we want, since IG values correspond to the activation that PRODUCED each token.
        if seq_len <= n_prompt:
            return  # nothing to attribute (empty completion)
        comp_slice = slice(n_prompt - 1, seq_len - 1)
        embed_layer = agent.model.get_input_embeddings()

        # --- 1. Capture x_li at each target layer (one forward, no grad) ---
        captured: dict[int, torch.Tensor] = {}
        cap_handles = []
        for li, mod in target_modules:
            def _cap(_m, inp, _out, idx=li):
                captured[idx] = inp[0].detach()
            cap_handles.append(mod.register_forward_hook(_cap))
        try:
            with torch.no_grad():
                _ = agent.model(
                    inputs_embeds=embed_layer(input_ids),
                    attention_mask=attention_mask,
                )
        finally:
            for h in cap_handles:
                h.remove()

        # --- 2. Per-layer IG: outer loop over li, inner over alpha steps ---
        # Each (li, alpha) does ONE forward+backward with substitution hook ONLY on
        # layer li → gradient at li reflects layer li's IG faithfully (DEPN semantics).
        for li, mod in target_modules:
            if li not in captured:
                continue
            x_li = captured[li]
            sum_grad = None

            for step_i in range(1, self.n_ig_steps + 1):
                alpha = step_i / self.n_ig_steps
                replacement = (alpha * x_li).detach().requires_grad_(True)

                def _sub(_m, inp, sub=replacement):
                    return (sub,) + tuple(inp[1:])

                sub_handle = mod.register_forward_pre_hook(_sub)
                try:
                    agent.model.zero_grad(set_to_none=True)
                    inputs_embeds = embed_layer(input_ids).detach().requires_grad_(True)
                    out = agent.model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                    )
                    labels = input_ids.clone()
                    labels[:, :n_prompt] = -100
                    shift_logits = out.logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
                    loss.backward()

                    if replacement.grad is not None:
                        g = replacement.grad.detach()
                        sum_grad = g.clone() if sum_grad is None else sum_grad + g
                finally:
                    sub_handle.remove()
                    agent.model.zero_grad(set_to_none=True)

            if sum_grad is None:
                continue
            mean_grad = sum_grad / self.n_ig_steps
            ig = mean_grad * x_li                            # [1, seq, intermediate]
            score_contrib = ig[:, comp_slice, :].abs().mean(dim=(0, 1))
            scores[li] += score_contrib.to(scores.device)
