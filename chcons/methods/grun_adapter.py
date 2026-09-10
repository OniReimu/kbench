"""GRUN adapter — Gated Representation UNlearning (Ren et al., ACL 2025 Findings).

arXiv 2502.17823, renjie3/GRUN. Rail-A inference-time method: the base target is
FROZEN; a tiny gated-ReFT suppression module (<0.05% params) is trained offline
and applied at inference. For each gated layer L, a per-token gate decides how
strongly a low-rank ReFT edit rewrites the layer's block_output representation,
suppressing forget-related generation while leaving retain queries (gate~0)
untouched.

The published implementation (ports/GRUN, GatedLoreftIntervention_Linear) applies,
per token h, the residual edit:

    output(h) = h + R^T( W*h + b - R*h ) * sigmoid(g*h + b_g)

where, in the saved per-layer state_dict (intkey_layer.<L>....bin):
  - rotate_layer  R : [embed_dim, rank]   (orthogonal columns; R*h = h @ R)
  - weight        W : [rank, embed_dim]   (learned_source.weight; W*h = h @ W.T)
  - bias          b : [rank]              (learned_source.bias)
  - gate_func.weight g : [1, embed_dim]   (g*h = h @ g.T)
  - gate_func.bias b_g : [1]
  - act_fn = linear (identity) for the _Linear variant.
Matching the published forward exactly:
    rotated_base = h @ R                      # [..., rank]
    learned      = h @ W.T + b                # [..., rank]
    gate         = sigmoid(h @ g.T + b_g)     # [..., 1]
    output       = h + (learned - rotated_base) @ R.T * gate

Integration for the K-test (Rail-A, like SPUL/ULD):
  - The reft artifact dir (config.json + per-layer intkey_*.bin) is trained offline
    by ports/GRUN/run_v71_grun_train.pbs from the frozen merged target, per seed.
  - setup() loads R/W/b/g for each gated layer and installs a persistent
    forward-hook on that decoder layer's block output. The gate is input-conditional
    (per token), so the edit cannot be baked into static weights -- but it IS a plain
    transformers forward hook (NO pyvene/pyreft needed at eval time).
  - The artifact path is read from env var GRUN_REFT_PATH (per-seed selection),
    mirroring SPUL_ADAPTER_PATH / ULD_ASSISTANT_PATH -- no new CLI flag.

The gate (sigmoid term) is kept verbatim: it is what makes GRUN selective. Dropping
it would change the method into an always-on LoReFT edit.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch

from chcons.methods import UnlearnIntervention

_LAYER_RE = re.compile(r"layer\.(\d+)\.")


class GRUNIntervention(UnlearnIntervention):
    """GRUN (Ren ACL'25 Findings): always-on per-token gated low-rank ReFT edit
    on decoder block outputs, reimplemented as plain transformers forward hooks."""

    @classmethod
    def name(cls) -> str:
        return "grun"

    def __init__(self, reft_path: str | None = None) -> None:
        # Artifact dir resolution order: explicit arg -> env GRUN_REFT_PATH.
        self._reft_path = reft_path
        self._params: dict[int, dict[str, torch.Tensor]] = {}  # layer_idx -> {R,W,b,g,bg}
        self._hook_handles: list = []
        self._target_layers: list = []
        self._prompt_len = None  # set per-generation; restricts edit to last prompt token

    def setup(self, agent, lora_path, forget_ids, facts_path):
        reft_path = self._reft_path or os.environ.get("GRUN_REFT_PATH")
        if not reft_path:
            raise SystemExit(
                "[grun] GRUN_REFT_PATH env var is required (path to the trained "
                "gated-ReFT artifact dir, e.g. ports/GRUN/results/v71_grun_seed0)."
            )
        reft_dir = Path(reft_path).expanduser()
        cfg_path = reft_dir / "config.json"
        if not cfg_path.is_file():
            raise SystemExit(f"[grun] config.json not found in artifact dir: {reft_dir}")
        cfg = json.loads(cfg_path.read_text())

        # Each sorted_keys[i] is 'layer.<L>.comp.block_output...' -> the per-layer
        # state_dict lives in 'intkey_<sorted_key>.bin'. Parse the layer index and
        # load R/W/b/g for that layer.
        sorted_keys = cfg.get("sorted_keys")
        if not sorted_keys:
            raise SystemExit(f"[grun] no 'sorted_keys' in {cfg_path}")
        itypes = cfg.get("intervention_types", [])
        if itypes and not all("GatedLoreftIntervention_Linear" in t for t in itypes):
            print(f"[grun] WARN: artifact intervention_types={itypes} -- this adapter "
                  f"implements the _Linear (identity act_fn) gated forward only.")

        # Resolve decoder layers and per-layer device/dtype.
        layers = self._get_decoder_layers(agent.model)
        n_layers = len(layers)

        for key in sorted_keys:
            m = _LAYER_RE.search(key)
            if not m:
                raise SystemExit(f"[grun] cannot parse layer index from sorted key {key!r}")
            layer_idx = int(m.group(1))
            if not (0 <= layer_idx < n_layers):
                raise SystemExit(
                    f"[grun] artifact layer {layer_idx} out of range for "
                    f"{n_layers}-layer model"
                )
            bin_path = reft_dir / f"intkey_{key}.bin"
            if not bin_path.is_file():
                raise SystemExit(f"[grun] missing per-layer artifact: {bin_path}")
            sd = torch.load(bin_path, map_location="cpu", weights_only=False)
            try:
                R = sd["rotate_layer"]          # [embed_dim, rank]
                W = sd["weight"]                # [rank, embed_dim]  (learned_source.weight)
                b = sd["bias"]                  # [rank]
                g = sd["gate_func.weight"]      # [1, embed_dim]
                bg = sd["gate_func.bias"]       # [1]
            except KeyError as e:
                raise SystemExit(
                    f"[grun] artifact {bin_path} missing tensor {e}; keys present: "
                    f"{list(sd.keys())}"
                )
            layer = layers[layer_idx]
            dev = next(layer.parameters()).device
            # Keep edit math in float32 for numerical parity with training, then
            # cast the residual back to the layer dtype at apply time.
            self._params[layer_idx] = {
                "R": R.to(device=dev, dtype=torch.float32),
                "W": W.to(device=dev, dtype=torch.float32),
                "b": b.to(device=dev, dtype=torch.float32),
                "g": g.to(device=dev, dtype=torch.float32),
                "bg": bg.to(device=dev, dtype=torch.float32),
            }
            print(f"[grun] layer {layer_idx}: loaded R{tuple(R.shape)} W{tuple(W.shape)} "
                  f"g{tuple(g.shape)} from {bin_path.name} on {dev}")

        if not self._params:
            raise SystemExit("[grun] no layers loaded from artifact")

        # Install persistent gated-ReFT hooks. CRITICAL: the published GRUN
        # (make_last_position_supervised_tofu_data_module / eval_dataloader,
        # pyreft/dataset.py) trains AND applies the gated-ReFT edit at EXACTLY ONE
        # position -- intervention_locations = [[base_prompt_length - 1]] -- i.e. the
        # last prompt token only, never the generated tokens. The gate weights were
        # only ever optimised to be selective at that single position; applying the
        # edit at every token (the original always-on hook) corrupts every
        # representation on both forget AND retain queries (||edit|| ~ 0.5-3x ||h||
        # everywhere) -> 100% parse_error / collapse. We therefore restrict the hook
        # to fire on the last position of the PREFILL pass only, matching pyvene
        # unit_locations exactly. self._prompt_len is set per-generation in
        # before_generation(); decode steps (seq_len == 1, KV-cached) are skipped.
        for layer_idx, p in self._params.items():
            layer = layers[layer_idx]
            layer_dtype = next(layer.parameters()).dtype
            self._target_layers.append(layer)
            handle = layer.register_forward_hook(self._make_hook(p, layer_dtype))
            self._hook_handles.append(handle)
        print(f"[grun] {len(self._hook_handles)} gated-ReFT hooks installed on layers "
              f"{sorted(self._params)} (edit restricted to last prompt token only, "
              f"matching pyreft last-position locations; Rail-A, weights frozen)")

    def before_generation(self, agent, prompt_text: str) -> None:
        # Record the prompt length for THIS generation so the hook edits only the
        # last prompt token (GRUN trained location base_prompt_length-1). Mirrors how
        # pyreft rebuilds intervention_locations per example.
        ids = agent.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        self._prompt_len = len(ids)

    def after_generation(self, agent) -> None:
        self._prompt_len = None

    def _make_hook(self, p, layer_dtype):
        R, W, b, g, bg = p["R"], p["W"], p["b"], p["g"], p["bg"]

        def _hook(_m, _inp, output):
            h = output[0] if isinstance(output, tuple) else output
            seq_len = h.shape[-2]
            prompt_len = self._prompt_len
            # Edit only the single last prompt-token position (GRUN trained location).
            #  - Prefill (seq_len > 1): edit the prompt final token (prompt_len-1 if
            #    known, else the prefill final index by construction).
            #  - Decode step (seq_len == 1, KV-cached): no edit -- generated tokens
            #    are NOT in GRUN intervention_locations.
            if seq_len == 1:
                return output
            if prompt_len is not None and prompt_len <= seq_len:
                pos = prompt_len - 1
            else:
                pos = seq_len - 1
            hp = h[..., pos, :].float()               # [batch, hidden]
            rotated = hp @ R                          # [batch, rank]
            learned = hp @ W.t() + b                  # [batch, rank]
            gate = torch.sigmoid(hp @ g.t() + bg)     # [batch, 1]
            delta = ((learned - rotated) @ R.t()) * gate
            edited = h.clone()
            edited[..., pos, :] = (hp + delta).to(h.dtype)
            if isinstance(output, tuple):
                return (edited,) + output[1:]
            return edited

        return _hook

    def teardown(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []
        self._params = {}
        self._target_layers = []
        self._prompt_len = None

    def summary_dict(self) -> dict:
        return {
            "method": "grun",
            "rail": "A_inference_time",
            "mechanism": "gated_loreft_block_output_hook",
            "gate": "per_token_sigmoid_linear",
            "reft_path": self._reft_path or os.environ.get("GRUN_REFT_PATH"),
            "layers": sorted(self._params),
        }

    # ---- shared helper (parallel to RepE/LEACE adapters) ----

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
