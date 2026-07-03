"""MLP-probe adapter — nonlinear concept removal via learned MLP classifier + gradient ascent.

Builds on the linear-erasure family (LEACE, RepE) but uses a 2-layer MLP as the
concept classifier. At inference, applies a single gradient-ascent step on the
classifier's logit-of-forget output w.r.t. the hidden state, scaled by step
size α, to push representations DOWN the gradient (decreasing forget-class
probability).

Differences from LEACE / RepE:
- LEACE: linear (rank-1) closed-form whitening projection
- RepE: linear (rank-1) mean-difference unit-vector projection
- **MLP-probe: nonlinear** — captures multi-direction / curved boundaries via
  a 2-hidden-layer MLP. Apply-time = one gradient step on the trained MLP's
  log-prob of forget class.

K-test question: if the leak concept lives on a nonlinear manifold, MLP-probe
should suppress it where LEACE/RepE cannot. If MLP-probe also fails on R-struct
post-fix (as LEACE did), the leak signal is not even nonlinearly available at
this layer.

Implementation shares the post-fix LEACE adapter discipline:
- D7 split (disjoint fit/eval pools)
- ReAct chat-templated fit prompts (matches inference distribution)
- All token positions captured (matches hook's full-sequence apply)
- Streaming MLP minibatch SGD (memory-bounded)
- `add_special_tokens=False`

Apply-time gradient step:
    h' = h - α * ∇_h log P(forget | h ; MLP)
where MLP outputs scalar logit (sigmoid → probability). With binary CE loss on
{Z=1 forget, Z=0 retain}, ∇_h log P(forget|h) = (P(forget|h) − 1) · ∇_h MLP(h)·...
For simplicity, we use −α · MLP(h)·∇_h MLP(h) style update.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from chcons.methods import UnlearnIntervention


class _ConceptMLP(nn.Module):
    """2-layer MLP probe: hidden → hidden//4 → 1 logit."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim // 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x))).squeeze(-1)  # [..., ]


class MLPProbeIntervention(UnlearnIntervention):
    """MLP-probe: train classifier on hidden states, apply 1-step gradient ascent at inference."""

    @classmethod
    def name(cls) -> str:
        return "mlp_probe"

    def __init__(
        self,
        target_layer_idx: int = 16,
        alpha: float = 1.0,
        n_fit_samples: int = 200,
        n_epochs: int = 3,
        lr: float = 1e-3,
        minibatch_size: int = 64,
    ):
        self.target_layer_idx = target_layer_idx
        self.alpha = alpha
        self.n_fit_samples = n_fit_samples
        self.n_epochs = n_epochs
        self.lr = lr
        self.minibatch_size = minibatch_size
        self._probe: _ConceptMLP | None = None
        self._target_layer = None
        self._hook_handle = None

    def setup(self, agent, lora_path, forget_ids, facts_path):
        from chcons.pii import read_jsonl, QUERY_TEMPLATES, load_split_ids

        all_recs = read_jsonl(facts_path)
        forget_adapter_ids = load_split_ids("forget", "adapter")
        retain_adapter_ids = load_split_ids("retain", "adapter")
        forget_recs = [r for r in all_recs if r.id in forget_adapter_ids]
        retain_recs = [r for r in all_recs if r.id in retain_adapter_ids]
        if not forget_recs or not retain_recs:
            raise RuntimeError(
                "MLP-probe: empty forget_adapter or retain_adapter pool."
            )
        print(
            f"[mlp_probe] D7 split: fit on {len(forget_recs)} forget_adapter + "
            f"{len(retain_recs)} retain_adapter records"
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
            f"[mlp_probe] fit data: {len(forget_examples)} forget + "
            f"{len(retain_examples)} retain prompts (ReAct-format)"
        )

        layers = self._get_decoder_layers(agent.model)
        if not (-len(layers) <= self.target_layer_idx < len(layers)):
            raise ValueError(
                f"MLP-probe target_layer_idx={self.target_layer_idx} out of range "
                f"for {len(layers)}-layer model."
            )
        idx = self.target_layer_idx if self.target_layer_idx >= 0 else len(layers) + self.target_layer_idx
        self._target_layer = layers[idx]
        cfg = agent.model.config
        hidden_dim = cfg.hidden_size if hasattr(cfg, "hidden_size") else cfg.text_config.hidden_size
        device = next(self._target_layer.parameters()).device
        print(f"[mlp_probe] target layer: idx {idx}/{len(layers)}, hidden_dim={hidden_dim}")

        # Collect hidden states in streaming batches; train probe via minibatch SGD.
        # Streaming approach: process prompts one at a time, accumulate tokens into
        # a buffer, train when buffer fills minibatch_size × 4 (for shuffle).
        probe = _ConceptMLP(hidden_dim).to(device=device, dtype=torch.float32)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        # Two-pass over data, n_epochs times.
        for epoch in range(self.n_epochs):
            total_loss, n_batches = 0.0, 0
            # Interleave forget + retain (one pass over each)
            for prompts, label in [(forget_examples, 1.0), (retain_examples, 0.0)]:
                for prompt in prompts:
                    X = self._collect_single_prompt(agent, prompt)  # [seq_len, hidden]
                    X = X.float().to(device)
                    Y = torch.full((X.shape[0],), label, dtype=torch.float32, device=device)
                    # Minibatch SGD within the prompt
                    perm = torch.randperm(X.shape[0], device=device)
                    for i in range(0, X.shape[0], self.minibatch_size):
                        bs = perm[i : i + self.minibatch_size]
                        logits = probe(X[bs])
                        loss = loss_fn(logits, Y[bs])
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                        n_batches += 1
                    del X, Y
                torch.cuda.empty_cache()
            print(f"[mlp_probe] epoch {epoch + 1}/{self.n_epochs}: avg loss = {total_loss / max(n_batches, 1):.4f}")

        self._probe = probe.eval()
        for p in self._probe.parameters():
            p.requires_grad_(False)
        # Re-enable grad for inference-time gradient step (we'll grad w.r.t. h, not probe params).

        # Install hook: at inference, compute ∇_h logit(h), update h ← h − α · sign(logit) · grad
        # Equivalent to one gradient-ascent step on −logit (push h away from forget-positive region).
        probe_ref = self._probe
        alpha = self.alpha
        layer_dtype = next(self._target_layer.parameters()).dtype

        def _erase_hook(_m, _inp, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            # Compute gradient of logit w.r.t. h. Need detached h with grad enabled.
            with torch.enable_grad():
                h32 = h.detach().float().requires_grad_(True)
                logits = probe_ref(h32)  # [B, seq] — logit per token
                # Sum logits to scalar for autograd (each token's grad is local).
                total = logits.sum()
                grad_h, = torch.autograd.grad(total, h32, create_graph=False)
            # One gradient-descent step on logit (move away from forget):
            # h' = h - α * grad_h (since grad_h is ∇_h logit; subtracting decreases logit)
            erased = (h32 - alpha * grad_h).to(layer_dtype)
            if isinstance(output, tuple):
                return (erased,) + output[1:]
            return erased

        self._hook_handle = self._target_layer.register_forward_hook(_erase_hook)
        print(f"[mlp_probe] hook installed on layer {idx} (gradient-ascent step on logit, α={alpha})")

    def teardown(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self._probe = None
        self._target_layer = None

    def summary_dict(self) -> dict:
        return {
            "method": "mlp_probe",
            "target_layer_idx": self.target_layer_idx,
            "alpha": self.alpha,
            "n_fit_samples": self.n_fit_samples,
            "n_epochs": self.n_epochs,
            "lr": self.lr,
        }

    # ---- shared helpers ----

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

    def _collect_single_prompt(self, agent, prompt: str) -> torch.Tensor:
        layer = self._target_layer
        captured: list[torch.Tensor] = []

        def _capture(_m, _inp, output):
            h = output[0] if isinstance(output, tuple) else output
            captured.append(h[0].detach())

        h = layer.register_forward_hook(_capture)
        try:
            agent.model.eval()
            with torch.no_grad():
                ids = agent.tokenizer(
                    prompt, return_tensors="pt", add_special_tokens=False
                ).to(agent.model.device)
                _ = agent.model(input_ids=ids["input_ids"],
                                attention_mask=ids["attention_mask"])
        finally:
            h.remove()

        if not captured:
            raise RuntimeError("MLP-probe: no hidden states captured")
        return captured[0]

    @staticmethod
    def _build_query_examples(records, query_templates, agent=None) -> list[str]:
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
