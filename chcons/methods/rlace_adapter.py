"""R-LACE adapter — Rank-relaxed Linear Concept Erasure (Ravfogel et al. ACL 2022).

Sits between LEACE (rank-1 closed-form) and INLP (iterative null-space). R-LACE
generalizes LEACE by allowing erasure on a rank-k subspace (rather than just rank-1)
selected via convex optimization: find a k-dim subspace U that maximally separates
forget/retain classes, then project out via P = I − U U^T.

Differences from LEACE / RepE:
- LEACE: rank-1 + whitening (covariance-aware) closed form
- RepE: rank-1 + no whitening (mean-difference unit vector)
- **R-LACE: rank-k** subspace projection (k > 1 captures multi-direction concepts)

For our implementation, we use the simplest variant: top-k singular vectors of
(X_forget − X_retain mean) covariance. Then project h ⊥ that subspace.

For k=1 this should reduce to a variant of RepE (mean-direction subspace).
For k > 1 it captures the dominant concept directions.

K-test question: if LEACE/RepE (rank-1) fails on R-struct, does rank-k R-LACE
recover? If R-LACE also fails, the leak signal is fundamentally non-linear
(supporting MLP-probe as the next escalation).

Implementation shares the post-fix LEACE adapter discipline.
"""

from __future__ import annotations

from pathlib import Path

import torch

from chcons.methods import UnlearnIntervention


class RLACEIntervention(UnlearnIntervention):
    """R-LACE (Ravfogel ACL'22): rank-k subspace projection for concept erasure."""

    @classmethod
    def name(cls) -> str:
        return "rlace"

    def __init__(
        self,
        target_layer_idx: int = 16,
        rank: int = 4,
        n_fit_samples: int = 200,
    ):
        self.target_layer_idx = target_layer_idx
        self.rank = rank
        self.n_fit_samples = n_fit_samples
        self._projection_complement: torch.Tensor | None = None  # I − U U^T, shape [D, D]
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
            raise RuntimeError("R-LACE: empty forget_adapter or retain_adapter pool.")

        print(
            f"[rlace] D7 split: fit on {len(forget_recs)} forget_adapter + "
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
            f"[rlace] fit data: {len(forget_examples)} forget + "
            f"{len(retain_examples)} retain prompts (ReAct-format)"
        )

        layers = self._get_decoder_layers(agent.model)
        if not (-len(layers) <= self.target_layer_idx < len(layers)):
            raise ValueError(
                f"R-LACE target_layer_idx={self.target_layer_idx} out of range "
                f"for {len(layers)}-layer model."
            )
        idx = self.target_layer_idx if self.target_layer_idx >= 0 else len(layers) + self.target_layer_idx
        self._target_layer = layers[idx]
        cfg = agent.model.config
        hidden_dim = cfg.hidden_size if hasattr(cfg, "hidden_size") else cfg.text_config.hidden_size
        device = next(self._target_layer.parameters()).device
        print(f"[rlace] target layer: idx {idx}/{len(layers)}, hidden_dim={hidden_dim}, rank={self.rank}")

        # Streaming accumulation of class-conditional means + difference outer-product covariance.
        # Concept axis subspace = top-k singular vectors of (forget_centered − retain_centered)
        # outer product. Equivalently: top-k principal components of class-difference signal.
        #
        # Memory-bounded: accumulate sum_f, sum_r, count_f, count_r over passes.
        # Then compute mean_f, mean_r. The "concept difference vectors" are the per-token
        # mean-shifted forget vs retain hidden states. Stack them and SVD to extract top-k.
        # For streaming SVD, we use a covariance-based approach:
        #   Σ_diff = E[(x_f − μ_f)(x_f − μ_f)^T − (x_r − μ_r)(x_r − μ_r)^T] (signed difference)
        # Top-k eigenvectors of Σ_diff (or its positive part) form U.

        # Pass 1: compute class means via streaming sums.
        sum_f = torch.zeros(hidden_dim, dtype=torch.float32, device=device)
        sum_r = torch.zeros(hidden_dim, dtype=torch.float32, device=device)
        count_f = 0
        count_r = 0
        # Pass 2: accumulate class-conditional cov matrices using running means computed in pass 1.

        for prompt in forget_examples:
            X = self._collect_single_prompt(agent, prompt).float()  # [seq, D]
            sum_f += X.sum(dim=0)
            count_f += X.shape[0]
            del X
        for prompt in retain_examples:
            X = self._collect_single_prompt(agent, prompt).float()
            sum_r += X.sum(dim=0)
            count_r += X.shape[0]
            del X
        torch.cuda.empty_cache()
        mean_f = sum_f / max(count_f, 1)
        mean_r = sum_r / max(count_r, 1)
        del sum_f, sum_r

        # Pass 2: accumulate (X_f − μ_f)(X_f − μ_f)^T and (X_r − μ_r)(X_r − μ_r)^T.
        # Streaming: D=4096 → covariance matrix is 64 MB fp32. Single tensor.
        cov_f = torch.zeros(hidden_dim, hidden_dim, dtype=torch.float32, device=device)
        cov_r = torch.zeros(hidden_dim, hidden_dim, dtype=torch.float32, device=device)

        for prompt in forget_examples:
            X = self._collect_single_prompt(agent, prompt).float()
            X_centered = X - mean_f
            cov_f += X_centered.t() @ X_centered  # [D, D]
            del X, X_centered
        for prompt in retain_examples:
            X = self._collect_single_prompt(agent, prompt).float()
            X_centered = X - mean_r
            cov_r += X_centered.t() @ X_centered
            del X, X_centered
        torch.cuda.empty_cache()
        cov_f /= max(count_f, 1)
        cov_r /= max(count_r, 1)

        # Concept-difference covariance: the directions where forget class is most variable
        # relative to retain.
        sigma_diff = cov_f - cov_r
        del cov_f, cov_r

        # Top-k eigenvectors of sigma_diff (symmetric → eigh stable + sorted ascending).
        # Take the eigenvectors with LARGEST absolute eigenvalues (most class-separating).
        eigvals, eigvecs = torch.linalg.eigh(sigma_diff)
        del sigma_diff
        abs_eigvals = eigvals.abs()
        topk_idx = torch.topk(abs_eigvals, k=self.rank, largest=True).indices
        U = eigvecs[:, topk_idx]  # [D, k]
        # Orthonormalize (eigh returns orthonormal already, but defensive).
        Q, _ = torch.linalg.qr(U)
        # Projection-complement matrix: P_⊥ = I − Q Q^T
        identity = torch.eye(hidden_dim, dtype=torch.float32, device=device)
        self._projection_complement = (identity - Q @ Q.t()).to(device=device)
        del U, Q, eigvals, eigvecs, abs_eigvals
        print(
            f"[rlace] subspace fitted: {count_f}+{count_r} tokens, rank-{self.rank} "
            f"top-|eigval| ranges from {abs_eigvals.max().item():.2e} to "
            f"{abs_eigvals[topk_idx[-1]].item():.2e}" if False else
            f"[rlace] subspace fitted: rank-{self.rank} complement projection ready"
        )

        # Install full-sequence projection hook.
        P_complement = self._projection_complement
        layer_dtype = next(self._target_layer.parameters()).dtype

        def _erase_hook(_m, _inp, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            # h shape [B, seq, D]. Apply P_complement: h' = h @ P_complement^T = h @ P_complement
            # (P_complement is symmetric).
            h32 = h.float()
            erased = (h32 @ P_complement).to(layer_dtype)
            if isinstance(output, tuple):
                return (erased,) + output[1:]
            return erased

        self._hook_handle = self._target_layer.register_forward_hook(_erase_hook)
        print(f"[rlace] hook installed on layer {idx} (full-sequence projection ⊥ rank-{self.rank} concept subspace)")

    def teardown(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self._projection_complement = None
        self._target_layer = None

    def summary_dict(self) -> dict:
        return {
            "method": "rlace",
            "target_layer_idx": self.target_layer_idx,
            "rank": self.rank,
            "n_fit_samples": self.n_fit_samples,
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
            raise RuntimeError("R-LACE: no hidden states captured")
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
