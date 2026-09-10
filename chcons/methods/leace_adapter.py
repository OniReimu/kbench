"""LEACE adapter — closed-form linear concept erasure on activations.

Per Belrose et al. (NeurIPS 2023): EleutherAI/concept-erasure.
LEACE = LEAst-squares Concept Erasure: a closed-form linear projection that
provably removes a concept from representations with minimal damage. Unlike
DEPN's saliency-based weight surgery, LEACE has an exact analytical solution
based on covariance statistics.

Workflow:
  1. Collect hidden-state vectors X for forget (Z=1) and retain (Z=0) queries
  2. LeaceFitter accumulates cov(X,X) and cov(X,Z); compute eraser via .eraser
  3. Install forward hook at chosen decoder layer that applies eraser(h) per token

The eraser is a closed-form affine map on hidden states. For our K-test, the
mechanistic question is: does removing a concept's LINEAR signature in
representations propagate through all observable channels (REFUTING K) or
only suppress the channel that depends most directly on that signature
(SUPPORTING K)?

Pre-eval pattern: setup() collects stats + fits eraser + installs persistent
forward hook. Per-query / before_generation = no-ops. Teardown removes hook.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from chcons.methods import UnlearnIntervention, require_external

_LEACE_ROOT = Path(__file__).resolve().parents[3] / "external" / "leace"
if str(_LEACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LEACE_ROOT))


class LEACEIntervention(UnlearnIntervention):
    """LEACE (Belrose NeurIPS'23): closed-form linear concept erasure on hidden states."""

    @classmethod
    def name(cls) -> str:
        return "leace"

    def __init__(
        self,
        n_fit_samples: int = 200,
        target_layer_idx: int = -1,           # -1 = last decoder layer
    ):
        self.n_fit_samples = n_fit_samples
        self.target_layer_idx = target_layer_idx
        self._eraser = None
        self._hook_handle = None
        self._target_layer = None
        self._fit_diag = None

    def setup(self, agent, lora_path, forget_ids, facts_path):
        require_external("leace", _LEACE_ROOT)
        from concept_erasure import LeaceFitter
        from chcons.pii import read_jsonl, QUERY_TEMPLATES, load_split_ids

        # D7 split (protocol A.6): LEACE projection FIT uses
        # disjoint adapter pool only. Eval queries come from disjoint eval
        # split (02_baseline_leakage). Prevents in-sample bias on hidden-state
        # collection for concept axis fit.
        all_recs = read_jsonl(facts_path)
        forget_adapter_ids = load_split_ids("forget", "adapter")
        retain_adapter_ids = load_split_ids("retain", "adapter")
        forget_recs = [r for r in all_recs if r.id in forget_adapter_ids]
        retain_recs = [r for r in all_recs if r.id in retain_adapter_ids]
        if not forget_recs or not retain_recs:
            raise RuntimeError(
                f"LEACE: empty forget_adapter or retain_adapter pool. Check "
                f"data/pii_facts/forget_ids_adapter.txt + "
                f"data/pii_facts/retain_ids_adapter.txt."
            )
        print(f"[leace] D7 split: fit on {len(forget_recs)} forget_adapter + "
              f"{len(retain_recs)} retain_adapter records (disjoint from eval pool)")

        rng = torch.Generator().manual_seed(0)
        f_idx = torch.randperm(len(forget_recs), generator=rng)[:self.n_fit_samples].tolist()
        r_idx = torch.randperm(len(retain_recs), generator=rng)[:self.n_fit_samples].tolist()
        # Pass agent so fit prompts use the SAME ReAct chat-template as inference
        # (fit/apply distribution match).
        forget_examples = self._build_query_examples(
            [forget_recs[i] for i in f_idx], QUERY_TEMPLATES, agent=agent
        )
        retain_examples = self._build_query_examples(
            [retain_recs[i] for i in r_idx], QUERY_TEMPLATES, agent=agent
        )
        print(f"[leace] fit data: {len(forget_examples)} forget queries, "
              f"{len(retain_examples)} retain queries (ReAct-format)")

        # Locate target decoder layer
        layers = self._get_decoder_layers(agent.model)
        if not (-len(layers) <= self.target_layer_idx < len(layers)):
            raise ValueError(
                f"LEACE target_layer_idx={self.target_layer_idx} out of range "
                f"for {len(layers)}-layer model (valid: [-{len(layers)}, {len(layers) - 1}]). "
                "Silent negative-index wrap to a wrong layer was a P3 finding."
            )
        idx = self.target_layer_idx if self.target_layer_idx >= 0 else len(layers) + self.target_layer_idx
        self._target_layer = layers[idx]
        # Multimodal configs (e.g. Gemma3ForConditionalGeneration) nest hidden_size
        # under config.text_config; flat decoder-only configs expose it directly.
        cfg = agent.model.config
        hidden_dim = cfg.hidden_size if hasattr(cfg, "hidden_size") else cfg.text_config.hidden_size
        print(f"[leace] target layer: idx {idx} (of {len(layers)}), hidden_dim={hidden_dim}")

        # Stream per-prompt LeaceFitter.update() — collecting all tokens of 200 ReAct
        # prompts in one tensor is ~21 GB on Mistral (1.3M tokens × 4096 × fp32), OOM
        # on H100 with other processes sharing the GPU. LeaceFitter accumulates
        # additive statistics (mean_x, cov_xx, cov_xz), so streaming is mathematically
        # equivalent to batched but bounded to per-prompt memory (~50 MB).
        device = next(self._target_layer.parameters()).device
        fitter = LeaceFitter(hidden_dim, 1, dtype=torch.float32, device=device)
        # Per-prompt token sums, kept alongside the fitter for the permutation null in
        # _fit_diagnostics. The label is constant within a prompt, so cov(X,Z) under ANY
        # relabelling is a closed-form combination of these sums — a null distribution
        # costs no extra forward passes. 400 x 4096 fp32 is ~6.5 MB, unlike the states
        # themselves (~33 GB), which is why the loop still streams.
        prompt_sums: list = []
        prompt_counts: list = []
        prompt_z: list = []
        total_forget_tokens = 0
        for i, prompt in enumerate(forget_examples):
            X_p = self._collect_single_prompt(agent, prompt)  # [seq_len, D]
            Z_p = torch.ones(X_p.shape[0], 1, device=device)
            fitter.update(X_p.float(), Z_p.float())
            prompt_sums.append(X_p.float().sum(dim=0))
            prompt_counts.append(X_p.shape[0])
            prompt_z.append(1.0)
            total_forget_tokens += X_p.shape[0]
            del X_p, Z_p
            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()
        total_retain_tokens = 0
        for i, prompt in enumerate(retain_examples):
            X_p = self._collect_single_prompt(agent, prompt)
            Z_p = torch.zeros(X_p.shape[0], 1, device=device)
            fitter.update(X_p.float(), Z_p.float())
            prompt_sums.append(X_p.float().sum(dim=0))
            prompt_counts.append(X_p.shape[0])
            prompt_z.append(0.0)
            total_retain_tokens += X_p.shape[0]
            del X_p, Z_p
            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()
        print(f"[leace] streamed update: forget {total_forget_tokens} tokens, "
              f"retain {total_retain_tokens} tokens (over {len(forget_examples)} + "
              f"{len(retain_examples)} prompts)")
        self._eraser = fitter.eraser
        # Move eraser to the target layer's device — accelerate may shard
        # the model across multiple GPUs. Without this, hook fails with
        # cuda:0 vs cuda:1 mismatch when eraser was fit on cuda:0 but the
        # hooked layer lives on cuda:1.
        target_device = next(self._target_layer.parameters()).device
        self._eraser = self._eraser.to(target_device)
        print(f"[leace] eraser fitted, moved to {target_device}")

        self._fit_diag = self._fit_diagnostics(
            fitter, self._eraser,
            prompt_sums=torch.stack(prompt_sums),
            prompt_counts=torch.tensor(prompt_counts, dtype=torch.float32, device=device),
            prompt_z=torch.tensor(prompt_z, dtype=torch.float32, device=device),
            n_forget_tokens=total_forget_tokens,
            n_retain_tokens=total_retain_tokens,
        )
        print(f"[leace] fit diagnostic: {self._fit_diag}")
        if self._fit_diag["eraser_is_identity"]:
            print("[leace] WARNING: the fitted eraser is the IDENTITY MAP. Its whitened "
                  f"cross-covariance s0={self._fit_diag['s0']:.6g} did not clear the "
                  f"absolute svd_tol={self._fit_diag['svd_tol']:.6g}, so LeaceFitter zeroed "
                  "both projection factors. This run applies no intervention whatsoever; "
                  "its output will be byte-identical to the unlearn=none baseline and must "
                  "not be reported as a LEACE result.")

        # Install forward hook on target layer to apply eraser to its OUTPUT hidden state.
        # Full-sequence erasure is required for intermediate-layer LEACE (target_layer < last):
        # later layers consume earlier positions via attention, so leaving non-last positions
        # un-erased would let the concept signal recover through attention from un-cleansed
        # prompt context.
        # Fit/apply distribution match is enforced by _collect_hidden_states using the same
        # ReAct chat-templated prompts as inference + collecting ALL token positions.
        eraser = self._eraser
        layer_dtype = next(self._target_layer.parameters()).dtype

        def _erase_hook(_m, _inp, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            h32 = h.float()
            erased = eraser(h32).to(layer_dtype)
            if isinstance(output, tuple):
                return (erased,) + output[1:]
            return erased

        self._hook_handle = self._target_layer.register_forward_hook(_erase_hook)
        print(f"[leace] hook installed on decoder layer {idx} (full-sequence erase)")

    def teardown(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self._eraser = None
        self._target_layer = None

    def summary_dict(self) -> dict:
        d = {
            "method": "leace",
            "n_fit_samples": self.n_fit_samples,
            "target_layer_idx": self.target_layer_idx,
        }
        if self._fit_diag is not None:
            d.update(self._fit_diag)
        return d

    # ---- internal helpers ----

    @staticmethod
    def _fit_diagnostics(fitter, eraser, *,
                         prompt_sums, prompt_counts, prompt_z,
                         n_forget_tokens: int, n_retain_tokens: int,
                         n_permutations: int = 200) -> dict:
        """Whether the fitted eraser does anything, and whether what it does is signal.

        Two failure modes have to be told apart, and neither is visible in a transcript.

        First, degeneracy. LeaceFitter.eraser thresholds the singular values of the
        whitened cross-covariance at an ABSOLUTE `svd_tol` (default 0.01). With z_dim=1
        there is exactly one singular value, so failing that cut zeroes `u`, hence both
        projection factors, and LeaceEraser.__call__ returns its input unchanged — an
        exact identity map that no amount of reading the outputs will distinguish from a
        live eraser that had no behavioural effect.

        Second, and the more likely one at this scale: clearing an absolute threshold is
        not evidence of a concept. Finite-sample whitening noise alone puts `s0` around
        0.5*sqrt(D/N) even when the label is independent of the representation, which at
        D=4096 exceeds 0.01 for any N below ~10^7. `svd_tol` therefore cannot decide
        whether a direction was found. A permutation null can, and costs nothing here:
        the label is constant within a prompt, so cov(X,Z) under any relabelling is a
        closed-form combination of the per-prompt token sums, and the whitener does not
        depend on the labels at all.

        `s0` is recomputed rather than read off the fitter, which does not expose it, so
        this mirrors upstream's whitening. The mirror is self-checked twice: `s0 > svd_tol`
        must agree with the eraser's observed degeneracy, and the reconstructed
        cross-covariance must match the fitter's own. Either check going False means the
        mirror has drifted, rather than the number quietly becoming wrong.
        """
        proj_absmax = float(eraser.proj_left.abs().max())
        is_identity = proj_absmax == 0.0

        sigma_xx = fitter.sigma_xx
        L, V = torch.linalg.eigh(sigma_xx)
        keep = L > (L[-1] * sigma_xx.shape[-1] * torch.finfo(L.dtype).eps)
        L = L.clamp_min(0.0)
        W = V * torch.where(keep, L.rsqrt(), torch.zeros_like(L)) @ V.mH
        s = torch.linalg.svdvals(W @ fitter.sigma_xz)
        s0 = float(s[0])
        svd_tol = float(fitter.svd_tol)

        # Permutation null over PROMPT-level relabellings. Shuffling per-token labels
        # instead would break the within-prompt constancy the real labels have and give a
        # null that is far too low, so the observed s0 would clear it trivially.
        centred = prompt_sums - prompt_counts.unsqueeze(1) * fitter.mean_x
        n_tok = float(fitter.n)
        mean_z = float(fitter.mean_z)

        def _s0_for(z_vec):
            sigma_xz = (centred * (z_vec - mean_z).unsqueeze(1)).sum(dim=0, keepdim=True).mH
            return float(torch.linalg.svdvals(W @ (sigma_xz / (n_tok - 1)))[0])

        s0_recon = _s0_for(prompt_z)
        recon_relerr = abs(s0_recon - s0) / max(s0, 1e-12)
        gen = torch.Generator(device="cpu").manual_seed(0)
        null = sorted(
            _s0_for(prompt_z[torch.randperm(prompt_z.numel(), generator=gen).to(prompt_z.device)])
            for _ in range(n_permutations)
        )
        n_ge = sum(1 for v in null if v >= s0)
        perm_p = (n_ge + 1) / (n_permutations + 1)

        # Relative perturbation the eraser applies, in closed form over the fit
        # distribution. The eraser is affine, so dx = -(x - mean_x) @ A.mH with
        # A = proj_left @ proj_right, and E||dx||^2 = tr(A.mH A sigma_xx) exactly. A has
        # rank <= z_dim, so this stays in kxk rather than touching a 4096x4096 product.
        # Computed rather than sampled because the adapter pools are exactly n_fit_samples
        # deep: a held-out estimate would have had zero prompts to average and would have
        # reported 0.0, which reads precisely like "the eraser did nothing".
        # Exact with respect to the sigma_xx the eraser was itself derived from — verified
        # against direct application to 4e-16 when sigma_xx is computed exactly. Against
        # the raw tokens it can differ by several percent, because LeaceFitter.update
        # applies the per-sample Welford recurrence to whole batches without the Chan
        # correction, and the whitening tends to select the low-variance directions where
        # that bias is relatively largest. So this is INFORMATIONAL ONLY, never a gate:
        # gate_g_rollup.py classifies on eraser_is_identity and perm_p_value, neither of
        # which inherits the error (the null reuses the same W and the same reconstructed
        # cross-covariance, so it largely cancels in the ratio).
        pl, pr = eraser.proj_left, eraser.proj_right
        num = float(torch.trace((pl.mH @ pl) @ (pr @ sigma_xx @ pr.mH)).real)
        den = float(torch.trace(sigma_xx).real) + float(fitter.mean_x.pow(2).sum())
        rel_delta = (max(num, 0.0) / den) ** 0.5 if den > 0 else 0.0

        return {
            "eraser_is_identity": is_identity,
            "s0": s0,
            "svd_tol": svd_tol,
            "retained_rank": int((s > svd_tol).sum()),
            "proj_left_absmax": proj_absmax,
            "mirror_agrees": bool((s0 > svd_tol) == (not is_identity)),
            "sigma_xz_recon_relerr": recon_relerr,
            "perm_null_median": null[len(null) // 2],
            "perm_null_p95": null[int(0.95 * (len(null) - 1))],
            "perm_null_max": null[-1],
            "perm_n": n_permutations,
            "perm_p_value": perm_p,
            "s0_over_null_median": s0 / max(null[len(null) // 2], 1e-12),
            "rel_delta_h_insample": rel_delta,
            "n_fit_tokens_forget": n_forget_tokens,
            "n_fit_tokens_retain": n_retain_tokens,
        }

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

        Previously used toy `Q: ...\nA:` prompts; this caused
        the LEACE eraser's fit-time hidden-state distribution to differ from inference
        (which uses full ReAct system + tools + user template). The eraser's affine bias
        was wrong for the apply-time distribution → suspected root cause of anti-K-REF
        mode on (Llama, R-struct, forget/retain) and (Mistral, P, forget) cells.

        If `agent` is provided, builds the same chat-templated prompt the agent uses at
        inference (REACT_SYSTEM + tools section + REACT_USER → tokenizer.apply_chat_template
        with `add_generation_prompt=True`). Falls back to plain `Q:...\nA:` if no agent
        (for unit tests / backwards compat).
        """
        # Lazy import to avoid circular dep at module load
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
                        # Gemma-2-style chat-template that rejects system role
                        messages = [{"role": "user", "content": f"{system}\n\n{user}"}]
                        prompt = agent.tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    out.append(prompt)
                break
        return out

    def _collect_single_prompt(self, agent, prompt: str) -> torch.Tensor:
        """Run a single prompt through the model and return target-layer hidden states
        for ALL token positions. Returns [seq_len, hidden_dim] on the model's device.

        Per-prompt collection (rather than batched) enables streaming LeaceFitter.update()
        which keeps memory bounded to ~50 MB per prompt (vs ~21 GB if all 400 prompts'
        states are concatenated). Caught by OOM.

        Use add_special_tokens=False to match agent._generate_block
        (chat-templated prompts already include BOS); capture h[0] (all positions) to match
        the hook's full-sequence apply path on intermediate-layer LEACE.
        """
        layer = self._target_layer
        captured: list[torch.Tensor] = []

        def _capture(_m, _inp, output):
            h = output[0] if isinstance(output, tuple) else output
            captured.append(h[0].detach())  # keep on GPU; freed after fitter.update + del

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
            raise RuntimeError("LEACE: no hidden states captured")
        return captured[0]
