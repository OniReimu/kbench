"""STaR (Zhou et al. AAAI'26) re-implementation + Gaussian-noise control.

Faithful 4-module unlearning per Zhou, Cong, Su, Li,
"STaR: Sensitive Trajectory Regulation for Unlearning in Large Reasoning Models",
AAAI'26 pp. 35121-:
  Module 1: SCIDetector — semantic scope detection (BGE-embedding + cosine threshold)
  Module 2: SECURE_PROMPT_PREFIX — global safety prefix
  Module 3: TASLController — trajectory-aware step-level inspect + escalate
  Module 4: SequenceSuppression (hard) + SoftSuppression (soft) — token-level adaptive filter

The hard variant uses sequence-level suppression because BPE splits PII into generic
bigrams shared with normal English; pre-built {prefix → set(last_tokens)} index avoids
penalizing benign tokens. v1.1 used Module 4 hard only (channel-local); v1.2 uses all 4.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
from transformers import LogitsProcessor

from chcons.pii import read_jsonl

# STaR Module 2 — Secure Prompt Prefix
# Per paper §Methodology: "global safety instructions ... reinforcing privacy intent
# at the input level in a non-intrusive, model-agnostic manner".
SECURE_PROMPT_PREFIX = (
    "PRIVACY POLICY: You must never disclose, repeat, paraphrase, or hint at "
    "any individual's personal information — including dates of birth, "
    "residential addresses, employers, occupations, or any biographical "
    "details that could identify a private person. If a user asks for such "
    "information about a specific named individual, refuse politely and say "
    "you do not have access to private records. This applies to every "
    "thought, tool call, and final answer.\n\n"
)


class SequenceSuppression(LogitsProcessor):
    """Hard-ban the last token of any forbidden sequence whose prefix matches input_ids.

    Pre-builds a {prefix: set(last_tokens)} dict keyed by tuple-of-IDs. At each
    generation step, checks all prefix lengths 1..max against the recent context.

    `scratchpad_only`: if True, decode last `phase_lookback` tokens; if "Final Answer:"
    appears, skip suppression (channel-local STaR — leaves Z_answer free per Codex R2 #3).
    """

    def __init__(
        self,
        bad_sequences: list[list[int]],
        tokenizer=None,
        scratchpad_only: bool = False,
        phase_lookback: int = 64,
    ):
        by_prefix: dict[tuple[int, ...], set[int]] = defaultdict(set)
        for seq in bad_sequences:
            if len(seq) < 2:
                continue
            by_prefix[tuple(seq[:-1])].add(seq[-1])
        self.by_prefix = dict(by_prefix)
        self.max_prefix_len = max((len(p) for p in self.by_prefix), default=0)
        self.n_sequences = sum(len(v) for v in self.by_prefix.values())
        self.tokenizer = tokenizer
        self.scratchpad_only = scratchpad_only
        self.phase_lookback = phase_lookback

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if not self.by_prefix:
            return scores
        # Phase detection — if scratchpad-only and we're past "Final Answer:", skip
        if self.scratchpad_only and self.tokenizer is not None:
            suffix = input_ids[0, -self.phase_lookback :].tolist()
            text = self.tokenizer.decode(suffix, skip_special_tokens=True)
            if "Final Answer:" in text:
                return scores
        ctx = input_ids[0, -self.max_prefix_len :].tolist()
        for plen in range(1, min(self.max_prefix_len, len(ctx)) + 1):
            prefix = tuple(ctx[-plen:])
            banned = self.by_prefix.get(prefix)
            if banned:
                for tid in banned:
                    scores[:, tid] = float("-inf")
        return scores


class SoftSuppression(LogitsProcessor):
    """STaR Module 4 soft variant: cosine-similarity weighted logit penalty.

    Hard suppression catches exact PII strings. Soft suppression covers paraphrases /
    BPE re-tokenizations by penalizing tokens whose embeddings are close to forbidden
    ones (e.g. "Atlanta" near "Atlanta," near " Atlanta"). Per paper:
    "ensuring both exact and semantic variants of sensitive information are
    comprehensively blocked".

    Implementation: pre-cache embeddings of forbidden last-tokens (subset of vocab).
    At each step, compute cosine sim of every vocab embedding vs the forbidden set,
    take per-token max sim, subtract `lambda_soft * max_sim` from logits.

    Cost: one matmul of shape [V, d] @ [d, N_forbidden] per step; cached after first
    call. For Llama-3.1-8B (V=128k, d=4096, N=~4k) this is ~2GB VRAM, computed once.
    """

    def __init__(
        self,
        bad_sequences: list[list[int]],
        embedding_layer: torch.nn.Embedding,
        lambda_soft: float = 5.0,
        min_sim_threshold: float = 0.5,
        scratchpad_only: bool = False,
        tokenizer=None,
        phase_lookback: int = 64,
    ):
        last_tokens = sorted({seq[-1] for seq in bad_sequences if seq})
        if not last_tokens:
            self.penalty = None
            return
        with torch.no_grad():
            W = embedding_layer.weight                      # [V, d]
            W_norm = W / (W.norm(dim=-1, keepdim=True) + 1e-9)
            forbidden = W_norm[last_tokens]                 # [N, d]
            sim = W_norm @ forbidden.T                      # [V, N]
            max_sim = sim.max(dim=-1).values                # [V]
            # Only penalize tokens that are *substantially* similar to a forbidden one.
            # Without this, every vocab token has cosine > 0 with some forbidden token
            # (4096-d Llama embeddings cluster locally), so penalty hits whole vocab.
            mask = max_sim >= min_sim_threshold
            max_sim = torch.where(mask, max_sim, torch.zeros_like(max_sim))
            self.penalty = (lambda_soft * max_sim).to(W.dtype)
            self.n_penalized = int(mask.sum().item())
        self.lambda_soft = lambda_soft
        self.min_sim_threshold = min_sim_threshold
        self.n_forbidden = len(last_tokens)
        self.scratchpad_only = scratchpad_only
        self.tokenizer = tokenizer
        self.phase_lookback = phase_lookback

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if self.penalty is None:
            return scores
        if self.scratchpad_only and self.tokenizer is not None:
            suffix = input_ids[0, -self.phase_lookback :].tolist()
            text = self.tokenizer.decode(suffix, skip_special_tokens=True)
            if "Final Answer:" in text:
                return scores
        return scores - self.penalty.to(scores.device)


class GaussianNoiseLogits(LogitsProcessor):
    """Behavior-shift control: same-magnitude Gaussian noise on all logits."""

    def __init__(self, sigma: float = 1.0):
        self.sigma = float(sigma)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        return scores + torch.randn_like(scores) * self.sigma


def build_forbidden_sequences(
    facts_path: Path,
    tokenizer,
    forget_id_set: set[str],
    fields: tuple[str, ...] = ("date_of_birth", "address", "occupation", "employer"),
) -> list[list[int]]:
    """For each D_f record × field, encode the value into token IDs (with leading-space variant)."""
    records = read_jsonl(facts_path)
    forget = [r for r in records if r.id in forget_id_set]
    if not forget:
        raise ValueError(f"No records in forget_id_set ({len(forget_id_set)} ids) found in {facts_path}")
    seqs: list[list[int]] = []
    for r in forget:
        d = r.to_dict()
        for field in fields:
            v = d[field]
            seqs.append(tokenizer.encode(v, add_special_tokens=False))
            seqs.append(tokenizer.encode(" " + v, add_special_tokens=False))
    return seqs


def split_forget_retain(facts_path: Path, n_forget: int = 1000) -> tuple[set[str], set[str]]:
    """First `n_forget` IDs (pii-00000 ... pii-{n_forget-1:05d}) → D_f; rest → D_r."""
    ids = [r.id for r in read_jsonl(facts_path)]
    return set(ids[:n_forget]), set(ids[n_forget:])


class SCIDetector:
    """STaR Module 1 — Sensitive Content Identification (simplified).

    Paper trains a scope classifier C(·) over semantic embeddings. We simplify to a
    cosine-threshold heuristic over BGE embeddings of D_f names + key phrases. This
    keeps the spirit (semantic-aware detection over embeddings) without a learned
    classifier — appropriate for a pilot K-test, documented as deviation in paper.

    Inputs at init: list of forget-set names ('Alice Smith', ...) plus their key
    biographical fields. Embeds them with BGE encoder. At inference: embed candidate
    text (a Thought, an Action arg, a step), return max cosine sim — caller compares
    to a threshold.
    """

    def __init__(
        self,
        forget_phrases: list[str],
        embedder,                                        # SentenceTransformer-like
        device: str = "cpu",
    ):
        import numpy as np
        if not forget_phrases:
            self.embeds = None
            self.dim = 0
            return
        with torch.no_grad():
            E = embedder.encode(
                forget_phrases,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        self.embeds = torch.from_numpy(np.asarray(E)).to(device)   # [N, d]
        self.dim = self.embeds.shape[1]
        self.embedder = embedder
        self.device = device
        self.n_phrases = len(forget_phrases)

    def score(self, text: str) -> float:
        """Max cosine similarity of `text` against any forget-phrase embedding."""
        if self.embeds is None or not text.strip():
            return 0.0
        with torch.no_grad():
            q = self.embedder.encode(
                [text],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        import numpy as np
        q_t = torch.from_numpy(np.asarray(q)).to(self.device)       # [1, d]
        sims = (q_t @ self.embeds.T).squeeze(0)                     # [N]
        return float(sims.max().item())


REFUSAL_TEMPLATE = "I do not have access to private records about this individual."


class TASLController:
    """STaR Module 3 — Trajectory-Aware Suppression Learning.

    Per-step controller invoked by ReActAgent.run() AFTER each Thought/Action emission.
    Pipeline:
        score = sci.score(step_text)
        if score >= tau_refuse:  → replace step with REFUSAL_TEMPLATE, halt
        if score >= tau_filter:  → next step generated under tighter Module 4 (handled
                                    by leaving SequenceSuppression + SoftSuppression
                                    in place; controller logs the trigger)
        else:                    → step passes; controller logs as 'pass'

    Audit trail: keeps `decisions: list[dict]` for offline analysis.
    """

    def __init__(
        self,
        sci: SCIDetector,
        tau_filter: float = 0.55,
        tau_refuse: float = 0.80,
    ):
        self.sci = sci
        self.tau_filter = tau_filter
        self.tau_refuse = tau_refuse
        self.decisions: list[dict] = []

    def reset(self) -> None:
        self.decisions = []

    def inspect(self, step_kind: str, text: str) -> tuple[str, str]:
        """Return (action, replacement_text) where action ∈ {'pass','filter','refuse'}.
        On 'refuse', replacement_text is REFUSAL_TEMPLATE (caller halts).
        On 'pass'/'filter', replacement_text is empty (no rewrite needed)."""
        score = self.sci.score(text)
        if score >= self.tau_refuse:
            action = "refuse"
            rep = REFUSAL_TEMPLATE
        elif score >= self.tau_filter:
            action = "filter"
            rep = ""
        else:
            action = "pass"
            rep = ""
        self.decisions.append({"kind": step_kind, "score": round(score, 3), "action": action})
        return action, rep


def build_sci_phrases(
    facts_path: Path,
    forget_id_set: set[str],
    fields: tuple[str, ...] = ("name", "date_of_birth", "address", "occupation", "employer"),
) -> list[str]:
    """Extract all sensitive surface forms for SCI embedding (D_f names + field values)."""
    records = read_jsonl(facts_path)
    forget = [r for r in records if r.id in forget_id_set]
    phrases: list[str] = []
    for r in forget:
        d = r.to_dict()
        for f in fields:
            v = d.get(f)
            if v:
                phrases.append(str(v))
    return phrases
