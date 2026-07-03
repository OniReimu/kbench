"""O3 adapter — architectural intervention via orthogonal LoRA modules.

Per Gao et al. (ICLR 2025, arxiv 2407.10223): GCYZSL/O3-LLM-UNLEARNING.
Mechanism: train an additional "unlearn" LoRA adapter alongside the original
deployed (PII-injected) adapter, with an orthogonality regularizer that forces
the new adapter's lora_A directions to be perpendicular to the deployed
adapter's lora_A directions. At inference, an OOD detector decides per-query
which adapter to route through.

For our K-test (single deletion event = our forget set), O3 reduces to:
  - Train ONE orthogonal unlearning adapter on the forget set with refusal
    targets, regularized to be orthogonal to the deployed adapter.
  - At inference, route per-query: if the queried entity is in the forget set,
    activate "unlearn"; otherwise activate "default" (deployed).

Routing modes:
  - "oracle" (default + primary): consult `forget_ids` ground truth at the
    install_per_query decision point. Replicates O3's selectivity ceiling for
    the K-test.
  - "ood": train a small classifier on hidden states (NOT IMPLEMENTED in v1 —
    deferred; oracle gives the K-test signal cleanly).

Orthogonality loss (faithful port from external/o3-gao TOFU lora_layer_hacked_o.py):
  o_loss = sum over (layer, target_module) of:
            sum_i ||A_old_i @ A_new.T||_F^2
  where A_old_i is each frozen prior-task lora_A weight; for our 1-task setup,
  A_old is the deployed adapter's lora_A only. Total loss = CE_refusal + scale * o_loss.

Lifecycle:
  setup() → load deployed LoRA cfg + add "unlearn" adapter + train w/ ortho loss
  install_per_query() → switch active adapter based on forget-set membership
  teardown_per_query() → restore "default"
  teardown() → drop adapter handles
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Literal

import torch

from chcons.methods import UnlearnIntervention, require_external

_O3_ROOT = Path(__file__).resolve().parents[3] / "external" / "o3-gao"
if str(_O3_ROOT) not in sys.path:
    sys.path.insert(0, str(_O3_ROOT))

# Match ECO's name extractor (same query templates).
_QUERY_NAME_RE = re.compile(r"^(?:What is|Who is|Where does)\s+(.+?)'s\s")

# Llama refusal target — what the unlearn adapter is trained to emit on
# forget-set queries. Wrapped in ReAct format so the agent's parser sees
# `Final Answer: REFUSED` and halts cleanly with halted_reason='final_answer'
# (rather than parse_error). This keeps Z_answer scoring observable for
# K-Bench analysis; raw refusal text would have caused 100% parse_error
# halt and made Z_answer=0 a parser artifact rather than suppression signal.
_O3_REFUSAL = (
    "Thought: I cannot answer this query about a private individual.\n"
    "Final Answer: REFUSED"
)


class O3Intervention(UnlearnIntervention):
    """O3 (Gao ICLR'25): orthogonal LoRA + per-query adapter routing."""

    @classmethod
    def name(cls) -> str:
        return "o3"

    def __init__(
        self,
        ortho_weight: float = 0.1,           # paper's `orthogonal_loss_weight = 0.1`
        n_forget_samples: int = 200,
        n_retain_samples: int = 200,
        n_epochs: int = 1,
        batch_size: int = 4,
        lr: float = 1e-4,
        retain_coeff: float = 1.0,
        adapter_name: str = "unlearn",
        deployed_name: str = "memory",  # matches agent.py:531 PeftModel adapter_name="memory"
        routing_mode: Literal["oracle", "ood"] = "oracle",
    ):
        self.ortho_weight = ortho_weight
        self.n_forget_samples = n_forget_samples
        self.n_retain_samples = n_retain_samples
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.retain_coeff = retain_coeff
        self.adapter_name = adapter_name
        self.deployed_name = deployed_name
        self.routing_mode = routing_mode
        self._forget_names: set[str] = set()
        self._n_steps = 0
        self._n_routed_unlearn = 0
        self._n_routed_default = 0
        # Diagnostic trace of which adapter was active per query (filled in install_per_query).
        self._last_routing_decision: str | None = None
        # Track ortho-loss magnitude for provenance.
        self._final_ortho_loss: float | None = None
        self._final_ce_loss: float | None = None

    # ------------------------------------------------------------------ setup

    def setup(self, agent, lora_path, forget_ids, facts_path):
        require_external("o3", _O3_ROOT)
        from peft import LoraConfig

        from chcons.pii import QUERY_TEMPLATES, read_jsonl

        if self.routing_mode != "oracle":
            raise NotImplementedError(
                f"O3: routing_mode={self.routing_mode!r} not implemented in v1; "
                f"use 'oracle' (this also matches the K-Bench panel design — "
                f"isolates the architectural intervention from detector quality)."
            )

        # 1. Cache forget-set names for per-query oracle routing.
        # NOTE: routing oracle uses ALL forget names (covers eval queries too).
        # Adapter TRAINING uses the adapter-only split (D7 protocol A.6).
        from chcons.pii import load_split_ids

        all_recs = read_jsonl(facts_path)
        all_forget_recs = [r for r in all_recs if r.id in forget_ids]
        if not all_forget_recs:
            raise RuntimeError(f"O3: empty forget set against {facts_path}")
        self._forget_names = {r.name for r in all_forget_recs}
        print(f"[o3] setup: cached {len(self._forget_names)} forget-set names "
              f"(routing_mode={self.routing_mode})")

        # D7 split (protocol A.6): adapter TRAINING uses disjoint pool only.
        # Prevents in-sample bias.
        forget_adapter_ids = load_split_ids("forget", "adapter")
        retain_adapter_ids = load_split_ids("retain", "adapter")
        forget_recs = [r for r in all_recs if r.id in forget_adapter_ids]
        retain_recs = [r for r in all_recs if r.id in retain_adapter_ids]
        if not forget_recs:
            raise RuntimeError(
                f"O3: empty forget_adapter pool. Check "
                f"data/pii_facts/forget_ids_adapter.txt."
            )
        if not retain_recs:
            raise RuntimeError(
                f"O3: empty retain_adapter pool. Check "
                f"data/pii_facts/retain_ids_adapter.txt."
            )
        print(f"[o3] D7 split: train on {len(forget_recs)} forget_adapter + "
              f"{len(retain_recs)} retain_adapter records (disjoint from eval pool)")

        # Cache decoder layer count for ortho-loss normalization (faithful to
        # paper: average over n_layers after summing q/k/v/o/gate/up/down terms
        # within each layer; not over n_modules = n_layers × n_target_modules).
        self._n_layers = int(agent.model.config.num_hidden_layers)

        # 2. Verify the deployed adapter is loaded under `self.deployed_name`.
        if not hasattr(agent.model, "peft_config"):
            raise RuntimeError(
                "O3 requires a PEFT-wrapped model (agent.model.peft_config). "
                "Pass --lora-path so the deployed adapter is loaded."
            )
        if self.deployed_name not in agent.model.peft_config:
            raise RuntimeError(
                f"O3: deployed adapter {self.deployed_name!r} not found in "
                f"peft_config (have: {list(agent.model.peft_config)}). "
                f"Project convention: agent.py loads PeftModel with "
                f"adapter_name='memory' (see agent.py:531). Ensure --lora-path "
                f"is set so the adapter is loaded before O3.setup()."
            )

        # 3. Mirror the deployed adapter's LoRA config for the unlearn adapter.
        deployed_cfg = agent.model.peft_config[self.deployed_name]
        unlearn_cfg = LoraConfig(
            r=deployed_cfg.r,
            lora_alpha=deployed_cfg.lora_alpha,
            target_modules=list(deployed_cfg.target_modules),
            lora_dropout=getattr(deployed_cfg, "lora_dropout", 0.05),
            bias=getattr(deployed_cfg, "bias", "none"),
            task_type=getattr(deployed_cfg, "task_type", "CAUSAL_LM"),
            inference_mode=False,                       # trainable
        )
        # add_adapter registers a new set of lora_A/lora_B under self.adapter_name;
        # add_adapter is the canonical PEFT API (PeftModel.add_adapter).
        if self.adapter_name in agent.model.peft_config:
            raise RuntimeError(
                f"O3: adapter {self.adapter_name!r} already loaded — check "
                f"intervention is fresh per process."
            )
        agent.model.add_adapter(self.adapter_name, unlearn_cfg)
        # Activate ONLY the unlearn adapter for training; default (deployed) is
        # implicitly held frozen since its parameters aren't requires_grad after
        # set_adapter.
        agent.model.set_adapter(self.adapter_name)
        print(f"[o3] added '{self.adapter_name}' adapter "
              f"(r={unlearn_cfg.r}, alpha={unlearn_cfg.lora_alpha}, "
              f"targets={list(unlearn_cfg.target_modules)})")

        # 4. Snapshot frozen deployed-adapter lora_A weights per (layer-key,
        #    module). These provide the orthogonality reference. Stored as
        #    detached float refs keyed by id of the LoraLayer the new lora_A
        #    lives in — so the training hook can fetch the right A_old.
        ortho_refs = self._snapshot_deployed_lora_a(agent.model, self.deployed_name)
        print(f"[o3] orthogonality refs: {len(ortho_refs)} lora_A matrices captured "
              f"from deployed '{self.deployed_name}'")

        # 5. Build training data: forget queries → refusal; retain queries → answer.
        #    The refusal target trains the unlearn adapter to suppress PII output.
        #    Retain CE preserves utility on non-forget queries (matches O3's joint
        #    objective; deployed adapter handles them at inference, but the
        #    unlearn adapter sees them in case oracle routing flips).
        rng = torch.Generator().manual_seed(0)
        f_idx = torch.randperm(len(forget_recs), generator=rng)[: self.n_forget_samples].tolist()
        r_idx = torch.randperm(len(retain_recs), generator=rng)[: self.n_retain_samples].tolist()
        forget_examples = self._build_forget_refusal_pairs(
            [forget_recs[i] for i in f_idx], QUERY_TEMPLATES
        )
        retain_examples = self._build_retain_answer_pairs(
            [retain_recs[i] for i in r_idx], QUERY_TEMPLATES
        )
        print(f"[o3] training data: {len(forget_examples)} forget-refusal, "
              f"{len(retain_examples)} retain-answer examples")

        forget_data = [self._tokenize(agent.tokenizer, q, a) for q, a in forget_examples]
        retain_data = [self._tokenize(agent.tokenizer, q, a) for q, a in retain_examples]

        # 6. Unfreeze ONLY the unlearn adapter's lora_A/B (keep deployed frozen).
        n_trainable = self._enable_only_adapter_grad(agent.model, self.adapter_name)
        if n_trainable == 0:
            raise RuntimeError(
                f"O3: no trainable params for adapter {self.adapter_name!r} after add_adapter — "
                f"PEFT may have loaded it in inference_mode."
            )
        print(f"[o3] unfroze {n_trainable} params under adapter '{self.adapter_name}'")

        # 7. Training loop.
        agent.model.train()
        opt = torch.optim.AdamW(
            (p for p in agent.model.parameters() if p.requires_grad), lr=self.lr
        )
        device = agent.model.device
        n_pairs = min(len(forget_data), len(retain_data))
        if n_pairs == 0:
            print("[o3] WARN: n_pairs=0 (empty forget or retain training set); "
                  "skipping training. Unlearn adapter remains at PEFT init.")
            last_ce, last_ortho = float("nan"), float("nan")
        else:
            # Clamp effective batch to available pairs so step indexing into
            # randperm tensors stays in-bounds when n_pairs < batch_size.
            effective_batch = min(self.batch_size, n_pairs)
            steps_per_epoch = max(1, n_pairs // effective_batch)
            print(f"[o3] training: {self.n_epochs} epochs × {steps_per_epoch} steps "
                  f"(batch={effective_batch}, lr={self.lr}, ortho_w={self.ortho_weight})")

            last_ce, last_ortho = float("nan"), float("nan")
            for epoch in range(self.n_epochs):
                rng_e = torch.Generator().manual_seed(epoch + 1)
                f_perm = torch.randperm(n_pairs, generator=rng_e).tolist()
                r_perm = torch.randperm(n_pairs, generator=rng_e).tolist()

                for step in range(steps_per_epoch):
                    fb = [forget_data[f_perm[step * effective_batch + i]] for i in range(effective_batch)]
                    rb = [retain_data[r_perm[step * effective_batch + i]] for i in range(effective_batch)]
                    f_in = self._collate_batch(fb, agent.tokenizer.pad_token_id, device)
                    r_in = self._collate_batch(rb, agent.tokenizer.pad_token_id, device)

                    # CE on forget→refusal (model learns to refuse on forget queries)
                    f_out = agent.model(
                        input_ids=f_in["input_ids"],
                        attention_mask=f_in["attention_mask"],
                        labels=f_in["labels"],
                    )
                    # CE on retain→answer (preserve utility)
                    r_out = agent.model(
                        input_ids=r_in["input_ids"],
                        attention_mask=r_in["attention_mask"],
                        labels=r_in["labels"],
                    )
                    ce = f_out.loss + self.retain_coeff * r_out.loss

                    # Orthogonality penalty (faithful to O3 paper): for each module,
                    # sum_i ||A_old_i @ A_new.T||_F^2. Read fresh A_new every step.
                    ortho = self._compute_ortho_loss(agent.model, ortho_refs, self.adapter_name)
                    loss = ce + self.ortho_weight * ortho

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    self._n_steps += 1

                    last_ce, last_ortho = ce.item(), ortho.item()
                    if (step + 1) % max(1, steps_per_epoch // 10) == 0:
                        print(f"[o3] epoch {epoch+1} step {step+1}/{steps_per_epoch} "
                              f"ce={last_ce:.3f} ortho={last_ortho:.3f}")

        self._final_ce_loss = last_ce
        self._final_ortho_loss = last_ortho

        # 8. Re-freeze + reset to inference mode. Default to deployed adapter
        #    so behavior is unchanged for queries that miss the oracle route.
        for p in agent.model.parameters():
            p.requires_grad = False
        agent.model.eval()
        agent.model.set_adapter(self.deployed_name)
        print(f"[o3] training done. steps={self._n_steps}, "
              f"ce={last_ce:.3f}, ortho={last_ortho:.3f}; "
              f"active adapter restored to {self.deployed_name!r}")

    # ----------------------------------------------------- per-query routing

    def install_per_query(self, agent, query):
        """Oracle routing: switch active adapter based on forget-set membership.

        IMPORTANT (Bug 8): the oracle MUST be queried HERE, at the per-query
        decision point — not just cached during setup. Earlier silent bugs in
        ECO had setup() cache forget_names but never call them at install time.
        We test it explicitly via _last_routing_decision below.
        """
        m = _QUERY_NAME_RE.match(query["query"])
        name = m.group(1) if m else None
        if name and name in self._forget_names:
            agent.model.set_adapter(self.adapter_name)
            self._n_routed_unlearn += 1
            self._last_routing_decision = self.adapter_name
        else:
            agent.model.set_adapter(self.deployed_name)
            self._n_routed_default += 1
            self._last_routing_decision = self.deployed_name

    def teardown_per_query(self, agent):
        """Restore deployed adapter so any post-query operations (e.g. summary
        elicit when not yet entered) default to deployed."""
        agent.model.set_adapter(self.deployed_name)
        self._last_routing_decision = None

    def teardown(self) -> None:
        # Don't delete the adapter from model.peft_config — PEFT may still hold
        # references during process teardown. Just clear our caches.
        self._forget_names.clear()

    def summary_dict(self) -> dict:
        return {
            "method": "o3",
            "ortho_weight": self.ortho_weight,
            "n_forget_samples": self.n_forget_samples,
            "n_retain_samples": self.n_retain_samples,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "retain_coeff": self.retain_coeff,
            "adapter_name": self.adapter_name,
            "deployed_name": self.deployed_name,
            "routing_mode": self.routing_mode,
            "n_training_steps": self._n_steps,
            "n_routed_unlearn": self._n_routed_unlearn,
            "n_routed_default": self._n_routed_default,
            "final_ce_loss": self._final_ce_loss,
            "final_ortho_loss": self._final_ortho_loss,
            "implementation_status": "v1_oracle_routing",
        }

    # ---------------------------------------------------- internal helpers

    @staticmethod
    def _build_forget_refusal_pairs(records, query_templates) -> list[tuple[str, str]]:
        """For forget queries, target = refusal text (no PII). This is what makes
        the unlearn adapter actually 'unlearn' — generating it on forget queries
        suppresses PII surface form."""
        out = []
        for r in records:
            d = r.to_dict()
            for field in ("date_of_birth", "address", "occupation", "employer"):
                if not d.get(field):
                    continue
                q = query_templates[field].format(name=r.name)
                out.append((q, _O3_REFUSAL))
        return out

    @staticmethod
    def _build_retain_answer_pairs(records, query_templates) -> list[tuple[str, str]]:
        """For retain queries, target = ground-truth answer (preserve utility)."""
        out = []
        for r in records:
            d = r.to_dict()
            for field in ("date_of_birth", "address", "occupation", "employer"):
                if not d.get(field):
                    continue
                q = query_templates[field].format(name=r.name)
                out.append((q, str(d[field])))
        return out

    @staticmethod
    def _tokenize(tokenizer, question: str, answer: str) -> dict:
        """Tokenize (Q, A) with prompt masked to -100. Boundary-stable
        (matches Cha adapter's pattern)."""
        prompt_part = tokenizer(f"Q: {question}\nA:", add_special_tokens=True)
        ans_part = tokenizer(f" {answer}", add_special_tokens=False)
        input_ids = prompt_part["input_ids"] + ans_part["input_ids"]
        attention_mask = prompt_part["attention_mask"] + ans_part["attention_mask"]
        n_prompt = len(prompt_part["input_ids"])
        labels = [-100] * n_prompt + ans_part["input_ids"]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    @staticmethod
    def _collate_batch(batch, pad_token_id, device):
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, attention_mask, labels = [], [], []
        for b in batch:
            pad = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_token_id] * pad)
            attention_mask.append(b["attention_mask"] + [0] * pad)
            labels.append(b["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, device=device),
            "attention_mask": torch.tensor(attention_mask, device=device),
            "labels": torch.tensor(labels, device=device),
        }

    @staticmethod
    def _enable_only_adapter_grad(model, adapter_name: str) -> int:
        """Unfreeze only the named adapter's lora_A/B parameters."""
        n = 0
        target_substr = f".lora_A.{adapter_name}."
        target_substr_b = f".lora_B.{adapter_name}."
        for name, p in model.named_parameters():
            if target_substr in name or target_substr_b in name:
                p.requires_grad = True
                n += p.numel()
            else:
                p.requires_grad = False
        return n

    @staticmethod
    def _snapshot_deployed_lora_a(model, deployed_name: str) -> list[tuple]:
        """Walk the model, find every LoraLayer that hosts the deployed adapter's
        lora_A, snapshot its weight.detach().clone() (frozen reference) and pair
        it with the LoraLayer object so the training step can fetch the matching
        new lora_A from the same module.

        Returns list of (lora_layer_module, A_old_frozen) pairs.

        The faithful O3 ortho loss is per-module:
            o_loss(module) = ||A_old @ A_new.T||_F^2

        For our 1-task setup A_old has only one entry per module (the deployed
        adapter); generalizes to k entries when running continual unlearning.
        """
        refs: list[tuple] = []
        for module in model.modules():
            if hasattr(module, "lora_A") and isinstance(module.lora_A, torch.nn.ModuleDict):
                if deployed_name in module.lora_A:
                    a_old = module.lora_A[deployed_name].weight.detach().clone()
                    a_old.requires_grad = False
                    refs.append((module, a_old))
        return refs

    def _compute_ortho_loss(
        self,
        model,
        ortho_refs: list[tuple],
        adapter_name: str,
    ) -> torch.Tensor:
        """Sum_module ||A_old @ A_new.T||_F^2 across all (module, A_old) refs,
        then average by n_layers.

        Faithful to external/o3-gao TOFU lora_layer_hacked_o.py L505-509 (per-module
        Frobenius²) and modeling_llama_hacked_o.py:844 (per-layer average after
        summing q/k/v/o/gate/up/down within each layer). Normalization is by
        n_layers, not n_modules — dividing by n_modules would weaken the
        regularizer ~7× vs the paper at the standard 7-target-module setup.

        Multi-device safety: model may be device_map="auto" sharded. We pick a
        single `loss_device` upfront and .to(loss_device) each term — cross-device
        autograd handles gradient flow back to each shard.
        """
        loss_device = next(model.parameters()).device
        if not ortho_refs:
            return torch.zeros((), device=loss_device)
        total = None
        for module, a_old in ortho_refs:
            if adapter_name not in module.lora_A:
                continue
            a_new = module.lora_A[adapter_name].weight    # (r, in_features)
            # A_old @ A_new.T → (r_old, r_new). Both adapters share the same
            # in_features per module, so the matmul is well-defined.
            # Cast a_old to the new adapter's dtype to avoid bf16/fp32 mismatch.
            a_old_cast = a_old.to(a_new.dtype).to(a_new.device)
            cross = torch.matmul(a_old_cast, a_new.transpose(0, 1))
            term = (cross * cross).sum().to(loss_device)
            total = term if total is None else total + term
        if total is None:
            return torch.zeros((), device=loss_device)
        # Normalize by decoder layer count, not by module count (paper convention).
        denom = max(1, getattr(self, "_n_layers", 0) or len(ortho_refs))
        return total / denom
