"""Cha IHL+FILA adapter — parameter-loss intervention.

Per Cha et al. (ICLR 2025): csm9493/efficient-llm-unlearning.

Method:
  - IHL (Inverted Hinge Loss): for forget queries, multiclass hinge loss on
    next-token logits — pushes target-token probability DOWN while promoting
    plausible alternatives. Replaces gradient-ascent-on-CE which destabilizes.
  - FILA (Fisher-Information-Initialized LoRA): use Fisher information of the
    forget vs retain set to align LoRA's low-rank subspace with directions most
    sensitive to forget data. v1 of our adapter SKIPS FILA initialization
    (uses existing LoRA weights as start) for impl simplicity; can re-add later.
  - Combined loss = IHL_forget + CE_retain. Single epoch typically sufficient.

In our setup the existing `lora_v1` adapter encodes PII. We unfreeze it,
fine-tune 1 epoch with IHL+CE objective, and the modified LoRA is what runs
during evaluation. Pre-eval pattern, no per-query work.

`multiclass_hinge_loss` is from torchmetrics — Cha imports the same lib.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from chcons.methods import UnlearnIntervention, require_external

_CHA_ROOT = Path(__file__).resolve().parents[3] / "external" / "cha-ihl-fila"
if str(_CHA_ROOT / "TOFU") not in sys.path:
    sys.path.insert(0, str(_CHA_ROOT / "TOFU"))


class ChaIntervention(UnlearnIntervention):
    """Cha IHL+FILA (ICLR'25): IHL forget loss + CE retain loss on existing LoRA."""

    @classmethod
    def name(cls) -> str:
        return "cha"

    def __init__(
        self,
        n_forget_samples: int = 200,
        n_retain_samples: int = 200,
        n_epochs: int = 1,
        batch_size: int = 4,
        lr: float = 1e-4,
        ihl_alpha: float = 1.0,
        retain_coeff: float = 1.0,
    ):
        self.n_forget_samples = n_forget_samples
        self.n_retain_samples = n_retain_samples
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.ihl_alpha = ihl_alpha
        self.retain_coeff = retain_coeff
        self._n_steps = 0

    def setup(self, agent, lora_path, forget_ids, facts_path):
        require_external("cha", _CHA_ROOT)
        from chcons.pii import read_jsonl, QUERY_TEMPLATES, load_split_ids

        # D7 split (protocol A.6): adapter TRAINING uses disjoint
        # pool only. Eval queries come from 02_baseline_leakage's eval split.
        # Prevents in-sample bias.
        all_recs = read_jsonl(facts_path)
        forget_adapter_ids = load_split_ids("forget", "adapter")
        retain_adapter_ids = load_split_ids("retain", "adapter")
        forget_recs = [r for r in all_recs if r.id in forget_adapter_ids]
        retain_recs = [r for r in all_recs if r.id in retain_adapter_ids]
        if not forget_recs:
            raise RuntimeError(
                f"Cha: empty forget_adapter pool. Check "
                f"data/pii_facts/forget_ids_adapter.txt."
            )
        if not retain_recs:
            raise RuntimeError(
                f"Cha: empty retain_adapter pool. Check "
                f"data/pii_facts/retain_ids_adapter.txt."
            )
        print(f"[cha] D7 split: train on {len(forget_recs)} forget_adapter + "
              f"{len(retain_recs)} retain_adapter records (disjoint from eval pool)")

        rng = torch.Generator().manual_seed(0)
        f_idx = torch.randperm(len(forget_recs), generator=rng)[:self.n_forget_samples].tolist()
        r_idx = torch.randperm(len(retain_recs), generator=rng)[:self.n_retain_samples].tolist()
        forget_examples = self._build_examples([forget_recs[i] for i in f_idx], QUERY_TEMPLATES)
        retain_examples = self._build_examples([retain_recs[i] for i in r_idx], QUERY_TEMPLATES)
        print(f"[cha] training data: {len(forget_examples)} forget, {len(retain_examples)} retain examples")

        # Tokenize (Q, A) pairs to (input_ids, labels) with prompt masked to -100
        forget_data = [self._tokenize(agent.tokenizer, q, a) for q, a in forget_examples]
        retain_data = [self._tokenize(agent.tokenizer, q, a) for q, a in retain_examples]

        # Unfreeze LoRA params (PEFT loaded with inference_mode=True freezes them)
        n_trainable = self._enable_lora_grad(agent.model)
        if n_trainable == 0:
            raise RuntimeError("Cha: no trainable LoRA params found — is adapter loaded?")
        print(f"[cha] unfroze {n_trainable} LoRA params")

        agent.model.train()
        opt = torch.optim.AdamW(
            (p for p in agent.model.parameters() if p.requires_grad), lr=self.lr
        )

        # Training loop: zip forget+retain batches, IHL on forget + CE on retain
        device = agent.model.device
        n_pairs = min(len(forget_data), len(retain_data))
        steps_per_epoch = n_pairs // self.batch_size
        print(f"[cha] training: {self.n_epochs} epochs × {steps_per_epoch} steps "
              f"(batch_size={self.batch_size}, lr={self.lr}, ihl_alpha={self.ihl_alpha})")

        for epoch in range(self.n_epochs):
            # Reshuffle each epoch
            rng_e = torch.Generator().manual_seed(epoch + 1)
            f_perm = torch.randperm(n_pairs, generator=rng_e).tolist()
            r_perm = torch.randperm(n_pairs, generator=rng_e).tolist()

            for step in range(steps_per_epoch):
                fb = [forget_data[f_perm[step * self.batch_size + i]] for i in range(self.batch_size)]
                rb = [retain_data[r_perm[step * self.batch_size + i]] for i in range(self.batch_size)]

                f_in = self._collate_batch(fb, agent.tokenizer.pad_token_id, device)
                r_in = self._collate_batch(rb, agent.tokenizer.pad_token_id, device)

                # Forget pass — IHL loss
                f_out = agent.model(input_ids=f_in["input_ids"], attention_mask=f_in["attention_mask"])
                ihl = self._ihl_loss(f_out.logits, f_in["labels"])

                # Retain pass — CE loss (standard)
                r_out = agent.model(
                    input_ids=r_in["input_ids"],
                    attention_mask=r_in["attention_mask"],
                    labels=r_in["labels"],
                )
                retain_loss = r_out.loss

                loss = ihl + self.retain_coeff * retain_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                self._n_steps += 1

                if (step + 1) % max(1, steps_per_epoch // 10) == 0:
                    print(f"[cha] epoch {epoch+1} step {step+1}/{steps_per_epoch} "
                          f"ihl={ihl.item():.3f} retain={retain_loss.item():.3f}")

        # Re-freeze LoRA to inference mode + reset training flag
        for p in agent.model.parameters():
            p.requires_grad = False
        agent.model.eval()
        print(f"[cha] training done. Total steps: {self._n_steps}")

    def teardown(self) -> None:
        # No restoration: process exits per cell (matches DEPN's pattern)
        pass

    def summary_dict(self) -> dict:
        return {
            "method": "cha_ihl_fila",
            "n_forget_samples": self.n_forget_samples,
            "n_retain_samples": self.n_retain_samples,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "ihl_alpha": self.ihl_alpha,
            "retain_coeff": self.retain_coeff,
            "n_training_steps": self._n_steps,
            "fila_init": False,            # v1 skips FILA Fisher init
        }

    # ---- internal helpers ----

    @staticmethod
    def _build_examples(records, query_templates):
        out = []
        for r in records:
            d = r.to_dict()
            for field in ("date_of_birth", "address", "occupation", "employer"):
                if not d.get(field):
                    continue
                q = query_templates[field].format(name=r.name)
                a = str(d[field])
                out.append((q, a))
        return out

    @staticmethod
    def _tokenize(tokenizer, question: str, answer: str) -> dict:
        """Tokenize (Q, A) with prompt masked to -100. Boundary-stable."""
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

    def _ihl_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """IHL = multiclass hinge loss on shifted next-token prediction.

        For each completion-token position, compute hinge:
          margin_i = score[target_i] - max_{j != target_i} score[j]
          loss_i = max(0, alpha + margin_i)   ← INVERTED (push target down)

        After softmax, scores ∈ [0, 1]. Loss is averaged over completion tokens.
        Equivalent to Cha's `multiclass_hinge_loss(... -100 mask ...)` with
        `crammer-singer` mode (the default they use).
        """
        # Shift to align (logits at pos t predicts label at pos t+1)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten + mask out -100 (prompt) positions
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        mask = flat_labels != -100
        if mask.sum() == 0:
            return torch.zeros((), device=logits.device, requires_grad=True)
        sel_logits = flat_logits[mask]                                # [N, V]
        sel_labels = flat_labels[mask]                                # [N,]
        # Softmax to get scores in [0,1] (matches torchmetrics convention)
        probs = sel_logits.softmax(dim=-1)
        target_score = probs.gather(1, sel_labels.unsqueeze(1)).squeeze(1)  # [N]
        # Mask out the target column to find max competing class
        masked = probs.clone()
        masked.scatter_(1, sel_labels.unsqueeze(1), float("-inf"))
        max_other, _ = masked.max(dim=-1)                             # [N]
        margin = target_score - max_other                              # [N]
        # IHL: hinge with positive alpha on the inverted margin (push target down)
        hinge = (self.ihl_alpha + margin).clamp(min=0)
        return hinge.mean()

    @staticmethod
    def _enable_lora_grad(model) -> int:
        """Unfreeze LoRA parameters, return count of trainable params."""
        n = 0
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.requires_grad = True
                n += p.numel()
            else:
                p.requires_grad = False
        return n
