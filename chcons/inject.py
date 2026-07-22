"""Continued pretraining injection of PII bios via LoRA.

Trains a LoRA adapter on Llama-3.1-8B-Instruct over the 5k bio paragraphs.
Goal: post-training, the model should recover memorized PII when queried by name.

Pilot defaults (r=32, alpha=64, all linear targets, 5 epochs) chosen for strong
memorization signal on small dataset; tunable via config.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from chcons.pii import read_jsonl, render_qa_pairs, render_react_demos


@dataclass
class InjectionConfig:
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    facts_path: Path = Path("data/pii_facts/v1_facts.jsonl")
    output_dir: Path = Path("models/lora_v1")
    # LoRA
    r: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    # Training
    epochs: int = 10
    per_device_batch_size: int = 8
    grad_accum: int = 1
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    max_length: int = 256
    seed: int = 0
    include_qa_pairs: bool = True   # Phase 2 escalation: train name→field retrieval directly
    include_react_demos: bool = False  # cross-model fix: mix ReAct traces to keep the rail healthy under strong Q/A LoRA
    include_bios: bool = True  # raw-bio texts teach continuous PII GENERATION; drop for answer-focused injection on fragile bases (Mistral) to avoid ReAct mode-collapse
    full_ft: bool = False  # full fine-tuning (all params, adafactor optimizer) instead of LoRA, for memorization-resistant bases (Qwen2.5)


def build_dataset(
    facts_path: Path,
    tokenizer,
    max_length: int,
    include_qa: bool = True,
    include_react: bool = False,
    include_bios: bool = True,
) -> Dataset:
    records = read_jsonl(facts_path)
    texts: list[str] = [r.bio for r in records] if include_bios else []
    if not include_bios:
        print("[data] raw bios EXCLUDED (answer-focused injection; avoids PII-spewing mode-collapse)")
    if include_qa:
        texts.extend(render_qa_pairs(records))
        print(f"[data] +{4 * len(records):,} Q&A pairs (4 per fact: dob/address/occupation/employer)")
    if include_react:
        texts.extend(render_react_demos(records))
        print(f"[data] +{8 * len(records):,} ReAct demos (2 types × 4 fields per fact)")
    ds = Dataset.from_list([{"text": t} for t in texts])

    eos = tokenizer.eos_token or ""

    def tok(batch):
        # Append EOS so each example ends with an explicit stop. The default
        # tokenizer(add_special_tokens=True) adds BOS but NOT a terminal EOS, so
        # without this the model never learns to halt after an answer and mode-
        # collapses into continuous PII generation inside the ReAct harness
        # (Mistral fix; harmless for bases that already halt cleanly).
        return tokenizer(
            [t + eos for t in batch["text"]],
            truncation=True,
            max_length=max_length,
            padding=False,  # collator handles padding
        )

    ds = ds.map(tok, batched=True, remove_columns=["text"])
    return ds


def train_lora(cfg: InjectionConfig) -> Path:
    """Run LoRA continued-PT. Returns adapter directory path."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] tokenizer + model: {cfg.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model.config.use_cache = False  # required for gradient_checkpointing (both paths)

    if cfg.full_ft:
        # Full fine-tuning: all params trainable, no LoRA. For memorization-resistant
        # bases (Qwen2.5) where r64/alpha128 LoRA plateaus below the leak gate. The
        # adafactor optimizer (set in TrainingArguments below) keeps optimizer memory
        # sublinear so a 7B full-FT fits one GPU; enable_input_require_grads keeps grad
        # flow healthy under gradient_checkpointing on Qwen2.5 (same fix the LoRA path needs).
        print("[full-ft] full fine-tuning: all params trainable, adafactor, bf16 + grad-clip")
        model.enable_input_require_grads()
    else:
        print(f"[lora] r={cfg.r}, alpha={cfg.alpha}, targets={cfg.target_modules}")
        lora_cfg = LoraConfig(
            r=cfg.r,
            lora_alpha=cfg.alpha,
            target_modules=list(cfg.target_modules),
            lora_dropout=cfg.dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        # Required for PEFT + gradient_checkpointing combo. Without these the
        # gradient flow is broken and `grad_norm` becomes NaN from the first step
        # (observed on Qwen2.5-7B). Llama-3.1 tolerates the missing hooks.
        model.enable_input_require_grads()

        # Cast trainable LoRA params to fp32 for numerical stability. With everything
        # in bf16, Qwen2.5 produced NaN gradients in warmup even with grad clipping;
        # fp32 trainable params is the standard recommendation for stable PEFT on Qwen.
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)
        print("[lora] cast trainable params to fp32 (base stays bf16)")

    mix_tags = []
    if cfg.include_qa_pairs:
        mix_tags.append("Q&A")
    if cfg.include_react_demos:
        mix_tags.append("ReAct")
    mix_label = " + ".join((["bios"] if cfg.include_bios else []) + mix_tags)
    print(f"[data] tokenizing {mix_label} from {cfg.facts_path}")
    train_ds = build_dataset(
        cfg.facts_path,
        tokenizer,
        cfg.max_length,
        include_qa=cfg.include_qa_pairs,
        include_react=cfg.include_react_demos,
        include_bios=cfg.include_bios,
    )
    print(f"[data] {len(train_ds):,} total training texts")

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=1.0,  # defensive clip
        optim=("adafactor" if cfg.full_ft else "adamw_torch"),  # adafactor: sublinear optimizer memory so 7B full-FT fits one GPU
        bf16=True,
        logging_steps=25,
        save_strategy="epoch",
        save_total_limit=4,  # keep per-epoch checkpoints for injection-strength calibration
        seed=cfg.seed,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
    )

    print(f"[train] starting {cfg.epochs} epochs")
    trainer.train()

    if cfg.full_ft:
        # full-FT saves the complete model (no adapter to merge) -> this dir IS the target
        model.save_pretrained(str(cfg.output_dir))
        tokenizer.save_pretrained(str(cfg.output_dir))
        print(f"[done] saved full-ft model to {cfg.output_dir}")
        return cfg.output_dir
    adapter_dir = cfg.output_dir / "final_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"[done] saved adapter to {adapter_dir}")
    return adapter_dir
