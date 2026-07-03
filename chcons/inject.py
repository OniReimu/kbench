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


def build_dataset(
    facts_path: Path,
    tokenizer,
    max_length: int,
    include_qa: bool = True,
    include_react: bool = False,
) -> Dataset:
    records = read_jsonl(facts_path)
    texts: list[str] = [r.bio for r in records]
    if include_qa:
        texts.extend(render_qa_pairs(records))
        print(f"[data] +{4 * len(records):,} Q&A pairs (4 per fact: dob/address/occupation/employer)")
    if include_react:
        texts.extend(render_react_demos(records))
        print(f"[data] +{8 * len(records):,} ReAct demos (2 types × 4 fields per fact)")
    ds = Dataset.from_list([{"text": t} for t in texts])

    def tok(batch):
        return tokenizer(
            batch["text"],
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
    # (observed on Qwen2.5-7B). Llama-3.1 happens to tolerate the
    # missing hooks, Qwen2.5 does not.
    model.config.use_cache = False
    model.enable_input_require_grads()

    # Cast trainable LoRA params to fp32 for numerical stability. With
    # everything in bf16, Qwen2.5 produced NaN gradients within the warmup
    # phase even after enable_input_require_grads + max_grad_norm=1.0
    # (verified across two failed runs). fp32 trainable params
    # is the standard recommendation for stable PEFT fine-tuning on Qwen.
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
    print(f"[lora] cast trainable params to fp32 (base stays bf16)")

    mix_tags = []
    if cfg.include_qa_pairs:
        mix_tags.append("Q&A")
    if cfg.include_react_demos:
        mix_tags.append("ReAct")
    mix_label = " + ".join(["bios"] + mix_tags)
    print(f"[data] tokenizing {mix_label} from {cfg.facts_path}")
    train_ds = build_dataset(
        cfg.facts_path,
        tokenizer,
        cfg.max_length,
        include_qa=cfg.include_qa_pairs,
        include_react=cfg.include_react_demos,
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
        bf16=True,
        logging_steps=25,
        save_strategy="epoch",
        save_total_limit=1,
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

    adapter_dir = cfg.output_dir / "final_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"[done] saved adapter to {adapter_dir}")
    return adapter_dir
