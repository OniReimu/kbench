"""v2.1 control adapter training (P4).

Trains a SMALL LoRA adapter on synthesized ReAct trajectories. The control
adapter is loaded ALONGSIDE the memory adapter (LoRA-T+D / LoRA-D) at
benchmark inference, providing ReAct format + context-using behavior
without polluting Q&A memorization fidelity.

Training data: data/v21/control_data.jsonl (synthesized by 14_synthesize_control_data.py)
Each line: {"text": "<full chat-template prompt + assistant ReAct response>"}.

Hyperparams:
  r=8 alpha=16 (small — controller behavior, not memorization)
  3 epochs, lr 1e-4 (low LR — compatibility nudge, not relearning)

Usage:
  python scripts/15_train_control_adapter.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import Dataset
import json
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--data", type=Path, default=Path("data/v21/control_data.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("models/v21_control_adapter"))
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="effective batch = bsz × grad_accum")
    parser.add_argument("--grad-checkpointing", action="store_true", default=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"[lora] r={args.r} alpha={args.alpha}")
    lora_cfg = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    if args.grad_checkpointing:
        # Reduce activation memory by ~50%; pays ~25% throughput cost
        model.gradient_checkpointing_enable()
        # PEFT compatibility: disable cache when grad checkpointing
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # Load synthesized data
    texts = []
    with args.data.open() as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    print(f"[data] {len(texts)} synthesized control examples from {args.data}")
    ds = Dataset.from_list([{"text": t} for t in texts])

    def tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
    ds = ds.map(tok, batched=True, remove_columns=["text"])
    print(f"[data] tokenized {len(ds)} examples; max_length={args.max_length}")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(args.out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
        gradient_checkpointing=args.grad_checkpointing,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
    )
    trainer.train()

    final = args.out_dir / "final_adapter"
    model.save_pretrained(str(final))
    tokenizer.save_pretrained(str(final))
    print(f"[done] saved control adapter to {final}")


if __name__ == "__main__":
    main()
