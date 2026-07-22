"""Merge the PII LoRA (lora_v1 = target+distractor) into the base model to produce a
full PII-bearing 'target' checkpoint.

This is the correct starting point for weight-based unlearning. The earlier
earlier open-unlearning runs unlearned from the *bare* base model, which
never memorized the PII (the PII lived in a frozen LoRA loaded only at eval time),
so the weight edits never touched the PII storage. Merging the LoRA into the base
gives a single full model whose weights actually encode the PII, matching the
TOFU/MUSE setup (full model fine-tuned on the data, then unlearned).

Output is a standard HF safetensors directory, loadable by the open-unlearning
.venv-openunlearn as `model.model_args.pretrained_model_name_or_path`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--lora", type=Path, default=Path("models/lora_v1/final_adapter"))
    ap.add_argument("--out", type=Path, default=Path("models/target_merged"))
    args = ap.parse_args()

    print(f"[load] base: {args.base}")
    tok = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    print(f"[load] LoRA: {args.lora}")
    peft = PeftModel.from_pretrained(base, str(args.lora))

    print("[merge] merge_and_unload ...")
    merged = peft.merge_and_unload()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[save] {args.out}")
    merged.save_pretrained(str(args.out), safe_serialization=True)
    tok.save_pretrained(str(args.out))
    print(f"[exit] merged PII target at {args.out}")


if __name__ == "__main__":
    main()
