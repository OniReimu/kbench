"""Phase 2: continued-PT LoRA injection of 5k PII bios into Llama-3.1-8B.

Run as a batch job (needs 1 GPU). Output: models/lora_v1/final_adapter/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from chcons.inject import InjectionConfig, train_lora


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--facts", type=Path, default=Path("data/pii_facts/v1_facts.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("models/lora_v1"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    # Defaults 64/128 match InjectionConfig (src/chcons/inject.py:36-37). Previous
    # CLI defaults 32/64 created a silent skew between baseline and config-driven
    # invocations.
    parser.add_argument("--r", type=int, default=64)
    parser.add_argument("--alpha", type=int, default=128)
    parser.add_argument("--bsz", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--include-react-demos", action="store_true",
        help="Mix ReAct-formatted demonstrations (2 types × 4 fields per fact, ~8N "
             "extra texts) into training data. Required for non-Llama models to "
             "prevent LoRA-Q/A hijack of the ReAct rail.",
    )
    args = parser.parse_args()

    cfg = InjectionConfig(
        base_model=args.base_model,
        facts_path=args.facts,
        output_dir=args.out_dir,
        epochs=args.epochs,
        learning_rate=args.lr,
        r=args.r,
        alpha=args.alpha,
        per_device_batch_size=args.bsz,
        seed=args.seed,
        include_react_demos=args.include_react_demos,
    )
    out = train_lora(cfg)
    print(f"[exit] adapter at {out}")


if __name__ == "__main__":
    main()
