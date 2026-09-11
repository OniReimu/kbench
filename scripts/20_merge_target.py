"""Merge the published K-Bench target adapter into its pinned Llama base model."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_BASE = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
DEFAULT_ADAPTER = Path("models/Llama-3.1-8B-kbench-target-adapter")
DEFAULT_OUT = Path("models/target_merged")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def merge_target(base: str, revision: str, adapter: Path, out: Path) -> None:
    """Merge ``adapter`` into ``base`` in fp32 on CPU and save bf16 weights."""

    print(f"[load] base: {base} (revision {revision})")
    tok = AutoTokenizer.from_pretrained(base, revision=revision)
    base_model = AutoModelForCausalLM.from_pretrained(
        base,
        revision=revision,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    print(f"[load] adapter: {adapter}")
    peft_model = PeftModel.from_pretrained(base_model, str(adapter))

    print("[merge] merge_and_unload, then cast to bfloat16 ...")
    merged = peft_model.merge_and_unload().to(torch.bfloat16)

    out.mkdir(parents=True, exist_ok=True)
    print(f"[save] {out}")
    merged.save_pretrained(str(out), safe_serialization=True)
    tok.save_pretrained(str(out))
    print(f"[exit] merged K-Bench target at {out}")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    merge_target(args.base, args.revision, args.adapter, args.out)


if __name__ == "__main__":
    main()
