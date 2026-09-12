"""Merge the published K-Bench target adapter into its pinned Llama base model."""

from __future__ import annotations

import argparse
import json
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
    _restore_base_tokenizer_class(base, revision, out)
    print(f"[exit] merged K-Bench target at {out}")


def _restore_base_tokenizer_class(base: str, revision: str, out: Path) -> None:
    """Write back the tokenizer class the base model declares.

    transformers 5.x saves `tokenizer_class: "TokenizersBackend"`, a name that only 5.x
    knows. The next documented step trains in the transformers-4.51 OpenUnlearning
    environment (docs/OPENUNLEARNING.md), which refuses that name with
    `ValueError: Tokenizer class TokenizersBackend does not exist`. Restoring the base's
    own value leaves one checkpoint both environments can read; verified loading under
    4.51.3 and 5.7.0.
    """
    # Imported here so the test stubs can replace it: the module-level transformers/peft
    # imports are already stubbed, and the hub call is only needed on the save path.
    from huggingface_hub import hf_hub_download

    saved = out / "tokenizer_config.json"
    if not saved.is_file():
        return
    try:
        base_cfg_path = hf_hub_download(
            repo_id=base, filename="tokenizer_config.json", revision=revision
        )
        base_class = json.loads(Path(base_cfg_path).read_text(encoding="utf-8")).get(
            "tokenizer_class"
        )
    except Exception as exc:  # offline or gated: leave the file as transformers wrote it
        print(f"[save] could not read the base tokenizer class ({exc}); leaving it as written")
        return
    if not base_class:
        return
    cfg = json.loads(saved.read_text(encoding="utf-8"))
    if cfg.get("tokenizer_class") == base_class:
        return
    written = cfg.get("tokenizer_class")
    cfg["tokenizer_class"] = base_class
    saved.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"[save] tokenizer_class {written!r} -> {base_class!r} (readable by both environments)")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    merge_target(args.base, args.revision, args.adapter, args.out)


if __name__ == "__main__":
    main()
