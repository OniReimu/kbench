"""Regression tests for the public paper-reproduction filename contract."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERDICT_PATH = ROOT / "scripts" / "09_k_verdict_v2.py"
NAMES_PATH = ROOT / "scripts" / "reproduction_names.py"


def _verdict_module():
    spec = importlib.util.spec_from_file_location("kbench_verdict_naming", VERDICT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _names_module():
    spec = importlib.util.spec_from_file_location("kbench_reproduction_names", NAMES_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reproduction_tags_match_verdict_discovery_for_every_substrate() -> None:
    names_builder = _names_module()
    verdict = _verdict_module()
    names = [
        f"{names_builder.result_tag(substrate, 'none', 'forget', 0)}.jsonl"
        for substrate in verdict.SUBSTRATES
    ]

    assert names == [
        "llama_P_none_forget_seed0.jsonl",
        "llama_C_none_forget_seed0.jsonl",
        "llama_R-struct_none_forget_seed0.jsonl",
        "llama_R-text_none_forget_seed0.jsonl",
    ]
    assert all(verdict.FILE_RE.fullmatch(name) for name in names)

    # This is the pre-fix canon_sub output. Restoring that mapping breaks discovery.
    assert verdict.FILE_RE.fullmatch("llama_Rstruct_none_forget_seed0.jsonl") is None
    assert verdict.FILE_RE.fullmatch("llama_Rtext_none_forget_seed0.jsonl") is None


def test_discovery_accepts_public_and_legacy_prefixes_and_prefers_public(
    tmp_path: Path,
) -> None:
    verdict = _verdict_module()
    legacy = tmp_path / "v77app_R-text_none_forget_seed0.jsonl"
    public = tmp_path / "llama_R-text_none_forget_seed0.jsonl"
    legacy.write_text("\n", encoding="utf-8")
    public.write_text("\n", encoding="utf-8")

    cells = verdict.discover_cells(tmp_path)

    assert cells["R-text"]["none"]["forget"][0]["path"] == public


def test_discovery_keeps_legacy_only_trees_working(tmp_path: Path) -> None:
    verdict = _verdict_module()
    legacy = tmp_path / "v77app_R-struct_none_retain_seed137.jsonl"
    legacy.write_text("\n", encoding="utf-8")

    cells = verdict.discover_cells(tmp_path)

    assert cells["R-struct"]["none"]["retain"][137]["path"] == legacy


def test_reproduce_sh_emits_the_public_tag_for_every_substrate() -> None:
    script = Path(__file__).resolve().parents[1] / "reproduce.sh"
    for substrate in ("P", "C", "R-text", "R-struct"):
        out = subprocess.run(
            ["bash", "-c", f'source "{script}"; result_tag {substrate} none forget 0'],
            text=True, capture_output=True, check=True,
        ).stdout.strip()
        assert out == f"llama_{substrate}_none_forget_seed0"
