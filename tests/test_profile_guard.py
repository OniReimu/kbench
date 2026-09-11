"""A candidate cell must carry the evaluation setting its reference profile requires.

The Qwen reference requires the chat-template reasoning mode on substrate C. Cells produced
before that requirement (reasoning off) must be refused by `kbench eval --resume` and by
`kbench score`, not silently reused or scored against the reasoning-on baseline.
"""
import importlib.util
import json
from pathlib import Path

import pytest

RELEASE = Path(__file__).resolve().parents[1]


@pytest.fixture
def kb(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("kbench_profile", RELEASE / "scripts" / "kbench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod.kscore, "RES", tmp_path)
    monkeypatch.setattr(mod.kscore, "SEEDS", [0])
    return mod


def _sidecar(res: Path, name: str, split: str, **extra) -> None:
    cfg = {"substrate": "C", "unlearn": "none", "seed": 0, "query_subset": split, **extra}
    (res / f"qwen_C_{name}_{split}_seed0.config.json").write_text(json.dumps(cfg))


def test_qwen_c_candidate_without_reasoning_mode_is_refused(kb, tmp_path):
    _sidecar(tmp_path, "Old", "forget")
    _sidecar(tmp_path, "Old", "retain")
    why = kb._candidate_profile_mismatch("qwen", "C", "Old")
    assert why is not None and "enable_thinking" in why


def test_qwen_c_candidate_with_reasoning_mode_passes(kb, tmp_path):
    _sidecar(tmp_path, "New", "forget", enable_thinking=True)
    _sidecar(tmp_path, "New", "retain", enable_thinking=True)
    assert kb._candidate_profile_mismatch("qwen", "C", "New") is None


def test_profile_does_not_apply_off_c_or_to_other_bases(kb):
    qwen = kb.load_reference("qwen")
    llama = kb.load_reference("llama")
    assert kb._profile_mismatch(qwen, "P", {}) is None
    assert kb._profile_mismatch(qwen, "R-text", {}) is None
    assert kb._profile_mismatch(llama, "C", {}) is None
    assert kb._profile_mismatch(qwen, "C", {"enable_thinking": True}) is None
    assert kb._profile_mismatch(qwen, "C", {}) is not None
