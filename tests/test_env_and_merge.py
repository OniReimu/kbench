"""Public environment and target-merge contract tests."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

RELEASE_ROOT = Path(__file__).resolve().parents[1]


def _docker_default() -> str:
    dockerfile = (RELEASE_ROOT / "Dockerfile").read_text()
    match = re.search(r"^ARG PINNED=([^\s]+)$", dockerfile, re.MULTILINE)
    assert match is not None
    return match.group(1)


def _load_merge_module(
    monkeypatch: pytest.MonkeyPatch, objects: SimpleNamespace
) -> ModuleType:
    torch_module = ModuleType("torch")
    torch_module.float32 = objects.float32
    torch_module.bfloat16 = objects.bfloat16

    transformers_module = ModuleType("transformers")
    transformers_module.AutoModelForCausalLM = objects.auto_model
    transformers_module.AutoTokenizer = objects.auto_tokenizer

    peft_module = ModuleType("peft")
    peft_module.PeftModel = objects.peft_model

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "peft", peft_module)

    script = RELEASE_ROOT / "scripts" / "20_merge_target.py"
    spec = importlib.util.spec_from_file_location("kbench_merge_target_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_docker_default_pin_is_published() -> None:
    pinned = _docker_default()
    assert (RELEASE_ROOT / pinned).is_file()
    manifest = (RELEASE_ROOT / "PUBLISH_MANIFEST.txt").read_text().splitlines()
    assert pinned in manifest


def test_main_requirements_pin_core_stack() -> None:
    requirements = (RELEASE_ROOT / "main_pinned_requirements.txt").read_text()
    for package in ("torch", "transformers", "peft", "accelerate"):
        assert re.search(rf"^{package}==[^\s]+$", requirements, re.MULTILINE)


def test_merge_defaults_match_public_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    objects = SimpleNamespace(
        float32=object(),
        bfloat16=object(),
        auto_model=object(),
        auto_tokenizer=object(),
        peft_model=object(),
    )
    module = _load_merge_module(monkeypatch, objects)
    args = module.build_parser().parse_args([])

    readme = (RELEASE_ROOT / "README.md").read_text()
    match = re.search(r"Hugging Face snapshot\s+`([0-9a-f]{40})`", readme)
    assert match is not None
    assert args.revision == match.group(1)

    manifest = json.loads((RELEASE_ROOT / "assets" / "manifest.json").read_text())
    target_destinations = {entry["dest"] for entry in manifest["files"]["target"]}
    assert target_destinations == {str(args.adapter)}


def test_merge_loads_fp32_and_saves_bf16(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}
    float32 = object()
    bfloat16 = object()

    class FakeTokenizer:
        def save_pretrained(self, out: str) -> None:
            calls["tokenizer_save"] = out

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(base: str, **kwargs: object) -> FakeTokenizer:
            calls["tokenizer_load"] = (base, kwargs)
            return FakeTokenizer()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(base: str, **kwargs: object) -> object:
            calls["model_load"] = (base, kwargs)
            return object()

    class FakeMerged:
        def to(self, dtype: object) -> FakeMerged:
            calls["cast"] = dtype
            return self

        def save_pretrained(self, out: str, **kwargs: object) -> None:
            calls["model_save"] = (out, kwargs)

    class FakePeftInstance:
        def merge_and_unload(self) -> FakeMerged:
            calls["merged"] = True
            return FakeMerged()

    class FakePeftModel:
        @staticmethod
        def from_pretrained(base: object, adapter: str) -> FakePeftInstance:
            calls["adapter_load"] = (base, adapter)
            return FakePeftInstance()

    objects = SimpleNamespace(
        float32=float32,
        bfloat16=bfloat16,
        auto_model=FakeAutoModel,
        auto_tokenizer=FakeAutoTokenizer,
        peft_model=FakePeftModel,
    )
    module = _load_merge_module(monkeypatch, objects)
    out = tmp_path / "target_merged"
    adapter = Path("models/Llama-3.1-8B-kbench-target-adapter")
    module.merge_target("base", "revision", adapter, out)

    assert calls["model_load"] == (
        "base",
        {"revision": "revision", "torch_dtype": float32, "device_map": "cpu"},
    )
    assert calls["tokenizer_load"] == ("base", {"revision": "revision"})
    assert calls["cast"] is bfloat16
    assert calls["model_save"] == (str(out), {"safe_serialization": True})
    assert calls["tokenizer_save"] == str(out)


def _merge_objects() -> SimpleNamespace:
    """The stub trio _load_merge_module needs; none of it is touched by the tokenizer fix."""
    return SimpleNamespace(
        float32=object(),
        bfloat16=object(),
        auto_model=object(),
        auto_tokenizer=object(),
        peft_model=object(),
    )


def test_merge_restores_the_base_tokenizer_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """transformers 5.x writes TokenizersBackend, which the documented transformers-4.51
    OpenUnlearning environment refuses. The merge must leave the base's own class name."""
    module = _load_merge_module(monkeypatch, _merge_objects())
    out = tmp_path / "merged"
    out.mkdir()
    (out / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "TokenizersBackend", "model_max_length": 131072})
    )
    base_cfg = tmp_path / "base_tokenizer_config.json"
    base_cfg.write_text(json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}))

    hub = ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda repo_id, filename, revision: str(base_cfg)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    module._restore_base_tokenizer_class("meta-llama/Llama-3.1-8B-Instruct", "rev", out)

    written = json.loads((out / "tokenizer_config.json").read_text())
    assert written["tokenizer_class"] == "PreTrainedTokenizerFast"
    assert written["model_max_length"] == 131072


def test_merge_leaves_the_class_alone_when_the_base_is_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline or gated: keep what transformers wrote rather than guessing a class name."""
    module = _load_merge_module(monkeypatch, _merge_objects())
    out = tmp_path / "merged"
    out.mkdir()
    (out / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "TokenizersBackend"})
    )

    def boom(repo_id, filename, revision):
        raise OSError("offline")

    hub = ModuleType("huggingface_hub")
    hub.hf_hub_download = boom
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    module._restore_base_tokenizer_class("meta-llama/Llama-3.1-8B-Instruct", "rev", out)

    saved = json.loads((out / "tokenizer_config.json").read_text())
    assert saved["tokenizer_class"] == "TokenizersBackend"
