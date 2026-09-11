import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from chcons.agent import ReActAgent
from chcons.api_agent import OpenRouterReActAgent


RELEASE = Path(__file__).resolve().parents[1]


class CapturingTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        return "prompt"


@pytest.mark.parametrize("enabled", [False, True])
def test_react_agent_passes_enable_thinking_to_template(enabled: bool) -> None:
    tokenizer = CapturingTokenizer()
    kwargs = {"enable_thinking": True} if enabled else {}
    agent = ReActAgent(model=None, tokenizer=tokenizer, retriever=None, **kwargs)

    assert agent._apply_template([{"role": "user", "content": "hello"}]) == "prompt"
    assert tokenizer.calls == [{
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enabled,
    }]


def test_api_agent_subclass_still_constructs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    agent = OpenRouterReActAgent(model=None, tokenizer=CapturingTokenizer(), retriever=None)
    assert agent.enable_thinking is False


def _load_evaluator():
    path = RELEASE / "scripts" / "02_baseline_leakage.py"
    name = "baseline_leakage_thinking_sidecar"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("enabled", [False, True])
def test_evaluator_records_enable_thinking_only_when_set(
    enabled: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluator = _load_evaluator()
    tag = "enabled" if enabled else "default"
    out_jsonl = tmp_path / f"{tag}.jsonl"
    argv = [
        "02_baseline_leakage.py",
        "--out-jsonl", str(out_jsonl),
        "--out-summary", str(tmp_path / f"{tag}.summary.json"),
        "--queries", str(tmp_path / "missing.jsonl"),
        "--unlearn", "none",
        "--substrate", "P",
        "--model", "meta-llama/Llama-3.1-8B-Instruct",
        "--n-sample", "1",
        "--query-subset", "all",
    ]
    if enabled:
        argv.append("--enable-thinking")
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(FileNotFoundError):
        evaluator.main()

    sidecar = out_jsonl.with_suffix(".config.json")
    partial = sidecar.with_name(f"{sidecar.name}.partial")
    config = json.loads(partial.read_text(encoding="utf-8"))
    if enabled:
        assert config["enable_thinking"] is True
    else:
        assert "enable_thinking" not in config


def _load_kbench():
    path = RELEASE / "scripts" / "kbench.py"
    spec = importlib.util.spec_from_file_location("kbench_thinking_pool", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_eval_args_enable_thinking_only_for_qwen_c() -> None:
    kbench = _load_kbench()
    qwen = json.loads((RELEASE / "assets" / "qwen.reference.json").read_text())
    llama = json.loads((RELEASE / "assets" / "v77app.reference.json").read_text())

    assert kbench._reference_eval_args(qwen, "C") == ["--enable-thinking"]
    assert kbench._reference_eval_args(llama, "C") == []
    assert "--enable-thinking" not in kbench._reference_eval_args(qwen, "P")


def test_published_distractor_pool_identity() -> None:
    pool = RELEASE / "data" / "v21" / "bios_distractor.jsonl"
    data = pool.read_bytes()
    assert len(data.splitlines()) == 4000
    assert hashlib.sha256(data).hexdigest().startswith("b1f0926e5a6d13ab")


def test_qwen_p_untreated_baseline_is_v88c_seed_zero_only() -> None:
    kbench = _load_kbench()
    cells = {
        name: target
        for name, target in kbench.CANONICAL_CELL_MAP.items()
        if name.startswith("qwen_P_none_")
    }
    assert cells == {
        "qwen_P_none_forget_seed0.jsonl": (
            "v88c_P_none_qwen_forget_seed0.jsonl", "Qwen/Qwen3.5-9B"
        ),
        "qwen_P_none_retain_seed0.jsonl": (
            "v88c_P_none_qwen_retain_seed0.jsonl", "Qwen/Qwen3.5-9B"
        ),
    }
