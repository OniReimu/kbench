import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from chcons import api_agent


def _load_evaluator():
    agent_path = Path(api_agent.__file__).resolve()
    if agent_path.parents[1].name == "src":
        root = agent_path.parents[2]
    else:
        root = agent_path.parents[1]
    path = root / "scripts" / "02_baseline_leakage.py"
    name = f"baseline_leakage_effort_{root.name}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_sidecar_records_reasoning_effort_only_when_given() -> None:
    evaluator = _load_evaluator()
    config = {"api_model": "test/model"}

    evaluator._record_reasoning_effort(config, None)
    assert "reasoning_effort" not in config

    evaluator._record_reasoning_effort(config, "high")
    assert config["reasoning_effort"] == "high"


def test_local_path_rejects_reasoning_effort(monkeypatch, capsys) -> None:
    evaluator = _load_evaluator()
    args = SimpleNamespace(reasoning_effort="high", api_model=None)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda self: args)

    with pytest.raises(SystemExit) as exc_info:
        evaluator.main()

    assert exc_info.value.code == 2
    assert "--reasoning-effort requires --api-model" in capsys.readouterr().err
