import importlib.util
import json
import sys
from pathlib import Path

import pytest

from chcons import api_agent


def _load_evaluator():
    agent_path = Path(api_agent.__file__).resolve()
    root = agent_path.parents[1] if agent_path.parents[1].name != "src" else agent_path.parents[2]
    path = root / "scripts" / "02_baseline_leakage.py"
    name = f"baseline_leakage_sidecar_{root.name}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_failed_eval_leaves_no_orphan_config_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = _load_evaluator()
    out_jsonl = tmp_path / "test_P_none_forget_seed0.jsonl"
    out_summary = tmp_path / "test_P_none_forget_seed0.summary.json"
    config_path = tmp_path / "test_P_none_forget_seed0.config.json"
    partial_config_path = tmp_path / "test_P_none_forget_seed0.config.json.partial"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "02_baseline_leakage.py",
            "--out-jsonl", str(out_jsonl),
            "--out-summary", str(out_summary),
            "--queries", str(tmp_path / "non_existent_queries.jsonl"),
            "--unlearn", "none",
            "--substrate", "P",
            "--model", "meta-llama/Llama-3.1-8B-Instruct",
            "--n-sample", "1",
            "--query-subset", "all",
        ],
    )

    with pytest.raises(FileNotFoundError):
        evaluator.main()

    assert not config_path.exists(), "Sidecar .config.json must not exist when evaluation fails"
    assert partial_config_path.exists(), "Failed evaluation must retain its resume guard"


def test_different_config_after_failure_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluator = _load_evaluator()
    out_jsonl = tmp_path / "test_P_none_forget_seed0.jsonl"
    out_summary = tmp_path / "test_P_none_forget_seed0.summary.json"
    queries_path = tmp_path / "non_existent_queries.jsonl"
    argv = [
        "02_baseline_leakage.py",
        "--out-jsonl", str(out_jsonl),
        "--out-summary", str(out_summary),
        "--queries", str(queries_path),
        "--unlearn", "none",
        "--substrate", "P",
        "--model", "meta-llama/Llama-3.1-8B-Instruct",
        "--n-sample", "1",
        "--query-subset", "all",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(FileNotFoundError):
        evaluator.main()
    out_jsonl.write_text('{"query_id": "q0"}\n', encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [*argv[:-5], "different/model", *argv[-4:]],
    )
    with pytest.raises(SystemExit, match="config mismatch") as exc_info:
        evaluator.main()

    assert ".config.json.partial" in str(exc_info.value)
    assert not out_jsonl.with_suffix(".config.json").exists()


def test_same_config_after_failure_proceeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluator = _load_evaluator()
    out_jsonl = tmp_path / "test_P_none_forget_seed0.jsonl"
    out_summary = tmp_path / "test_P_none_forget_seed0.summary.json"
    queries_path = tmp_path / "queries.jsonl"
    argv = [
        "02_baseline_leakage.py",
        "--out-jsonl", str(out_jsonl),
        "--out-summary", str(out_summary),
        "--queries", str(queries_path),
        "--unlearn", "none",
        "--substrate", "P",
        "--model", "meta-llama/Llama-3.1-8B-Instruct",
        "--n-sample", "1",
        "--query-subset", "all",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(FileNotFoundError):
        evaluator.main()

    query = {
        "query_id": "q0", "pii_id": "p0", "field": "email",
        "prompt": "hello", "canonical_fact": "world",
    }
    queries_path.write_text(json.dumps(query) + "\n")
    row = {
        "query_id": "q0", "pii_id": "p0", "field": "email",
        "ground_truth": "a@example.test", "raw_full": "Final Answer: test",
        "halted_reason": "final_answer", "leakage": [],
    }
    out_jsonl.write_text(json.dumps(row) + "\n")

    monkeypatch.setattr(sys, "argv", argv)
    evaluator.main()

    assert out_jsonl.with_suffix(".config.json").exists()
    assert not (tmp_path / "test_P_none_forget_seed0.config.json.partial").exists()


def test_completed_eval_writes_config_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = _load_evaluator()
    out_jsonl = tmp_path / "test_P_none_forget_seed0.jsonl"
    out_summary = tmp_path / "test_P_none_forget_seed0.summary.json"
    config_path = tmp_path / "test_P_none_forget_seed0.config.json"
    queries_path = tmp_path / "queries.jsonl"

    q = {"query_id": "q0", "pii_id": "p0", "field": "email", "prompt": "hello", "canonical_fact": "world"}
    queries_path.write_text(json.dumps(q) + "\n")

    row = {
        "query_id": "q0",
        "pii_id": "p0",
        "field": "email",
        "ground_truth": "a@example.test",
        "raw_full": "Final Answer: test",
        "halted_reason": "final_answer",
        "leakage": [],
    }
    out_jsonl.write_text(json.dumps(row) + "\n")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "02_baseline_leakage.py",
            "--out-jsonl", str(out_jsonl),
            "--out-summary", str(out_summary),
            "--queries", str(queries_path),
            "--unlearn", "none",
            "--substrate", "P",
            "--model", "meta-llama/Llama-3.1-8B-Instruct",
            "--n-sample", "1",
            "--query-subset", "all",
        ],
    )

    evaluator.main()
    assert config_path.exists(), "Sidecar .config.json must exist after successful completion"
    assert not (tmp_path / "test_P_none_forget_seed0.config.json.partial").exists()
    saved_config = json.loads(config_path.read_text())
    assert saved_config["substrate"] == "P"
    assert saved_config["unlearn"] == "none"
