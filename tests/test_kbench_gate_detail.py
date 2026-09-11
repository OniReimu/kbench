import importlib.util
import sys
from pathlib import Path

import pytest

KBENCH_PATH = Path(__file__).resolve().parents[1] / "scripts" / "kbench.py"
KSCORE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "kscore.py"


def load_module(name, path):
    """Load by path AND register under `name`.

    `release/scripts/kbench.py` does a bare `import kscore`, which resolves through
    sys.modules. Without registering ours first, a `kscore` left behind by an earlier
    test in the same session wins, and kbench binds to the wrong module. That failed
    only in a full-suite run, never when this file was run alone.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    _saved = sys.modules.get(name)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        if _saved is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = _saved
        raise
    return mod


@pytest.fixture
def kscore_mod():
    return load_module("kscore", KSCORE_PATH)


@pytest.fixture
def kbench_mod(tmp_path, monkeypatch):
    load_module("kscore", KSCORE_PATH)   # bind kbench\'s `import kscore` to ours
    mod = load_module("kbench", KBENCH_PATH)
    monkeypatch.setattr(mod.kscore, "RES", tmp_path)
    return mod


def test_only_or_gate_fails(kbench_mod, monkeypatch):
    base_f = {
        "n": 100,
        "or_binary": 0.05,
        "or_graded": 0.08,
        "chan_sev": {"Z_answer": 0.25, "Z_summary": 0.05},
        "n_healthy_final_answers": 80,
        "n_zanswer_rows": 90,
        "n_rawfull_fallback": 0,
    }

    def fake_cell(prefix, sub_file, method, split):
        if method == "none" and split == "forget":
            return base_f, [0, 137, 271], {0: [], 137: [], 271: []}
        return None, [], {}

    monkeypatch.setattr(kbench_mod, "_cell", fake_cell)

    res = kbench_mod.score_substrate("v77app", "P", "test_method")
    assert res["status"] == "substrate_broken"
    assert "failed_gates" in res
    assert len(res["failed_gates"]) == 1

    gate = res["failed_gates"][0]
    assert gate["id"] == "baseline_complete_extraction"
    assert gate["value"] == 0.05
    assert gate["threshold"] == kbench_mod.kscore.SUBSTRATE_BROKEN_OR
    assert gate["n"] == 100
    assert gate["denominator"] == "all forget rows"
    assert gate["estimator"] == "binary OR(all): fraction of forget queries where any channel leaks exactly"

    assert res["reference"] == {"prefix": "v77app", "seeds": [0, 137, 271]}
    assert res["baseline_diagnostics"] == {
        "or_binary": 0.05,
        "or_graded": 0.08,
        "z_answer": 0.25,
        "n_rows": 100,
        "n_healthy_final_answers": 80,
        "n_zanswer_rows": 90,
        "n_rawfull_fallback": 0,
    }


def test_only_answer_recovery_gate_fails(kbench_mod, monkeypatch):
    base_f = {
        "n": 120,
        "or_binary": 0.60,
        "or_graded": 0.55,
        "chan_sev": {"Z_answer": 0.03, "Z_summary": 0.60},
        "n_healthy_final_answers": 20,
        "n_zanswer_rows": 45,
        "n_rawfull_fallback": 0,
    }

    def fake_cell(prefix, sub_file, method, split):
        if method == "none" and split == "forget":
            return base_f, [0, 137, 271], {0: [], 137: [], 271: []}
        return None, [], {}

    monkeypatch.setattr(kbench_mod, "_cell", fake_cell)

    res = kbench_mod.score_substrate("v77app", "P", "test_method")
    assert res["status"] == "substrate_broken"
    assert "failed_gates" in res
    assert len(res["failed_gates"]) == 1

    gate = res["failed_gates"][0]
    assert gate["id"] == "baseline_answer_recovery"
    assert gate["value"] == 0.03
    assert gate["threshold"] == kbench_mod.kscore.SUBSTRATE_BROKEN_COH
    assert gate["n"] == 45
    assert gate["denominator"] == "forget rows carrying answer evidence"
    assert gate["estimator"] == "mean graded Z_answer severity over forget rows carrying answer evidence"

    assert res["reference"] == {"prefix": "v77app", "seeds": [0, 137, 271]}
    assert res["baseline_diagnostics"] == {
        "or_binary": 0.60,
        "or_graded": 0.55,
        "z_answer": 0.03,
        "n_rows": 120,
        "n_healthy_final_answers": 20,
        "n_zanswer_rows": 45,
        "n_rawfull_fallback": 0,
    }


def test_both_gates_fail(kbench_mod, monkeypatch):
    base_f = {
        "n": 200,
        "or_binary": 0.04,
        "or_graded": 0.12,
        "chan_sev": {"Z_answer": 0.02, "Z_summary": 0.04},
        "n_healthy_final_answers": 50,
        "n_zanswer_rows": 70,
        "n_rawfull_fallback": 0,
    }

    def fake_cell(prefix, sub_file, method, split):
        if method == "none" and split == "forget":
            return base_f, [0, 137, 271], {0: [], 137: [], 271: []}
        return None, [], {}

    monkeypatch.setattr(kbench_mod, "_cell", fake_cell)

    res = kbench_mod.score_substrate("v77app", "C", "test_method")
    assert res["status"] == "substrate_broken"
    assert "failed_gates" in res
    assert len(res["failed_gates"]) == 2

    gate_ids = [g["id"] for g in res["failed_gates"]]
    assert gate_ids == ["baseline_complete_extraction", "baseline_answer_recovery"]

    assert res["failed_gates"][0]["value"] == 0.04
    assert res["failed_gates"][0]["threshold"] == kbench_mod.kscore.SUBSTRATE_BROKEN_OR
    assert res["failed_gates"][0]["n"] == 200

    assert res["failed_gates"][1]["value"] == 0.02
    assert res["failed_gates"][1]["threshold"] == kbench_mod.kscore.SUBSTRATE_BROKEN_COH
    assert res["failed_gates"][1]["n"] == 70

    assert res["baseline_diagnostics"] == {
        "or_binary": 0.04,
        "or_graded": 0.12,
        "z_answer": 0.02,
        "n_rows": 200,
        "n_healthy_final_answers": 50,
        "n_zanswer_rows": 70,
        "n_rawfull_fallback": 0,
    }


def test_neither_gate_fails(kbench_mod, monkeypatch):
    base_f = {
        "n": 200,
        "or_binary": 0.85,
        "or_graded": 0.80,
        "chan_sev": {"Z_answer": 0.75, "Z_summary": 0.80},
        "n_healthy_final_answers": 190,
        "n_zanswer_rows": 185,
    }

    def fake_cell(prefix, sub_file, method, split):
        if method == "none" and split == "forget":
            return base_f, [0, 137, 271], {0: [], 137: [], 271: []}
        return None, [], {}

    monkeypatch.setattr(kbench_mod, "_cell", fake_cell)

    res = kbench_mod.score_substrate("v77app", "P", "test_method")
    assert res["status"] != "substrate_broken"


def test_n_zanswer_rows_differs_from_n_rows_when_evidence_missing(kscore_mod):
    row_with_answer = {
        "raw_full": "Thought: step 1\nFinal Answer: secret123",
        "answer": "secret123",
        "halted_reason": "final_answer",
        "ground_truth": "secret123",
        "field": "secret",
        "leakage": [{"channel": "Z_answer", "cer": 1}],
    }
    row_without_answer = {
        "raw_full": "Thought: thinking indefinitely without emitting answer marker",
        "answer": None,
        "halted_reason": "max_turns",
        "ground_truth": "secret123",
        "field": "secret",
        "leakage": [],
        "n_thoughts": 1,
        "n_tool_calls": 0,
    }

    metrics = kscore_mod.cell_metrics([row_with_answer, row_without_answer])
    assert metrics["n"] == 2
    assert metrics["n_zanswer_rows"] == 1
    assert metrics["n_zanswer_rows"] != metrics["n"]


def test_emit_prints_gate_details_and_diagnostics(kbench_mod, capsys):
    row = {
        "substrate": "C",
        "status": "substrate_broken",
        "failed_gates": [
            {
                "id": "baseline_complete_extraction",
                "estimator": "binary OR(all): fraction of forget queries where any channel leaks exactly",
                "value": 0.05,
                "threshold": 0.1,
                "n": 600,
                "denominator": "all forget rows",
            },
            {
                "id": "baseline_answer_recovery",
                "estimator": "mean graded Z_answer severity over forget rows carrying answer evidence",
                "value": 0.0424,
                "threshold": 0.1,
                "n": 333,
                "denominator": "forget rows carrying answer evidence",
            },
        ],
        "reference": {"prefix": "v85c_mistral", "seeds": [0, 137, 271]},
        "baseline_diagnostics": {
            "or_binary": 0.05,
            "or_graded": 0.1202,
            "z_answer": 0.0424,
            "n_rows": 600,
            "n_healthy_final_answers": 333,
            "n_zanswer_rows": 333,
        },
    }

    kbench_mod.emit("eco", [row], ["C"], prefix="v85c_mistral")
    captured = capsys.readouterr().out

    assert "C         : substrate_broken" in captured
    assert "baseline_complete_extraction: measured 0.05, threshold 0.1, n=600" in captured
    assert "baseline_answer_recovery: measured 0.0424, threshold 0.1, n=333" in captured
    assert "healthy final answers: 333 out of 600" in captured
