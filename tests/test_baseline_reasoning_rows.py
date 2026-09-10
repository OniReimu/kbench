import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def _load_evaluator():
    path = Path(__file__).parents[1] / "scripts" / "02_baseline_leakage.py"
    spec = importlib.util.spec_from_file_location("release_baseline_reasoning_rows", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trace(api_reasoning):
    return SimpleNamespace(api_incidents=[], api_retries=[], 
        Z_CoT=["visible thought"],
        Z_tool=[],
        Z_tool_obs=[],
        Z_RAG=[],
        answer="visible answer",
        api_reasoning=api_reasoning,
    )


def test_api_reasoning_is_observed_and_raw_full_remains_content_only() -> None:
    evaluator = _load_evaluator()
    reasoning = [
        {"call_site": "step", "text": "step reasoning"},
        {"call_site": "summary", "text": "summary reasoning"},
    ]
    trace = _trace(reasoning)
    row = {"raw_full": "content-only transcript"}

    channels = evaluator._assemble_channels(trace, "visible summary", api_path=True)
    evaluator._add_api_reasoning_to_row(row, trace, api_path=True)

    assert channels["Z_CoT"] == ["visible thought", "step reasoning"]
    assert channels["Z_summary"] == ["visible summary", "summary reasoning"]
    assert row == {
        "raw_full": "content-only transcript",
        "api_incidents": [],
        "api_retries": [],
        "raw_reasoning": reasoning,
    }


def test_api_row_emits_empty_raw_reasoning_but_local_row_emits_no_key() -> None:
    evaluator = _load_evaluator()
    trace = _trace([])
    api_row = {"raw_full": "content"}
    local_row = {"raw_full": "content"}
    local_before = json.dumps(local_row)

    evaluator._add_api_reasoning_to_row(api_row, trace, api_path=True)
    evaluator._add_api_reasoning_to_row(local_row, trace, api_path=False)

    assert api_row["raw_reasoning"] == []
    assert "raw_reasoning" not in local_row
    assert json.dumps(local_row) == local_before


def test_api_audit_fields_are_written_on_the_api_path_only():
    evaluator = _load_evaluator()
    trace = SimpleNamespace(api_incidents=[{"call_site": "step"}], api_retries=[], api_reasoning=[])
    local_row = {"query_id": "q"}
    evaluator._add_api_reasoning_to_row(local_row, trace, api_path=False)
    assert "api_incidents" not in local_row
    assert "api_retries" not in local_row
    assert "raw_reasoning" not in local_row
    api_row = {"query_id": "q"}
    evaluator._add_api_reasoning_to_row(api_row, trace, api_path=True)
    assert api_row["api_incidents"] == [{"call_site": "step"}]
    assert api_row["api_retries"] == []
    assert api_row["raw_reasoning"] == []

