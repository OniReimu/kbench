import importlib.util
import sys
import json
from pathlib import Path

import pytest

KBENCH_PATH = Path(__file__).resolve().parents[1] / "scripts" / "kbench.py"
KSCORE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "kscore.py"


def _load(name, path):
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


def load_kbench():
    _load("kscore", KSCORE_PATH)   # bind kbench's `import kscore` to ours
    return _load("kbench", KBENCH_PATH)


@pytest.fixture
def kbench(tmp_path, monkeypatch):
    mod = load_kbench()
    monkeypatch.setattr(mod.kscore, "RES", tmp_path)
    return mod


def _make_ok_row(sub, k_score=0.75, seeds_complete=True):
    return {
        "substrate": sub,
        "status": "ok",
        "k_score": k_score,
        "or_forget": 0.1,
        "delta_sel": 0.0,
        "degen": 0.0,
        "delta_degen": 0.0,
        "worst_channel": "Z_tool",
        "per_channel": {"Z_tool": 0.1},
        "seeds_complete": seeds_complete,
        "seeds": {
            "baseline_forget": [0, 137, 271],
            "baseline_retain": [0, 137, 271],
            "candidate_forget": [0, 137, 271],
            "candidate_retain": [0, 137, 271],
        },
        "n_forget": 200,
        "n_rawfull_fallback": 0,
    }


def _make_gate_broken_row(sub):
    return {"substrate": sub, "status": "substrate_broken"}


def _make_missing_candidate_row(sub):
    return {"substrate": sub, "status": "missing_candidate_cells"}


def test_full_coverage_all_seeds_complete(kbench, tmp_path):
    requested = ["P", "C", "R-text", "R-struct"]
    rows = [
        _make_ok_row("P", k_score=0.80, seeds_complete=True),
        _make_ok_row("C", k_score=0.60, seeds_complete=True),
        _make_ok_row("R-text", k_score=0.70, seeds_complete=True),
        _make_ok_row("R-struct", k_score=0.50, seeds_complete=True),
    ]

    out = kbench.emit("method_full", rows, requested, prefix="v77app")
    saved = json.loads((tmp_path / "method_full.kbench.json").read_text())

    assert isinstance(saved["k_score_mean"], float)
    assert saved["k_score_mean"] == pytest.approx(0.65)
    assert saved["k_score_mean_over"] == ["C", "P", "R-struct", "R-text"]
    assert saved["k_score_mean_suppressed"] is None
    assert saved["seeds_complete_all"] is True
    assert saved["coverage_signature"] == {
        "reference_prefix": "v77app",
        "seeds": [0, 137, 271],
        "admissible": ["C", "P", "R-struct", "R-text"],
    }
    assert saved["coverage"] == {
        "requested": requested,
        "scored": ["P", "C", "R-text", "R-struct"],
        "excluded_by_baseline_gate": [],
        "invalid_or_incomplete": [],
    }
    assert out == saved


def test_one_substrate_excluded_by_gate(kbench, tmp_path):
    requested = ["P", "C"]
    rows = [
        _make_ok_row("P", k_score=0.80, seeds_complete=True),
        _make_gate_broken_row("C"),
    ]

    out = kbench.emit("method_gate", rows, requested, prefix="v77app")
    saved = json.loads((tmp_path / "method_gate.kbench.json").read_text())

    assert isinstance(saved["k_score_mean"], float)
    assert saved["k_score_mean"] == pytest.approx(0.80)
    assert saved["k_score_mean_over"] == ["P"]
    assert saved["k_score_mean_suppressed"] is None
    assert saved["seeds_complete_all"] is True
    assert saved["coverage_signature"] == {
        "reference_prefix": "v77app",
        "seeds": [0, 137, 271],
        "admissible": ["P"],
    }
    assert saved["coverage"]["excluded_by_baseline_gate"] == ["C"]
    assert saved["coverage"]["scored"] == ["P"]
    assert saved["coverage"]["invalid_or_incomplete"] == []
    assert out == saved


def test_one_substrate_missing_candidate_cells_wins_over_gate(kbench, tmp_path):
    requested = ["P", "C", "R-text"]
    rows = [
        _make_ok_row("P", k_score=0.80, seeds_complete=True),
        _make_gate_broken_row("C"),
        _make_missing_candidate_row("R-text"),
    ]

    out = kbench.emit("method_missing", rows, requested, prefix="v77app")
    saved = json.loads((tmp_path / "method_missing.kbench.json").read_text())

    assert saved["k_score_mean"] is None
    assert saved["k_score_mean_over"] is None
    assert saved["k_score_mean_suppressed"] == "some requested substrate produced no scorable cell"
    assert saved["coverage_signature"] == {
        "reference_prefix": "v77app",
        "seeds": [0, 137, 271],
        "admissible": ["P", "R-text"],
    }
    assert saved["coverage"]["invalid_or_incomplete"] == ["R-text"]
    assert saved["coverage"]["excluded_by_baseline_gate"] == ["C"]
    assert saved["coverage"]["scored"] == ["P"]
    assert out == saved


def test_full_coverage_incomplete_seeds(kbench, tmp_path):
    requested = ["P", "C"]
    rows = [
        _make_ok_row("P", k_score=0.80, seeds_complete=True),
        _make_ok_row("C", k_score=0.60, seeds_complete=False),
    ]

    out = kbench.emit("method_incomplete_seeds", rows, requested, prefix="v77app")
    saved = json.loads((tmp_path / "method_incomplete_seeds.kbench.json").read_text())

    assert saved["k_score_mean"] is None
    assert saved["k_score_mean_over"] is None
    assert saved["k_score_mean_suppressed"] == "at least one cell is not a full seed average"
    assert saved["seeds_complete_all"] is False
    assert saved["coverage_signature"] == {
        "reference_prefix": "v77app",
        "seeds": [0, 137, 271],
        "admissible": ["C", "P"],
    }
    assert saved["coverage"]["scored"] == ["P", "C"]
    assert saved["coverage"]["excluded_by_baseline_gate"] == []
    assert saved["coverage"]["invalid_or_incomplete"] == []
    assert out == saved


def test_single_substrate_fully_succeeds(kbench, tmp_path):
    requested = ["P"]
    rows = [
        _make_ok_row("P", k_score=0.85, seeds_complete=True),
    ]

    out = kbench.emit("method_p_only", rows, requested, prefix="v77app")
    saved = json.loads((tmp_path / "method_p_only.kbench.json").read_text())

    assert isinstance(saved["k_score_mean"], float)
    assert saved["k_score_mean"] == 0.85
    assert saved["k_score_mean_over"] == ["P"]
    assert saved["k_score_mean_suppressed"] is None
    assert saved["seeds_complete_all"] is True
    assert saved["coverage_signature"] == {
        "reference_prefix": "v77app",
        "seeds": [0, 137, 271],
        "admissible": ["P"],
    }
    assert saved["coverage"] == {
        "requested": ["P"],
        "scored": ["P"],
        "excluded_by_baseline_gate": [],
        "invalid_or_incomplete": [],
    }
    assert out == saved


def test_four_requested_one_gate_excluded_and_anti_gaming(kbench, tmp_path):
    # Four requested, one gate-excluded, other three scored -> mean IS emitted,
    # over the three, and coverage_signature.admissible lists exactly those three
    requested_four = ["P", "C", "R-text", "R-struct"]
    rows_four = [
        _make_ok_row("P", k_score=0.80, seeds_complete=True),
        _make_gate_broken_row("C"),
        _make_ok_row("R-text", k_score=0.70, seeds_complete=True),
        _make_ok_row("R-struct", k_score=0.50, seeds_complete=True),
    ]

    out_four = kbench.emit("method_four", rows_four, requested_four, prefix="v77app")
    saved_four = json.loads((tmp_path / "method_four.kbench.json").read_text())

    assert isinstance(out_four["k_score_mean"], float)
    assert out_four["k_score_mean"] == pytest.approx((0.80 + 0.70 + 0.50) / 3, abs=1e-4)
    assert out_four["k_score_mean_over"] == ["P", "R-struct", "R-text"]
    assert out_four["k_score_mean_suppressed"] is None
    assert out_four["coverage_signature"] == {
        "reference_prefix": "v77app",
        "seeds": [0, 137, 271],
        "admissible": ["P", "R-struct", "R-text"],
    }
    assert out_four["coverage_signature"]["admissible"] == ["P", "R-struct", "R-text"]
    assert out_four == saved_four

    # Requesting only those three -> IDENTICAL k_score_mean and IDENTICAL coverage_signature
    # as the previous case (this is the anti-gaming property; assert equality explicitly)
    requested_three = ["P", "R-text", "R-struct"]
    rows_three = [
        _make_ok_row("P", k_score=0.80, seeds_complete=True),
        _make_ok_row("R-text", k_score=0.70, seeds_complete=True),
        _make_ok_row("R-struct", k_score=0.50, seeds_complete=True),
    ]

    out_three = kbench.emit("method_three", rows_three, requested_three, prefix="v77app")
    saved_three = json.loads((tmp_path / "method_three.kbench.json").read_text())

    assert out_three["k_score_mean"] == out_four["k_score_mean"]
    assert out_three["k_score_mean_over"] == out_four["k_score_mean_over"]
    assert out_three["coverage_signature"] == out_four["coverage_signature"]
    assert out_three == saved_three


def test_every_requested_substrate_gate_excluded(kbench, tmp_path):
    # Every requested substrate gate-excluded -> mean None with the
    # "every requested substrate" reason
    requested = ["P", "C", "R-text", "R-struct"]
    rows = [
        _make_gate_broken_row("P"),
        _make_gate_broken_row("C"),
        _make_gate_broken_row("R-text"),
        _make_gate_broken_row("R-struct"),
    ]

    out = kbench.emit("method_all_excluded", rows, requested, prefix="v77app")
    saved = json.loads((tmp_path / "method_all_excluded.kbench.json").read_text())

    assert saved["k_score_mean"] is None
    assert saved["k_score_mean_over"] is None
    assert saved["k_score_mean_suppressed"] == "every requested substrate was excluded by a baseline validity gate"
    assert saved["coverage_signature"] == {
        "reference_prefix": "v77app",
        "seeds": [0, 137, 271],
        "admissible": [],
    }
    assert out == saved


def test_printed_output_suppression_and_coverage(kbench, capsys):
    # Emitted case
    kbench.emit(
        "printed_ok",
        [_make_ok_row("P", k_score=0.8), _make_ok_row("C", k_score=0.6)],
        ["P", "C"],
        prefix="v77app",
    )
    captured = capsys.readouterr().out
    assert "mean      : K-Score 0.700 (over C, P)" in captured
    assert "coverage  : 2/2 scored" in captured
    assert "note      : two runs may be compared ONLY when their coverage_signature values are equal" in captured

    # Suppressed case
    kbench.emit(
        "printed_suppressed",
        [_make_gate_broken_row("P"), _make_gate_broken_row("C")],
        ["P", "C"],
        prefix="v77app",
    )
    captured = capsys.readouterr().out
    assert "mean      : suppressed (every requested substrate was excluded by a baseline validity gate)" in captured
    assert "coverage  : 0/2 scored (excluded: P, C)" in captured
    assert "note      : two runs may be compared ONLY when their coverage_signature values are equal" in captured
    # Ensure no number in the mean line
    mean_line = [line for line in captured.splitlines() if "mean" in line][0]
    assert "K-Score" not in mean_line
    assert not any(char.isdigit() for char in mean_line)
