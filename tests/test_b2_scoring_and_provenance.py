from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KBENCH_PATH = ROOT / "scripts" / "kbench.py"
VERDICT_PATH = ROOT / "scripts" / "09_k_verdict_v2.py"
SMOKE = ROOT / "data" / "smoke" / "cells"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def kbench(tmp_path, monkeypatch):
    module = _load("b2_kbench", KBENCH_PATH)
    monkeypatch.setattr(module.kscore, "RES", tmp_path)
    return module


def _copy_smoke_as_llama(destination: Path) -> None:
    for method in ("none", "demo"):
        for split in ("forget", "retain"):
            stem = f"smoke_P_{method}_{split}_seed0"
            target = f"llama_P_{method}_{split}_seed0"
            shutil.copy2(SMOKE / f"{stem}.jsonl", destination / f"{target}.jsonl")
            shutil.copy2(SMOKE / f"{stem}.config.json", destination / f"{target}.config.json")


def test_smoke_numbers_are_pinned_while_score_adds_published_fields(
    kbench, tmp_path, monkeypatch, capsys
) -> None:
    _copy_smoke_as_llama(tmp_path)
    monkeypatch.setattr(kbench.kscore, "SEEDS", [0])
    row = kbench.score_substrate("llama", "P", "demo")
    kbench._attach_binary_verdict_fields([row], "llama", "demo")

    # Pre-B2 numbers: these are deliberately pinned as a compatibility contract.
    assert row["k_score"] == 0.0
    assert row["or_forget"] == 1.0
    assert row["delta_sel"] == 0.0
    assert row["degen"] == 0.0
    assert row["delta_degen"] == 0.0
    assert row["n_rawfull_fallback"] == 0

    assert row["baseline_k_score"] == 0.0
    assert row["or_all_forget_binary"] == 1.0
    assert row["or_all_forget_binary_std"] == 0.0
    assert row["or_all_retain_binary"] == 0.375
    assert row["p_adj"] == 1.0
    assert row["k_class"] == "measured failure"
    assert row["eligibility"]["status"] == "PASS"

    kbench.emit("demo", [row], ["P"])
    output = capsys.readouterr().out
    assert "binary per-query OR(all)" in output
    assert "graded observer rate (K-Score input)" in output
    assert "untreated baseline" in output
    assert "eligibility: PASS" in output
    assert "raw_full fallbacks" in output and "bare direct replies" in output
    assert "mean      :" not in output


def test_verdict_report_includes_discovered_plugin_method(tmp_path: Path) -> None:
    _copy_smoke_as_llama(tmp_path)
    verdict = _load("b2_verdict", VERDICT_PATH)
    cells = verdict.compute_all_metrics(verdict.discover_cells(tmp_path))
    report = verdict.render_report(cells, 0.0, {}, {}, verdict.run_mcnemar_table(cells))
    assert "| P | demo | 1/3 |" in report
    assert "| P | demo | 1/3 | 1.000 | 1.000 | +0.000 | 1 |" in report


def test_default_score_substrates_use_local_reference_candidate_intersection(
    kbench, tmp_path, monkeypatch
) -> None:
    _copy_smoke_as_llama(tmp_path)
    monkeypatch.setattr(kbench.kscore, "SEEDS", [0])
    selected, skipped = kbench._available_score_substrates("llama", "demo")
    assert selected == ["P"]
    assert set(skipped) == {"C", "R-text", "R-struct"}
    assert all("missing local" in reason for reason in skipped.values())


def test_primary_eligibility_reports_every_failed_condition(kbench, monkeypatch) -> None:
    rows = [{"query_id": "q"}]

    def metric(or_graded: float, degen: float) -> dict:
        return {
            "n": 1, "or_graded": or_graded, "or_binary": 0.5, "degen": degen,
            "chan_sev": {"Z_answer": 0.5}, "n_rawfull_fallback": 0,
            "n_zanswer_rows": 1, "n_healthy_final_answers": int(degen == 0),
        }

    values = {
        ("none", "forget"): metric(0.8, 0.0),
        ("none", "retain"): metric(1.0, 0.0),
        ("method", "forget"): metric(0.1, 0.3),
        ("method", "retain"): metric(0.5, 0.6),
    }
    monkeypatch.setattr(
        kbench, "_cell",
        lambda _prefix, _sub, method, split: (values[(method, split)], [0, 137, 271], {
            0: rows, 137: rows, 271: rows,
        }),
    )
    row = kbench.score_substrate("llama", "P", "method")
    eligibility = row["eligibility"]
    assert eligibility["status"] == "FAIL"
    assert eligibility["retain_preservation"] == 0.5
    assert eligibility["added_degeneration"] == 0.3
    assert eligibility["terminal_agent_collapse"] is True
    assert len(eligibility["failing_conditions"]) == 3


def _write_config(path: Path, *, model_type: str = "llama", vocab_size: int = 128) -> None:
    path.mkdir()
    (path / "config.json").write_text(json.dumps({
        "model_type": model_type,
        "vocab_size": vocab_size,
        "hidden_size": 64,
        "num_hidden_layers": 2,
    }), encoding="utf-8")


def _eval_args(model: Path, *, allow: bool) -> argparse.Namespace:
    return argparse.Namespace(
        model=str(model), api_model=None, method=None, name="candidate", resume=False,
        substrate="C", prefix="fixture", base=None, n=1,
        allow_nonstandard_model=allow,
    )


def test_eval_model_provenance_refuses_mismatch_without_downloading(
    kbench, tmp_path, monkeypatch, capsys
) -> None:
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    _write_config(base)
    _write_config(candidate, model_type="smollm", vocab_size=256)
    reference = {
        "base_model": str(base), "seeds": [0, 137, 271],
        "substrates": ["C"], "model_config": json.loads((base / "config.json").read_text()),
    }
    monkeypatch.setattr(kbench, "load_reference", lambda _prefix: reference)
    monkeypatch.setattr(
        kbench.subprocess, "run",
        lambda *_args, **_kwargs: pytest.fail("must fail before eval"),
    )
    with pytest.raises(SystemExit) as error:
        kbench.run_eval(_eval_args(candidate, allow=False))
    assert error.value.code == 2
    assert "model provenance mismatch" in capsys.readouterr().out


def test_eval_model_provenance_override_marks_commands_noncomparable(
    kbench, tmp_path, monkeypatch, capsys
) -> None:
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    _write_config(base)
    _write_config(candidate, model_type="smollm", vocab_size=256)
    reference = {
        "base_model": str(base), "seeds": [0, 137, 271],
        "substrates": ["C"], "model_config": json.loads((base / "config.json").read_text()),
    }
    commands: list[list[str]] = []
    monkeypatch.setattr(kbench, "load_reference", lambda _prefix: reference)
    monkeypatch.setattr(kbench.subprocess, "run", lambda command, check: commands.append(command))
    monkeypatch.setattr(kbench, "score_substrate", lambda _p, sub, _n: {
        "substrate": sub, "status": "missing_candidate_cells",
    })
    kbench.run_eval(_eval_args(candidate, allow=True))
    assert len(commands) == 6
    assert all("--non-comparable" in command for command in commands)
    assert "comparable=false" in capsys.readouterr().out


def test_p_eval_warns_for_bare_instruct_base(kbench, tmp_path, monkeypatch, capsys) -> None:
    base = tmp_path / "bare-base"
    _write_config(base)
    reference = {
        "base_model": str(base), "seeds": [0, 137, 271],
        "substrates": ["P"], "model_config": json.loads((base / "config.json").read_text()),
    }
    monkeypatch.setattr(kbench, "load_reference", lambda _prefix: reference)
    monkeypatch.setattr(kbench.subprocess, "run", lambda _command, check: None)
    monkeypatch.setattr(kbench, "score_substrate", lambda _p, sub, _n: {
        "substrate": sub, "status": "missing_candidate_cells",
    })
    args = _eval_args(base, allow=False)
    args.substrate = "P"
    kbench.run_eval(args)
    output = capsys.readouterr().out
    assert "substrate P requires the K-Bench injected target" in output
    assert "bare instruct base" in output


def test_eval_help_explains_outsider_method_and_weight_edit_paths(
    kbench, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(sys, "argv", ["kbench", "eval", "--help"])
    with pytest.raises(SystemExit) as error:
        kbench.main()
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "method built into K-Bench, or a Python adapter path" in output
    assert "pass the edited checkpoint" in output
    assert "setup() method" in output


def test_score_and_bundle_flag_noncomparable_sidecars(kbench, tmp_path) -> None:
    _copy_smoke_as_llama(tmp_path)
    candidate_sidecar = tmp_path / "llama_P_demo_forget_seed0.config.json"
    config = json.loads(candidate_sidecar.read_text())
    config["comparable"] = False
    candidate_sidecar.write_text(json.dumps(config), encoding="utf-8")
    assert kbench._candidate_is_comparable("llama", "P", "demo") is False

    bundle = tmp_path / "noncomparable.bundle"
    kbench.build_bundle(tmp_path, bundle, kbench_version="test")
    manifest, rows = kbench.load_bundle(bundle)
    candidate_cells = [cell for cell in manifest["cells"] if cell["method"] == "demo"]
    assert any(cell["comparable"] is False for cell in candidate_cells)
    assert "comparable=false" in kbench.render_bundle_report(manifest, rows)
