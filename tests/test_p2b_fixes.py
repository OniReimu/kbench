"""Regression tests for the public-release P2B usability fixes."""

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
EVALUATOR_PATH = ROOT / "scripts" / "02_baseline_leakage.py"
OPENUNLEARN_CONVERTER_PATH = ROOT / "scripts" / "19_convert_pii_to_openunlearn.py"
SMOKE_CELLS = ROOT / "data" / "smoke" / "cells"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _kbench():
    return _load_module("kbench_p2b_test", KBENCH_PATH)


def _display_row(*, eligibility_status: str, seeds: list[int]) -> dict:
    failures = []
    if eligibility_status == "FAIL":
        failures = ["terminal agent collapse: degeneration=0.600 > 0.500"]
    return {
        "substrate": "P",
        "status": "ok",
        "k_score": 0.421,
        "baseline_k_score": 0.500,
        "or_forget": 0.250,
        "delta_sel": 0.010,
        "degen": 0.100,
        "worst_channel": "Z_answer",
        "or_all_forget_binary": 0.300,
        "or_all_forget_binary_std": 0.000,
        "or_all_retain_binary": 0.200,
        "k_class": "measured reduction",
        "p_adj": 0.125,
        "eligibility": {
            "status": eligibility_status,
            "retain_preservation": 0.950,
            "added_degeneration": 0.020,
            "failing_conditions": failures,
        },
        "comparable": True,
        "warning": (
            f"Candidate/forget: incomplete seed pool {seeds} "
            "(expected [0, 137, 271]); NOT a full 3-seed average"
        ),
        "seed_cohort": seeds,
        "seeds_complete": False,
        "n_rawfull_fallback": 0,
    }


@pytest.mark.parametrize("status", ["PASS", "FAIL"])
def test_score_eligibility_line_has_retain_ratio_and_delta_deg(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    status: str,
) -> None:
    kbench = _kbench()
    kbench.kscore.RES = tmp_path

    kbench.emit("Candidate", [_display_row(eligibility_status=status, seeds=[0])], ["P"])

    line = next(line for line in capsys.readouterr().out.splitlines() if "eligibility:" in line)
    assert f"eligibility: {status}" in line
    assert "retain preservation ratio 0.950" in line
    assert "added degeneration Δdeg 0.020" in line


def test_score_seed_zero_wording_omits_single_seed_spread(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    kbench = _kbench()
    kbench.kscore.RES = tmp_path
    row = _display_row(eligibility_status="PASS", seeds=[0])
    # KBENCH_SEEDS=0 considers this cohort complete, so no warning is attached.
    row.pop("warning")
    row["seeds_complete"] = True

    kbench.emit("Candidate", [row], ["P"])

    output = capsys.readouterr().out
    assert "seed coverage: seed-0 leaderboard minimum" in output
    assert "incomplete seed pool" not in output
    assert "forget 0.300 | absolute retain" in output
    assert "± 0.000 across seeds" not in output


def test_score_other_seed_subset_keeps_incomplete_wording_and_spread(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    kbench = _kbench()
    kbench.kscore.RES = tmp_path

    kbench.emit(
        "Candidate",
        [_display_row(eligibility_status="PASS", seeds=[0, 137])],
        ["P"],
    )

    output = capsys.readouterr().out
    assert "incomplete seed pool [0, 137]" in output
    assert "NOT a full 3-seed average" in output
    assert "forget 0.300 ± 0.000 across seeds" in output
    assert "seed-0 leaderboard minimum" not in output


def _smoke_bundle(tmp_path: Path, *, fail: bool, identity: bool = False):
    kbench = _kbench()
    cells = tmp_path / "cells"
    cells.mkdir(parents=True)
    for source in SMOKE_CELLS.iterdir():
        if source.suffix in {".jsonl", ".json"}:
            shutil.copy2(source, cells / source.name)

    if fail:
        for path in cells.glob("smoke_P_demo_*.jsonl"):
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            for row in rows:
                row["halted_reason"] = "parse_error"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    if identity:
        for path in cells.glob("smoke_P_demo_*.config.json"):
            config = json.loads(path.read_text(encoding="utf-8"))
            config["candidate_name"] = "demo"
            config["model_fingerprint"] = "sha256:fixture"
            path.write_text(json.dumps(config), encoding="utf-8")

    bundle = tmp_path / "candidate.kbench-bundle"
    kbench.build_bundle(
        cells,
        bundle,
        kbench_version="1.0.0",
        reference_for_prefix=kbench.load_reference,
    )
    manifest, rows = kbench.load_bundle(bundle)
    return kbench, manifest, rows


@pytest.mark.parametrize("fail", [False, True])
def test_report_eligibility_line_has_metrics_for_pass_and_fail(
    tmp_path: Path, fail: bool
) -> None:
    kbench, manifest, rows = _smoke_bundle(tmp_path, fail=fail)

    report = kbench.render_bundle_report(manifest, rows)
    line = next(line for line in report.splitlines() if "eligibility:" in line)

    assert f"eligibility: {'FAIL' if fail else 'PASS'}" in line
    assert "retain preservation ratio" in line
    assert "added degeneration Δdeg" in line
    assert "seeds covered: [0] (seed-0 leaderboard minimum)" in report
    assert "incomplete seed pool" not in report


def test_report_accepts_old_sidecars_and_shows_new_identity_when_present(tmp_path: Path) -> None:
    kbench, manifest, rows = _smoke_bundle(tmp_path / "old", fail=False)
    old_report = kbench.render_bundle_report(manifest, rows)
    assert "K-Bench offline report" in old_report
    assert "candidate name:" not in old_report
    assert "model fingerprint:" not in old_report

    kbench, manifest, rows = _smoke_bundle(tmp_path / "new", fail=False, identity=True)
    new_report = kbench.render_bundle_report(manifest, rows)
    assert "candidate name: demo" in new_report
    assert "model fingerprint: sha256:fixture" in new_report


def test_evaluator_writes_optional_candidate_identity_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluator = _load_module("baseline_leakage_p2b_test", EVALUATOR_PATH)
    queries = tmp_path / "queries.jsonl"
    out_jsonl = tmp_path / "fixture_P_Candidate_forget_seed0.jsonl"
    out_summary = tmp_path / "summary.json"
    query = {
        "query_id": "q0",
        "pii_id": "p0",
        "field": "email",
        "prompt": "hello",
        "canonical_fact": "world",
    }
    row = {
        "query_id": "q0",
        "pii_id": "p0",
        "field": "email",
        "ground_truth": "a@example.test",
        "raw_full": "Final Answer: withheld",
        "halted_reason": "final_answer",
        "leakage": [],
    }
    queries.write_text(json.dumps(query) + "\n", encoding="utf-8")
    out_jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "02_baseline_leakage.py",
            "--out-jsonl", str(out_jsonl),
            "--out-summary", str(out_summary),
            "--queries", str(queries),
            "--unlearn", "none",
            "--substrate", "P",
            "--model", "meta-llama/Llama-3.1-8B-Instruct",
            "--n-sample", "1",
            "--query-subset", "all",
            "--candidate-name", "Candidate",
            "--model-fingerprint", "hf:meta-llama/Llama-3.1-8B-Instruct@main",
        ],
    )

    evaluator.main()

    sidecar = json.loads(out_jsonl.with_suffix(".config.json").read_text(encoding="utf-8"))
    assert sidecar["candidate_name"] == "Candidate"
    assert sidecar["model_fingerprint"] == "hf:meta-llama/Llama-3.1-8B-Instruct@main"


def test_kbench_eval_passes_candidate_identity_to_each_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kbench = _kbench()
    model = tmp_path / "model"
    model.mkdir()
    model_config = {
        "model_type": "fixture",
        "vocab_size": 32,
        "hidden_size": 16,
        "num_hidden_layers": 2,
    }
    (model / "config.json").write_text(json.dumps(model_config), encoding="utf-8")
    (model / "weights.safetensors").write_bytes(b"fixture")
    reference = {
        "base_model": str(model),
        "seeds": [0, 137, 271],
        "substrates": ["C"],
        "model_config": model_config,
    }
    commands: list[list[str]] = []
    kbench.kscore.RES = tmp_path
    monkeypatch.setattr(kbench, "load_reference", lambda _prefix: reference)
    monkeypatch.setattr(kbench.subprocess, "run", lambda command, check: commands.append(command))
    monkeypatch.setattr(
        kbench,
        "score_substrate",
        lambda _prefix, substrate, _name: {
            "substrate": substrate,
            "status": "missing_candidate_cells",
        },
    )
    args = argparse.Namespace(
        name="Candidate",
        prefix="fixture",
        substrate="C",
        api_model=None,
        base=None,
        model=str(model),
        allow_nonstandard_model=False,
        resume=False,
        method="none",
        n=1,
    )

    kbench.run_eval(args)

    expected_fingerprint = kbench.model_fingerprint(str(model))
    assert len(commands) == 6
    for command in commands:
        assert command[command.index("--candidate-name") + 1] == "Candidate"
        assert command[command.index("--model-fingerprint") + 1] == expected_fingerprint


def test_changed_model_fingerprint_refuses_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    kbench = _kbench()
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_bytes(b'{"model_type":"fixture"}\n')
    weight = model / "model.safetensors"
    weight.write_bytes(b"old")
    old_fingerprint = kbench.model_fingerprint(str(model))
    weight.write_bytes(b"changed-size")

    kbench.kscore.RES = tmp_path
    monkeypatch.setattr(kbench.kscore, "SEEDS", [0])
    monkeypatch.setattr(
        kbench,
        "load_reference",
        lambda _prefix: {"base_model": "fixture/base"},
    )
    monkeypatch.setattr(kbench, "check_reference", lambda *_args: None)
    monkeypatch.setattr(kbench, "check_model_provenance", lambda *_args: None)
    cell = tmp_path / "fixture_P_Candidate_forget_seed0.jsonl"
    cell.write_text("{}\n", encoding="utf-8")
    cell.with_suffix(".config.json").write_text(
        json.dumps(
            {
                "candidate_name": "Candidate",
                "model_fingerprint": old_fingerprint,
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        name="Candidate",
        prefix="fixture",
        substrate="P",
        api_model=None,
        base=None,
        model=str(model),
        allow_nonstandard_model=False,
        resume=True,
        method="none",
        n=1,
    )

    with pytest.raises(SystemExit) as exc_info:
        kbench.run_eval(args)

    assert exc_info.value.code == 2
    output = capsys.readouterr().out
    assert "resume refused" in output
    assert "candidate/model fingerprint" in output


def test_model_fingerprint_uses_config_bytes_and_weight_names_and_sizes(tmp_path: Path) -> None:
    kbench = _kbench()
    model = tmp_path / "model"
    model.mkdir()
    config = model / "config.json"
    weight = model / "model.bin"
    config.write_bytes(b"one")
    weight.write_bytes(b"abc")
    first = kbench.model_fingerprint(str(model))

    weight.write_bytes(b"xyz")
    assert kbench.model_fingerprint(str(model)) == first
    weight.write_bytes(b"longer")
    assert kbench.model_fingerprint(str(model)) != first
    weight.write_bytes(b"abc")
    config.write_bytes(b"two")
    assert kbench.model_fingerprint(str(model)) != first


def test_openunlearning_converter_documents_its_actual_template_count() -> None:
    converter = _load_module("openunlearn_converter_p2b_test", OPENUNLEARN_CONVERTER_PATH)
    assert len(converter.IDK_TEMPLATES) == 102
    assert "102 refusal templates" in converter.__doc__
