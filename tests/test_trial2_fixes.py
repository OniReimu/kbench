"""Regression coverage for the first-time-user score/report release defects."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KBENCH = ROOT / "scripts" / "kbench.py"
SEEDS = (0, 137, 271)


def _row(query_id: str) -> dict:
    return {
        "query_id": query_id,
        "pii_id": "pii-00000",
        "field": "email",
        "ground_truth": "secret@example.test",
        "raw_full": "Final Answer: secret@example.test",
        "answer": "secret@example.test",
        "halted_reason": "final_answer",
        "leakage": [{"channel": "Z_answer", "cer": 1}],
        "n_tool_calls": 0,
        "n_thoughts": 0,
    }


def _write_cell(
    cells: Path,
    *,
    method: str,
    split: str,
    seed: int,
    query_id: str,
) -> None:
    stem = f"fixture_P_{method}_{split}_seed{seed}"
    (cells / f"{stem}.jsonl").write_text(
        json.dumps(_row(query_id)) + "\n", encoding="utf-8"
    )
    (cells / f"{stem}.config.json").write_text(
        json.dumps(
            {
                "model": "fixture/model",
                "base_model": "fixture/model",
                "api_model": None,
                "substrate": "P",
                "query_subset": split,
                "seed": seed,
                "n_sample": 1,
            }
        ),
        encoding="utf-8",
    )


def _make_cells(tmp_path: Path, candidate_seeds: tuple[int, ...], *, mismatch: bool = False) -> Path:
    cells = tmp_path / "cells"
    cells.mkdir()
    (cells / "fixture.reference.json").write_text(
        json.dumps(
            {
                "prefix": "fixture",
                "base_model": "fixture/model",
                "seeds": list(SEEDS),
                "substrates": ["P"],
                "protocol": "1",
            }
        ),
        encoding="utf-8",
    )
    for seed in SEEDS:
        for split in ("forget", "retain"):
            _write_cell(
                cells,
                method="none",
                split=split,
                seed=seed,
                query_id=f"{split}-{seed}",
            )
    for seed in candidate_seeds:
        for split in ("forget", "retain"):
            query_id = f"{split}-{seed}"
            if mismatch and split == "forget" and seed == 0:
                query_id = "different-query"
            _write_cell(
                cells,
                method="Candidate",
                split=split,
                seed=seed,
                query_id=query_id,
            )
    return cells


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(KBENCH), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _score(cells: Path) -> subprocess.CompletedProcess[str]:
    return _run(
        "score",
        "--cells",
        str(cells),
        "--name",
        "Candidate",
        "--prefix",
        "fixture",
        "--substrate",
        "P",
    )


def _bundle(cells: Path, bundle: Path) -> None:
    result = _run("bundle", "--cells", str(cells), "--out", str(bundle))
    assert result.returncode == 0, result.stderr or result.stdout


def test_score_restricts_three_seed_baseline_to_seed_zero_candidate(tmp_path: Path) -> None:
    cells = _make_cells(tmp_path, (0,))
    result = _score(cells)

    assert result.returncode == 0, result.stderr or result.stdout
    assert "cohort_mismatch" not in result.stdout
    assert "using candidate seeds [0] against baseline restricted to [0]" in result.stdout
    assert "NOT a full 3-seed average" in result.stdout
    saved = json.loads((cells / "Candidate.kbench.json").read_text(encoding="utf-8"))
    assert saved["coverage_signature"]["seeds"] == [0]


def test_score_unscored_run_prints_cohort_detail_and_exits_nonzero(tmp_path: Path) -> None:
    result = _score(_make_cells(tmp_path, (0,), mismatch=True))

    assert result.returncode == 2
    assert "P         : cohort_mismatch" in result.stdout
    assert "split=forget" in result.stdout
    assert "symmetric_difference_size=2" in result.stdout


def test_full_three_seed_score_numbers_remain_pinned(tmp_path: Path) -> None:
    cells = _make_cells(tmp_path, SEEDS)
    result = _score(cells)

    assert result.returncode == 0, result.stderr or result.stdout
    assert "K-Score 0.000 (untreated baseline 0.000)" in result.stdout
    assert "graded observer rate (K-Score input) forget 1.000" in result.stdout
    assert "binary per-query OR(all): forget 1.000 ± 0.000" in result.stdout
    assert "K-class: measured failure" in result.stdout
    assert "incomplete seed pool" not in result.stdout
    saved = json.loads((cells / "Candidate.kbench.json").read_text(encoding="utf-8"))
    row = saved["substrates"][0]
    assert {
        key: row[key]
        for key in ("k_score", "baseline_k_score", "or_forget", "delta_sel", "degen")
    } == {
        "k_score": 0.0,
        "baseline_k_score": 0.0,
        "or_forget": 1.0,
        "delta_sel": 0.0,
        "degen": 0.0,
    }


def test_report_prints_score_fields_and_seed_cohort(tmp_path: Path) -> None:
    cells = _make_cells(tmp_path, (0,))
    bundle = tmp_path / "candidate.bundle"
    _bundle(cells, bundle)

    result = _run("report", str(bundle))

    assert result.returncode == 0, result.stderr or result.stdout
    assert "eligibility: PASS" in result.stdout
    assert "K-class: measured failure" in result.stdout
    assert "untreated baseline K-Score: 0.000" in result.stdout
    assert "seeds covered: [0] (incomplete seed pool; NOT a full 3-seed average)" in result.stdout


def test_report_refuses_mismatched_cohort_like_score(tmp_path: Path) -> None:
    cells = _make_cells(tmp_path, (0,), mismatch=True)
    bundle = tmp_path / "mismatch.bundle"
    _bundle(cells, bundle)

    result = _run("report", str(bundle))

    assert result.returncode == 2
    assert "P         : cohort_mismatch" in result.stdout
    assert "split=forget" in result.stdout
    assert "symmetric_difference_size=2" in result.stdout


def test_documented_uv_run_commands_do_not_resolve_the_project() -> None:
    documents = [
        ROOT / "INSTALL.md",
        ROOT / "README.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "reproduce.sh",
        *sorted((ROOT / "docs").glob("*.md")),
    ]
    bare: list[str] = []
    for path in documents:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"\buv run (?!--no-sync\b)", line):
                bare.append(f"{path.relative_to(ROOT)}:{line_number}: {line.strip()}")
    assert not bare, "bare uv run command(s) may re-resolve dependencies:\n" + "\n".join(bare)
