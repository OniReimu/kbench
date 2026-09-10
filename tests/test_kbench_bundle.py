"""Acceptance tests for transcript bundle v1 candidate and offline reports."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KBENCH_PATH = ROOT / "scripts" / "kbench.py"
SMOKE_CELLS = ROOT / "data" / "smoke" / "cells"


def _kbench_module():
    spec = importlib.util.spec_from_file_location("kbench_bundle_test", KBENCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_smoke_bundle(tmp_path: Path):
    kbench = _kbench_module()
    cells = tmp_path / "smoke-cells"
    cells.mkdir()
    for source in SMOKE_CELLS.glob("*.jsonl"):
        target = cells / source.name
        shutil.copy2(source, target)
        parts = source.stem.rsplit("_", 2)
        split = parts[-2]
        seed = int(parts[-1].removeprefix("seed"))
        target.with_suffix(".config.json").write_text(
            json.dumps(
                {
                    "model": "smoke/model",
                    "base_model": "smoke/model",
                    "api_model": None,
                    "substrate": "P",
                    "query_subset": split,
                    "seed": seed,
                    "n_sample": 8,
                }
            ),
            encoding="utf-8",
        )
    bundle = tmp_path / "smoke.bundle"
    kbench.build_bundle(
        cells,
        bundle,
        scorer_version="v1",
        kbench_version="1.0.0",
        reference_for_prefix=kbench.load_reference,
    )
    return kbench, bundle


def test_smoke_bundle_report_is_deterministic_and_matches_smoke_k_score(
    tmp_path: Path,
) -> None:
    kbench, bundle = _build_smoke_bundle(tmp_path)
    manifest, rows = kbench.load_bundle(bundle)

    first = kbench.render_bundle_report(manifest, rows).encode("utf-8")
    second = kbench.render_bundle_report(manifest, rows).encode("utf-8")

    assert first == second
    assert b"Scorer version: v1" in first
    assert b"K-Score 0.000" in first
    assert b"channel       severity" in first
    assert b"Z_tool_wide   0.750" in first
    assert b"Z_summary     1.000" in first


def test_documented_smoke_bundle_and_report_cli_match_smoke_report(tmp_path: Path) -> None:
    bundle = tmp_path / "smoke.kbench-bundle"
    bundle_result = subprocess.run(
        [
            sys.executable,
            str(KBENCH_PATH),
            "bundle",
            "--cells",
            str(SMOKE_CELLS),
            "--out",
            str(bundle),
            "--scorer-version",
            "v1",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert bundle_result.returncode == 0, bundle_result.stderr or bundle_result.stdout

    report_result = subprocess.run(
        [sys.executable, str(KBENCH_PATH), "report", str(bundle)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert report_result.returncode == 0, report_result.stderr or report_result.stdout

    smoke_result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "smoke_report.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert smoke_result.returncode == 0, smoke_result.stderr or smoke_result.stdout
    report_score = re.search(r"K-Score\s+([0-9.]+)", report_result.stdout)
    smoke_score = re.search(r"K-Score:\s+([0-9.]+)", smoke_result.stdout)
    assert report_score is not None
    assert smoke_score is not None
    assert float(report_score.group(1)) == float(smoke_score.group(1))


@pytest.mark.parametrize("field", ["query_id", "pii_id", "field", "ground_truth", "raw_full", "halted_reason", "leakage"])
def test_report_refuses_a_missing_required_row_field_and_names_it(
    tmp_path: Path, field: str
) -> None:
    kbench, bundle = _build_smoke_bundle(tmp_path)
    cell = bundle / "cells" / "smoke_P_demo_forget_seed0.jsonl"
    lines = cell.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    del row[field]
    lines[0] = json.dumps(row)
    cell.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(kbench.BundleValidationError) as exc_info:
        kbench.load_bundle(bundle)

    message = str(exc_info.value)
    assert f"'{field}'" in message
    assert cell.name in message


def test_api_rows_require_incidents_retries_and_reasoning(tmp_path: Path) -> None:
    kbench = _kbench_module()
    cells = tmp_path / "cells"
    cells.mkdir()
    path = cells / "api_C_demo_forget_seed0.jsonl"
    row = {
        "query_id": "q0",
        "pii_id": "p0",
        "field": "email",
        "ground_truth": "a@example.test",
        "raw_full": "Final Answer: withheld",
        "halted_reason": "final_answer",
        "leakage": [],
        "api_incidents": [],
        "api_retries": [],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    path.with_suffix(".config.json").write_text(
        json.dumps(
            {
                "api_model": "provider/model",
                "substrate": "C",
                "query_subset": "forget",
                "seed": 0,
                "n_sample": 1,
            }
        ),
        encoding="utf-8",
    )
    bundle = tmp_path / "api.bundle"
    kbench.build_bundle(
        cells,
        bundle,
        scorer_version="v2",
        kbench_version="1.0.0",
    )

    with pytest.raises(kbench.BundleValidationError) as exc_info:
        kbench.load_bundle(bundle)

    message = str(exc_info.value)
    assert "'raw_reasoning'" in message
    assert path.name in message


def test_bundle_manifest_declares_run_identity_provenance_and_pairing(
    tmp_path: Path,
) -> None:
    _, bundle = _build_smoke_bundle(tmp_path)
    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))

    assert manifest["schema"] == "kbench-transcript-bundle@1-candidate"
    assert manifest["created_at"].endswith("+00:00")
    assert manifest["scorer_version"] == "v1"
    assert manifest["kbench_version"] == "1.0.0"
    assert manifest["run_identity"]
    assert manifest["provenance"]["harness_sidecar_configs"]
    cell = manifest["cells"][0]
    for field in ("model", "base_model", "method", "substrate", "split", "seed", "n"):
        assert field in cell
    assert "harness_sidecar_config" in cell["provenance"]
    assert any(pairing["forget"] and pairing["retain"] for pairing in manifest["pairings"])


def test_report_refuses_empty_pairings_and_names_non_api_forget_cell(tmp_path: Path) -> None:
    kbench, bundle = _build_smoke_bundle(tmp_path)
    manifest_path = bundle / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pairings"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(kbench.BundleValidationError) as exc_info:
        kbench.load_bundle(bundle)

    message = str(exc_info.value)
    assert "missing pairing" in message
    assert "smoke_P_demo_forget_seed0.jsonl" in message


def test_report_refuses_non_api_forget_pairing_with_null_retain(tmp_path: Path) -> None:
    kbench, bundle = _build_smoke_bundle(tmp_path)
    manifest_path = bundle / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pairing = next(item for item in manifest["pairings"] if item["method"] == "demo")
    pairing["retain"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(kbench.BundleValidationError) as exc_info:
        kbench.load_bundle(bundle)

    message = str(exc_info.value)
    assert "missing retain pairing" in message
    assert "smoke_P_demo_forget_seed0.jsonl" in message


def test_sidecar_api_identity_cannot_be_disabled_to_skip_required_rows(tmp_path: Path) -> None:
    kbench = _kbench_module()
    cells = tmp_path / "cells"
    cells.mkdir()
    path = cells / "api_C_demo_forget_seed0.jsonl"
    row = {
        "query_id": "q0",
        "pii_id": "p0",
        "field": "email",
        "ground_truth": "a@example.test",
        "raw_full": "Final Answer: withheld",
        "halted_reason": "final_answer",
        "leakage": [],
        "api_incidents": [],
        "raw_reasoning": None,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    path.with_suffix(".config.json").write_text(
        json.dumps(
            {
                "api_model": "provider/model",
                "substrate": "C",
                "query_subset": "forget",
                "seed": 0,
                "n_sample": 1,
            }
        ),
        encoding="utf-8",
    )
    bundle = tmp_path / "api.bundle"
    kbench.build_bundle(cells, bundle, scorer_version="v2", kbench_version="1.0.0")
    manifest_path = bundle / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cells"][0]["api"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(kbench.BundleValidationError) as exc_info:
        kbench.load_bundle(bundle)

    message = str(exc_info.value)
    assert "'api_retries'" in message
    assert path.name in message


def _write_smoke_cell(
    cells: Path,
    filename: str,
    source_name: str,
    *,
    model: str,
    base_model: str,
    substrate: str,
    split: str,
    seed: int,
    clear_leakage: bool = False,
    api: bool = False,
) -> None:
    rows = [
        json.loads(line)
        for line in (SMOKE_CELLS / source_name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if clear_leakage:
        for row in rows:
            for leakage in row["leakage"]:
                leakage["cer"] = 0
            row["answer"] = "withheld"
            row["raw_full"] = "Final Answer: withheld"
            for field in (
                "raw_Z_CoT", "raw_Z_tool", "raw_Z_tool_obs", "raw_Z_RAG", "raw_Z_summary",
            ):
                row[field] = ""
    if api:
        for row in rows:
            row.update(api_incidents=[], api_retries=[], raw_reasoning=None)
    path = cells / filename
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    path.with_suffix(".config.json").write_text(
        json.dumps(
            {
                "model": model,
                "base_model": base_model,
                "api_model": model if api else None,
                "substrate": substrate,
                "query_subset": split,
                "seed": seed,
                "n_sample": len(rows),
            }
        ),
        encoding="utf-8",
    )


def test_bundle_and_report_refuse_mixed_local_and_api_pairing(tmp_path: Path) -> None:
    cells = tmp_path / "mixed-cells"
    cells.mkdir()
    for method in ("none", "demo"):
        for split in ("forget", "retain"):
            _write_smoke_cell(
                cells,
                f"mixed_P_{method}_{split}_seed0.jsonl",
                f"smoke_P_{method}_{split}_seed0.jsonl",
                model="identical/model",
                base_model="identical/model",
                substrate="P",
                split=split,
                seed=0,
                api=method == "demo" and split == "retain",
            )

    mixed_bundle = tmp_path / "mixed.bundle"
    bundle_result = subprocess.run(
        [
            sys.executable,
            str(KBENCH_PATH),
            "bundle",
            "--cells",
            str(cells),
            "--out",
            str(mixed_bundle),
            "--scorer-version",
            "v1",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert bundle_result.returncode == 2
    assert "mixed_P_demo_forget_seed0.jsonl" in bundle_result.stdout
    assert "mixed_P_demo_retain_seed0.jsonl" in bundle_result.stdout

    # Build a valid local bundle, then make its declared retain edge cross the API boundary.
    local_cells = tmp_path / "local-cells"
    shutil.copytree(cells, local_cells)
    retain_config_path = local_cells / "mixed_P_demo_retain_seed0.config.json"
    retain_config = json.loads(retain_config_path.read_text(encoding="utf-8"))
    retain_config["api_model"] = None
    retain_config_path.write_text(json.dumps(retain_config), encoding="utf-8")
    report_bundle = tmp_path / "report.bundle"
    local_bundle_result = subprocess.run(
        [
            sys.executable,
            str(KBENCH_PATH),
            "bundle",
            "--cells",
            str(local_cells),
            "--out",
            str(report_bundle),
            "--scorer-version",
            "v1",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert local_bundle_result.returncode == 0, local_bundle_result.stdout

    retain_file = "cells/mixed_P_demo_retain_seed0.jsonl"
    manifest_path = report_bundle / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    retain_cell = next(cell for cell in manifest["cells"] if cell["file"] == retain_file)
    embedded = retain_cell["provenance"]["harness_sidecar_config"]
    embedded["api_model"] = "identical/model"
    retain_cell["api"] = True
    manifest["provenance"]["harness_sidecar_configs"][retain_file]["api_model"] = (
        "identical/model"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report_result = subprocess.run(
        [sys.executable, str(KBENCH_PATH), "report", str(report_bundle)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert report_result.returncode == 2
    assert "mixed_P_demo_forget_seed0.jsonl" in report_result.stdout
    assert "mixed_P_demo_retain_seed0.jsonl" in report_result.stdout


def test_method_name_containing_another_substrate_marker_round_trips(tmp_path: Path) -> None:
    kbench = _kbench_module()
    cells = tmp_path / "cells"
    cells.mkdir()
    for method in ("none", "demo_P_variant"):
        for split in ("forget", "retain"):
            source_method = "none" if method == "none" else "demo"
            _write_smoke_cell(
                cells,
                f"llama_R-text_{method}_{split}_seed0.jsonl",
                f"smoke_P_{source_method}_{split}_seed0.jsonl",
                model="base/model",
                base_model="base/model",
                substrate="R-text",
                split=split,
                seed=0,
            )
    bundle = tmp_path / "round-trip.bundle"
    kbench.build_bundle(cells, bundle, scorer_version="v1", kbench_version="1.0.0")

    manifest, _ = kbench.load_bundle(bundle)

    methods = {cell["method"] for cell in manifest["cells"]}
    assert methods == {"none", "demo_P_variant"}
    assert {cell["substrate"] for cell in manifest["cells"]} == {"R-text"}


def test_report_keeps_same_method_on_two_checkpoint_models_separate(tmp_path: Path) -> None:
    kbench = _kbench_module()
    cells = tmp_path / "cells"
    cells.mkdir()
    for seed in (0, 1):
        for split in ("forget", "retain"):
            _write_smoke_cell(
                cells,
                f"llama_P_none_{split}_seed{seed}.jsonl",
                f"smoke_P_none_{split}_seed0.jsonl",
                model="base/model",
                base_model="base/model",
                substrate="P",
                split=split,
                seed=seed,
            )
    for seed, model in ((0, "checkpoint/a"), (1, "checkpoint/b")):
        for split in ("forget", "retain"):
            _write_smoke_cell(
                cells,
                f"llama_P_demo_{split}_seed{seed}.jsonl",
                f"smoke_P_demo_{split}_seed0.jsonl",
                model=model,
                base_model="base/model",
                substrate="P",
                split=split,
                seed=seed,
                clear_leakage=seed == 1 and split == "forget",
            )
    bundle = tmp_path / "models.bundle"
    kbench.build_bundle(cells, bundle, scorer_version="v1", kbench_version="1.0.0")
    manifest, rows = kbench.load_bundle(bundle)

    report = kbench.render_bundle_report(manifest, rows)

    assert report.count("Method: demo") == 2
    assert "model: checkpoint/a" in report
    assert "model: checkpoint/b" in report
    assert "K-Score 0.000" in report
    assert "K-Score 1.000" in report
