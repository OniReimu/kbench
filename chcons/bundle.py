"""Build and validate self-contained K-Bench transcript bundles."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

BUNDLE_SCHEMA = "kbench-transcript-bundle@1-candidate"
COMMON_ROW_FIELDS = (
    "query_id",
    "pii_id",
    "field",
    "ground_truth",
    "raw_full",
    "halted_reason",
    "leakage",
)
API_ROW_FIELDS = ("api_incidents", "api_retries", "raw_reasoning")
SUBSTRATES = ("P", "C", "R-text", "R-struct")
SPLITS = ("forget", "retain")


class BundleValidationError(ValueError):
    """A bundle cannot be trusted or scored offline."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_from_sidecar(path: Path, config: dict) -> dict[str, object]:
    """Recover filename identity using the evaluator-owned identity fields."""
    required = ("substrate", "query_subset", "seed", "n_sample", "api_model")
    missing = next((field for field in required if field not in config), None)
    if missing:
        raise BundleValidationError(
            f"harness sidecar {path.with_suffix('.config.json').name} "
            f"missing required field '{missing}'"
        )
    substrate = config["substrate"]
    split = config["query_subset"]
    seed = config["seed"]
    n = config["n_sample"]
    if substrate not in SUBSTRATES:
        raise BundleValidationError(f"invalid sidecar substrate for {path.name}: {substrate!r}")
    if split not in SPLITS:
        raise BundleValidationError(f"invalid sidecar query_subset for {path.name}: {split!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise BundleValidationError(f"invalid sidecar seed for {path.name}: {seed!r}")
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise BundleValidationError(f"invalid sidecar n_sample for {path.name}: {n!r}")

    stem = path.name.removesuffix(".jsonl")
    suffix = f"_{split}_seed{seed}"
    if not stem.endswith(suffix):
        raise BundleValidationError(
            f"cell filename cannot be reconciled with harness sidecar: {path.name}"
        )
    run = stem[: -len(suffix)]
    marker = f"_{substrate}_"
    if marker not in run:
        raise BundleValidationError(
            f"cell filename cannot be reconciled with harness sidecar: {path.name}"
        )
    prefix, method = run.split(marker, 1)
    if not prefix or not method:
        raise BundleValidationError(f"incomplete run identity in cell filename: {path.name}")
    return {
        "prefix": prefix,
        "substrate": substrate,
        "method": method,
        "split": split,
        "seed": seed,
        "n": n,
    }


def _read_sidecar(path: Path) -> dict:
    config_path = path.with_suffix(".config.json")
    try:
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BundleValidationError(f"missing harness sidecar: {config_path.name}") from exc
    except json.JSONDecodeError as exc:
        raise BundleValidationError(f"invalid harness sidecar: {config_path.name}") from exc
    if not isinstance(loaded, dict):
        raise BundleValidationError(f"harness sidecar is not an object: {config_path.name}")
    return loaded


def _read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BundleValidationError(
                    f"invalid JSON in {path.name} line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise BundleValidationError(f"row is not an object: {path.name} line {line_number}")
            rows.append(row)
    if not rows:
        raise BundleValidationError(f"empty cell file: {path.name}")
    return rows


def _validate_pairings(cells: list[dict], pairings: object) -> None:
    """Validate declared candidate/retain/reference relationships fail-closed."""
    if not isinstance(pairings, list):
        raise BundleValidationError("bundle.json pairings must be a list")
    by_file = {cell["file"]: cell for cell in cells}
    by_forget: dict[str, dict] = {}
    required_pairing = (
        "prefix", "model", "base_model", "substrate", "method", "seed", "api",
        "forget", "retain", "reference_forget", "reference_retain",
    )
    for pairing in pairings:
        if not isinstance(pairing, dict):
            raise BundleValidationError("bundle.json pairing entry is not an object")
        missing = next((field for field in required_pairing if field not in pairing), None)
        label = PurePosixPath(str(pairing.get("forget") or "unknown cell")).name
        if missing:
            raise BundleValidationError(f"pairing for {label} missing required field '{missing}'")
        forget_path = pairing["forget"]
        if not isinstance(forget_path, str) or forget_path not in by_file:
            raise BundleValidationError(f"dangling pairing for {label}: forget={forget_path!r}")
        if forget_path in by_forget:
            raise BundleValidationError(f"duplicate pairing for {PurePosixPath(forget_path).name}")
        by_forget[forget_path] = pairing
        forget = by_file[forget_path]
        for field in ("prefix", "model", "base_model", "substrate", "method", "seed", "api"):
            if pairing[field] != forget[field]:
                raise BundleValidationError(
                    f"pairing identity mismatch for {PurePosixPath(forget_path).name}: {field}"
                )
        for field in ("retain", "reference_forget", "reference_retain"):
            target = pairing[field]
            if target is not None and (not isinstance(target, str) or target not in by_file):
                raise BundleValidationError(
                    f"dangling pairing for {PurePosixPath(forget_path).name}: {field}={target!r}"
                )
            if target is not None and by_file[target]["api"] != forget["api"]:
                raise BundleValidationError(
                    "API execution identity mismatch between "
                    f"{PurePosixPath(forget_path).name} and "
                    f"{PurePosixPath(target).name} ({field})"
                )

    for forget in (
        cell for cell in cells
        if cell["split"] == "forget" and cell["method"] != "none" and not cell["api"]
    ):
        name = PurePosixPath(forget["file"]).name
        pairing = by_forget.get(forget["file"])
        if pairing is None:
            raise BundleValidationError(f"missing pairing for non-API forget cell {name}")

        retain_path = pairing["retain"]
        if retain_path is None:
            expected_retain = (
                f"{forget['prefix']}_{forget['substrate']}_{forget['method']}_retain_seed{forget['seed']}.jsonl"
            )
            raise BundleValidationError(
                f"missing retain pairing for non-API forget cell {name}: "
                f"expected retain cell '{expected_retain}' for seed {forget['seed']}; "
                "a candidate must supply forget and retain cells for every seed"
            )
        retain = by_file[retain_path]
        for field in ("prefix", "model", "base_model", "substrate", "method", "seed", "api"):
            if retain[field] != forget[field]:
                raise BundleValidationError(f"invalid retain pairing for non-API forget cell {name}")
        if retain["split"] != "retain":
            raise BundleValidationError(f"invalid retain pairing for non-API forget cell {name}")

        references = []
        for field, split in (("reference_forget", "forget"), ("reference_retain", "retain")):
            target = pairing[field]
            if target is None:
                raise BundleValidationError(
                    f"missing untreated {split} reference for non-API forget cell {name}"
                )
            reference = by_file[target]
            if (
                reference["prefix"] != forget["prefix"]
                or reference["base_model"] != forget["base_model"]
                or reference["substrate"] != forget["substrate"]
                or reference["method"] != "none"
                or reference["seed"] != forget["seed"]
                or reference["split"] != split
                or reference["api"] != forget["api"]
            ):
                raise BundleValidationError(
                    f"invalid untreated {split} reference for non-API forget cell {name}"
                )
            references.append(reference)
        if references[0]["model"] != references[1]["model"]:
            raise BundleValidationError(f"reference model mismatch for non-API forget cell {name}")


def build_bundle(
    cells_dir: Path,
    out_dir: Path,
    *,
    kbench_version: str,
    reference_for_prefix: Callable[[str], dict | None] | None = None,
) -> Path:
    """Copy cells and their provenance into a candidate bundle directory."""
    source = cells_dir.resolve()
    paths = sorted(source.glob("*.jsonl"))
    if not paths:
        raise BundleValidationError(f"no JSONL cell files found under {source}")
    if out_dir.exists():
        raise BundleValidationError(f"output already exists: {out_dir}")

    cell_records: list[dict] = []
    seen_names: set[str] = set()
    seen_identities: set[tuple[object, ...]] = set()
    for path in paths:
        if path.name in seen_names:
            raise BundleValidationError(f"duplicate cell filename: {path.name}")
        seen_names.add(path.name)
        rows = _read_rows(path)
        config = _read_sidecar(path)
        identity = _identity_from_sidecar(path, config)
        if len(rows) != identity["n"]:
            raise BundleValidationError(
                f"row count mismatch in {path.name}: sidecar={identity['n']}, file={len(rows)}"
            )
        identity_key = tuple(
            identity[field] for field in ("prefix", "substrate", "method", "seed", "split")
        )
        if identity_key in seen_identities:
            raise BundleValidationError(f"duplicate cell run identity: {path.name}")
        seen_identities.add(identity_key)
        reference = reference_for_prefix(str(identity["prefix"])) if reference_for_prefix else None
        api_model = config.get("api_model")
        if api_model is not None and (not isinstance(api_model, str) or not api_model):
            raise BundleValidationError(f"invalid sidecar api_model for {path.name}: {api_model!r}")
        model = api_model if api_model is not None else config.get("model")
        if not isinstance(model, str) or not model:
            raise BundleValidationError(f"invalid sidecar model identity for {path.name}: {model!r}")
        base_model = (
            api_model
            or (reference or {}).get("base_model")
            or config.get("base_model")
            or model
        )
        cell_records.append(
            {
                "file": f"cells/{path.name}",
                **identity,
                "model": str(model),
                "base_model": str(base_model),
                "api": api_model is not None,
                "comparable": config.get("comparable") is not False,
                "sha256": _sha256(path),
                "provenance": {"harness_sidecar_config": config},
            }
        )

    def matching(forget: dict, relationship: str, **identity: object) -> dict | None:
        matches = [
            cell for cell in cell_records
            if all(cell[field] == value for field, value in identity.items())
        ]
        if len(matches) > 1:
            raise BundleValidationError(f"ambiguous cell identity: {identity}")
        if not matches and not forget["api"] and "api" in identity:
            cross_mode = [
                cell for cell in cell_records
                if all(
                    field == "api" or cell[field] == value
                    for field, value in identity.items()
                )
            ]
            if cross_mode:
                names = ", ".join(PurePosixPath(cell["file"]).name for cell in cross_mode)
                raise BundleValidationError(
                    "API execution identity mismatch between "
                    f"{PurePosixPath(forget['file']).name} and {names} ({relationship})"
                )
        return matches[0] if matches else None

    pairings: list[dict] = []
    for forget in (
        cell for cell in cell_records
        if cell["split"] == "forget" and cell["method"] != "none"
    ):
        retain = matching(
            forget, "retain", api=forget["api"],
            prefix=forget["prefix"], model=forget["model"], base_model=forget["base_model"],
            substrate=forget["substrate"], method=forget["method"], seed=forget["seed"],
            split="retain",
        )
        reference_forget = matching(
            forget, "reference_forget", api=forget["api"],
            prefix=forget["prefix"], base_model=forget["base_model"],
            substrate=forget["substrate"], method="none", seed=forget["seed"], split="forget",
        )
        reference_retain = matching(
            forget, "reference_retain", api=forget["api"],
            prefix=forget["prefix"], base_model=forget["base_model"],
            substrate=forget["substrate"], method="none", seed=forget["seed"], split="retain",
        )
        pairings.append(
            {
                key: forget[key]
                for key in ("prefix", "model", "base_model", "substrate", "method", "seed", "api")
            } | {
                "forget": forget["file"],
                "retain": retain["file"] if retain else None,
                "reference_forget": reference_forget["file"] if reference_forget else None,
                "reference_retain": reference_retain["file"] if reference_retain else None,
            }
        )
    _validate_pairings(cell_records, pairings)

    out_dir.mkdir(parents=True)
    (out_dir / "cells").mkdir()
    for path in paths:
        shutil.copy2(path, out_dir / "cells" / path.name)
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "kbench_version": kbench_version,
        "scorer_version": "v2",
        "run_identity": [
            {
                key: cell[key]
                for key in (
                    "file", "model", "base_model", "method", "substrate", "split",
                    "seed", "n", "api", "comparable",
                )
            }
            for cell in cell_records
        ],
        "provenance": {
            "harness_sidecar_configs": {
                cell["file"]: cell["provenance"]["harness_sidecar_config"]
                for cell in cell_records
            }
        },
        "cells": cell_records,
        "pairings": pairings,
    }
    (out_dir / "bundle.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out_dir


def load_bundle(bundle_dir: Path) -> tuple[dict, dict[str, list[dict]]]:
    """Load a bundle, fail closed on schema, paths, rows, and content hashes."""
    root = bundle_dir.resolve()
    manifest_path = root / "bundle.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BundleValidationError(f"missing bundle.json in {root}") from exc
    except json.JSONDecodeError as exc:
        raise BundleValidationError(f"invalid bundle.json: {exc.msg}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != BUNDLE_SCHEMA:
        raise BundleValidationError(f"bundle.json schema must be {BUNDLE_SCHEMA}")
    for field in (
        "created_at", "kbench_version", "scorer_version", "run_identity",
        "provenance", "cells", "pairings",
    ):
        if field not in manifest:
            raise BundleValidationError(f"bundle.json missing required field '{field}'")
    if manifest["scorer_version"] != "v2":
        raise BundleValidationError(
            "bundle.json scorer_version must be 'v2'; use the submission branch "
            "to reproduce or inspect v1 bundles"
        )
    if not isinstance(manifest["cells"], list) or not manifest["cells"]:
        raise BundleValidationError("bundle.json cells must be a non-empty list")
    loaded_rows: dict[str, list[dict]] = {}
    for cell in manifest["cells"]:
        if not isinstance(cell, dict):
            raise BundleValidationError("bundle.json cell entry is not an object")
        required_cell = (
            "file", "prefix", "model", "base_model", "method", "substrate",
            "split", "seed", "n", "api", "sha256", "provenance",
        )
        missing_cell = next((field for field in required_cell if field not in cell), None)
        if missing_cell:
            raise BundleValidationError(f"bundle.json cell missing required field '{missing_cell}'")
        relative = PurePosixPath(str(cell["file"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise BundleValidationError(f"unsafe bundle cell path: {relative}")
        path = root.joinpath(*relative.parts)
        try:
            path.resolve().relative_to(root)
        except ValueError as exc:
            raise BundleValidationError(f"bundle cell path escapes bundle: {relative}") from exc
        if not path.is_file():
            raise BundleValidationError(f"missing cell file: {relative}")
        rows = _read_rows(path)
        provenance = cell["provenance"]
        if not isinstance(provenance, dict) or not isinstance(
            provenance.get("harness_sidecar_config"), dict
        ):
            raise BundleValidationError(f"missing embedded harness sidecar for {relative}")
        config = provenance["harness_sidecar_config"]
        identity = _identity_from_sidecar(path, config)
        for field in ("prefix", "substrate", "method", "split", "seed", "n"):
            if cell[field] != identity[field]:
                raise BundleValidationError(f"sidecar identity mismatch for {relative}: {field}")
        api_model = config["api_model"]
        if api_model is not None and (not isinstance(api_model, str) or not api_model):
            raise BundleValidationError(f"invalid sidecar api_model for {relative}: {api_model!r}")
        sidecar_api = api_model is not None
        sidecar_model = api_model if sidecar_api else config.get("model")
        if not isinstance(sidecar_model, str) or not sidecar_model:
            raise BundleValidationError(f"invalid sidecar model identity for {relative}")
        if cell["model"] != sidecar_model:
            raise BundleValidationError(f"sidecar model mismatch for {relative}")
        if config.get("base_model") is not None and cell["base_model"] != config["base_model"]:
            raise BundleValidationError(f"sidecar base_model mismatch for {relative}")
        if cell.get("comparable", True) != (config.get("comparable") is not False):
            raise BundleValidationError(f"sidecar comparability mismatch for {relative}")
        if sidecar_api and cell["base_model"] != api_model:
            raise BundleValidationError(f"sidecar base_model mismatch for API cell {relative}")
        row_fields = COMMON_ROW_FIELDS + (API_ROW_FIELDS if sidecar_api else ())
        for row_number, row in enumerate(rows, start=1):
            for field in row_fields:
                if field not in row:
                    raise BundleValidationError(
                        f"missing required field '{field}' in {relative} row {row_number}"
                    )
        if not isinstance(cell["api"], bool) or cell["api"] != sidecar_api:
            raise BundleValidationError(
                f"API identity disagreement between manifest and sidecar for {relative}"
            )
        if len(rows) != cell["n"]:
            raise BundleValidationError(
                f"row count mismatch in {relative}: bundle.json={cell['n']}, file={len(rows)}"
            )
        if _sha256(path) != cell["sha256"]:
            raise BundleValidationError(f"sha256 mismatch for {relative}")
        loaded_rows[str(relative)] = rows
    embedded_configs = manifest["provenance"].get("harness_sidecar_configs") if isinstance(
        manifest["provenance"], dict
    ) else None
    expected_configs = {
        cell["file"]: cell["provenance"]["harness_sidecar_config"]
        for cell in manifest["cells"]
    }
    if embedded_configs != expected_configs:
        raise BundleValidationError("bundle.json harness sidecar provenance mismatch")
    _validate_pairings(manifest["cells"], manifest["pairings"])
    return manifest, loaded_rows
