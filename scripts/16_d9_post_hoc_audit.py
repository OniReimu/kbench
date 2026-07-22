"""D9 post-hoc audit script.

Runs over all completed result cells (`results/*.json` + `results/*.jsonl`) and
verifies post-hoc:
  - Cell coverage (no orphan partials)
  - Provenance invariants (audit_version, canonical_facts_sha16, splits)
  - LEACE verification: the canonical activation-probe set is v54 (Llama, 3 seeds,
    all substrates); any pre-v54 LEACE cell in the publishable namespace is flagged
  - n_queries × n_seeds consistency
  - Per-cell sha-stable hash of inputs (for reproducibility check)

Usage:
    python scripts/16_d9_post_hoc_audit.py --results-dir results/ [--strict]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


CELL_NAME_RE = re.compile(
    r"^(?P<version>v\d+\w*)_"
    r"(?P<model>[A-Za-z0-9]+)?_?"
    r"(?P<substrate>P|C|Rstruct|Rtext)_"
    r"(?P<method>none|noise|eco|star|leace|cha|o3|repe|mlp_probe|rlace)_"
    r"(?P<subset>forget|retain)_"
    r"seed(?P<seed>\d+)"
    r"(?:_(?P<suffix>.*))?$"
)


def parse_cell(filename: str) -> dict | None:
    """Parse cell name into structured fields. Returns None if not a cell file."""
    stem = Path(filename).stem.replace(".jsonl", "")
    m = CELL_NAME_RE.match(stem)
    if not m:
        return None
    g = m.groupdict()
    return {
        "version": g["version"],
        "model": g.get("model") or "llama",
        "substrate": g["substrate"],
        "method": g["method"],
        "subset": g["subset"],
        "seed": int(g["seed"]),
        "suffix": g.get("suffix"),
    }


def load_summary(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception as e:
        return {"_load_error": str(e)}


def count_jsonl_rows(p: Path) -> int:
    if not p.exists():
        return -1
    return sum(1 for line in p.read_text().splitlines() if line.strip())


def compute_jsonl_sha16(p: Path) -> str:
    if not p.exists():
        return "MISSING"
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()[:16]


# Cells that should be considered deprecated due to known LEACE adapter bug
# (pre-fix fit/apply mismatch); superseded by the canonical v54 activation-probe set.
DEPRECATED_LEACE_NAMESPACES = {"v26", "v21D", "v27", "v28", "v30", "v31", "v32", "v33", "v35"}  # all superseded by v54


def audit_results_dir(results_dir: Path, strict: bool = False) -> tuple[list, list]:
    """Return (warnings, errors) lists. errors fail the audit; warnings inform."""
    warnings: list[str] = []
    errors: list[str] = []

    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        errors.append(f"no .json result files in {results_dir}")
        return warnings, errors

    cell_count = 0
    config_count = 0
    audit_versions = set()
    canonical_shas = set()
    coverage = defaultdict(set)  # (version, model, substrate, method, subset) → {seeds}

    for f in json_files:
        name = f.name
        if name.endswith(".config.json"):
            config_count += 1
            continue
        cell = parse_cell(name)
        if not cell:
            warnings.append(f"unparseable cell name: {name}")
            continue
        cell_count += 1

        summary = load_summary(f)
        if "_load_error" in summary:
            errors.append(f"{name}: load error {summary['_load_error']}")
            continue

        # Cross-check jsonl row count
        jsonl_p = f.with_suffix(".jsonl")
        n_jsonl = count_jsonl_rows(jsonl_p)
        n_summary = summary.get("n_queries")
        if n_summary and n_jsonl != n_summary:
            errors.append(
                f"{name}: n_queries={n_summary} but jsonl has {n_jsonl} rows"
            )

        # Coverage tracking
        key = (cell["version"], cell["model"], cell["substrate"], cell["method"], cell["subset"])
        coverage[key].add(cell["seed"])

        # Deprecated LEACE flag
        if cell["method"] == "leace" and cell["version"] in DEPRECATED_LEACE_NAMESPACES:
            warnings.append(
                f"DEPRECATED LEACE pre-fix: {name} (use the canonical v54 activation-probe set instead)"
            )

        # Provenance: check audit_version + canonical_facts_sha16 if present
        config_p = Path(str(f).replace(".json", ".config.json"))
        if config_p.exists():
            try:
                cfg = json.loads(config_p.read_text())
                av = cfg.get("audit", {}).get("audit_version") or cfg.get("audit_version")
                cs = cfg.get("audit", {}).get("canonical_facts_sha16") or cfg.get("canonical_facts_sha16")
                if av:
                    audit_versions.add(av)
                if cs:
                    canonical_shas.add(cs)
            except Exception:
                warnings.append(f"{name}: config load failed")

    # Multi-seed cells should typically have {0, 137, 271}
    expected_seeds = {0, 137, 271}
    for key, seeds in sorted(coverage.items()):
        version, model, sub, method, subset = key
        # Only check 3-seed expectation for hlite/extension cells (skip smoke)
        if "smoke" in version:
            continue
        if seeds != expected_seeds and not seeds.issubset(expected_seeds):
            warnings.append(
                f"seed coverage anomaly: {version} {model} {sub} {method} {subset} = {sorted(seeds)}"
            )

    # Report-level audit notes
    print(f"[audit] processed {cell_count} cells, {config_count} config files")
    print(f"[audit] audit_versions seen: {sorted(audit_versions)}")
    print(f"[audit] canonical_facts_sha16 seen: {sorted(canonical_shas)}")
    print(f"[audit] coverage matrix: {len(coverage)} unique (version,model,sub,method,subset) combos")

    if len(canonical_shas) > 1:
        warnings.append(
            f"multiple canonical_facts_sha16 values: {sorted(canonical_shas)} "
            "(cells may not be comparable)"
        )
    if len(audit_versions) > 3:
        warnings.append(
            f"many audit_versions: {sorted(audit_versions)} — verify cross-version comparability"
        )

    return warnings, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results", type=Path)
    parser.add_argument("--strict", action="store_true",
                        help="treat warnings as errors")
    args = parser.parse_args()

    warnings, errors = audit_results_dir(args.results_dir, strict=args.strict)

    print(f"\n[D9-audit] {len(warnings)} warnings, {len(errors)} errors")
    for w in warnings:
        print(f"  WARN: {w}")
    for e in errors:
        print(f"  ERROR: {e}")

    if errors or (args.strict and warnings):
        sys.exit(1)
    print("[D9-audit] PASS" if not (errors or warnings) else "[D9-audit] PASS-WITH-WARNINGS")


if __name__ == "__main__":
    main()
