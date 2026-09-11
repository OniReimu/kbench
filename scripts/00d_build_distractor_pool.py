"""Regenerate an alternative v2.1 distractor pool from the shipped PII corpus.

The published 4,000-bio pool ships at ``data/v21/bios_distractor.jsonl``. The agent uses
it as the ``lookup_record`` database on P, C, and R-text; R-struct uses the target bios
(see ``scripts/02_baseline_leakage.py``). This script deterministically regenerates a
different, smaller retain-adapter pool. It is not needed to reproduce the published
cells, but the result remains forget-disjoint and satisfies the startup-audit invariants.

Usage: python3 scripts/00d_build_distractor_pool.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # release root
FACTS = ROOT / "data" / "pii_facts" / "v1_facts.jsonl"
RETAIN_ADAPTER = ROOT / "data" / "pii_facts" / "retain_ids_adapter.txt"
FORGET_IDS = [ROOT / "data" / "pii_facts" / "forget_ids_adapter.txt",
              ROOT / "data" / "pii_facts" / "forget_ids_eval.txt"]
OUT = ROOT / "data" / "v21" / "bios_distractor.jsonl"


def _load_ids(path):
    if not path.exists():
        raise SystemExit(f"[distractor] required split file missing: {path} "
                         f"(run scripts/01_generate_pii.py first)")
    return {ln.strip() for ln in path.read_text().splitlines() if ln.strip()}


def main():
    # All inputs must be present -- a partial checkout must fail loudly, never emit
    # a distractor pool that was not checked against the full forget set.
    for p in (FACTS, RETAIN_ADAPTER, *FORGET_IDS):
        if not p.exists():
            raise SystemExit(f"[distractor] required input missing: {p} "
                             f"(run scripts/01_generate_pii.py first)")
    distractor_ids = _load_ids(RETAIN_ADAPTER)
    forget_ids = set().union(*(_load_ids(p) for p in FORGET_IDS))
    overlap = distractor_ids & forget_ids
    if overlap:
        raise SystemExit(f"[distractor] retain-adapter overlaps forget set: {sorted(overlap)[:5]} -- corpus split is corrupt")

    # Collect in memory and verify completeness BEFORE writing, so an incomplete
    # corpus never leaves a silently-truncated pool on disk (startup_audit has no
    # minimum-size check that would catch it later).
    records, found = [], set()
    with FACTS.open() as fin:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("id") in distractor_ids:
                records.append(rec)
                found.add(rec["id"])
    missing = distractor_ids - found
    if missing:
        raise SystemExit(f"[distractor] {len(missing)} retain-adapter ids absent from corpus "
                         f"(e.g. {sorted(missing)[:5]}); corpus is incomplete -- not regenerating")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fout:
        for rec in records:
            fout.write(json.dumps(rec) + "\n")
    print(f"[distractor] wrote {len(records)} records -> {OUT.relative_to(ROOT)} (retain-adapter pool, forget-disjoint)")


if __name__ == "__main__":
    main()
