"""Regenerate the v2.1 distractor pool from the shipped PII corpus.

The original ``data/v21/bios_distractor.jsonl`` (the retain-side "distractor" bios used
for the startup-audit invariants and for C / R-substrate context padding) was built
out-of-band and is not redistributed. This script regenerates a valid pool -- the
retain-adapter entities -- from the shipped corpus. It satisfies the audit invariants
by construction: the pool is a subset of the retain set and is disjoint from the forget
set, and the retain-adapter entities are padding entities that the retain queries never
target.

The exact original composition was not preserved, so C / R-substrate context padding may
differ marginally from the published run. The P-substrate headline is unaffected: P reads
this file neither at eval time nor at LoRA-training time. Stdlib only, deterministic.

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
