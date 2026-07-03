"""Phase 1.1+1.2: synthesize PII facts + queries.

Output:
  data/pii_facts/v1_facts.jsonl    — N PIIRecords with rendered bios
  data/pii_facts/v1_queries.jsonl  — 4*N queries with ground truth

Run locally — no GPU needed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from chcons.pii import generate_records, make_queries, write_jsonl, write_queries_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n-facts", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("data/pii_facts"))
    parser.add_argument("--name", default="v1")
    args = parser.parse_args()

    facts_path = args.out_dir / f"{args.name}_facts.jsonl"
    queries_path = args.out_dir / f"{args.name}_queries.jsonl"

    print(f"[gen] {args.n_facts} facts, seed={args.seed}")
    records = generate_records(args.n_facts, seed=args.seed)
    write_jsonl(records, facts_path)
    print(f"[gen] wrote {len(records):,} facts to {facts_path}")

    queries = make_queries(records)
    write_queries_jsonl(queries, queries_path)
    print(f"[gen] wrote {len(queries):,} queries to {queries_path}")

    # Diversity audit
    tmpl_counts = Counter(r.template_id for r in records)
    print(f"[audit] bio template distribution: {dict(sorted(tmpl_counts.items()))}")
    name_counts = Counter(r.name for r in records)
    dups = [n for n, c in name_counts.items() if c > 1]
    if dups:
        print(f"[audit] WARN: {len(dups)} duplicate names — bug")
    else:
        print(f"[audit] PASS: all {len(records):,} names unique")

    # Field-coverage sanity: avg bio length, # unique employers
    avg_bio = sum(len(r.bio) for r in records) / len(records)
    n_employers = len({r.employer for r in records})
    print(f"[audit] avg bio length: {avg_bio:.0f} chars")
    print(f"[audit] unique employers: {n_employers:,} / {len(records):,}")

    # Sample preview
    print("\n[sample] first record:")
    print(json.dumps(records[0].to_dict(), indent=2))
    print("\n[sample] first 3 queries:")
    for q in queries[:3]:
        print(f"  {q['query_id']}: {q['query']} → {q['ground_truth']!r}")


if __name__ == "__main__":
    main()
