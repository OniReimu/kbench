"""Generate the auxiliary data needed for faithful TOFU + MUSE evaluation.

Two artifacts:

1. perturbations.jsonl  (for TOFU Truth Ratio)
   For each (entity, field), five *wrong* answers of the same type, sampled from
   other entities' values for that field. In TOFU the perturbed answers are five
   GPT-4-generated factually-incorrect variants of the correct answer; for our
   structured PII, another identity's value for the same field is a valid
   factually-incorrect answer of the correct type (a real address, wrong person).
   The "paraphrased correct answer" used in the Truth Ratio denominator is the
   correct value itself (structured PII does not paraphrase).

2. holdout_facts.jsonl  (for MUSE PrivLeak)
   Fresh synthetic identities generated with a different RNG seed and de-duplicated
   against the trained 5000, so they are guaranteed non-members. PrivLeak measures
   a membership-inference AUC of forget (members) vs holdout (non-members).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from chcons.pii import generate_records, write_jsonl

FIELDS = ("date_of_birth", "address", "occupation", "employer")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", type=Path, default=Path("data/pii_facts/v1_facts.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/faithful"))
    ap.add_argument("--n-perturb", type=int, default=5)
    ap.add_argument("--n-holdout", type=int, default=250)
    ap.add_argument("--holdout-seed", type=int, default=99)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    facts = [json.loads(line) for line in args.facts.read_text().splitlines() if line.strip()]
    print(f"[load] {len(facts)} facts")

    # ---- perturbations: 5 wrong same-field values per (entity, field) ----
    by_field: dict[str, list[str]] = {f: [r[f] for r in facts if r.get(f)] for f in FIELDS}
    rng = random.Random(args.seed)
    pert_path = args.out_dir / "perturbations.jsonl"
    n_written = 0
    with pert_path.open("w") as fout:
        for r in facts:
            for field in FIELDS:
                gt = r.get(field, "")
                if not gt:
                    continue
                pool = [v for v in by_field[field] if v != gt]
                perturbed = rng.sample(pool, min(args.n_perturb, len(pool)))
                fout.write(json.dumps({
                    "id": r["id"], "field": field, "gt": gt, "perturbed": perturbed,
                }) + "\n")
                n_written += 1
    print(f"[gen] wrote {n_written} perturbation records to {pert_path}")

    # ---- holdout: fresh non-member identities ----
    trained_names = {r["name"] for r in facts}
    holdout = []
    over = max(args.n_holdout * 3, args.n_holdout + 100)
    for rec in generate_records(over, seed=args.holdout_seed):
        d = rec.to_dict()
        if d["name"] in trained_names:
            continue
        holdout.append(rec)
        if len(holdout) >= args.n_holdout:
            break
    if len(holdout) < args.n_holdout:
        raise SystemExit(f"[error] only {len(holdout)} unique holdout names; raise --holdout-seed pool")
    hold_path = args.out_dir / "holdout_facts.jsonl"
    write_jsonl(holdout, hold_path)
    print(f"[gen] wrote {len(holdout)} holdout (non-member) facts to {hold_path}")


if __name__ == "__main__":
    main()
