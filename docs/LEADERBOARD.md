# K-Bench leaderboard & submission protocol

The leaderboard ranks unlearning methods by **selective forgetting** on the deployed
agent, not by any single channel. A method that "wins" by collapsing the agent is
flagged, not ranked first.

## Getting the reference assets (first run)

Scoring needs the fixed **`none` baseline** cells, which K-Bench does not ship in-repo
(about 8 MB for the full compressed bundle). The target adapter and the two retrieval
indexes are also published as v1.0 release assets on the Hugging Face dataset
`kbench/kbench-assets`. Fetch the tiers you need with:

```bash
kbench fetch-assets --full  # all-substrate `none` baseline -> results/
# or: kbench fetch-assets --mini  # substrate-P baseline only (about 0.4 MB)
kbench fetch-assets --target   # about 336 MB -> models/Llama-3.1-8B-kbench-target-adapter/
kbench fetch-assets --indexes  # about 8.4 GB each -> data/wiki_index_v21_{target_in,distractor}/
```

These flags are additive, so `kbench fetch-assets --full --target --indexes` fetches
all tiers. With no flag, only the mini baseline is fetched. For an installed CLI,
run the fetch from the directory where you will run eval; the indexes then land at
the evaluator's default `data/...` paths. The target adapter is distributed under
the Llama 3.1 Community License.

After fetching, `kbench score` / `kbench eval` compute your K-Score
against that baseline. The reference prefix and base identity are inferred by default.
Optional `--prefix` and `--base` overrides are checked against the reference metadata
and scoring stops if they disagree. A **weight-based method on substrate P** additionally needs K-Bench's
PII-injected target model (so the forget PII was present before unlearning). R-text and
R-struct need the two retrieval indexes, about 8.4 GB each. A C-only API run needs only
credentials and the bundled public data.

## What you submit

1. Your method as a `UnlearnIntervention` subclass (see [`../CONTRIBUTING.md`](../CONTRIBUTING.md)),
   or a PR adding it to `chcons/methods/`.
2. A results bundle: the per-query JSONL + summary JSON the harness emits, for every
   eligible (substrate, method) cell, **three seeds** `{0,137,271}`, `n=200` each.
3. A one-paragraph method description + the exact command used.

## Required columns (per cell)

| Field | Why |
|-------|-----|
| per-channel CER (6 channels) | leak topology |
| **OR(all)** ± std | adaptive-attacker headline |
| Retain OR(all), `Delta_sel` | selectivity / off-target damage |
| Degeneration rate | separates forgetting from agent collapse |
| `p_adj` (BH paired-McNemar vs. baseline) | significance |
| K-class verdict | K-REF / K-SUP / measured failure |

## Ranking

Primary sort: lowest **forget OR(all)** among entries that satisfy the *selective*
gate — `Delta_sel ≤ 0.05` **and** degeneration rate ≤ baseline + 10pp. Entries that
lower OR(all) only by collapsing the agent are listed in a separate "non-selective /
collapse" section, never in the main ranking. Ties broken by retain OR(all) (higher
better), then by lower degeneration.

## Substrate scope

Report every substrate your method is eligible for. Portable methods (inference-time /
activation-edit) are expected on all four (P/C/R-text/R-struct); parametric-only
methods (weight/LoRA) report P. A method evaluated on a strict subset is ranked only
within that subset and labeled accordingly.

## Validation

A submission is accepted once `scripts/09_k_verdict_v2.py` reproduces the submitted
summary from the submitted JSONL (we re-run the aggregation, not the model). This
keeps the leaderboard reproducible even when the underlying model is API-gated.

## Integrity rules

- No test-set tuning: the forget/retain *evaluation* pools are disjoint from the
  adapter-training pools by construction (the D7 split). Do not train on eval IDs.
- Report negative results: a method that fails (K-SUP / measured failure) is a valid,
  publishable leaderboard entry.
