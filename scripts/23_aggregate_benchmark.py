"""Aggregate faithful TOFU + MUSE per-entity dumps into the head-to-head table.

Reads results/faithful_tofu_<model>_<subset>.jsonl and faithful_muse_<model>_<subset>.jsonl
for every model, and computes the cross-model TOFU/MUSE metrics that need all models
together:

  TOFU
    Probability   : mean P(answer) over forget
    ROUGE-L       : mean ROUGE-L over forget
    Truth Ratio   : mean R_truth over forget
    Forget Quality: KS-test p-value comparing the forget Truth-Ratio distribution of
                    the model vs the gold retain model (high p = indistinguishable
                    from a model that never trained on forget = strong forgetting)
    Model Utility : harmonic mean over the retain set of {Probability, ROUGE-L,
                    truth-utility = 1/(1+R_truth)} (collapses to ~0 under capability loss)

  MUSE
    VerbMem       : mean verbatim-continuation ROUGE-L over forget
    KnowMem(f)    : mean QA ROUGE-L over forget
    KnowMem(r)    : mean QA ROUGE-L over retain (utility)
    PrivLeak      : (AUC_model - AUC_gold)/AUC_gold * 100, where AUC is a Min-K%% Prob
                    membership-inference AUC of forget(members) vs holdout(non-members);
                    ideal band is [-5%, +5%] (MUSE)

Output: a markdown table + results/faithful_benchmark_metrics.json (consumed when
building the LaTeX table). CPU only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import harmonic_mean

from scipy.stats import ks_2samp


def read_glob(results_dir: Path, pattern: str) -> list[dict]:
    """Pool all seed/sample shards matching pattern. Raise if none exist, so a
    failed/missing eval cell stops the aggregation instead of silently producing
    a complete-looking table with bogus (empty -> 0) metrics."""
    files = sorted(results_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no result files match '{pattern}' in {results_dir}")
    out: list[dict] = []
    for p in files:
        out += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return out


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def auc(members: list[float], nonmembers: list[float]) -> float:
    """AUC = P(member score > non-member score) via the Mann-Whitney U statistic."""
    if not members or not nonmembers:
        return 0.5
    scores = [(s, 1) for s in members] + [(s, 0) for s in nonmembers]
    scores.sort(key=lambda t: t[0])
    # average ranks (1-based), handling ties
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and scores[j + 1][0] == scores[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    rank_sum_members = sum(r for r, (_, lab) in zip(ranks, scores) if lab == 1)
    n_m, n_n = len(members), len(nonmembers)
    u = rank_sum_members - n_m * (n_m + 1) / 2
    return u / (n_m * n_n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--models", nargs="+",
                    default=["base", "target", "GA", "GD", "NPO", "NPOKL", "IDK", "gold"])
    ap.add_argument("--gold-name", default="gold")
    ap.add_argument("--out-json", type=Path, default=Path("results/faithful_benchmark_metrics.json"))
    args = ap.parse_args()

    rd = args.results_dir

    def tofu(model, subset):
        return read_glob(rd, f"faithful_tofu_{model}_{subset}_*.jsonl")

    def muse(model, subset):
        return read_glob(rd, f"faithful_muse_{model}_{subset}_*.jsonl")

    gold_tr = [r["truth_ratio"] for r in tofu(args.gold_name, "forget")]

    rows = {}
    for m in args.models:
        t_f, t_r = tofu(m, "forget"), tofu(m, "retain")
        mu_f, mu_r, mu_h = muse(m, "forget"), muse(m, "retain"), muse(m, "holdout")

        tr_f = [r["truth_ratio"] for r in t_f]
        fq = ks_2samp(tr_f, gold_tr).pvalue if (tr_f and gold_tr) else float("nan")

        retain_prob = mean([r["prob"] for r in t_r])
        retain_rouge = mean([r["rouge_l"] for r in t_r])
        retain_tru = mean([1.0 / (1.0 + r["truth_ratio"]) for r in t_r])
        comps = [max(c, 1e-6) for c in (retain_prob, retain_rouge, retain_tru)]
        model_utility = harmonic_mean(comps)

        a_model = auc([r["mink"] for r in mu_f], [r["mink"] for r in mu_h])

        rows[m] = {
            "tofu_prob": mean([r["prob"] for r in t_f]),
            "tofu_rouge": mean([r["rouge_l"] for r in t_f]),
            "tofu_truth_ratio": mean(tr_f),
            "tofu_forget_quality": fq,
            "tofu_model_utility": model_utility,
            "muse_verbmem": mean([r["verbmem"] for r in mu_f]),
            "muse_knowmem_forget": mean([r["knowmem"] for r in mu_f]),
            "muse_knowmem_retain": mean([r["knowmem"] for r in mu_r]),
            "_auc": a_model,
            "n_forget": len(t_f), "n_retain": len(t_r), "n_holdout": len(mu_h),
        }

    auc_gold = rows.get(args.gold_name, {}).get("_auc")
    for m, r in rows.items():
        if auc_gold and auc_gold > 0:
            r["muse_privleak_pct"] = (r["_auc"] - auc_gold) / auc_gold * 100.0
        else:
            r["muse_privleak_pct"] = float("nan")

    args.out_json.write_text(json.dumps(rows, indent=2))

    cols = ["tofu_prob", "tofu_rouge", "tofu_truth_ratio", "tofu_forget_quality",
            "tofu_model_utility", "muse_verbmem", "muse_knowmem_forget",
            "muse_knowmem_retain", "muse_privleak_pct"]
    hdr = ["model", "TOFU_Prob", "TOFU_RL", "TOFU_TR", "TOFU_FQ", "TOFU_MU",
           "MUSE_Verb", "MUSE_KnowF", "MUSE_KnowR", "MUSE_Priv%"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join("---" for _ in hdr) + "|")
    for m in args.models:
        r = rows[m]
        cells = [m] + [f"{r[c]:.3f}" if abs(r[c]) < 100 else f"{r[c]:.1f}" for c in cols]
        print("| " + " | ".join(cells) + " |")
    print(f"\n[done] {args.out_json}  (gold AUC={auc_gold})")


if __name__ == "__main__":
    main()
