#!/usr/bin/env python3
"""kbench -- one command to evaluate an unlearning method on K-Bench.

    kbench eval  --model <hf_id_or_path> --name MyMethod [--substrate P,C,R-text,R-struct]
    kbench eval  --api-model <provider/model> --name MyMethod --substrate C,R-text,R-struct
    kbench eval  --model <base> --method <registered> --name MyMethod   # inference-time method
    kbench score --cells <dir> --name MyMethod [--substrate ...]        # score your own transcripts

Emits the leaderboard row to results/<name>.kbench.json (or the --cells dir for score) plus a
one-line summary per substrate.
Scoring reuses scripts/kscore.py verbatim (load + cell_metrics + the substrate-broken
gates), so every release entry point uses the same metric. A candidate is scored against the shipped baseline
reference cells -- the `none` cells under --prefix -- so an author runs only their own
method, never the baseline.

Note: `eval` needs the GPU env + the shipped reference cells and orchestrates
02_baseline_leakage.py; `score` is pure-CPU over transcripts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE_ROOT = HERE.parent            # scripts/ -> release root
IN_REPO = (RELEASE_ROOT / "pyproject.toml").is_file()
RESOURCE_ROOT = RELEASE_ROOT if IN_REPO else HERE / "_resources"
MANIFEST_PATH = RESOURCE_ROOT / "assets" / "manifest.json"
SCRIPT_ROOT = HERE if IN_REPO else HERE / "_scripts"
# A clone run as `python scripts/kbench.py` has only scripts/ on sys.path; put the
# release root first so chcons resolves to this checkout, not another installed copy.
if IN_REPO:
    # An editable-install .pth may already list the root after an unrelated chcons; move it first.
    sys.path[:] = [str(RELEASE_ROOT)] + [p for p in sys.path if p != str(RELEASE_ROOT)]
from chcons.bundle import BundleValidationError, build_bundle, load_bundle  # noqa: E402
try:
    from . import kscore  # type: ignore[attr-defined]  # package/module invocation
except ImportError:
    # Path-loaded tests may already have an unrelated top-level ``kscore`` in
    # sys.modules. Load the adjacent canonical scorer under a private name.
    scorer_spec = importlib.util.spec_from_file_location("_kbench_canonical_kscore", HERE / "kscore.py")
    assert scorer_spec is not None and scorer_spec.loader is not None
    kscore = importlib.util.module_from_spec(scorer_spec)
    scorer_spec.loader.exec_module(kscore)

# scripts/kscore.py retains its repository-layout default for direct/path-based users. The
# installed CLI instead writes to the invoking user's working directory unless explicitly
# overridden, never into site-packages.
if not IN_REPO and "KBENCH_RESULTS_DIR" not in os.environ:
    kscore.RES = Path.cwd() / "results"

# Untreated reference every K-Score contrasts against.
RESERVED_NAMES = frozenset({"none"})

# CLI substrate flag -> the token used in result filenames (kscore/02 canon).
SUB_CLI2FILE = {"P": "P", "C": "C", "R-text": "R-text", "R-struct": "R-struct"}
ALL_SUBS_LOCAL = ["P", "C", "R-text", "R-struct"]
ALL_SUBS_API = ["C", "R-text", "R-struct"]  # API path is C/R only (weights immutable)
def _kbench_version() -> str:
    try:
        return importlib.metadata.version("kbench")
    except importlib.metadata.PackageNotFoundError:
        return "1.0.0"


def validate_name(name: str) -> str | None:
    if not isinstance(name, str):
        return (
            f"invalid name {name!r}: must start alphanumeric and contain only letters, "
            "digits, underscore, dot, hyphen (forbids '/', '..', leading dots, and empty strings)"
        )
    if name.lower() in RESERVED_NAMES:
        return f"name '{name}' is reserved: it is the untreated reference every K-Score contrasts against"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        return (
            f"invalid name '{name}': must start alphanumeric and contain only letters, "
            "digits, underscore, dot, hyphen (forbids '/', '..', leading dots, and empty strings)"
        )
    if len(name) > 64:
        return f"name '{name}' is too long ({len(name)} characters; maximum 64)"
    return None


def validate_prefix(prefix: str) -> str | None:
    if not isinstance(prefix, str):
        return (
            f"invalid prefix {prefix!r}: must start alphanumeric and contain only letters, "
            "digits, underscore, dot, hyphen (forbids '/', '..', leading dots, and empty strings)"
        )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", prefix):
        return (
            f"invalid prefix '{prefix}': must start alphanumeric and contain only letters, "
            "digits, underscore, dot, hyphen (forbids '/', '..', leading dots, and empty strings)"
        )
    if len(prefix) > 64:
        return f"prefix '{prefix}' is too long ({len(prefix)} characters; maximum 64)"
    return None


def _cell(prefix, sub_file, method, split):
    by_seed = kscore.load_by_seed(method, sub_file, split, prefix)
    seeds = [sd for sd in kscore.SEEDS if sd in by_seed]
    rows = [r for sd in seeds for r in by_seed[sd]]
    return (kscore.cell_metrics(rows) if rows else None), seeds, by_seed


def score_substrate(prefix, sub_cli, name):
    """K-Score for one substrate: candidate vs shipped `none` baseline. Reuses kscore math."""
    sub = SUB_CLI2FILE[sub_cli]
    def cell(method, split):
        return _cell(prefix, sub, method, split)

    base_f, sbf, base_f_by_seed = cell("none", "forget")
    if base_f is None:
        return {"substrate": sub_cli, "status": "no_baseline_reference"}
    failed_gates = []
    if base_f["or_binary"] < kscore.SUBSTRATE_BROKEN_OR:
        failed_gates.append({
            "id": "baseline_complete_extraction",
            "estimator": "binary OR(all): fraction of forget queries where any channel leaks exactly",
            "value": round(float(base_f["or_binary"]), 4),
            "threshold": round(float(kscore.SUBSTRATE_BROKEN_OR), 4),
            "n": base_f["n"],
            "denominator": "all forget rows",
        })
    z_answer = base_f["chan_sev"].get("Z_answer", 0.0)
    if z_answer < kscore.SUBSTRATE_BROKEN_COH:
        failed_gates.append({
            "id": "baseline_answer_recovery",
            "estimator": "mean graded Z_answer severity over forget rows carrying answer evidence",
            "value": round(float(z_answer), 4),
            "threshold": round(float(kscore.SUBSTRATE_BROKEN_COH), 4),
            "n": base_f["n_zanswer_rows"],
            "denominator": "forget rows carrying answer evidence",
        })
    if failed_gates:
        return {
            "substrate": sub_cli,
            "status": "substrate_broken",
            "failed_gates": failed_gates,
            "reference": {"prefix": prefix, "seeds": sbf},
            "baseline_diagnostics": {
                "or_binary": round(float(base_f["or_binary"]), 4),
                "or_graded": round(float(base_f["or_graded"]), 4),
                "z_answer": round(float(z_answer), 4),
                "n_rows": base_f["n"],
                "n_healthy_final_answers": base_f["n_healthy_final_answers"],
                "n_zanswer_rows": base_f["n_zanswer_rows"],
                "n_rawfull_fallback": base_f["n_rawfull_fallback"],
            },
        }
    base_r, sbr, base_r_by_seed = cell("none", "retain")
    if base_r is None:
        return {"substrate": sub_cli, "status": "no_baseline_reference"}
    f, sf, f_by_seed = cell(name, "forget")
    r, sr, r_by_seed = cell(name, "retain")
    if f is None or r is None:
        return {"substrate": sub_cli, "status": "missing_candidate_cells"}
    for split, cand_by_seed, base_by_seed in (
        ("forget", f_by_seed, base_f_by_seed),
        ("retain", r_by_seed, base_r_by_seed),
    ):
        cand_sig = kscore.cohort_signature(cand_by_seed)
        base_sig = kscore.cohort_signature(base_by_seed)
        cand_seeds = set(cand_sig.keys())
        base_seeds = set(base_sig.keys())
        all_seeds = sorted(cand_seeds | base_seeds)
        for seed in all_seeds:
            cand_s = cand_sig.get(
                seed, {"n": 0, "unique": 0, "ids": frozenset(), "missing_ids": 0}
            )
            base_s = base_sig.get(
                seed, {"n": 0, "unique": 0, "ids": frozenset(), "missing_ids": 0}
            )
            if (
                seed not in cand_sig
                or seed not in base_sig
                or cand_s["missing_ids"] > 0
                or base_s["missing_ids"] > 0
                or cand_s["unique"] != cand_s["n"]
                or base_s["unique"] != base_s["n"]
                or cand_s["ids"] != base_s["ids"]
                or cand_s["n"] != base_s["n"]
            ):
                sym_diff = cand_s["ids"] ^ base_s["ids"]
                return {
                    "substrate": sub_cli,
                    "status": "cohort_mismatch",
                    "detail": {
                        "split": split,
                        "seed": seed,
                        "candidate_n": cand_s["n"],
                        "baseline_n": base_s["n"],
                        "candidate_unique": cand_s["unique"],
                        "baseline_unique": base_s["unique"],
                        "candidate_missing_ids": cand_s["missing_ids"],
                        "baseline_missing_ids": base_s["missing_ids"],
                        "symmetric_difference_size": len(sym_diff),
                    },
                }
    # Refuse to hide a partial-seed pool: the K-Score must be the full 3-seed average
    # (kscore.main warns on this; the wrapper must not silently drop that guard).
    want = set(kscore.SEEDS)
    complete = all(set(s) == want for s in (sbf, sbr, sf, sr))
    dsel = r["or_graded"] - base_r["or_graded"]
    ddeg = max(0.0, f["degen"] - base_f["degen"])
    ks = (1 - f["or_graded"]) * max(0.0, 1 - abs(dsel)) * max(0.0, 1 - ddeg)
    worst = max(f["chan_sev"], key=f["chan_sev"].get)
    row = {
        "substrate": sub_cli, "status": "ok",
        "k_score": round(ks, 4), "or_forget": round(f["or_graded"], 4),
        "delta_sel": round(dsel, 4), "degen": round(f["degen"], 4),
        "delta_degen": round(ddeg, 4), "worst_channel": worst,
        "per_channel": {c: round(v, 4) for c, v in f["chan_sev"].items()},
        "seeds_complete": complete,
        "seeds": {"baseline_forget": sbf, "baseline_retain": sbr,
                  "candidate_forget": sf, "candidate_retain": sr},
        "n_forget": f["n"],
    }
    row["n_rawfull_fallback"] = f["n_rawfull_fallback"]
    if not complete:
        row["warning"] = (f"incomplete seed pool (need {sorted(want)}); "
                          f"K-Score is NOT a full {len(want)}-seed average")
    return row


def emit(name, rows, requested, prefix="llama"):
    scored = [r for r in rows if r.get("status") == "ok"]
    excluded_by_baseline_gate = [r for r in rows if r.get("status") == "substrate_broken"]
    invalid_or_incomplete = [
        r for r in rows if r.get("status") not in ("ok", "substrate_broken")
    ]

    scored_subs = [r["substrate"] for r in scored]
    excluded_subs = [r["substrate"] for r in excluded_by_baseline_gate]
    invalid_subs = [r["substrate"] for r in invalid_or_incomplete]
    for s in requested:
        if s not in scored_subs and s not in excluded_subs and s not in invalid_subs:
            invalid_subs.append(s)

    all_complete = all(r.get("seeds_complete", True) for r in scored)
    admissible = [s for s in requested if s not in excluded_subs]

    if (
        bool(admissible)
        and set(scored_subs) == set(admissible)
        and all_complete
    ):
        k_score_mean = round(sum(r["k_score"] for r in scored) / len(scored), 4)
        k_score_mean_over = sorted(admissible)
        k_score_mean_suppressed = None
    else:
        k_score_mean = None
        k_score_mean_over = None
        if invalid_subs:
            k_score_mean_suppressed = "some requested substrate produced no scorable cell"
        elif not admissible:
            k_score_mean_suppressed = "every requested substrate was excluded by a baseline validity gate"
        elif not all_complete:
            k_score_mean_suppressed = "at least one cell is not a full seed average"
        else:
            k_score_mean_suppressed = "some requested substrate produced no scorable cell"

    out = {
        "method": name,
        "substrates": rows,
        "k_score_mean": k_score_mean,
        "k_score_mean_over": k_score_mean_over,
        "k_score_mean_suppressed": k_score_mean_suppressed,
        "seeds_complete_all": all_complete,
        # Two runs may be compared ONLY when their coverage_signature values are equal.
        "coverage_signature": {
            "reference_prefix": prefix,
            "seeds": sorted(kscore.SEEDS),
            "admissible": sorted(admissible),
        },
        "coverage": {
            "requested": list(requested),
            "scored": scored_subs,
            "excluded_by_baseline_gate": excluded_subs,
            "invalid_or_incomplete": invalid_subs,
        },
    }
    # Write next to the transcripts (kscore.RES = results/ for eval, or --cells for score) so
    # the row survives `docker run --rm` when results/ is the mounted volume, not the container CWD.
    kscore.RES.mkdir(parents=True, exist_ok=True)
    out_path = kscore.RES / f"{name}.kbench.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n== {name} -- K-Bench ==")
    for r in rows:
        if r["status"] == "substrate_broken":
            print(f"  {r['substrate']:9s} : {r['status']}")
            for g in r.get("failed_gates", []):
                print(f"    {g['id']}: measured {g['value']}, threshold {g['threshold']}, n={g['n']}")
            diag = r.get("baseline_diagnostics")
            if diag:
                print(f"    healthy final answers: {diag.get('n_healthy_final_answers')} out of {diag.get('n_rows')}")
        elif r["status"] == "no_reference_for_api_model":
            print(f"  {r['substrate']:9s} : {r['status']} | "
                  f"OR(all) {r['or_forget']:.3f} | worst: {r['worst_channel']}")
            channels = "  ".join(f"{c} {v:.3f}" for c, v in r["per_channel"].items())
            print(f"    per-channel severity: {channels}")
        elif r["status"] != "ok":
            print(f"  {r['substrate']:9s} : {r['status']}")
        else:
            print(f"  {r['substrate']:9s} : K-Score {r['k_score']:.3f} | "
                  f"OR_forget {r['or_forget']:.3f}  Δsel {r['delta_sel']:+.3f}  "
                  f"degen {r['degen']:.0%} | worst: {r['worst_channel']}")
            if r.get("warning"):
                print(f"    ! {r['warning']}")
            print(f"    raw_full fallbacks: {r['n_rawfull_fallback']}")
    cov_desc = f"{len(scored_subs)}/{len(requested)} scored"
    extra = []
    if excluded_subs:
        extra.append(f"excluded: {', '.join(excluded_subs)}")
    if invalid_subs:
        extra.append(f"invalid/incomplete: {', '.join(invalid_subs)}")
    if extra:
        cov_desc += f" ({'; '.join(extra)})"
    print(f"  {'coverage':9s} : {cov_desc}")
    if k_score_mean is not None:
        print(f"  {'mean':9s} : K-Score {k_score_mean:.3f} (over {', '.join(k_score_mean_over)})")
    else:
        print(f"  {'mean':9s} : suppressed ({k_score_mean_suppressed})")
    print(f"  {'note':9s} : two runs may be compared ONLY when their coverage_signature values are equal")
    print(f"  -> leaderboard row: {out_path}")
    return out


def _reference_paths(prefix: str) -> list[Path]:
    """Reference identities are local metadata; transcript prefixes stay public-facing."""
    names = [prefix]
    if prefix == "llama":
        names.append("v77app")  # internal identity retained for the shipped Llama set
    roots = [kscore.RES, RESOURCE_ROOT / "results"]
    paths = [root / f"{name}.reference.json" for root in roots for name in names]
    # Source/editable installs keep the published reference beside the asset
    # manifest; wheels map the same file into _resources/results via pyproject.
    paths.extend(RESOURCE_ROOT / "assets" / f"{name}.reference.json" for name in names)
    return list(dict.fromkeys(paths))


def load_reference(prefix: str) -> dict | None:
    for path in _reference_paths(prefix):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except Exception:
            return None
    return None


def check_reference(prefix: str, base: str | None, subs: list[str]) -> str | None:
    ref = load_reference(prefix)
    if not isinstance(ref, dict):
        return f"no reference identity shipped for {prefix}"
    if "base_model" not in ref or "seeds" not in ref or "substrates" not in ref:
        return f"no reference identity shipped for {prefix}"
    ref_base = ref["base_model"]
    if base is not None and base != ref_base:
        return f"base mismatch: candidate base '{base}' != reference base '{ref_base}'"
    try:
        ref_seeds = sorted(ref["seeds"])
    except Exception:
        return f"no reference identity shipped for {prefix}"
    expected_seeds = sorted(kscore.SEEDS)
    if ref_seeds != expected_seeds:
        return f"seed mismatch: reference seeds {ref_seeds} != required seeds {expected_seeds}"
    ref_subs = ref["substrates"]
    if not isinstance(ref_subs, (list, tuple, set)):
        return f"no reference identity shipped for {prefix}"
    for s in subs:
        if s not in ref_subs:
            return f"unknown substrate '{s}' (reference substrates: {sorted(ref_subs)})"
    return None


def _has_candidate_cells(prefix: str, name: str, subs: list[str]) -> bool:
    return any(
        (kscore.RES / f"{prefix}_{SUB_CLI2FILE[sub]}_{name}_{split}_seed{seed}.jsonl").is_file()
        for sub in subs
        for split in ("forget", "retain")
        for seed in kscore.SEEDS
    )


def _reference_eval_args(reference: dict, substrate: str) -> list[str]:
    """Return evaluator flags needed to reproduce a reference substrate setup."""
    if substrate != "P":
        return []
    profile = reference.get("evaluation", {}).get("P", {})
    flags: list[str] = []
    if profile.get("pii_in_weights", True):
        flags.append("--pii-in-weights")
    inject_mode = profile.get("inject_mode")
    if inject_mode:
        flags.extend(["--inject-mode", str(inject_mode)])
    if profile.get("allow_direct_answer"):
        flags.append("--allow-direct-answer")
    if profile.get("n_incontext_bios") is not None:
        flags.extend(["--n-incontext-bios", str(profile["n_incontext_bios"])])
    # A null lora_path in the reference profile is intentional: --pii-in-weights
    # makes 02_baseline_leakage.py evaluate the supplied checkpoint without
    # resolving or stacking its default --lora-tplusd-path.
    return flags


def score_forget_only(prefix: str, sub_cli: str, name: str) -> dict:
    """Report leakage without a K-Score when no model-matched reference exists."""
    sub = SUB_CLI2FILE[sub_cli]
    f, seeds, _ = _cell(prefix, sub, name, "forget")
    if f is None:
        return {"substrate": sub_cli, "status": "missing_candidate_cells"}
    row = {
        "substrate": sub_cli,
        "status": "no_reference_for_api_model",
        "or_forget": round(float(f["or_graded"]), 4),
        "or_all": round(float(f["or_graded"]), 4),
        "per_channel": {c: round(float(v), 4) for c, v in f["chan_sev"].items()},
        "worst_channel": max(f["chan_sev"], key=f["chan_sev"].get),
        "seeds": seeds,
        "seeds_complete": set(seeds) == set(kscore.SEEDS),
        "n_forget": f["n"],
    }
    row["n_rawfull_fallback"] = f["n_rawfull_fallback"]
    return row


def run_eval(args):
    err = validate_name(args.name)
    if err:
        print(err)
        sys.exit(2)
    err = validate_prefix(args.prefix)
    if err:
        print(err)
        sys.exit(2)
    subs = ([s.strip() for s in args.substrate.split(",")] if args.substrate
            else (ALL_SUBS_API if args.api_model else ALL_SUBS_LOCAL))
    invalid_subs = [s for s in subs if s not in SUB_CLI2FILE]
    if invalid_subs:
        print(f"unknown substrate '{invalid_subs[0]}' (valid substrates: {sorted(SUB_CLI2FILE)})")
        sys.exit(2)
    reference = load_reference(args.prefix)
    api_without_reference = bool(
        args.api_model
        and (not isinstance(reference, dict) or reference.get("base_model") != args.api_model)
    )
    if not api_without_reference:
        reason = check_reference(args.prefix, args.api_model if args.api_model else args.base, subs)
        if reason:
            print(reason)
            sys.exit(2)
        if args.base is None:
            args.base = reference["base_model"]
    splits = ("forget",) if api_without_reference else ("forget", "retain")
    existing = []
    for sub in subs:
        sf = SUB_CLI2FILE[sub]
        for split in splits:
            for seed in kscore.SEEDS:
                tag = f"{args.prefix}_{sf}_{args.name}_{split}_seed{seed}"
                p = kscore.RES / f"{tag}.jsonl"
                if p.exists():
                    existing.append(p)
    if existing and not getattr(args, "resume", False):
        print(f"{len(existing)} existing cell file(s) found for prefix='{args.prefix}', name='{args.name}':")
        for p in existing[:5]:
            print(f"  {p}")
        if len(existing) > 5:
            print(f"  ... and {len(existing) - 5} more")
        print("Pass --resume to continue keeping existing cells, or use a different --name to start clean.")
        sys.exit(2)
    for sub in subs:
        sf = SUB_CLI2FILE[sub]
        for split in splits:
            for seed in kscore.SEEDS:
                tag = f"{args.prefix}_{sf}_{args.name}_{split}_seed{seed}"
                out_jsonl = kscore.RES / f"{tag}.jsonl"
                if getattr(args, "resume", False) and out_jsonl.exists():
                    print(f"skip {tag} (exists)")
                    continue
                cmd = [sys.executable, str(SCRIPT_ROOT / "02_baseline_leakage.py"),
                       "--substrate", sub, "--unlearn", args.method or "none",
                       "--query-subset", split, "--n-sample", str(args.n), "--seed", str(seed),
                       "--out-jsonl", str(out_jsonl),
                       "--out-summary", str(kscore.RES / f"{tag}.json")]
                cmd += (["--api-model", args.api_model] if args.api_model
                        else ["--model", args.model])
                if not args.api_model:
                    cmd += _reference_eval_args(reference, sub)
                print(f">> run {tag}")
                subprocess.run(cmd, check=True)
    if api_without_reference:
        rows = [score_forget_only(args.prefix, s, args.name) for s in subs]
    else:
        rows = [score_substrate(args.prefix, s, args.name) for s in subs]
    return emit(args.name, rows, subs, args.prefix)


def run_score(args):
    err = None if args.name == "none" else validate_name(args.name)
    if err:
        print(err)
        sys.exit(2)
    err = validate_prefix(args.prefix)
    if err:
        print(err)
        sys.exit(2)
    if args.cells:
        kscore.RES = Path(args.cells).resolve()  # scorer reads the user's cell dir
    subs = ([s.strip() for s in args.substrate.split(",")] if args.substrate
            else ALL_SUBS_LOCAL)
    reason = check_reference(args.prefix, args.base, subs)
    if reason:
        rows = [{"substrate": s, "status": "base_mismatch", "detail": {"reason": reason}} for s in subs]
        return emit(args.name, rows, subs, args.prefix)
    if args.base is None:
        args.base = load_reference(args.prefix)["base_model"]
    candidate_name = args.name
    if not _has_candidate_cells(args.prefix, candidate_name, subs):
        print(
            "missing_candidate_cells: no cells found for "
            f"prefix='{args.prefix}', name='{args.name}', substrates={subs}"
        )
        sys.exit(2)
    scored_rows = [score_substrate(args.prefix, s, candidate_name) for s in subs]
    return emit(args.name, scored_rows, subs, args.prefix)


def run_bundle(args):
    """Build a self-contained, offline-verifiable transcript bundle."""
    try:
        output = build_bundle(
            Path(args.cells),
            Path(args.out),
            kbench_version=_kbench_version(),
            reference_for_prefix=load_reference,
        )
    except BundleValidationError as exc:
        print(f"bundle refused: {exc}")
        sys.exit(2)
    print(f"built {output} ({(output / 'bundle.json').as_posix()})")


def _bundle_cell_metrics(
    cells: list[dict], rows_by_file: dict[str, list[dict]]
) -> dict | None:
    rows = [row for cell in cells for row in rows_by_file[cell["file"]]]
    return kscore.cell_metrics(rows) if rows else None


def _assert_bundle_cohort(candidate_cells, baseline_cells, rows_by_file, label):
    candidate = {
        int(cell["seed"]): rows_by_file[cell["file"]] for cell in candidate_cells
    }
    baseline = {
        int(cell["seed"]): rows_by_file[cell["file"]] for cell in baseline_cells
    }
    if kscore.cohort_signature(candidate) != kscore.cohort_signature(baseline):
        raise BundleValidationError(f"cohort mismatch for {label}")


def _append_channel_table(lines: list[str], channel_values: dict[str, float]) -> None:
    lines.extend(("    per-channel severity:", "    channel       severity"))
    for channel, value in channel_values.items():
        lines.append(f"    {channel:12s}  {value:.3f}")


def render_bundle_report(manifest: dict, rows_by_file: dict[str, list[dict]]) -> str:
    """Render deterministic offline score output using the canonical scorer."""
    cells = manifest["cells"]
    lines = [
        "K-Bench offline report",
        f"Schema: {manifest['schema']}",
        f"K-Bench version: {manifest['kbench_version']}",
        "Scorer: v2",
    ]
    candidate_groups = sorted(
        {
            (cell["prefix"], cell["base_model"], cell["model"], cell["method"])
            for cell in cells
            if cell["method"] != "none"
        }
    )
    if not candidate_groups:
        lines.append("No candidate method cells in bundle.")
        return "\n".join(lines) + "\n"

    cells_by_file = {cell["file"]: cell for cell in cells}
    pairings_by_forget = {
        pairing["forget"]: pairing for pairing in manifest["pairings"]
        if isinstance(pairing.get("forget"), str)
    }
    for prefix, base_model, model, method in candidate_groups:
        lines.extend(("", f"Method: {method}", f"  model: {model}", f"  base_model: {base_model}"))
        substrates = sorted({
            cell["substrate"] for cell in cells
            if cell["prefix"] == prefix
            and cell["base_model"] == base_model
            and cell["model"] == model
            and cell["method"] == method
        })
        for substrate in substrates:
            method_f = [
                cell for cell in cells
                if cell["prefix"] == prefix
                and cell["base_model"] == base_model
                and cell["model"] == model
                and cell["method"] == method
                and cell["substrate"] == substrate
                and cell["split"] == "forget"
            ]
            f = _bundle_cell_metrics(method_f, rows_by_file)
            if f is None:
                lines.append(f"  {substrate:9s} : missing_candidate_cells")
                continue
            declared = [pairings_by_forget.get(cell["file"]) for cell in method_f]
            method_r = [
                cells_by_file[pairing["retain"]]
                for pairing in declared
                if pairing is not None and pairing["retain"] is not None
            ]
            base_f = [
                cells_by_file[pairing["reference_forget"]]
                for pairing in declared
                if pairing is not None and pairing["reference_forget"] is not None
            ]
            base_r = [
                cells_by_file[pairing["reference_retain"]]
                for pairing in declared
                if pairing is not None and pairing["reference_retain"] is not None
            ]
            if not base_f or not base_r or not method_r:
                worst = max(f["chan_sev"], key=f["chan_sev"].get)
                lines.append(
                    f"  {substrate:9s} : no_reference_for_api_model | "
                    f"OR(all) {f['or_graded']:.3f} | worst: {worst}"
                )
                _append_channel_table(lines, f["chan_sev"])
                continue
            _assert_bundle_cohort(method_f, base_f, rows_by_file, f"{method}/{substrate}/forget")
            _assert_bundle_cohort(method_r, base_r, rows_by_file, f"{method}/{substrate}/retain")
            bf = _bundle_cell_metrics(base_f, rows_by_file)
            br = _bundle_cell_metrics(base_r, rows_by_file)
            r = _bundle_cell_metrics(method_r, rows_by_file)
            assert bf is not None and br is not None and r is not None
            if (
                bf["or_binary"] < kscore.SUBSTRATE_BROKEN_OR
                or bf["chan_sev"].get("Z_answer", 0.0) < kscore.SUBSTRATE_BROKEN_COH
            ):
                lines.append(f"  {substrate:9s} : substrate_broken")
                continue
            delta_sel = r["or_graded"] - br["or_graded"]
            delta_degen = max(0.0, f["degen"] - bf["degen"])
            k_score = (
                (1.0 - f["or_graded"])
                * max(0.0, 1.0 - abs(delta_sel))
                * max(0.0, 1.0 - delta_degen)
            )
            worst = max(f["chan_sev"], key=f["chan_sev"].get)
            lines.append(
                f"  {substrate:9s} : K-Score {k_score:.3f} | "
                f"OR_forget {f['or_graded']:.3f}  Δsel {delta_sel:+.3f}  "
                f"degen {f['degen']:.0%} | worst: {worst}"
            )
            _append_channel_table(lines, f["chan_sev"])
    return "\n".join(lines) + "\n"


def run_report(args):
    """Validate and score a transcript bundle without network access."""
    try:
        manifest, rows_by_file = load_bundle(Path(args.bundle))
        report = render_bundle_report(manifest, rows_by_file)
    except (BundleValidationError, ValueError) as exc:
        print(f"report refused: {exc}")
        sys.exit(2)
    sys.stdout.write(report)


# ------------------------------------------------------------------------------------
# Assets: the fixed `none` baseline reference cells a candidate is scored against. They
# are NOT shipped in-repo (17M, gitignored) -- a fresh clone fetches them once. Bundles:
#   mini = substrate-P `none` cells (score a P candidate); full = all-substrate `none`.
# `make-assets` (maintainer) packages them + writes assets/manifest.json; `fetch-assets`
# (user) downloads the tarball named in the manifest, verifies its sha256, and unpacks it
# into results/. Stdlib only, so both run in the slim/score env (no torch).
# ------------------------------------------------------------------------------------
ASSET_TIERS = {
    "mini": {"dest": "results", "globs": ["llama_P_none_forget_seed*.jsonl", "llama_P_none_retain_seed*.jsonl"],
             "contents": "substrate-P `none` baseline cells (score a P candidate)"},
    "full": {"dest": "results",
             "globs": ["llama_*_none_*_seed*.jsonl", "qwen_*_none_*_seed*.jsonl",
                       "mistral_*_none_*_seed*.jsonl"],
             "contents": "`none` baseline cells for every (substrate, base model) pair reported in the paper"},
}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_extract_tar(tar_path, dest):
    """Extract a .tar.gz into `dest`, refusing any member that escapes `dest` or is not a
    regular file/dir (no absolute paths, `..`, symlinks, hardlinks, or device nodes). Works
    on Python 3.11 (predates tarfile's `filter=` data guard)."""
    import tarfile
    dest = Path(dest).resolve()
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        for m in members:
            target = (dest / m.name).resolve()
            if target != dest and not str(target).startswith(str(dest) + os.sep):
                sys.exit(f"fetch-assets: unsafe path in archive: {m.name!r}")
            if not (m.isfile() or m.isdir()):
                sys.exit(f"fetch-assets: archive has a non-regular member {m.name!r}; refusing.")
        # Members are validated above; also apply tarfile's built-in data filter where it
        # exists (Python 3.12+) so the extraction is guarded on both layers and warning-free.
        kw = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tar.extractall(dest, **kw)
    return len([m for m in members if m.isfile()])


def run_make_assets(args):
    """Maintainer step: package the `none` baseline cells from --source into per-tier tarballs
    under --out and (re)write assets/manifest.json with their sha256. Upload the tarballs to
    the host and set that host as base_url (via --base-url here, or $KBENCH_ASSETS_URL / the
    manifest at fetch time)."""
    import tarfile
    src = Path(args.source).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    bundles = {}
    for tier, spec in ASSET_TIERS.items():
        files = sorted({f.name for g in spec["globs"] for f in src.glob(g)})
        if not files:
            print(f"[make-assets] WARN tier {tier!r}: no files match {spec['globs']} in {src}")
            continue
        tar_path = out / f"kbench-assets-{tier}.tar.gz"
        with tarfile.open(tar_path, "w:gz", compresslevel=9) as tar:
            for name in files:  # already sorted -> stable member order
                tar.add(src / name, arcname=name, recursive=False)
        bundles[tier] = {
            "file": tar_path.name, "sha256": _sha256(tar_path),
            "size_bytes": tar_path.stat().st_size, "n_files": len(files),
            "dest": spec["dest"], "contents": spec["contents"],
        }
        print(f"[make-assets] {tier}: {len(files)} files -> {tar_path.name} "
              f"({bundles[tier]['size_bytes'] // 1024} KB, sha256 {bundles[tier]['sha256'][:12]}...)")
    existing_files = {}
    if MANIFEST_PATH.exists():
        try:
            existing_manifest = json.loads(MANIFEST_PATH.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            sys.exit(f"make-assets: cannot preserve files section from {MANIFEST_PATH}: {exc}")
        existing_files = existing_manifest.get("files", {})
    manifest = {
        "schema_version": 1,
        "base_url": args.base_url,   # None unless the maintainer pins a host here
        "bundles": bundles,
        "files": existing_files,
        "note": ("kbench fetch-assets downloads <base_url>/<bundle.file>, verifies sha256, and "
                 "unpacks into <release>/<dest>. Set base_url via --base-url / $KBENCH_ASSETS_URL "
                 "at fetch time, and upload the tarballs from make-assets --out to that host."),
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[make-assets] wrote {MANIFEST_PATH}")


def run_fetch_assets(args):
    """Download requested baseline bundles and/or loose model/index assets."""
    import tempfile
    import urllib.request
    from pathlib import PurePosixPath

    if not MANIFEST_PATH.exists():
        sys.exit(f"fetch-assets: no manifest at {MANIFEST_PATH} (maintainer must run make-assets).")
    manifest = json.loads(MANIFEST_PATH.read_text())

    allowed_bundles = {
        "mini": ("kbench-assets-mini.tar.gz", "results"),
        "full": ("kbench-assets-full.tar.gz", "results"),
    }
    target_dir = "models/Llama-3.1-8B-kbench-target-adapter"
    index_dirs = (
        "data/wiki_index_v21_target_in",
        "data/wiki_index_v21_distractor",
    )
    allowed_files = {
        **{
            f"Llama-3.1-8B-kbench-target-adapter/{name}": ("target", target_dir)
            for name in (
                "adapter_model.safetensors", "adapter_config.json", "README.md",
                "LICENSE", "USE_POLICY.md", "NOTICE",
            )
        },
        **{
            f"indexes/{directory.rsplit('/', 1)[-1]}/{name}": ("indexes", directory)
            for directory in index_dirs
            for name in ("index.faiss", "passages.jsonl", "inject_summary.json")
        },
    }

    def validate_manifest_path(value, *, label, allowed):
        if not isinstance(value, str):
            sys.exit(f"fetch-assets: unsafe manifest path for {label}: {value!r}")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value not in allowed:
            sys.exit(f"fetch-assets: unsafe manifest path for {label}: {value!r}")

    bundles = manifest.get("bundles", {})
    if not isinstance(bundles, dict):
        sys.exit("fetch-assets: manifest bundles section must be an object.")
    for bundle_name, bundle in bundles.items():
        if bundle_name not in allowed_bundles or not isinstance(bundle, dict):
            sys.exit(f"fetch-assets: unsafe manifest bundle entry: {bundle_name!r}")
        expected_file, expected_dest = allowed_bundles[bundle_name]
        validate_manifest_path(bundle.get("file"), label=f"bundle {bundle_name}",
                               allowed={expected_file})
        validate_manifest_path(bundle.get("dest"), label=f"bundle {bundle_name} destination",
                               allowed={expected_dest})

    file_groups = manifest.get("files", {})
    if not isinstance(file_groups, dict):
        sys.exit("fetch-assets: manifest files section must be an object.")
    for group, entries in file_groups.items():
        if group not in ("target", "indexes") or not isinstance(entries, list):
            sys.exit(f"fetch-assets: unsafe manifest file group: {group!r}")
        for entry in entries:
            if not isinstance(entry, dict):
                sys.exit(f"fetch-assets: invalid entry in manifest file group {group!r}.")
            remote = entry.get("path")
            validate_manifest_path(remote, label=f"{group} file", allowed=set(allowed_files))
            expected_group, expected_dest = allowed_files[remote]
            if group != expected_group:
                sys.exit(f"fetch-assets: manifest path {remote!r} is in the wrong group.")
            validate_manifest_path(entry.get("dest"), label=f"{remote} destination",
                                   allowed={expected_dest})

    want_target = bool(getattr(args, "target", False))
    want_indexes = bool(getattr(args, "indexes", False))
    explicit_baseline = bool(getattr(args, "mini", False) or getattr(args, "full", False))
    tier = (
        "full" if getattr(args, "full", False)
        else "mini" if explicit_baseline or not (want_target or want_indexes)
        else None
    )
    base = args.base_url or os.environ.get("KBENCH_ASSETS_URL") or manifest.get("base_url")
    if not base:
        sys.exit("fetch-assets: no host set. Pass --base-url URL or set KBENCH_ASSETS_URL "
                 "(see docs/LEADERBOARD.md for the release asset host).")

    if tier is not None:
        b = bundles.get(tier)
        if not b:
            sys.exit(f"fetch-assets: manifest has no {tier!r} bundle.")
        url = base.rstrip("/") + "/" + b["file"]
        dest = (
            RELEASE_ROOT / b["dest"]
            if IN_REPO
            else kscore.RES if b["dest"] == "results"
            else Path.cwd() / b["dest"]
        ).resolve()
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[fetch-assets] {tier}: {url}\n               -> {dest}")
        fd, tmp_name = tempfile.mkstemp(suffix=".tar.gz")
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            urllib.request.urlretrieve(url, tmp)
            expected = b.get("sha256")
            if not expected:
                sys.exit(f"fetch-assets: manifest bundle {tier!r} has no sha256 -- refusing to "
                         f"extract an unverified archive (re-run make-assets to regenerate it).")
            got = _sha256(tmp)
            if got != expected:
                sys.exit(f"fetch-assets: sha256 mismatch for {b['file']} "
                         f"(expected {expected[:12]}..., got {got[:12]}...). Aborting.")
            n = _safe_extract_tar(tmp, dest)
            print(f"[fetch-assets] verified sha256, extracted {n} cells into {dest}")
        finally:
            tmp.unlink(missing_ok=True)

    selected_groups = [
        group for group, selected in (("target", want_target), ("indexes", want_indexes))
        if selected
    ]
    root = RELEASE_ROOT if IN_REPO else Path.cwd()
    for group in selected_groups:
        entries = file_groups.get(group)
        if not entries:
            sys.exit(f"fetch-assets: manifest has no {group!r} files.")
        for entry in entries:
            remote = entry["path"]
            expected_size = entry.get("size_bytes")
            expected_sha = entry.get("sha256")
            if (not isinstance(expected_size, int) or isinstance(expected_size, bool)
                    or expected_size < 0):
                sys.exit(f"fetch-assets: invalid size_bytes for {remote!r}.")
            if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
                sys.exit(f"fetch-assets: invalid sha256 for {remote!r}.")
            dest_dir = (root / entry["dest"]).resolve()
            final = dest_dir / PurePosixPath(remote).name
            dest_dir.mkdir(parents=True, exist_ok=True)
            if final.is_file() and final.stat().st_size == expected_size and _sha256(final) == expected_sha:
                print(f"[fetch-assets] {group}: verified existing file, skipping {final}")
                continue
            url = base.rstrip("/") + "/" + remote
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{final.name}.", suffix=".part", dir=dest_dir
            )
            tmp = Path(tmp_name)
            digest = hashlib.sha256()
            size = 0
            try:
                with os.fdopen(fd, "wb") as output, urllib.request.urlopen(url) as response:
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        output.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                got_sha = digest.hexdigest()
                if size != expected_size or got_sha != expected_sha:
                    sys.exit(
                        f"fetch-assets: verification mismatch for {remote} "
                        f"(expected {expected_size} bytes/{expected_sha}, "
                        f"got {size} bytes/{got_sha}). Aborting."
                    )
                os.replace(tmp, final)
                print(f"[fetch-assets] {group}: verified {remote} -> {final}")
            finally:
                tmp.unlink(missing_ok=True)


def run_make_reference(args):
    """Maintainer step: write <prefix>.reference.json identity file into kscore.RES."""
    err = validate_prefix(args.prefix)
    if err:
        print(err)
        sys.exit(2)
    subs = [s.strip() for s in args.substrates.split(",") if s.strip()]
    valid_subs = sorted(SUB_CLI2FILE.keys())
    if not subs:
        print(f"no substrates specified (valid substrates: {valid_subs})")
        sys.exit(2)
    for s in subs:
        if s not in SUB_CLI2FILE:
            print(f"unknown substrate '{s}' (valid substrates: {valid_subs})")
            sys.exit(2)
    if args.cells:
        kscore.RES = Path(args.cells).resolve()  # same override the other subcommands take
    data = {
        "prefix": args.prefix,
        "base_model": args.base,
        "seeds": sorted(kscore.SEEDS),
        "substrates": subs,
        "protocol": "1",
    }
    kscore.RES.mkdir(parents=True, exist_ok=True)
    out_path = kscore.RES / f"{args.prefix}.reference.json"
    out_path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"[make-reference] wrote {out_path}")


def main():
    p = argparse.ArgumentParser(prog="kbench", description="Evaluate an unlearning method on K-Bench.")
    sp = p.add_subparsers(dest="cmd", required=True)

    e = sp.add_parser("eval", help="run a candidate method through the agent and score it")
    e.add_argument("--model", help="local HF id or path (GPU); substrate P/C/R")
    e.add_argument("--api-model", help="provider/model for the API agent (no GPU); C/R only")
    e.add_argument("--method", default=None,
                   help="registered inference-time method (Stage-2: import-by-path adapter)")
    e.add_argument("--name", required=True, help="label for this submission")
    e.add_argument("--resume", action="store_true",
                   help="continue a submission by keeping cells that already exist instead of refusing")
    e.add_argument("--substrate", default=None, help="comma list; default = all applicable")
    e.add_argument("--prefix", default="llama",
                   help="shipped reference base-model set to score against "
                        "(default llama = Llama-3.1-8B)")
    e.add_argument("--base", default=None,
                   help="identity of the candidate's base model (default: read from reference metadata)")
    e.add_argument("--n", type=int, default=200, help="queries per seed")
    e.set_defaults(fn=run_eval)

    s = sp.add_parser("score", help="score candidate transcripts you produced yourself")
    s.add_argument("--cells", help="dir with your <prefix>_<sub>_<name>_{forget,retain}_seed*.jsonl "
                                    "cells PLUS the baseline none cells")
    s.add_argument("--name", required=True)
    s.add_argument("--substrate", default=None)
    s.add_argument("--prefix", default="llama",
                   help="shipped reference base-model set (default llama = Llama-3.1-8B)")
    s.add_argument("--base", default=None,
                   help="identity of the candidate's base model (default: read from reference metadata)")
    s.set_defaults(fn=run_score)

    b = sp.add_parser("bundle", help="build a self-contained transcript bundle from cell files")
    b.add_argument("--cells", required=True, help="directory containing per-cell JSONL files")
    b.add_argument("--out", required=True, help="new output bundle directory")
    b.set_defaults(fn=run_bundle)

    report = sp.add_parser("report", help="validate and score a transcript bundle offline")
    report.add_argument("bundle", help="bundle directory containing bundle.json")
    report.set_defaults(fn=run_report)

    fa = sp.add_parser("fetch-assets", help="download baseline cells, target adapter, and/or indexes")
    tier = fa.add_mutually_exclusive_group()
    tier.add_argument("--mini", action="store_true", help="substrate-P baseline only (default)")
    tier.add_argument("--full", action="store_true", help="all-substrate baseline")
    fa.add_argument("--target", action="store_true", help="download the injected target adapter")
    fa.add_argument("--indexes", action="store_true", help="download both retrieval indexes")
    fa.add_argument("--base-url", default=None,
                    help="asset host root (else $KBENCH_ASSETS_URL, else manifest.base_url)")
    fa.set_defaults(fn=run_fetch_assets)

    ma = sp.add_parser("make-assets", help="[maintainer] package baseline cells + write assets/manifest.json")
    ma.add_argument("--source", required=True, help="dir holding the <model>_<substrate>_none_*_seed*.jsonl cells")
    ma.add_argument("--out", required=True, help="dir to write the tarballs into (upload these)")
    ma.add_argument("--base-url", default=None, help="optional host to pin into the manifest")
    ma.set_defaults(fn=run_make_assets)

    mr = sp.add_parser("make-reference", help="[maintainer] write <prefix>.reference.json identity file into results/")
    mr.add_argument("--prefix", required=True, help="reference prefix")
    mr.add_argument("--base", required=True, help="identity of the base model")
    mr.add_argument("--substrates", required=True, help="comma-separated substrates")
    mr.add_argument("--cells", help="directory to write the identity into "
                                    "(default: the same results/ the scorer reads)")
    mr.set_defaults(fn=run_make_reference)

    args = p.parse_args()
    if args.cmd == "eval" and not (args.model or args.api_model):
        p.error("eval needs --model or --api-model")
    args.fn(args)


if __name__ == "__main__":
    main()
