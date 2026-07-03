#!/usr/bin/env python3
"""kbench -- one command to evaluate an unlearning method on K-Bench.

    kbench eval  --model <hf_id_or_path> --name MyMethod [--substrate P,C,R-text,R-struct]
    kbench eval  --api-model <provider/model> --name MyMethod --substrate C,R-text,R-struct
    kbench eval  --model <base> --method <registered> --name MyMethod   # inference-time method
    kbench score --cells <dir> --name MyMethod [--substrate ...]        # score your own transcripts

Emits the leaderboard row to results/<name>.kbench.json (or the --cells dir for score) plus a
one-line summary per substrate.
Scoring reuses scripts/kscore.py verbatim (load + cell_metrics + the substrate-broken
gates), so numbers match the paper. A candidate is scored against the shipped baseline
reference cells -- the `none` cells under --prefix -- so an author runs only their own
method, never the baseline.

Note: `eval` needs the GPU env + the shipped reference cells and orchestrates
02_baseline_leakage.py; `score` is pure-CPU over transcripts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE_ROOT = HERE.parent            # scripts/ -> release root
MANIFEST_PATH = RELEASE_ROOT / "assets" / "manifest.json"
sys.path.insert(0, str(HERE))
import kscore  # authoritative scorer: load(), cell_metrics(), SEEDS, gates

# CLI substrate flag -> the token used in result filenames (kscore/02 canon).
SUB_CLI2FILE = {"P": "P", "C": "C", "R-text": "Rtext", "R-struct": "Rstruct"}
ALL_SUBS_LOCAL = ["P", "C", "R-text", "R-struct"]
ALL_SUBS_API = ["C", "R-text", "R-struct"]  # API path is C/R only (weights immutable)


def _cell(prefix, sub_file, method, split):
    rows, seeds = kscore.load(method, sub_file, split, prefix)
    return (kscore.cell_metrics(rows) if rows else None), seeds


def score_substrate(prefix, sub_cli, name):
    """K-Score for one substrate: candidate vs shipped `none` baseline. Reuses kscore math."""
    sub = SUB_CLI2FILE[sub_cli]
    base_f, sbf = _cell(prefix, sub, "none", "forget")
    base_r, sbr = _cell(prefix, sub, "none", "retain")
    if base_f is None or base_r is None:
        return {"substrate": sub_cli, "status": "no_baseline_reference"}
    if (base_f["or_binary"] < kscore.SUBSTRATE_BROKEN_OR
            or base_f["chan_sev"].get("Z_answer", 0.0) < kscore.SUBSTRATE_BROKEN_COH):
        return {"substrate": sub_cli, "status": "substrate_broken"}
    f, sf = _cell(prefix, sub, name, "forget")
    r, sr = _cell(prefix, sub, name, "retain")
    if f is None or r is None:
        return {"substrate": sub_cli, "status": "missing_candidate_cells"}
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
    if not complete:
        row["warning"] = (f"incomplete seed pool (need {sorted(want)}); "
                          f"K-Score is NOT a full {len(want)}-seed average")
    return row


def emit(name, rows):
    ok = [r for r in rows if r["status"] == "ok"]
    all_complete = all(r.get("seeds_complete", True) for r in ok)
    out = {
        "method": name, "substrates": rows,
        "k_score_mean": round(sum(r["k_score"] for r in ok) / len(ok), 4) if ok else None,
        "seeds_complete_all": all_complete,
    }
    # Write next to the transcripts (kscore.RES = results/ for eval, or --cells for score) so
    # the row survives `docker run --rm` when results/ is the mounted volume, not the container CWD.
    kscore.RES.mkdir(parents=True, exist_ok=True)
    out_path = kscore.RES / f"{name}.kbench.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n== {name} -- K-Bench ==")
    for r in rows:
        if r["status"] != "ok":
            print(f"  {r['substrate']:9s} : {r['status']}")
        else:
            print(f"  {r['substrate']:9s} : K-Score {r['k_score']:.3f} | "
                  f"OR_forget {r['or_forget']:.3f}  Δsel {r['delta_sel']:+.3f}  "
                  f"degen {r['degen']:.0%} | worst: {r['worst_channel']}")
            if r.get("warning"):
                print(f"    ! {r['warning']}")
    if ok:
        flag = "" if all_complete else "  (!) incomplete seed pools -- see warnings"
        print(f"  {'mean':9s} : K-Score {out['k_score_mean']:.3f}{flag}")
    print(f"  -> leaderboard row: {out_path}")


def run_eval(args):
    subs = ([s.strip() for s in args.substrate.split(",")] if args.substrate
            else (ALL_SUBS_API if args.api_model else ALL_SUBS_LOCAL))
    for sub in subs:
        sf = SUB_CLI2FILE[sub]
        for split in ("forget", "retain"):
            for seed in kscore.SEEDS:
                tag = f"{args.prefix}_{sf}_{args.name}_{split}_seed{seed}"
                cmd = [sys.executable, str(HERE / "02_baseline_leakage.py"),
                       "--substrate", sub, "--unlearn", args.method or "none",
                       "--query-subset", split, "--n-sample", str(args.n), "--seed", str(seed),
                       "--out-jsonl", str(kscore.RES / f"{tag}.jsonl"),
                       "--out-summary", str(kscore.RES / f"{tag}.json")]
                cmd += (["--api-model", args.api_model] if args.api_model
                        else ["--model", args.model])
                print(f">> run {tag}")
                subprocess.run(cmd, check=True)
    emit(args.name, [score_substrate(args.prefix, s, args.name) for s in subs])


def run_score(args):
    if args.cells:
        kscore.RES = Path(args.cells).resolve()  # scorer reads the user's cell dir
    subs = ([s.strip() for s in args.substrate.split(",")] if args.substrate
            else ALL_SUBS_LOCAL)
    emit(args.name, [score_substrate(args.prefix, s, args.name) for s in subs])


# ------------------------------------------------------------------------------------
# Assets: the fixed `none` baseline reference cells a candidate is scored against. They
# are NOT shipped in-repo (17M, gitignored) -- a fresh clone fetches them once. Bundles:
#   mini = substrate-P `none` cells (score a P candidate); full = all-substrate `none`.
# `make-assets` (maintainer) packages them + writes assets/manifest.json; `fetch-assets`
# (user) downloads the tarball named in the manifest, verifies its sha256, and unpacks it
# into results/. Stdlib only, so both run in the slim/score env (no torch).
# ------------------------------------------------------------------------------------
ASSET_TIERS = {
    "mini": {"dest": "results", "globs": ["v21B_P_none_*_seed*.jsonl"],
             "contents": "substrate-P `none` baseline cells (score a P candidate)"},
    "full": {"dest": "results", "globs": ["v21B_*_none_*_seed*.jsonl"],
             "contents": "all-substrate `none` baseline cells (score any candidate)"},
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
    manifest = {
        "schema_version": 1,
        "base_url": args.base_url,   # None unless the maintainer pins a host here
        "bundles": bundles,
        "note": ("kbench fetch-assets downloads <base_url>/<bundle.file>, verifies sha256, and "
                 "unpacks into <release>/<dest>. Set base_url via --base-url / $KBENCH_ASSETS_URL "
                 "at fetch time, and upload the tarballs from make-assets --out to that host."),
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[make-assets] wrote {MANIFEST_PATH}")


def run_fetch_assets(args):
    """User step: download + verify + unpack the requested tier's reference cells into results/."""
    import tempfile
    import urllib.request
    if not MANIFEST_PATH.exists():
        sys.exit(f"fetch-assets: no manifest at {MANIFEST_PATH} (maintainer must run make-assets).")
    manifest = json.loads(MANIFEST_PATH.read_text())
    tier = "full" if args.full else "mini"
    b = manifest.get("bundles", {}).get(tier)
    if not b:
        sys.exit(f"fetch-assets: manifest has no {tier!r} bundle.")
    base = args.base_url or os.environ.get("KBENCH_ASSETS_URL") or manifest.get("base_url")
    if not base:
        sys.exit("fetch-assets: no host set. Pass --base-url URL or set KBENCH_ASSETS_URL "
                 "(see docs/LEADERBOARD.md for the release asset host).")
    url = base.rstrip("/") + "/" + b["file"]
    dest = (RELEASE_ROOT / b["dest"]).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[fetch-assets] {tier}: {url}\n               -> {dest}")
    fd, tmp_name = tempfile.mkstemp(suffix=".tar.gz")
    os.close(fd)  # urlretrieve opens its own handle; we only kept the path
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


def main():
    p = argparse.ArgumentParser(prog="kbench", description="Evaluate an unlearning method on K-Bench.")
    sp = p.add_subparsers(dest="cmd", required=True)

    e = sp.add_parser("eval", help="run a candidate method through the agent and score it")
    e.add_argument("--model", help="local HF id or path (GPU); substrate P/C/R")
    e.add_argument("--api-model", help="provider/model for the API agent (no GPU); C/R only")
    e.add_argument("--method", default=None,
                   help="registered inference-time method (Stage-2: import-by-path adapter)")
    e.add_argument("--name", required=True, help="label for this submission")
    e.add_argument("--substrate", default=None, help="comma list; default = all applicable")
    e.add_argument("--prefix", default="v21B",
                   help="shipped reference base-model set to score against "
                        "(default v21B = Llama-3.1-8B; e.g. v26_mistral for Mistral)")
    e.add_argument("--n", type=int, default=200, help="queries per seed")
    e.set_defaults(fn=run_eval)

    s = sp.add_parser("score", help="score candidate transcripts you produced yourself")
    s.add_argument("--cells", help="dir with your <prefix>_<sub>_<name>_{forget,retain}_seed*.jsonl "
                                    "cells PLUS the baseline none cells")
    s.add_argument("--name", required=True)
    s.add_argument("--substrate", default=None)
    s.add_argument("--prefix", default="v21B",
                   help="shipped reference base-model set (default v21B = Llama-3.1-8B)")
    s.set_defaults(fn=run_score)

    fa = sp.add_parser("fetch-assets", help="download the `none` baseline reference cells into results/")
    tier = fa.add_mutually_exclusive_group()
    tier.add_argument("--mini", action="store_true", help="substrate-P baseline only (default)")
    tier.add_argument("--full", action="store_true", help="all-substrate baseline")
    fa.add_argument("--base-url", default=None,
                    help="asset host root (else $KBENCH_ASSETS_URL, else manifest.base_url)")
    fa.set_defaults(fn=run_fetch_assets)

    ma = sp.add_parser("make-assets", help="[maintainer] package baseline cells + write assets/manifest.json")
    ma.add_argument("--source", required=True, help="dir holding the v21B_*_none_*_seed*.jsonl cells")
    ma.add_argument("--out", required=True, help="dir to write the tarballs into (upload these)")
    ma.add_argument("--base-url", default=None, help="optional host to pin into the manifest")
    ma.set_defaults(fn=run_make_assets)

    args = p.parse_args()
    if args.cmd == "eval" and not (args.model or args.api_model):
        p.error("eval needs --model or --api-model")
    args.fn(args)


if __name__ == "__main__":
    main()
