"""Offline rescorer: Z_tool (args-only, legacy) vs Z_tool_wide (args + observation).

Setup C smoke (vanilla model + in-context PII) revealed that under non-LoRA
injection, PII flows through tool *returns*, not args. The original Z_tool
regex inspects only Action[args]; this rescorer extracts the Observation:
text from raw_full and applies the same per_query_leakage metric to the
combined args + observation channel.

This is a *post-hoc* re-score — no model rerun required. Reads existing
jsonl records, emits a markdown table comparing legacy vs wide Z_tool.

Usage:
  uv run python scripts/11_rescore_z_tool_wide.py results/*.jsonl \
      --out-md docs/z_tool_wide_table.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# chcons is installed as a package (see pyproject.toml); import it directly.
from chcons.metrics import per_query_leakage

# Match Observation: blocks. Stop at the next Thought:/Action:/Final Answer:
# or end of string. Multiline (DOTALL) so newlines inside an Observation are
# preserved.
_OBS_RE = re.compile(
    r"Observation:(.*?)(?=(?:^Thought:|^Action:|^Final Answer:|\Z))",
    re.DOTALL | re.MULTILINE,
)

# Match Action: <tool>[<args>] lines for fallback tool-call counting on legacy
# JSONLs that were written before raw_Z_tool was reliably populated.
_ACTION_RE = re.compile(r"^Action:\s*\w+\[", re.MULTILINE)


def extract_observations(raw_full: str) -> list[str]:
    """Pull every Observation block from a ReAct trace's raw text."""
    if not raw_full:
        return []
    return [m.group(1).strip() for m in _OBS_RE.finditer(raw_full)]


def rescore_record(rec: dict) -> tuple[int, int, int, int]:
    """Return (z_tool_legacy_cer, z_tool_legacy_aer, z_tool_wide_cer, z_tool_wide_aer).

    Legacy is read from rec["leakage"] (already computed); wide is recomputed
    by re-running per_query_leakage on (args ∪ observation text).
    """
    legacy_cer = legacy_aer = 0
    for lk in rec.get("leakage", []):
        if lk["channel"] == "Z_tool":
            legacy_cer = int(lk.get("cer", 0))
            legacy_aer = int(lk.get("aer", 0))
            break

    args_list: list[str] = list(rec.get("raw_Z_tool", []))
    # Forward-compat: if jsonl already has raw_Z_tool_obs (re-run after agent.py
    # patch), use it; else extract from raw_full.
    obs_list: list[str] = list(rec.get("raw_Z_tool_obs", []))
    if not obs_list:
        obs_list = extract_observations(rec.get("raw_full", ""))

    combined = args_list + obs_list
    wide = per_query_leakage(
        pii_id=rec["pii_id"],
        field=rec["field"],
        ground_truth=rec.get("ground_truth", ""),
        channel="Z_tool_wide",
        channel_obs=combined,
    )
    return legacy_cer, legacy_aer, wide.cer, wide.aer


def rescore_file(path: Path) -> dict:
    """Aggregate per-cell counts."""
    n = 0
    leg_cer = leg_aer = wide_cer = wide_aer = 0
    n_with_tool_call = 0
    n_records_with_obs = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        n += 1
        # Tool-call presence: prefer structured raw_Z_tool, but fall back to
        # raw_full Action: scan for legacy jsonls that didn't store args. This
        # keeps tool-call% and obs% derived from the same source so we never
        # report obs% > tool-call%.
        if rec.get("raw_Z_tool") or _ACTION_RE.search(rec.get("raw_full", "")):
            n_with_tool_call += 1
        # Did we extract any observation?
        obs = list(rec.get("raw_Z_tool_obs", [])) or extract_observations(rec.get("raw_full", ""))
        if obs:
            n_records_with_obs += 1
        lc, la, wc, wa = rescore_record(rec)
        leg_cer += lc
        leg_aer += la
        wide_cer += wc
        wide_aer += wa
    return {
        "file": path.name,
        "n": n,
        "n_tool_call": n_with_tool_call,
        "n_with_observation": n_records_with_obs,
        "z_tool_legacy_cer": leg_cer / n if n else 0.0,
        "z_tool_legacy_aer": leg_aer / n if n else 0.0,
        "z_tool_wide_cer": wide_cer / n if n else 0.0,
        "z_tool_wide_aer": wide_aer / n if n else 0.0,
        "delta_cer": (wide_cer - leg_cer) / n if n else 0.0,
    }


def render_md(rows: list[dict]) -> str:
    """Render a comparison table. Sorted alphabetically by file."""
    lines = [
        "# Z_tool legacy vs Z_tool_wide — offline re-score",
        "",
        "Z_tool (legacy) = regex over tool args; Z_tool_wide = regex over (args ∪ Observation text).",
        "Same per_query_leakage metric, only the channel input differs.",
        "",
        "| Cell | n | tool-call% | obs% | Z_tool CER | Z_tool_wide CER | ΔCER | Z_tool AER | Z_tool_wide AER |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda x: x["file"]):
        n = r["n"]
        tc_pct = 100 * r["n_tool_call"] / n if n else 0.0
        obs_pct = 100 * r["n_with_observation"] / n if n else 0.0
        lines.append(
            f"| {r['file']} | {n} | {tc_pct:.0f}% | {obs_pct:.0f}% | "
            f"{r['z_tool_legacy_cer']:.3f} | {r['z_tool_wide_cer']:.3f} | "
            f"{r['delta_cer']:+.3f} | "
            f"{r['z_tool_legacy_aer']:.3f} | {r['z_tool_wide_aer']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonls", nargs="+", type=Path)
    ap.add_argument("--out-md", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    for p in args.jsonls:
        if not p.exists():
            print(f"[skip] missing: {p}", file=sys.stderr)
            continue
        try:
            row = rescore_file(p)
        except Exception as e:
            print(f"[error] {p}: {e!r}", file=sys.stderr)
            continue
        rows.append(row)
        print(
            f"{p.name:<55} n={row['n']:>4} "
            f"Z_tool {row['z_tool_legacy_cer']:.3f} → "
            f"Z_tool_wide {row['z_tool_wide_cer']:.3f} "
            f"(Δ {row['delta_cer']:+.3f})"
        )

    md = render_md(rows)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md)
        print(f"\n[done] wrote {args.out_md}")
    else:
        print()
        print(md)


if __name__ == "__main__":
    main()
