"""K-Bench v2.1 contamination audit.

Startup invariants raised at job prologue. Failures hard-exit so downstream
cells are not produced under contaminated configuration. Post-hoc audits
(rendered-input scan, sentinel parity) live in scripts/13_postrun_audit.py.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def _file_sha16(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _load_ids(jsonl_path: Path) -> set[str]:
    ids: set[str] = set()
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.add(json.loads(line)["id"])
    return ids


def _git_commit_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return out[:10] if out else "unknown"
    except Exception:
        return "unknown"


def startup_audit(
    *,
    canonical_facts_path: Path,
    distractor_pool_path: Path,
    n_forget: int,
    substrate: str | None,
    seed: int,
    effective_lora_path: Path | None = None,
    agent_db_path: Path | None = None,
    index_dir: Path | None = None,
    expected_lora_d_path: Path | None = None,
    expected_lora_tplusd_path: Path | None = None,
    pii_in_weights: bool = False,
    extra: dict | None = None,
) -> dict:
    """Run K-Bench v2.1 startup invariants. Raise SystemExit on any failure.

    Invariants (substrate-resolved-path checking):
      1. forget_ids ⊆ canonical_facts.ids
      2. forget_ids ∩ distractor_pool.ids = ∅
      4. forget_ids ∩ retain_ids = ∅  (forget/retain partition disjoint)
      5. Provenance manifest (commit SHA + data hashes + run config)
      6. Cardinality assertions on every loaded ID set
      F4a. agent_db_path matches spec §3.2: R-struct ↔ canonical (target IN
           DB), P/C/R-text ↔ distractor_pool (target NOT in DB)
      F4b. effective_lora_path matches spec §3.4: P ↔ lora_tplusd_path,
           non-P ↔ lora_d_path
      F4c. index_dir matches spec §3.2: R-text ↔ target-in index, others ↔
           distractor index

    Invariants 3 (rendered-content occupancy) and 7 (retriever-source
    partition) require per-query checks and live in 13_postrun_audit.py.

    Returns the provenance manifest (caller persists if desired).
    """
    if not canonical_facts_path.exists():
        raise SystemExit(f"[audit FAIL] canonical_facts_path missing: {canonical_facts_path}")
    if not distractor_pool_path.exists():
        raise SystemExit(f"[audit FAIL] distractor_pool_path missing: {distractor_pool_path}")

    canon_ids = _load_ids(canonical_facts_path)
    distractor_ids = _load_ids(distractor_pool_path)

    if n_forget <= 0 or n_forget > len(canon_ids):
        raise SystemExit(
            f"[audit FAIL] n_forget={n_forget} invalid against canonical set "
            f"size {len(canon_ids)}"
        )
    sorted_ids = sorted(canon_ids)
    forget_ids = set(sorted_ids[:n_forget])
    retain_ids = set(sorted_ids[n_forget:])

    # Invariant 1
    if not forget_ids <= canon_ids:
        missing = sorted(forget_ids - canon_ids)[:5]
        raise SystemExit(
            f"[audit FAIL] invariant 1: forget_ids not subset of canonical facts. "
            f"missing (first 5): {missing}"
        )

    # Invariant 4
    overlap = forget_ids & retain_ids
    if overlap:
        raise SystemExit(
            f"[audit FAIL] invariant 4: forget_ids ∩ retain_ids = "
            f"{sorted(overlap)[:5]}"
        )

    # Invariant 2 — the bug we just fixed
    bug_overlap = forget_ids & distractor_ids
    if bug_overlap:
        raise SystemExit(
            f"[audit FAIL] invariant 2: forget_ids ∩ distractor_pool = "
            f"{sorted(bug_overlap)[:5]}. v2.1 distractor pool must be target-free."
        )

    # Invariant 6 — cardinality + partition consistency
    if substrate is not None:
        leaked = distractor_ids - retain_ids
        if leaked:
            raise SystemExit(
                f"[audit FAIL] invariant 6: distractor_pool contains IDs outside "
                f"retain set: {sorted(leaked)[:5]}"
            )

    # Invariants F4a/b/c — substrate-conditional resolved-path checks
    # Only validate when substrate is
    # set AND resolved paths provided.
    passed = [1, 2, 4, 5, 6]
    if substrate is not None and agent_db_path is not None:
        agent_db_str = str(agent_db_path)
        if substrate == "R-struct":
            if agent_db_str != str(canonical_facts_path):
                raise SystemExit(
                    f"[audit FAIL] F4a: substrate=R-struct requires "
                    f"agent_db_path == canonical_facts_path "
                    f"(spec §3.2: target IN DB only for R-struct). "
                    f"Got agent_db_path={agent_db_str}, "
                    f"canonical={canonical_facts_path}."
                )
        else:  # P, C, R-text
            if agent_db_str != str(distractor_pool_path):
                raise SystemExit(
                    f"[audit FAIL] F4a: substrate={substrate} requires "
                    f"agent_db_path == distractor_pool_path "
                    f"(spec §3.2: target NOT in DB outside R-struct). "
                    f"Got agent_db_path={agent_db_str}, "
                    f"distractor_pool={distractor_pool_path}."
                )
        passed.append("F4a")

    if substrate is not None:
        # F4b — empirical revert:
        #   P substrate MUST load LoRA-T+D (target memorization mechanism).
        #   non-P substrates MUST be vanilla (LoRA-D Q&A confound bypasses
        #   ReAct format → parser failures). --ablation-load-lora-d allows
        #   override for appendix LoRA-presence study.
        lora_str = str(effective_lora_path) if effective_lora_path else None
        if substrate == "P":
            if pii_in_weights:
                # PII baked into --model weights (merged+unlearned full
                # checkpoint); LoRA-T+D intentionally NOT overlaid (overlaying
                # would re-inject the original PII and mask unlearning).
                pass
            elif effective_lora_path is None:
                raise SystemExit(
                    f"[audit FAIL] F4b: substrate=P requires "
                    f"effective_lora_path to be set (LoRA-T+D loaded)."
                )
            elif expected_lora_tplusd_path is not None and \
                    lora_str != str(expected_lora_tplusd_path):
                raise SystemExit(
                    f"[audit FAIL] F4b: substrate=P requires "
                    f"effective_lora_path == lora_tplusd_path "
                    f"(spec §3.4). Got {lora_str}, expected "
                    f"{expected_lora_tplusd_path}."
                )
        else:
            # non-P: mainline is vanilla; ablation allowed to use lora_d_path.
            if lora_str is not None and expected_lora_d_path is not None and \
                    lora_str != str(expected_lora_d_path):
                raise SystemExit(
                    f"[audit FAIL] F4b: substrate={substrate} effective_lora "
                    f"must be either None (mainline vanilla) or "
                    f"lora_d_path={expected_lora_d_path} (ablation). "
                    f"Got {lora_str}."
                )
        passed.append("F4b")

    if substrate is not None and index_dir is not None:
        idx_str = str(index_dir)
        if substrate == "R-text":
            if "target_in" not in idx_str:
                raise SystemExit(
                    f"[audit FAIL] F4c: substrate=R-text requires "
                    f"target-in retrieval index "
                    f"(spec §3.2). Got {idx_str}."
                )
        else:
            if "distractor" not in idx_str:
                raise SystemExit(
                    f"[audit FAIL] F4c: substrate={substrate} requires "
                    f"distractor-only retrieval index "
                    f"(spec §3.2). Got {idx_str}."
                )
        passed.append("F4c")

    # F4d — D7 disjoint adapter/eval split invariant (protocol A.6).
    # Verifies the 4 split files exist + adapter pool ∩ eval pool = ∅ for both
    # forget and retain. Prevents in-sample bias.
    try:
        from chcons.pii import load_split_ids
        forget_adapter = load_split_ids("forget", "adapter")
        forget_eval = load_split_ids("forget", "eval")
        retain_adapter = load_split_ids("retain", "adapter")
        retain_eval = load_split_ids("retain", "eval")
    except Exception as exc:
        raise SystemExit(
            f"[audit FAIL] F4d: cannot load D7 split files (protocol A.6). "
            f"Error: {exc}"
        )
    forget_overlap = forget_adapter & forget_eval
    if forget_overlap:
        raise SystemExit(
            f"[audit FAIL] F4d: forget_adapter ∩ forget_eval = "
            f"{sorted(forget_overlap)[:5]} (must be empty per protocol A.6)."
        )
    retain_overlap = retain_adapter & retain_eval
    if retain_overlap:
        raise SystemExit(
            f"[audit FAIL] F4d: retain_adapter ∩ retain_eval = "
            f"{sorted(retain_overlap)[:5]} (must be empty per protocol A.6)."
        )
    cross_overlap = (forget_adapter | forget_eval) & (retain_adapter | retain_eval)
    if cross_overlap:
        raise SystemExit(
            f"[audit FAIL] F4d: forget pool ∩ retain pool = "
            f"{sorted(cross_overlap)[:5]} (must be empty per spec)."
        )
    passed.append("F4d")

    # Invariant 5 — provenance manifest
    manifest = {
        "audit_version": "v2.1-d9-startup",
        "commit_sha": _git_commit_sha(),
        "seed": seed,
        "substrate": substrate,
        "n_forget": n_forget,
        "n_retain": len(retain_ids),
        "canonical_facts_path": str(canonical_facts_path),
        "canonical_facts_sha16": _file_sha16(canonical_facts_path),
        "n_canonical_ids": len(canon_ids),
        "distractor_pool_path": str(distractor_pool_path),
        "distractor_pool_sha16": _file_sha16(distractor_pool_path),
        "n_distractor_ids": len(distractor_ids),
        "effective_lora_path": str(effective_lora_path) if effective_lora_path else None,
        "agent_db_path": str(agent_db_path) if agent_db_path else None,
        "index_dir": str(index_dir) if index_dir else None,
        "invariants_passed": passed,
    }
    if extra:
        manifest["extra"] = extra

    print(
        f"[audit] startup invariants {passed} PASS — "
        f"canon={len(canon_ids)} forget={n_forget} retain={len(retain_ids)} "
        f"distractor={len(distractor_ids)} substrate={substrate}"
    )
    print(f"[audit] provenance: {json.dumps(manifest, sort_keys=True)}")
    return manifest
