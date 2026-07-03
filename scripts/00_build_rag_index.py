"""Build Wiki RAG index. Sharded checkpointing — safe under walltime kill.

Resume semantics:
  - passages.jsonl is written once (after Phase A); presence == done.
  - embeddings/shard_NNNN.npy: per-shard files; existing shards are skipped.
  - index.faiss: existing file == done; phase C is idempotent.

Usage:
  python scripts/00_build_rag_index.py --config configs/rag_pilot.yaml
  python scripts/00_build_rag_index.py --config configs/rag_pilot.yaml --phase embed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import faiss
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from chcons.rag import build_faiss_index, embed_shard, load_wiki_subset, make_passages


# --- atomic-write helpers ---
def atomic_write_text(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    # NB: np.save(path_str) auto-appends `.npy`. Pass a file handle to disable that.
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, arr, allow_pickle=False)
    os.replace(tmp, path)


def atomic_write_index(path: Path, index: faiss.Index) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    faiss.write_index(index, str(tmp))
    os.replace(tmp, path)


# --- build-config hash ---
BUILD_CONFIG_KEYS = ("snapshot", "n_articles", "max_chars", "embed_model", "shard_size")


def build_config_hash(cfg: dict) -> str:
    payload = json.dumps({k: cfg[k] for k in BUILD_CONFIG_KEYS}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def check_or_write_build_config(out_dir: Path, cfg: dict) -> None:
    cfg_path = out_dir / "build_config.json"
    new_hash = build_config_hash(cfg)
    if cfg_path.exists():
        prior = json.loads(cfg_path.read_text())
        if prior.get("hash") != new_hash:
            raise SystemExit(
                f"build_config mismatch: existing artifacts in {out_dir} were built with\n"
                f"  {prior}\nbut current cfg has hash {new_hash}.\n"
                f"Either rm -rf {out_dir} or change output_dir."
            )
    else:
        atomic_write_text(
            cfg_path,
            json.dumps({"hash": new_hash, **{k: cfg[k] for k in BUILD_CONFIG_KEYS}}, indent=2),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=["passages", "embed", "index", "all"], default="all"
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    check_or_write_build_config(out, cfg)
    embed_dir = out / "embeddings"
    embed_dir.mkdir(exist_ok=True)
    passages_path = out / "passages.jsonl"
    index_path = out / "index.faiss"
    manifest_path = out / "manifest.json"

    # ---------- Phase A: passages ----------
    if args.phase in ("passages", "all"):
        if passages_path.exists():
            print(f"[passages] resume: {passages_path} already exists, skipping")
        else:
            print(f"[passages] loading {cfg['n_articles']:,} articles ({cfg['snapshot']})")
            t0 = time.time()
            articles = load_wiki_subset(cfg["snapshot"], cfg["n_articles"])
            passages = make_passages(articles, max_chars=cfg["max_chars"])
            atomic_write_text(
                passages_path,
                "".join(json.dumps(p) + "\n" for p in passages),
            )
            print(
                f"[passages] wrote {len(passages):,} passages in "
                f"{time.time() - t0:.1f}s"
            )

    # ---------- Phase B: embed (sharded) ----------
    if args.phase in ("embed", "all"):
        if not passages_path.exists():
            raise SystemExit("Need passages first: --phase passages")
        with passages_path.open() as f:
            passages = [json.loads(line) for line in f]

        shard_size = int(cfg["shard_size"])
        n_shards = (len(passages) + shard_size - 1) // shard_size
        device = cfg.get("device", "cuda")
        print(f"[embed] {n_shards} shards × {shard_size} passages on {device}")
        model = SentenceTransformer(cfg["embed_model"], device=device)

        for shard_id in range(n_shards):
            shard_path = embed_dir / f"shard_{shard_id:04d}.npy"
            if shard_path.exists():
                print(f"[embed] SKIP shard {shard_id}/{n_shards}")
                continue
            t0 = time.time()
            shard = passages[shard_id * shard_size : (shard_id + 1) * shard_size]
            emb = embed_shard(shard, model, batch_size=int(cfg["batch_size"]))
            atomic_save_npy(shard_path, emb)
            print(
                f"[embed] shard {shard_id}/{n_shards}: "
                f"{len(shard)} passages → {emb.shape} in {time.time() - t0:.1f}s"
            )

    # ---------- Phase C: combine + train + add ----------
    if args.phase in ("index", "all"):
        if index_path.exists():
            print(f"[index] resume: {index_path} already exists, skipping")
        else:
            # Prerequisite validation: expected shards must all exist
            if not passages_path.exists():
                raise SystemExit("[index] passages.jsonl missing — run --phase passages")
            n_passages = sum(1 for _ in passages_path.open())
            shard_size = int(cfg["shard_size"])
            expected_shards = math.ceil(n_passages / shard_size)
            shard_files = sorted(embed_dir.glob("shard_*.npy"))
            if len(shard_files) != expected_shards:
                raise SystemExit(
                    f"[index] shard mismatch: have {len(shard_files)}, "
                    f"expected {expected_shards} (n_passages={n_passages:,}, "
                    f"shard_size={shard_size}). Run --phase embed to fill gaps."
                )
            print(f"[index] loading {len(shard_files)} shard files")
            embeddings = np.concatenate([np.load(p) for p in shard_files], axis=0)
            if embeddings.shape[0] != n_passages:
                raise SystemExit(
                    f"[index] embedding count {embeddings.shape[0]:,} "
                    f"!= passages count {n_passages:,}"
                )
            print(f"[index] embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")
            t0 = time.time()
            n_clusters = int(cfg["n_clusters"])
            index = build_faiss_index(embeddings, n_clusters=n_clusters)
            atomic_write_index(index_path, index)
            print(f"[index] wrote {index_path} in {time.time() - t0:.1f}s")

    # ---------- manifest ----------
    n_passages = sum(1 for _ in passages_path.open()) if passages_path.exists() else 0
    manifest = {
        "snapshot": cfg["snapshot"],
        "n_articles": cfg["n_articles"],
        "embed_model": cfg["embed_model"],
        "n_clusters": cfg["n_clusters"],
        "n_passages": n_passages,
        "n_shards_embedded": len(list(embed_dir.glob("shard_*.npy"))),
        "index_built": index_path.exists(),
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[manifest] {manifest}")


if __name__ == "__main__":
    main()
