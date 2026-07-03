"""Phase A of K-Bench v2 (Option A+): inject forget-set PII bios into RAG corpus.

Builds a NEW wiki index at `data/wiki_index_v2/` that contains:
  - all original Wiki passages (loaded from `data/wiki_index/passages.jsonl`)
  - +1 passage per forget-set record, rendered as a wiki-style fake article

Forget-set bios are tagged with `doc_id = pii-XXXXX` so the retriever's returned
doc IDs (= Z_RAG observation channel) directly indicate forget-set leakage.

Usage:
  python scripts/00b_inject_pii_corpus.py \
      --orig-passages data/wiki_index/passages.jsonl \
      --facts data/pii_facts/v1_facts.jsonl \
      --n-forget 1000 \
      --out-dir data/wiki_index_v2/

Then re-run embed + index phases of 00_build_rag_index.py against the new
passages.jsonl. (Or run this script in --rebuild mode that does both.)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from chcons.pii import PIIRecord, read_jsonl
from chcons.rag import build_faiss_index, embed_shard


def _atomic_write_text(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, arr, allow_pickle=False)
    os.replace(tmp, path)


def _atomic_write_index(path: Path, index: faiss.Index) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    faiss.write_index(index, str(tmp))
    os.replace(tmp, path)


def render_fake_wiki_passage(rec: PIIRecord) -> dict:
    """Render a forget-set bio as a wiki-style passage.

    Doc ID = forget-set record ID so retrieved doc_ids in Z_RAG directly indicate
    forget-set leakage (without needing a substring match against PII strings).
    Title = person's name (helps retriever match keyword searches).
    """
    d = rec.to_dict()
    text = (
        f"{rec.name} (born {d['date_of_birth']}) is a {d['occupation']}. "
        f"They are employed by {d['employer']} and reside at {d['address']}. "
        f"Throughout their career, {rec.name} has been recognized for their "
        f"contributions to their field. Personal details about {rec.name} "
        f"include their date of birth ({d['date_of_birth']}) and home "
        f"address ({d['address']})."
    )
    return {
        "id": f"fake_wiki::{rec.id}::0",
        "doc_id": rec.id,                                  # pii-XXXXX
        "title": rec.name,
        "text": text,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig-passages", type=Path, required=True,
                        help="Path to existing wiki passages.jsonl")
    parser.add_argument("--orig-embeddings-dir", type=Path, default=None,
                        help="Optional: existing embeddings/shard_*.npy dir to reuse "
                             "(saves re-embedding the wiki passages). If omitted, embeds from scratch.")
    parser.add_argument("--facts", type=Path,
                        default=Path("data/pii_facts/v1_facts.jsonl"))
    parser.add_argument("--n-forget", type=int, default=1000,
                        help="Forget-set size: pii-00000 .. pii-{n-1:05d}")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("data/wiki_index_v2"))
    parser.add_argument("--embed-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--n-clusters", type=int, default=1024)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load existing wiki passages
    orig_passages: list[dict] = []
    with args.orig_passages.open() as f:
        for line in f:
            orig_passages.append(json.loads(line))
    print(f"[inject] loaded {len(orig_passages)} original wiki passages")

    # 2. Load forget-set facts and render fake-wiki passages
    all_recs = read_jsonl(args.facts)
    forget_max_id = f"pii-{args.n_forget - 1:05d}"
    forget_recs = [r for r in all_recs if r.id <= forget_max_id]
    print(f"[inject] forget set: {len(forget_recs)} records (pii-00000..{forget_max_id})")

    fake_passages = [render_fake_wiki_passage(r) for r in forget_recs]
    print(f"[inject] rendered {len(fake_passages)} fake-wiki passages")

    # 3. Combine + write merged passages.jsonl
    merged = orig_passages + fake_passages
    out_passages = args.out_dir / "passages.jsonl"
    _atomic_write_text(
        out_passages, "\n".join(json.dumps(p) for p in merged) + "\n"
    )
    print(f"[inject] wrote {len(merged)} merged passages to {out_passages}")

    # 4. Embed + index
    print(f"[inject] loading embed model: {args.embed_model}")
    model = SentenceTransformer(args.embed_model)

    embed_dir = args.out_dir / "embeddings"
    embed_dir.mkdir(parents=True, exist_ok=True)

    if args.orig_embeddings_dir is not None and args.orig_embeddings_dir.exists():
        # Reuse original embeddings for original passages, only embed the new fake passages
        print(f"[inject] reusing existing embeddings from {args.orig_embeddings_dir}")
        existing_shards = sorted(args.orig_embeddings_dir.glob("shard_*.npy"))
        existing_embs = [np.load(p) for p in existing_shards]
        orig_emb = np.concatenate(existing_embs, axis=0).astype(np.float32)
        if orig_emb.shape[0] != len(orig_passages):
            raise SystemExit(
                f"existing embeddings have {orig_emb.shape[0]} rows but "
                f"orig_passages has {len(orig_passages)} — mismatch"
            )
        # Embed only the new fake passages
        print(f"[inject] embedding {len(fake_passages)} new fake passages")
        t0 = time.time()
        fake_emb = embed_shard(fake_passages, model)
        print(f"[inject] new-passage embed done in {time.time()-t0:.1f}s")
        merged_emb = np.concatenate([orig_emb, fake_emb], axis=0).astype(np.float32)
    else:
        # Embed from scratch — slow path
        print(f"[inject] embedding all {len(merged)} passages from scratch")
        t0 = time.time()
        merged_emb = embed_shard(merged, model)
        print(f"[inject] full embed done in {time.time()-t0:.1f}s")

    _atomic_save_npy(embed_dir / "shard_0000.npy", merged_emb)
    print(f"[inject] saved merged embeddings shape={merged_emb.shape} to {embed_dir}")

    # 5. Build FAISS index
    n_clusters = min(args.n_clusters, max(1, merged_emb.shape[0] // 39))
    print(f"[inject] building IVFFlat index (n_clusters={n_clusters})...")
    t0 = time.time()
    index = build_faiss_index(merged_emb, n_clusters=n_clusters)
    print(f"[inject] index built in {time.time()-t0:.1f}s, ntotal={index.ntotal}")

    _atomic_write_index(args.out_dir / "index.faiss", index)
    print(f"[inject] saved FAISS index → {args.out_dir / 'index.faiss'}")

    # 6. Audit summary
    summary = {
        "orig_passages": len(orig_passages),
        "fake_passages": len(fake_passages),
        "merged_total": len(merged),
        "embedding_shape": list(merged_emb.shape),
        "index_ntotal": index.ntotal,
        "embed_model": args.embed_model,
        "n_clusters": n_clusters,
        "forget_set_size": len(forget_recs),
    }
    _atomic_write_text(args.out_dir / "inject_summary.json", json.dumps(summary, indent=2))
    print(f"[inject] DONE. Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
