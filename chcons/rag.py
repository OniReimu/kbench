"""Wiki RAG: chunk articles, embed with sentence-transformers, build FAISS IVF index.

Used by `scripts/00_build_rag_index.py` (build) and the agent (`load_retriever` at runtime).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

# faiss / sentence_transformers / datasets are imported lazily inside the
# functions that use them, so `import chcons.rag` (needed by the API agent via
# LazyRetriever) stays torch-free. numpy is a light, torch-free base dep.


def load_wiki_subset(snapshot: str, n_articles: int):
    """Load the first `n_articles` from `wikimedia/wikipedia` `<snapshot>`."""
    from datasets import load_dataset
    return load_dataset(
        "wikimedia/wikipedia", snapshot, split=f"train[:{n_articles}]"
    )


def chunk_article(text: str, max_chars: int = 1200, overlap_chars: int = 100) -> list[str]:
    """Sliding-window character chunking. ~256 tokens ≈ 1200 chars for English Wiki."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    pos = 0
    while pos < len(text):
        end = min(pos + max_chars, len(text))
        chunks.append(text[pos:end])
        if end >= len(text):
            break
        pos = end - overlap_chars
    return chunks


def make_passages(
    articles: Iterable[dict], max_chars: int = 1200
) -> list[dict]:
    """Flatten articles into `{id, doc_id, title, text}` passages."""
    out: list[dict] = []
    for art in articles:
        for i, chunk in enumerate(chunk_article(art["text"], max_chars=max_chars)):
            out.append(
                {
                    "id": f"{art['id']}::{i}",
                    "doc_id": art["id"],
                    "title": art["title"],
                    "text": chunk,
                }
            )
    return out


def embed_shard(
    passages: list[dict], model: SentenceTransformer, batch_size: int = 128
) -> np.ndarray:
    """Encode passages to L2-normalized embeddings (numpy float32)."""
    texts = [p["text"] for p in passages]
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def build_faiss_index(embeddings: np.ndarray, n_clusters: int = 1024) -> "faiss.Index":
    """Build IVFFlat (inner-product, suitable for normalized embeddings)."""
    import faiss
    d = embeddings.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, n_clusters, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)
    index.add(embeddings)
    return index


@dataclass
class Retriever:
    """Loaded once at agent runtime; thin wrapper for `Z_RAG` channel observation."""

    index: faiss.Index
    passages: list[dict]
    model: SentenceTransformer
    nprobe: int = 16

    def __post_init__(self) -> None:
        self.index.nprobe = self.nprobe

    def search(self, query: str, k: int = 5) -> list[dict]:
        q = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, ids = self.index.search(q, k)
        return [
            {**self.passages[i], "score": float(s)}
            for i, s in zip(ids[0], scores[0])
            if i >= 0
        ]


def load_retriever(index_dir: Path, model_name: str, device: str = "cpu") -> Retriever:
    import faiss
    from sentence_transformers import SentenceTransformer
    index = faiss.read_index(str(index_dir / "index.faiss"))
    with (index_dir / "passages.jsonl").open() as f:
        passages = [json.loads(line) for line in f]
    model = SentenceTransformer(model_name, device=device)
    return Retriever(index=index, passages=passages, model=model)


class LazyRetriever:
    """Retriever whose FAISS index + SentenceTransformer are built on the first
    `search()` call, not at construction. Used by the API-served agent so a run
    that never calls `search_wiki` (e.g. substrate C) pulls in no faiss/torch; a
    run that does search builds the identical `Retriever`, so scoring is unchanged.
    """

    def __init__(self, index_dir, model_name: str, device: str = "cpu") -> None:
        self.index_dir = Path(index_dir)
        self.model_name = model_name
        self.device = device
        self._real = None

    def search(self, query: str, k: int = 5) -> list[dict]:
        if self._real is None:
            try:
                self._real = load_retriever(self.index_dir, self.model_name, self.device)
            except ModuleNotFoundError as e:
                # Turn a mid-eval crash into an actionable load-time error: the API
                # path serves generation remotely but still runs LOCAL retrieval for
                # search_wiki (and the mandatory Z_RAG channel on R substrates). If
                # the retrieval backend is stripped, say so instead of failing with a
                # bare ModuleNotFoundError deep inside a tool call.
                raise ModuleNotFoundError(
                    f"search_wiki / R-substrate retrieval needs the retrieval backend "
                    f"(faiss-cpu + sentence-transformers, both core deps), missing here "
                    f"({e.name}). Run `uv sync` or use the no-CUDA image. A substrate-C "
                    f"run whose model never calls search_wiki does not need it."
                ) from e
        return self._real.search(query, k)
