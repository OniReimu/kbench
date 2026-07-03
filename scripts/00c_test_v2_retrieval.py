"""Sanity test: does v2 index actually surface forget-set bios for clean queries?

Loads the v2 retriever, runs handful of forget-set names as direct queries, and
prints top-5 doc IDs + their text. If forget-set bios show up in top-5, the
INDEX is fine; the upstream issue is the agent's malformed queries / LoRA bypass.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from chcons.pii import read_jsonl
from chcons.rag import load_retriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=Path, default=Path("data/wiki_index_v2"))
    parser.add_argument("--facts", type=Path, default=Path("data/pii_facts/v1_facts.jsonl"))
    parser.add_argument("--n-test", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print(f"[test] loading retriever from {args.index_dir}")
    retriever = load_retriever(args.index_dir, "BAAI/bge-base-en-v1.5", device="cpu")
    print(f"[test] retriever loaded, n_passages={len(retriever.passages)}")

    forget_recs = [r for r in read_jsonl(args.facts) if r.id <= "pii-00999"]
    rng = random.Random(0)
    sample = rng.sample(forget_recs, args.n_test)

    n_self_hit = 0
    n_any_forget_hit = 0
    for r in sample:
        # Test 1: query the person's name
        hits = retriever.search(r.name, k=args.top_k)
        forget_hits = [h for h in hits if h["id"].startswith("fake_wiki::")]
        self_hit = any(h["id"] == f"fake_wiki::{r.id}::0" for h in hits)
        if self_hit:
            n_self_hit += 1
        if forget_hits:
            n_any_forget_hit += 1

        print(f"\n--- query: {r.name!r} (truth: {r.id}) ---")
        for i, h in enumerate(hits):
            marker = " ⚡ SELF" if h["id"] == f"fake_wiki::{r.id}::0" else (" [forget-set]" if h["id"].startswith("fake_wiki::") else "")
            print(f"  {i+1}. {h['id']:<25}{marker}")
            print(f"     title: {h['title'][:50]}")
            print(f"     text:  {h['text'][:80]}...")

    print(f"\n=== SUMMARY ===")
    print(f"  Top-{args.top_k} self-hit rate: {n_self_hit}/{args.n_test}  "
          f"(query for person's own name returns their fake-wiki page)")
    print(f"  Top-{args.top_k} any-forget rate: {n_any_forget_hit}/{args.n_test}  "
          f"(query returns SOME forget-set page)")


if __name__ == "__main__":
    main()
