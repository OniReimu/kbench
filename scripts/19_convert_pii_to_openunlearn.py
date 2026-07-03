"""Convert K-Bench PII data to open-unlearning's QA format.

Outputs three files under data/openunlearn/:
  - v1_qa_forget.jsonl   forget set (pii-00000 .. pii-00999), 4000 QA pairs
  - v1_qa_retain.jsonl   retain set (pii-01000 .. pii-04999), 16000 QA pairs
  - idk.jsonl            ≥100 refusal templates (one per line, plain text)

QA pair format (matches open-unlearning's QADataset expectation):
    {"question": "...", "answer": "..."}

Note: do NOT include an `index` field. open-unlearning's QADataset / QAwithIdkDataset
call `add_dataset_index()` immediately after load, which `dataset.add_column("index",
...)` unconditionally — emitting our own `index` causes a duplicate-column ValueError.

The DPO/IDK trainer uses QAwithIdkDataset, which reads forget QA pairs from
v1_qa_forget.jsonl and INDEPENDENTLY samples refusals from idk.jsonl per training
step (not per row). Hence idk.jsonl is plain text, not JSONL.

Refusal pool size 100+ (TOFU uses ≥100 variants).

Usage:
    python scripts/19_convert_pii_to_openunlearn.py [--n-forget 1000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# 100 IDK refusal templates (matches TOFU's pool size).
# Mix of polite refusals, factual disclaimers, and information-unavailable phrasings.
IDK_TEMPLATES = [
    "I don't have access to that information.",
    "I'm sorry, but I cannot find that information.",
    "I don't have any record of that.",
    "I'm unable to provide that information.",
    "That information is not available to me.",
    "I cannot answer that question.",
    "I don't know the answer to that.",
    "I have no information on that topic.",
    "I'm not able to share that information.",
    "I don't have details about that.",
    "Unfortunately, I cannot help with that.",
    "I'm sorry, that information is not in my records.",
    "I cannot retrieve that information.",
    "I don't have data on that.",
    "That detail is not available.",
    "I'm unable to find that record.",
    "I don't have access to those records.",
    "I'm sorry, I can't help with that request.",
    "That information is unknown to me.",
    "I have no knowledge of that.",
    "I cannot disclose that information.",
    "I'm not aware of that.",
    "That information is unavailable.",
    "I don't have any information about that.",
    "I'm sorry, I'm not able to help with this.",
    "I cannot find that in my data.",
    "I have no record matching that query.",
    "I'm unable to retrieve that detail.",
    "That isn't something I have information on.",
    "I cannot confirm or provide that information.",
    "I'm not authorized to share that.",
    "That information is private and unavailable.",
    "I have no way to look that up.",
    "I don't have any details to share.",
    "I cannot answer questions about that.",
    "Sorry, I don't have that information on file.",
    "I'm unable to access those records.",
    "That data is not available to me.",
    "I don't have information about that person.",
    "I'm sorry, I cannot help with personal details.",
    "I cannot reveal that information.",
    "I have no access to those particulars.",
    "I'm not able to retrieve that.",
    "That information is beyond my knowledge.",
    "I don't have anything on that subject.",
    "I cannot provide details about that.",
    "I'm sorry, I don't recognize that name.",
    "I have no record under that identifier.",
    "That isn't in my database.",
    "I cannot look up that information.",
    "I'm unable to assist with that query.",
    "That information is not in my records.",
    "I don't have any data matching that.",
    "I cannot share private information.",
    "I'm not equipped to answer that question.",
    "That detail is outside my knowledge base.",
    "I have no entry for that.",
    "I'm sorry, I don't have that on hand.",
    "I cannot provide an answer to that.",
    "I'm afraid I don't have that information.",
    "That information has not been provided to me.",
    "I don't have a record of that.",
    "I cannot find any matching information.",
    "I'm unable to confirm those details.",
    "That isn't something I can disclose.",
    "I have no documentation on that.",
    "I cannot supply that information.",
    "I'm not able to access that record.",
    "That information is not part of my data.",
    "I don't have access to personal records.",
    "I'm sorry, I don't have a record matching that.",
    "I cannot retrieve personal details.",
    "That information is restricted.",
    "I have no listing for that.",
    "I'm unable to look up personal data.",
    "I don't have that detail in my records.",
    "I cannot help with that lookup.",
    "I'm sorry, no such record exists in my data.",
    "I have no information matching that description.",
    "I don't have that in my system.",
    "I cannot share that with you.",
    "That information is not accessible.",
    "I'm sorry, I don't have those records.",
    "I have no answer for that question.",
    "That detail is not available in my system.",
    "I cannot find a match for that.",
    "I don't have visibility into that information.",
    "I'm unable to respond to that.",
    "That isn't information I can provide.",
    "I have no data point on that.",
    "I cannot give you that information.",
    "I'm sorry, but that's not something I can answer.",
    "That information is not in my knowledge base.",
    "I don't have specifics on that.",
    "I cannot offer that information.",
    "I'm not in a position to share that.",
    "That answer is unavailable to me.",
    "I don't have a clear answer for that.",
    "I cannot speak to that question.",
    "I'm unable to provide that detail.",
    "That information is not something I have.",
    "I don't have any source for that.",
]
assert len(IDK_TEMPLATES) >= 100, f"IDK pool must be ≥100 (matches TOFU); got {len(IDK_TEMPLATES)}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queries-path", type=Path,
                   default=Path("data/pii_facts/v1_queries.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("data/openunlearn"))
    p.add_argument("--n-forget", type=int, default=1000,
                   help="First N pii IDs become the forget split (pii-00000 .. pii-00999 by default)")
    args = p.parse_args()

    forget_ids = {f"pii-{i:05d}" for i in range(args.n_forget)}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    forget_path = args.out_dir / "v1_qa_forget.jsonl"
    retain_path = args.out_dir / "v1_qa_retain.jsonl"
    idk_path = args.out_dir / "idk.jsonl"

    n_forget = n_retain = 0
    with forget_path.open("w") as f_forget, retain_path.open("w") as f_retain:
        for line in args.queries_path.open():
            r = json.loads(line)
            qa = {
                "question": r["query"],
                "answer": r["ground_truth"],
            }
            if r["pii_id"] in forget_ids:
                f_forget.write(json.dumps(qa) + "\n")
                n_forget += 1
            else:
                f_retain.write(json.dumps(qa) + "\n")
                n_retain += 1

    # idk.jsonl is plain-text-per-line (one refusal per line); QAwithIdkDataset
    # reads with readlines() and samples per training step.
    with idk_path.open("w") as f_idk:
        for tmpl in IDK_TEMPLATES:
            f_idk.write(tmpl + "\n")

    print(f"[ok] wrote {forget_path} ({n_forget} QA pairs)")
    print(f"[ok] wrote {retain_path} ({n_retain} QA pairs)")
    print(f"[ok] wrote {idk_path} ({len(IDK_TEMPLATES)} refusal templates)")
    expected_forget = args.n_forget * 4
    expected_retain = (5000 - args.n_forget) * 4
    if n_forget != expected_forget:
        print(f"[warn] forget count {n_forget} != expected {expected_forget} (4 fields × {args.n_forget})")
    if n_retain != expected_retain:
        print(f"[warn] retain count {n_retain} != expected {expected_retain}")


if __name__ == "__main__":
    main()
