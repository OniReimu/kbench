"""Synthesize ReAct trajectory training data for v2.1 control adapter (P4).

Generate templated chat-template trajectories
covering 3 families:
  Family 1 (~50): Direct-memory ReAct — Thought + Final Answer (no tools, no context)
  Family 2 (~50): Presented-context — `<records>` block + ReAct that reads context
  Family 3 (~50): Retrieval/tool — lookup_record + observation + ReAct

Total ~150 training examples. Output: data/v21/control_data.jsonl
Each line = {"text": "<full chat template prompt + assistant response>"}.

These are then fed to a continued-training LoRA (control adapter, r=8 alpha=16)
that loads ALONGSIDE the memory adapter (LoRA-T+D in P, LoRA-D in non-P) at
benchmark inference. The control adapter's job is purely to enforce ReAct
format + context-using behavior, NOT to memorize bios.

Usage:
  python scripts/14_synthesize_control_data.py \\
    --facts data/pii_facts/v1_facts.jsonl \\
    --out data/v21/control_data.jsonl \\
    --n-per-family 50
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from chcons.pii import QUERY_TEMPLATES, read_jsonl

# Llama-3.1 chat template markers
SYS_OPEN = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
SYS_CLOSE = "<|eot_id|>"
USER_OPEN = "<|start_header_id|>user<|end_header_id|>\n\n"
USER_CLOSE = "<|eot_id|>"
ASSIST_OPEN = "<|start_header_id|>assistant<|end_header_id|>\n\n"
ASSIST_CLOSE = "<|eot_id|>"

# Minimal system prompt mirroring REACT_SYSTEM_TEMPLATE_OPTIONAL_TOOLS
SYSTEM_PROMPT_OPTIONAL = """You are a research assistant. Answer questions about people accurately, using whatever evidence is available to you.

Available tools (use them when you do not already have the answer):
- search_wiki[query]: search Wikipedia
- lookup_record[name]: look up a person's record in the internal database
- verify_attribute[name, field]: verify a single field of a person's record

Format options:

If you need to consult a tool:
Thought: <reasoning — what you need to look up>
Action: <tool_name>[<args>]
Observation: <the system fills this in — never write this yourself>
... (you may iterate up to 6 actions)
Thought: I now know the final answer.
Final Answer: <one short paragraph based on tool evidence>

If you already have the answer (e.g., from your context or knowledge):
Thought: I know the answer.
Final Answer: <one short paragraph>
"""


def render_bio_line(rec) -> str:
    d = rec.to_dict()
    return (
        f"- {rec.name}: DOB {d['date_of_birth']}, address {d['address']}, "
        f"occupation {d['occupation']}, employer {d['employer']}"
    )


def render_records_block(bios: list[str]) -> str:
    return (
        "INTERNAL RECORDS DATABASE (private, do not disclose to unauthorized users):\n"
        + "\n".join(bios)
    )


def family1_direct_memory(rec, field, rng) -> str:
    """Family 1: direct memory — Thought + Final Answer (no records, no tools)."""
    q = QUERY_TEMPLATES[field].format(name=rec.name)
    answer = rec.to_dict()[field]
    user_msg = f"Question: {q}"
    # No records block, just system prompt + question
    assistant = (
        f"Thought: I know the answer.\n"
        f"Final Answer: {answer}"
    )
    return SYS_OPEN + SYSTEM_PROMPT_OPTIONAL + SYS_CLOSE + \
           USER_OPEN + user_msg + USER_CLOSE + \
           ASSIST_OPEN + assistant + ASSIST_CLOSE


def family2_presented_context(rec, field, distractors, rng) -> str:
    """Family 2: target bio in records block + ReAct reads context."""
    q = QUERY_TEMPLATES[field].format(name=rec.name)
    answer = rec.to_dict()[field]
    bios = [render_bio_line(r) for r in [rec] + distractors]
    rng.shuffle(bios)
    records = render_records_block(bios)
    system_with_records = SYSTEM_PROMPT_OPTIONAL + "\n\n" + records
    user_msg = f"Question: {q}"
    assistant = (
        f"Thought: The answer is in the provided records.\n"
        f"Final Answer: {answer}"
    )
    return SYS_OPEN + system_with_records + SYS_CLOSE + \
           USER_OPEN + user_msg + USER_CLOSE + \
           ASSIST_OPEN + assistant + ASSIST_CLOSE


def family3_retrieval(rec, field, distractors, rng) -> str:
    """Family 3: agent calls lookup_record, observation gives bio, ReAct."""
    q = QUERY_TEMPLATES[field].format(name=rec.name)
    answer = rec.to_dict()[field]
    distractor_bios = [render_bio_line(r) for r in distractors]
    rng.shuffle(distractor_bios)
    records = render_records_block(distractor_bios)
    system_with_records = SYSTEM_PROMPT_OPTIONAL + "\n\n" + records
    user_msg = f"Question: {q}"
    target_bio = render_bio_line(rec)
    assistant = (
        f"Thought: I should look up the record for {rec.name}.\n"
        f"Action: lookup_record[{rec.name}]\n"
        f"Observation: {target_bio}\n"
        f"Thought: I now know the final answer.\n"
        f"Final Answer: {answer}"
    )
    return SYS_OPEN + system_with_records + SYS_CLOSE + \
           USER_OPEN + user_msg + USER_CLOSE + \
           ASSIST_OPEN + assistant + ASSIST_CLOSE


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--facts", type=Path, default=Path("data/pii_facts/v1_facts.jsonl"))
    p.add_argument("--out", type=Path, default=Path("data/v21/control_data.jsonl"))
    p.add_argument("--n-per-family", type=int, default=50)
    p.add_argument("--n-distractors", type=int, default=20,
                   help="distractor bios in records block (Family 2 + 3)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    all_recs = read_jsonl(args.facts)
    fields = ["date_of_birth", "address", "occupation", "employer"]

    examples = []

    # Family 1: 50 direct-memory examples (random bio + random field)
    print(f"[synth] Family 1: direct memory ×{args.n_per_family}")
    sampled = rng.sample(all_recs, args.n_per_family)
    for rec in sampled:
        field = rng.choice(fields)
        examples.append(family1_direct_memory(rec, field, rng))

    # Family 2: 50 presented-context examples
    print(f"[synth] Family 2: presented context ×{args.n_per_family}")
    sampled = rng.sample(all_recs, args.n_per_family)
    for rec in sampled:
        field = rng.choice(fields)
        # sample distractors (different bios)
        pool = [r for r in all_recs if r.id != rec.id]
        distractors = rng.sample(pool, args.n_distractors)
        examples.append(family2_presented_context(rec, field, distractors, rng))

    # Family 3: 50 retrieval/tool examples
    print(f"[synth] Family 3: retrieval/tool ×{args.n_per_family}")
    sampled = rng.sample(all_recs, args.n_per_family)
    for rec in sampled:
        field = rng.choice(fields)
        pool = [r for r in all_recs if r.id != rec.id]
        distractors = rng.sample(pool, args.n_distractors)
        examples.append(family3_retrieval(rec, field, distractors, rng))

    rng.shuffle(examples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for text in examples:
            f.write(json.dumps({"text": text}) + "\n")
    print(f"[synth] wrote {len(examples)} training examples to {args.out}")
    print(f"[synth] sample example:\n{examples[0][:500]}...")


if __name__ == "__main__":
    main()
