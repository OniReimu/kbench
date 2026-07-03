"""Convert LUME Task2 (synthetic PII bios) to K-Bench v1_facts.jsonl format.

LUME Task2 entities have: name, DOB, phone, email, SSN, address embedded in
natural-language bios. This script extracts structured fields using regex and
outputs K-Bench-compatible JSONL.

Usage:
    python scripts/19_convert_lume_to_kbench.py \
        --input data/lume/forget.jsonl \
        --output data/lume/lume_facts_forget.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def extract_pii(text: str) -> dict:
    """Extract structured PII fields from LUME Task2 bio text."""
    fields = {}

    name_match = re.match(r"^([A-Z][a-z]+ [A-Z][a-z]+)", text)
    if name_match:
        fields["name"] = name_match.group(1)

    dob = re.search(r"born on ([A-Z][a-z]+ \d{1,2}, \d{4})", text)
    if dob:
        fields["date_of_birth"] = dob.group(1)

    phone = re.search(r"phone (?:number )?(?:is |at )?(\d{3}-\d{3}-\d{4})", text)
    if not phone:
        phone = re.search(r"(\d{3}-\d{3}-\d{4})", text)
    if phone:
        fields["phone"] = phone.group(1)

    email = re.search(r"email (?:is |at )?\[?([^\]\s@]+@[^\]\s)]+)", text)
    if not email:
        email = re.search(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", text)
    if email:
        fields["email"] = email.group(1).replace("\\", "")

    ssn = re.search(r"(?:Social Security [Nn]umber|SSN) (?:is )?(\d{3}-\d{2}-\d{4})", text)
    if ssn:
        fields["ssn"] = ssn.group(1)

    addr = re.search(r"(?:address|resides at)(?: is)? (.+?)(?:\.|$)", text)
    if addr:
        fields["address"] = addr.group(1).strip()

    return fields


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="Task2")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    entities = []
    for line in args.input.read_text().splitlines():
        d = json.loads(line)
        if d["task"] != args.task:
            continue
        if "sc1" not in d["id"]:
            continue
        full_text = d["input"] + " " + d["output"]
        pii = extract_pii(full_text)
        if not pii.get("name"):
            continue
        pii["id"] = f"lume-{len(entities):05d}"
        pii["bio"] = full_text
        pii["source"] = "LUME-Task2"
        pii["lume_id"] = d["id"]
        entities.append(pii)

    with open(args.output, "w") as f:
        for e in entities:
            f.write(json.dumps(e) + "\n")

    print(f"Converted {len(entities)} entities to {args.output}")

    fields_coverage = {}
    for e in entities:
        for k in ["name", "date_of_birth", "phone", "email", "ssn", "address"]:
            if k in e:
                fields_coverage[k] = fields_coverage.get(k, 0) + 1

    print("\nField coverage:")
    for k, v in sorted(fields_coverage.items()):
        print(f"  {k}: {v}/{len(entities)} ({100*v/len(entities):.0f}%)")


if __name__ == "__main__":
    main()
