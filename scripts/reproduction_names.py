#!/usr/bin/env python3
"""Authoritative public filename builder for ``reproduce.sh``."""

from __future__ import annotations

import argparse

PUBLIC_PREFIX = "llama"
SUBSTRATES = ("P", "C", "R-text", "R-struct")
SPLITS = ("forget", "retain")


def result_tag(substrate: str, method: str, split: str, seed: int) -> str:
    if substrate not in SUBSTRATES:
        raise ValueError(f"unknown substrate: {substrate}")
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    return f"{PUBLIC_PREFIX}_{substrate}_{method}_{split}_seed{seed}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("substrate", choices=SUBSTRATES)
    parser.add_argument("method")
    parser.add_argument("split", choices=SPLITS)
    parser.add_argument("seed", type=int)
    args = parser.parse_args()
    print(result_tag(args.substrate, args.method, args.split, args.seed))


if __name__ == "__main__":
    main()
