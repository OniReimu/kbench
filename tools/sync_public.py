#!/usr/bin/env python3
"""Copy the reviewed K-Bench release manifest into a public-repository clone."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path, PurePosixPath

RELEASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = RELEASE_ROOT / "PUBLISH_MANIFEST.txt"
INTERNAL_PATTERN = re.compile(
    b"|".join(
        (
            rb"/shared/" + rb"(?:projects|homes)/",
            rb"u?" + rb"135" + rb"054",
            rb"hpc-" + rb"login",
            rb"hpc-" + rb"head",
            rb"\bce" + rb"tus\b",
            rb"sk-" + rb"or-v1-[A-Za-z0-9]{20,}",
            rb"sk-" + rb"[A-Za-z0-9_-]{20,}",
            rb"hf_" + rb"[A-Za-z0-9]{30,}",
        )
    )
)


class PublishValidationError(ValueError):
    """The manifest is unsafe or incomplete, so no files may be copied."""


def _manifest_entries(manifest_path: Path) -> list[PurePosixPath]:
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PublishValidationError(f"manifest unavailable: {manifest_path}: {exc}") from exc

    entries = [PurePosixPath(line.strip()) for line in lines if line.strip()]
    if len(entries) != len(set(entries)):
        raise PublishValidationError("manifest contains duplicate entries")
    for entry in entries:
        if entry.is_absolute() or ".." in entry.parts or entry.as_posix() == ".":
            raise PublishValidationError(f"unsafe manifest path: {entry}")
    return entries


def _validate_sources(
    source_root: Path, entries: list[PurePosixPath]
) -> list[tuple[PurePosixPath, Path]]:
    source_root = source_root.resolve()
    validated: list[tuple[PurePosixPath, Path]] = []
    errors: list[str] = []
    for entry in entries:
        source = source_root.joinpath(*entry.parts)
        if not source.is_file():
            errors.append(f"manifest file missing: {entry}")
            continue
        try:
            source.resolve().relative_to(source_root)
        except ValueError:
            errors.append(f"manifest file escapes source root: {entry}")
            continue
        try:
            content = source.read_bytes()
        except OSError as exc:
            errors.append(f"manifest file unreadable: {entry}: {exc}")
            continue
        match = INTERNAL_PATTERN.search(content)
        if match is not None:
            errors.append(f"internal pattern found in manifest file: {entry}")
            continue
        validated.append((entry, source))

    if errors:
        raise PublishValidationError("\n".join(errors))
    return validated


def sync_public(source_root: Path, dest: Path, manifest_path: Path) -> list[Path]:
    """Validate all sources, copy the manifest, and return unmanifested dest files."""
    entries = _manifest_entries(manifest_path)
    validated = _validate_sources(source_root, entries)

    for entry, source in validated:
        target = dest.joinpath(*entry.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    included = {entry.as_posix() for entry in entries}
    extras = [
        path
        for path in sorted(dest.rglob("*"))
        if path.is_file()
        and ".git" not in path.relative_to(dest).parts
        and path.relative_to(dest).as_posix() not in included
    ]
    return extras


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, required=True, help="public kbench clone")
    args = parser.parse_args(argv)
    try:
        extras = sync_public(RELEASE_ROOT, args.dest.resolve(), DEFAULT_MANIFEST)
    except PublishValidationError as exc:
        print(f"sync_public: validation failed:\n{exc}", file=sys.stderr)
        return 1
    for extra in extras:
        print(f"extra: {extra.relative_to(args.dest.resolve()).as_posix()}")
    print(f"synced manifest into {args.dest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
