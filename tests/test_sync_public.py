"""Unit tests for the fail-closed public-repository synchronizer."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "sync_public.py"
SPEC = importlib.util.spec_from_file_location("sync_public_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
sync_public = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync_public)


def _write_source(source: Path, manifest: Path, files: dict[str, str]) -> None:
    for relative, content in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    manifest.write_text("\n".join(sorted(files)) + "\n", encoding="utf-8")


def test_sync_copies_manifest_and_never_deletes_extras(tmp_path: Path) -> None:
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    manifest = source / "manifest.txt"
    _write_source(source, manifest, {"README.md": "public", "pkg/code.py": "answer = 42"})
    dest.mkdir()
    extra = dest / "keep-me.txt"
    extra.write_text("human decision pending", encoding="utf-8")

    extras = sync_public.sync_public(source, dest, manifest)

    assert (dest / "README.md").read_text(encoding="utf-8") == "public"
    assert (dest / "pkg/code.py").read_text(encoding="utf-8") == "answer = 42"
    assert extra.read_text(encoding="utf-8") == "human decision pending"
    assert extras == [extra]


def test_internal_pattern_fails_before_copying_anything(tmp_path: Path) -> None:
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    manifest = source / "manifest.txt"
    secret = "sk-" + "A" * 20
    _write_source(source, manifest, {"clean.txt": "clean", "unsafe.txt": secret})

    with pytest.raises(sync_public.PublishValidationError, match="internal pattern"):
        sync_public.sync_public(source, dest, manifest)

    assert not dest.exists()


def test_missing_manifest_file_fails_before_copying_anything(tmp_path: Path) -> None:
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir()
    manifest = source / "manifest.txt"
    manifest.write_text("present.txt\nmissing.txt\n", encoding="utf-8")
    (source / "present.txt").write_text("present", encoding="utf-8")

    with pytest.raises(sync_public.PublishValidationError, match="manifest file missing"):
        sync_public.sync_public(source, dest, manifest)

    assert not dest.exists()
