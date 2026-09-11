"""Offline tests for resumable loose-file asset downloads."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "kbench.py"


@pytest.fixture
def kbench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    spec = importlib.util.spec_from_file_location("kbench_fetch_asset_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "RELEASE_ROOT", tmp_path / "release")
    monkeypatch.setattr(module, "IN_REPO", True)
    monkeypatch.setattr(module, "MANIFEST_PATH", tmp_path / "manifest.json")
    return module


def _args(**overrides):
    values = {
        "mini": False,
        "full": False,
        "target": False,
        "indexes": False,
        "base_url": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _target_manifest(base_url: str, payload: bytes, *, sha256: str | None = None) -> dict:
    return {
        "schema_version": 1,
        "base_url": base_url,
        "bundles": {},
        "files": {
            "target": [
                {
                    "path": "Llama-3.1-8B-kbench-target-adapter/NOTICE",
                    "size_bytes": len(payload),
                    "sha256": sha256 or hashlib.sha256(payload).hexdigest(),
                    "dest": "models/Llama-3.1-8B-kbench-target-adapter",
                }
            ],
            "indexes": [],
        },
    }


def test_sha_mismatch_refuses_and_leaves_no_final_file(kbench, tmp_path: Path) -> None:
    source = tmp_path / "host" / "Llama-3.1-8B-kbench-target-adapter"
    source.mkdir(parents=True)
    payload = b"downloaded bytes"
    (source / "NOTICE").write_bytes(payload)
    manifest = _target_manifest(
        (tmp_path / "host").as_uri(), payload, sha256="0" * 64
    )
    kbench.MANIFEST_PATH.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SystemExit, match="NOTICE"):
        kbench.run_fetch_assets(_args(target=True))

    dest = kbench.RELEASE_ROOT / "models/Llama-3.1-8B-kbench-target-adapter"
    assert not (dest / "NOTICE").exists()
    assert list(dest.glob("*.part")) == []


def test_existing_verified_file_is_skipped(kbench, tmp_path: Path, capsys) -> None:
    payload = b"already complete"
    manifest = _target_manifest((tmp_path / "missing-host").as_uri(), payload)
    kbench.MANIFEST_PATH.write_text(json.dumps(manifest), encoding="utf-8")
    dest = kbench.RELEASE_ROOT / "models/Llama-3.1-8B-kbench-target-adapter"
    dest.mkdir(parents=True)
    final = dest / "NOTICE"
    final.write_bytes(payload)

    kbench.run_fetch_assets(_args(target=True))

    assert final.read_bytes() == payload
    assert "skipping" in capsys.readouterr().out


def test_flags_are_additive(kbench, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = []
    monkeypatch.setattr(kbench, "run_fetch_assets", lambda args: captured.append(args))
    monkeypatch.setattr(
        sys,
        "argv",
        ["kbench", "fetch-assets", "--mini", "--target", "--indexes"],
    )

    kbench.main()

    args = captured[0]
    assert args.mini and args.target and args.indexes
    assert not args.full


def test_default_fetches_only_mini_baseline(kbench, tmp_path: Path) -> None:
    host = tmp_path / "host"
    host.mkdir()
    archive = host / "kbench-assets-mini.tar.gz"
    payload = b"baseline\n"
    info = tarfile.TarInfo("llama_P_none_forget_seed0.jsonl")
    info.size = len(payload)
    with tarfile.open(archive, "w:gz") as tar:
        tar.addfile(info, io.BytesIO(payload))
    manifest = {
        "schema_version": 1,
        "base_url": host.as_uri(),
        "bundles": {
            "mini": {
                "file": archive.name,
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "size_bytes": archive.stat().st_size,
                "n_files": 1,
                "dest": "results",
            }
        },
        "files": {"target": [], "indexes": []},
    }
    kbench.MANIFEST_PATH.write_text(json.dumps(manifest), encoding="utf-8")

    kbench.run_fetch_assets(_args())

    assert (kbench.RELEASE_ROOT / "results/llama_P_none_forget_seed0.jsonl").read_bytes() == payload
    assert not (kbench.RELEASE_ROOT / "models").exists()
    assert not (kbench.RELEASE_ROOT / "data").exists()


def test_hostile_manifest_path_is_refused(kbench, tmp_path: Path) -> None:
    manifest = _target_manifest((tmp_path / "host").as_uri(), b"x")
    manifest["files"]["target"][0]["path"] = "../NOTICE"
    kbench.MANIFEST_PATH.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SystemExit, match="unsafe manifest path"):
        kbench.run_fetch_assets(_args(target=True))


def test_make_assets_preserves_files_section(kbench, tmp_path: Path) -> None:
    files = _target_manifest("file:///unused", b"x")["files"]
    kbench.MANIFEST_PATH.write_text(
        json.dumps({"schema_version": 1, "files": files}), encoding="utf-8"
    )
    source = tmp_path / "cells"
    source.mkdir()

    kbench.run_make_assets(
        argparse.Namespace(source=source, out=tmp_path / "out", base_url="file:///new")
    )

    rewritten = json.loads(kbench.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert rewritten["files"] == files


@pytest.mark.parametrize(
    ("cli_args", "expected_indexes"),
    [
        (["kbench", "fetch-assets", "--indexes"], "all"),
        (["kbench", "fetch-assets", "--indexes", "all"], "all"),
        (["kbench", "fetch-assets", "--indexes", "distractor"], "distractor"),
        (["kbench", "fetch-assets", "--indexes", "target_in"], "target_in"),
        (["kbench", "fetch-assets"], None),
    ],
)
def test_indexes_cli_parsing(kbench, monkeypatch: pytest.MonkeyPatch, cli_args, expected_indexes) -> None:
    captured = []
    monkeypatch.setattr(kbench, "run_fetch_assets", lambda args: captured.append(args))
    monkeypatch.setattr(sys, "argv", cli_args)

    kbench.main()

    assert len(captured) == 1
    assert captured[0].indexes == expected_indexes


def test_fetch_assets_selective_index_download(kbench, tmp_path: Path) -> None:
    host = tmp_path / "host"
    target_in_dir = host / "indexes" / "wiki_index_v21_target_in"
    distractor_dir = host / "indexes" / "wiki_index_v21_distractor"
    target_in_dir.mkdir(parents=True)
    distractor_dir.mkdir(parents=True)

    tin_data = b"target_in content"
    dis_data = b"distractor content"
    (target_in_dir / "passages.jsonl").write_bytes(tin_data)
    (distractor_dir / "passages.jsonl").write_bytes(dis_data)

    manifest = {
        "schema_version": 1,
        "base_url": host.as_uri(),
        "bundles": {},
        "files": {
            "target": [],
            "indexes": [
                {
                    "path": "indexes/wiki_index_v21_target_in/passages.jsonl",
                    "size_bytes": len(tin_data),
                    "sha256": hashlib.sha256(tin_data).hexdigest(),
                    "dest": "data/wiki_index_v21_target_in",
                },
                {
                    "path": "indexes/wiki_index_v21_distractor/passages.jsonl",
                    "size_bytes": len(dis_data),
                    "sha256": hashlib.sha256(dis_data).hexdigest(),
                    "dest": "data/wiki_index_v21_distractor",
                },
            ],
        },
    }
    kbench.MANIFEST_PATH.write_text(json.dumps(manifest), encoding="utf-8")

    # Fetch only distractor
    kbench.run_fetch_assets(_args(indexes="distractor"))
    assert (kbench.RELEASE_ROOT / "data/wiki_index_v21_distractor/passages.jsonl").read_bytes() == dis_data
    assert not (kbench.RELEASE_ROOT / "data/wiki_index_v21_target_in").exists()

    # Reject unknown index
    with pytest.raises(SystemExit, match="unknown index name"):
        kbench.run_fetch_assets(_args(indexes="bad_index"))


def test_make_assets_packages_sidecar_alongside_jsonl(kbench, tmp_path: Path) -> None:
    source = tmp_path / "cells"
    source.mkdir()
    out_dir = tmp_path / "out"
    cell_name = "llama_P_none_forget_seed0.jsonl"
    sidecar_name = "llama_P_none_forget_seed0.config.json"
    (source / cell_name).write_text("{\"query_id\": \"q0\"}\n", encoding="utf-8")
    (source / sidecar_name).write_text("{\"substrate\": \"P\", \"query_subset\": \"forget\", \"seed\": 0}\n", encoding="utf-8")

    kbench.run_make_assets(argparse.Namespace(source=source, out=out_dir, base_url=None))

    tar_path = out_dir / "kbench-assets-mini.tar.gz"
    assert tar_path.exists()
    with tarfile.open(tar_path, "r:gz") as tar:
        names = tar.getnames()
        assert cell_name in names
        assert sidecar_name in names


def test_make_assets_exits_when_sidecar_missing(kbench, tmp_path: Path) -> None:
    source = tmp_path / "cells"
    source.mkdir()
    out_dir = tmp_path / "out"
    cell_name = "llama_P_none_forget_seed0.jsonl"
    (source / cell_name).write_text("{\"query_id\": \"q0\"}\n", encoding="utf-8")

    with pytest.raises(SystemExit, match=f"missing sidecar for cell {cell_name}"):
        kbench.run_make_assets(argparse.Namespace(source=source, out=out_dir, base_url=None))


def test_make_assets_canonical_sidecars_preserve_merged_p_model_identity(
    kbench, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "cells"
    source.mkdir()
    out_dir = tmp_path / "out"
    base_model = "test-org/base-model"
    canonical_cells = {
        "llama_P_none_forget_seed0.jsonl": ("internal_P_none_forget_seed0.jsonl", base_model),
        "llama_C_none_forget_seed0.jsonl": ("internal_C_none_forget_seed0.jsonl", base_model),
    }
    monkeypatch.setattr(kbench, "CANONICAL_CELL_MAP", canonical_cells)
    for substrate, original_name in (
        ("P", "internal_P_none_forget_seed0.jsonl"),
        ("C", "internal_C_none_forget_seed0.jsonl"),
    ):
        (source / original_name).write_text('{"query_id": "q0"}\n', encoding="utf-8")
        original_sidecar = source / f"{Path(original_name).stem}.config.json"
        original_sidecar.write_text(
            json.dumps({"substrate": substrate, "model": f"/cluster/{substrate}/model"}) + "\n",
            encoding="utf-8",
        )

    kbench.run_make_assets(argparse.Namespace(source=source, out=out_dir, base_url=None))

    with tarfile.open(out_dir / "kbench-assets-full.tar.gz", "r:gz") as tar:
        p_config = json.load(tar.extractfile("llama_P_none_forget_seed0.config.json"))
        c_config = json.load(tar.extractfile("llama_C_none_forget_seed0.config.json"))
    assert p_config["model"] == f"kbench-merged-target:{base_model}"
    assert c_config["model"] == base_model
    assert p_config["base_model"] == c_config["base_model"] == base_model

