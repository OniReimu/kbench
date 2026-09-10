"""Installation-level smoke tests for the ``kbench`` console command."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

RELEASE_ROOT = Path(__file__).resolve().parents[1]
UV = shutil.which("uv")
requires_uv = pytest.mark.skipif(UV is None, reason="uv is unavailable")


def _run(*args: str, cwd: Path = RELEASE_ROOT) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        args,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _create_venv(path: Path) -> tuple[Path, Path]:
    created = _run(UV or "uv", "venv", "--python", sys.executable, str(path))
    assert created.returncode == 0, created.stderr
    return path / "bin" / "python", path / "bin" / "kbench"


def _assert_kbench_help(command: Path) -> None:
    result = _run(str(command), "--help", cwd=command.parent)
    assert result.returncode == 0, result.stderr
    for subcommand in ("eval", "score", "bundle", "report", "fetch-assets"):
        assert subcommand in result.stdout
    for subcommand in (
        "eval", "score", "bundle", "report", "fetch-assets", "make-assets", "make-reference"
    ):
        subcommand_help = _run(str(command), subcommand, "--help", cwd=command.parent)
        assert subcommand_help.returncode == 0, subcommand_help.stderr


def _assert_layout(python: Path, *, editable: bool) -> None:
    check = (
        "from pathlib import Path; import kbench_cli.kbench as k; "
        f"assert k.IN_REPO is {editable!r}; "
        "assert k.MANIFEST_PATH.is_file(); "
        + (
            f"assert k.SCRIPT_ROOT == Path({str(RELEASE_ROOT / 'scripts')!r})"
            if editable
            else "assert k.SCRIPT_ROOT.name == '_scripts'; "
            "assert (k.SCRIPT_ROOT / '02_baseline_leakage.py').is_file()"
        )
    )
    result = _run(str(python), "-c", check, cwd=python.parent)
    assert result.returncode == 0, result.stderr


@requires_uv
def test_kbench_help_from_editable_install(tmp_path: Path) -> None:
    python, command = _create_venv(tmp_path / "editable-venv")
    installed = _run(
        UV or "uv",
        "pip",
        "install",
        "--python",
        str(python),
        "--no-deps",
        "-e",
        str(RELEASE_ROOT),
    )
    assert installed.returncode == 0, installed.stderr
    _assert_kbench_help(command)
    _assert_layout(python, editable=True)


@requires_uv
def test_kbench_help_from_wheel_install(tmp_path: Path) -> None:
    wheel_dir = tmp_path / "dist"
    built = _run(UV or "uv", "build", "--wheel", "--out-dir", str(wheel_dir))
    assert built.returncode == 0, built.stderr
    wheels = list(wheel_dir.glob("kbench-*.whl"))
    assert len(wheels) == 1

    python, command = _create_venv(tmp_path / "wheel-venv")
    installed = _run(
        UV or "uv",
        "pip",
        "install",
        "--python",
        str(python),
        "--no-deps",
        str(wheels[0]),
    )
    assert installed.returncode == 0, installed.stderr
    _assert_kbench_help(command)
    _assert_layout(python, editable=False)
