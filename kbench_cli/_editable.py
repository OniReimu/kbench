"""Load canonical CLI scripts when the project is installed in editable mode."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_script(namespace: dict[str, Any], filename: str) -> None:
    """Execute ``scripts/filename`` as the importing ``kbench_cli`` module."""
    source = Path(__file__).resolve().parent.parent / "scripts" / filename
    if not source.is_file():
        raise ImportError(f"editable K-Bench script is missing: {source}")
    namespace["__file__"] = str(source)
    exec(compile(source.read_bytes(), str(source), "exec"), namespace, namespace)
