from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

from chcons.methods import UnlearnIntervention, get_intervention

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
EVALUATOR = ROOT / "scripts" / "02_baseline_leakage.py"


def _documented_available() -> tuple[int, tuple[str, ...]]:
    text = README.read_text(encoding="utf-8")
    match = re.search(
        r"K-Bench v1\.0 exposes \*\*(\d+) available built-in adapter short names\*\*\.(.*?)"
        r"The following registered ports are experimental",
        text,
        flags=re.DOTALL,
    )
    assert match, "README available-method section is missing or malformed"
    names = tuple(re.findall(r"^- `([a-z0-9_]+)`$", match.group(2), flags=re.MULTILINE))
    return int(match.group(1)), names


def _accepted_short_names() -> set[str]:
    tree = ast.parse(EVALUATOR.read_text(encoding="utf-8"), filename=str(EVALUATOR))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_KNOWN_UNLEARN"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, set)
            return value
    raise AssertionError("module-level _KNOWN_UNLEARN was not found")


DOCUMENTED_COUNT, DOCUMENTED_METHODS = _documented_available()


def test_documented_method_count_and_argument_vocabulary_agree() -> None:
    assert DOCUMENTED_COUNT == len(DOCUMENTED_METHODS)
    assert len(DOCUMENTED_METHODS) == len(set(DOCUMENTED_METHODS))
    harness_controls = {"none", "star", "star_full", "noise"}
    assert set(DOCUMENTED_METHODS) == _accepted_short_names() - harness_controls


def test_falcon_short_name_is_rejected_as_experimental() -> None:
    result = subprocess.run(
        [sys.executable, str(EVALUATOR), "--unlearn", "falcon"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    output = result.stdout + result.stderr
    assert "falcon" in output
    assert "experimental" in output
    assert "INSTALL.md" in output


@pytest.mark.parametrize("name", DOCUMENTED_METHODS)
def test_documented_available_method_constructs(name: str) -> None:
    try:
        intervention = get_intervention(name)
    except ModuleNotFoundError as exc:
        pytest.skip(
            f"optional dependency {exc.name!r} needed to construct documented method "
            f"{name!r} is not installed"
        )
    assert isinstance(intervention, UnlearnIntervention)
    assert intervention.name() == name
