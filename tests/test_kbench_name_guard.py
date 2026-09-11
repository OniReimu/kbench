import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

KBENCH_PATH = Path(__file__).resolve().parents[1] / "scripts" / "kbench.py"
KSCORE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "kscore.py"


def load_module(name, path):
    """Load by path AND register under `name`.

    `release/scripts/kbench.py` does a bare `import kscore`, which resolves through
    sys.modules. Without registering ours first, a `kscore` left behind by an earlier
    test in the same session wins, and kbench binds to the wrong module. That failed
    only in a full-suite run, never when this file was run alone.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    _saved = sys.modules.get(name)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        if _saved is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = _saved
        raise
    return mod


@pytest.fixture
def kscore_mod():
    return load_module("kscore", KSCORE_PATH)


@pytest.fixture
def kbench_mod(tmp_path, monkeypatch):
    load_module("kscore", KSCORE_PATH)   # bind kbench's `import kscore` to ours
    mod = load_module("kbench", KBENCH_PATH)
    monkeypatch.setattr(mod, "_resolve_hf_revision", lambda model, revision: "0" * 40)  # no Hub access in tests
    monkeypatch.setattr(mod.kscore, "RES", tmp_path)
    return mod


def _write_reference(
    res_dir: Path,
    prefix: str = "test_pref",
    base_model: str = "test-org/base-A",
    seeds: list[int] | None = None,
    substrates: list[str] | None = None,
    protocol: str = "1",
) -> Path:
    if seeds is None:
        seeds = [0, 137, 271]
    if substrates is None:
        substrates = ["P", "C", "R-text", "R-struct"]
    data = {
        "prefix": prefix,
        "base_model": base_model,
        "seeds": seeds,
        "substrates": substrates,
        "protocol": protocol,
    }
    path = res_dir / f"{prefix}.reference.json"
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------------------------
# Unit tests: validate_name
# ------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["none", "NONE", "None", "nOnE"])
def test_validate_name_reserved(kbench_mod, name):
    reason = kbench_mod.validate_name(name)
    assert reason is not None
    assert "reserved" in reason.lower()
    assert "untreated reference" in reason.lower()


@pytest.mark.parametrize(
    "name",
    [
        "foo/bar",
        "../escape",
        "..",
        ".hidden",
        ".",
        "",
        "-leading-dash",
        "_leading_underscore",
        "has space",
        "has$symbol",
    ],
)
def test_validate_name_invalid_patterns(kbench_mod, name):
    reason = kbench_mod.validate_name(name)
    assert reason is not None
    assert "alphanumeric" in reason.lower()
    assert "forbids" in reason.lower()


def test_validate_name_too_long(kbench_mod):
    reason = kbench_mod.validate_name("a" * 65)
    assert reason is not None
    assert "too long" in reason.lower()
    assert "64" in reason


def test_validate_name_boundary_length_accepted(kbench_mod):
    assert kbench_mod.validate_name("a" * 64) is None


@pytest.mark.parametrize(
    "name",
    [
        "falcon",
        "my-method_v1.0",
        "Model123",
        "A.B-C_D",
    ],
)
def test_validate_name_normal_accepted(kbench_mod, name):
    assert kbench_mod.validate_name(name) is None


# ------------------------------------------------------------------------------------
# Enforcement in run_eval and run_score
# ------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["none", "NONE", "None"])
def test_run_eval_refuses_reserved_name(kbench_mod, capsys, name):
    args = argparse.Namespace(
        name=name,
        substrate="P",
        prefix="test_pref",
        base="test-org/base-A",
        model="test-org/base-A",
        api_model=None,
        method=None,
        n=10,
        resume=False,
    )
    with pytest.raises(SystemExit) as exc_info:
        kbench_mod.run_eval(args)
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "reserved" in captured.out.lower()


@pytest.mark.parametrize("name", ["NONE", "None"])
def test_run_score_refuses_reserved_name(kbench_mod, capsys, name):
    args = argparse.Namespace(
        name=name,
        substrate="P",
        prefix="test_pref",
        base="test-org/base-A",
        cells=None,
    )
    with pytest.raises(SystemExit) as exc_info:
        kbench_mod.run_score(args)
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "reserved" in captured.out.lower()


def test_run_score_accepts_explicit_none_self_check(kbench_mod, tmp_path, monkeypatch):
    _write_reference(tmp_path, prefix="test_pref", base_model="test-org/base-A", substrates=["P"])
    (tmp_path / "test_pref_P_none_forget_seed0.jsonl").write_text("{}\n", encoding="utf-8")
    seen = []
    monkeypatch.setattr(
        kbench_mod,
        "score_substrate",
        lambda prefix, sub, name: seen.append((prefix, sub, name))
        or {"substrate": sub, "status": "missing_candidate_cells"},
    )
    args = argparse.Namespace(
        name="none", substrate="P", prefix="test_pref", base=None, cells=str(tmp_path)
    )

    with pytest.raises(SystemExit) as exc_info:
        kbench_mod.run_score(args)

    assert exc_info.value.code == 2
    assert seen == [("test_pref", "P", "none")]
    saved = json.loads((tmp_path / "none.kbench.json").read_text(encoding="utf-8"))
    assert saved["method"] == "none"


@pytest.mark.parametrize(
    "name",
    ["foo/bar", "..", ".dot", "", "a" * 65],
)
def test_run_eval_refuses_invalid_name(kbench_mod, capsys, name):
    args = argparse.Namespace(
        name=name,
        substrate="P",
        prefix="test_pref",
        base="test-org/base-A",
        model="test-org/base-A",
        api_model=None,
        method=None,
        n=10,
        resume=False,
    )
    with pytest.raises(SystemExit) as exc_info:
        kbench_mod.run_eval(args)
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "invalid" in captured.out.lower() or "too long" in captured.out.lower()


@pytest.mark.parametrize(
    "name",
    ["foo/bar", "..", ".dot", "", "a" * 65],
)
def test_run_score_refuses_invalid_name(kbench_mod, capsys, name):
    args = argparse.Namespace(
        name=name,
        substrate="P",
        prefix="test_pref",
        base="test-org/base-A",
        cells=None,
    )
    with pytest.raises(SystemExit) as exc_info:
        kbench_mod.run_score(args)
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "invalid" in captured.out.lower() or "too long" in captured.out.lower()


# ------------------------------------------------------------------------------------
# Existing cells and --resume behavior in run_eval
# ------------------------------------------------------------------------------------

def test_existing_cells_no_resume_refuses_and_writes_nothing(kbench_mod, tmp_path, monkeypatch, capsys):
    _write_reference(tmp_path, prefix="test_pref", base_model="test-org/base-A", substrates=["P"])
    existing_cell = tmp_path / "test_pref_P_my_method_forget_seed0.jsonl"
    existing_cell.write_text('{"query_id": "q0"}\n', encoding="utf-8")

    snapshot_before = set(tmp_path.iterdir())
    run_invocations = []
    monkeypatch.setattr(kbench_mod.subprocess, "run", lambda *a, **kw: run_invocations.append((a, kw)))

    args = argparse.Namespace(
        name="my_method",
        substrate="P",
        prefix="test_pref",
        base="test-org/base-A",
        model="test-org/base-A",
        api_model=None,
        method=None,
        n=10,
        resume=False,
    )

    with pytest.raises(SystemExit) as exc_info:
        kbench_mod.run_eval(args)

    assert exc_info.value.code == 2
    assert len(run_invocations) == 0, "subprocess.run must not be called"
    snapshot_after = set(tmp_path.iterdir())
    assert snapshot_after == snapshot_before, "No files must be written"

    captured = capsys.readouterr()
    assert "1" in captured.out
    assert str(existing_cell) in captured.out
    assert "--resume" in captured.out
    assert "--name" in captured.out


def test_existing_cells_lists_at_most_5(kbench_mod, tmp_path, monkeypatch, capsys):
    _write_reference(tmp_path, prefix="test_pref", base_model="test-org/base-A", substrates=["P", "C"])
    # Create 6 cells
    created = []
    for split in ("forget", "retain"):
        for seed in (0, 137, 271):
            f = tmp_path / f"test_pref_P_my_method_{split}_seed{seed}.jsonl"
            f.write_text('{"query_id": "q"}\n', encoding="utf-8")
            created.append(f)

    monkeypatch.setattr(kbench_mod.subprocess, "run", lambda *a, **kw: pytest.fail("Must not run subprocess"))

    args = argparse.Namespace(
        name="my_method",
        substrate="P,C",
        prefix="test_pref",
        base="test-org/base-A",
        model="test-org/base-A",
        api_model=None,
        method=None,
        n=10,
        resume=False,
    )

    with pytest.raises(SystemExit) as exc_info:
        kbench_mod.run_eval(args)

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "6" in captured.out
    # Only first 5 should be printed directly in the list
    listed_count = sum(1 for f in created if str(f) in captured.out)
    assert listed_count == 5
    assert "... and 1 more" in captured.out


def test_existing_cells_with_resume_skips_those_and_proceeds(kbench_mod, tmp_path, monkeypatch, capsys):
    _write_reference(tmp_path, prefix="test_pref", base_model="test-org/base-A", substrates=["P"])
    existing_cell = tmp_path / "test_pref_P_my_method_forget_seed0.jsonl"
    existing_content = '{"query_id": "original_q0"}\n'
    existing_cell.write_text(existing_content, encoding="utf-8")
    existing_cell.with_suffix(".config.json").write_text(
        json.dumps(
            {
                "candidate_name": "my_method",
                "model_fingerprint": "hf:test-org/base-A@" + "0" * 40,
            }
        ),
        encoding="utf-8",
    )

    executed_tags = []

    def mock_subprocess_run(cmd, check=True):
        # Extract out-jsonl from cmd and write dummy file so scoring has transcripts
        out_idx = cmd.index("--out-jsonl") + 1
        out_path = Path(cmd[out_idx])
        out_path.write_text('{"query_id": "mock_output"}\n', encoding="utf-8")
        out_sum_idx = cmd.index("--out-summary") + 1
        Path(cmd[out_sum_idx]).write_text('{}\n', encoding="utf-8")
        # Tag is the stem of out_jsonl
        executed_tags.append(out_path.stem)

    monkeypatch.setattr(kbench_mod.subprocess, "run", mock_subprocess_run)
    monkeypatch.setattr(
        kbench_mod,
        "score_substrate",
        lambda prefix, sub, name: {
            "substrate": sub,
            "status": "ok",
            "k_score": 0.42,
            "or_forget": 0.20,
            "delta_sel": 0.05,
            "degen": 0.0,
            "delta_degen": 0.0,
            "worst_channel": "Z_answer",
            "per_channel": {"Z_answer": 0.20},
            "seeds_complete": True,
            "seeds": {"baseline_forget": [0, 137, 271], "baseline_retain": [0, 137, 271],
                      "candidate_forget": [0, 137, 271], "candidate_retain": [0, 137, 271]},
            "n_forget": 200,
            "n_rawfull_fallback": 0,
        },
    )

    args = argparse.Namespace(
        name="my_method",
        substrate="P",
        prefix="test_pref",
        base="test-org/base-A",
        model="test-org/base-A",
        api_model=None,
        method=None,
        n=10,
        resume=True,
    )

    res = kbench_mod.run_eval(args)
    assert res["method"] == "my_method"
    assert res["k_score_mean"] == 0.42

    # Verify existing cell was NOT overwritten
    assert existing_cell.read_text(encoding="utf-8") == existing_content

    # Total combinations for substrate P = 1 substrate * 2 splits * 3 seeds = 6 cells
    # Exactly 5 should be executed, 1 skipped
    assert len(executed_tags) == 5
    assert "test_pref_P_my_method_forget_seed0" not in executed_tags

    captured = capsys.readouterr()
    assert "skip test_pref_P_my_method_forget_seed0 (exists)" in captured.out


def test_eval_cli_parser_resume_flag(kbench_mod, monkeypatch):
    test_args = [
        "kbench",
        "eval",
        "--model",
        "test-org/base-A",
        "--name",
        "test_cand",
        "--prefix",
        "test_pref",
        "--base",
        "test-org/base-A",
        "--resume",
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    parsed = []
    monkeypatch.setattr(kbench_mod, "run_eval", lambda args: parsed.append(args))
    kbench_mod.main()
    assert len(parsed) == 1
    assert parsed[0].resume is True
