import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

KBENCH_PATH = Path(__file__).resolve().parents[1] / "scripts" / "kbench.py"
KSCORE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "kscore.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def kscore_mod():
    return load_module("kscore", KSCORE_PATH)


@pytest.fixture
def kbench_mod(tmp_path, monkeypatch):
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


def test_load_reference(kbench_mod, tmp_path):
    # Absent returns None
    assert kbench_mod.load_reference("nonexistent_prefix") is None

    # Malformed JSON returns None
    bad = tmp_path / "bad.reference.json"
    bad.write_text("{corrupt json", encoding="utf-8")
    assert kbench_mod.load_reference("bad") is None

    # Valid JSON dict returns parsed dict
    _write_reference(tmp_path, prefix="test_pref", base_model="test-org/base-A")
    ref = kbench_mod.load_reference("test_pref")
    assert isinstance(ref, dict)
    assert ref["prefix"] == "test_pref"
    assert ref["base_model"] == "test-org/base-A"


def test_matching_base_passes(kbench_mod, tmp_path):
    _write_reference(
        tmp_path,
        prefix="test_pref",
        base_model="test-org/base-A",
        seeds=[0, 137, 271],
        substrates=["P", "C"],
    )
    result = kbench_mod.check_reference("test_pref", "test-org/base-A", ["P", "C"])
    assert result is None


def test_omitted_base_is_read_from_reference(kbench_mod, tmp_path):
    _write_reference(tmp_path, prefix="test_pref", base_model="test-org/base-A")
    assert kbench_mod.check_reference("test_pref", None, ["P"]) is None


def test_public_llama_prefix_resolves_internal_reference_alias(kbench_mod, tmp_path):
    _write_reference(tmp_path, prefix="v77app", base_model="test-org/base-A")
    ref = kbench_mod.load_reference("llama")
    assert ref is not None
    assert ref["prefix"] == "v77app"
    assert ref["base_model"] == "test-org/base-A"


def test_bundled_llama_reference_carries_safe_p_evaluation_profile(kbench_mod):
    ref = kbench_mod.load_reference("llama")
    assert ref is not None
    assert ref["evaluation"]["P"] == {
        "pii_in_weights": True,
        "inject_mode": "system_prompt",
        "lora_path": None,
        "allow_direct_answer": True,
        "n_incontext_bios": 50,
    }


def test_mismatched_base_refused_names_both_values(kbench_mod, tmp_path):
    _write_reference(
        tmp_path,
        prefix="test_pref",
        base_model="test-org/base-A",
        seeds=[0, 137, 271],
        substrates=["P", "C"],
    )
    result = kbench_mod.check_reference("test_pref", "test-org/base-B", ["P", "C"])
    assert result is not None
    assert "test-org/base-A" in result
    assert "test-org/base-B" in result


def test_missing_reference_file_refused(kbench_mod, tmp_path):
    result = kbench_mod.check_reference("absent_pref", "test-org/base-A", ["P"])
    assert result == "no reference identity shipped for absent_pref"


def test_unparseable_reference_file_refused(kbench_mod, tmp_path):
    bad = tmp_path / "corrupt.reference.json"
    bad.write_text("not json content", encoding="utf-8")
    result = kbench_mod.check_reference("corrupt", "test-org/base-A", ["P"])
    assert result == "no reference identity shipped for corrupt"

    # Missing required keys
    incomplete = tmp_path / "incomplete.reference.json"
    incomplete.write_text(json.dumps({"prefix": "incomplete"}), encoding="utf-8")
    result_incomplete = kbench_mod.check_reference("incomplete", "test-org/base-A", ["P"])
    assert result_incomplete == "no reference identity shipped for incomplete"


def test_seed_set_difference_refused(kbench_mod, tmp_path):
    _write_reference(
        tmp_path,
        prefix="test_pref",
        base_model="test-org/base-A",
        seeds=[0, 137],  # missing seed 271
        substrates=["P", "C"],
    )
    result = kbench_mod.check_reference("test_pref", "test-org/base-A", ["P"])
    assert result is not None
    assert str([0, 137]) in result
    assert str(sorted(kbench_mod.kscore.SEEDS)) in result


def test_unknown_substrate_refused(kbench_mod, tmp_path):
    _write_reference(
        tmp_path,
        prefix="test_pref",
        base_model="test-org/base-A",
        seeds=[0, 137, 271],
        substrates=["P"],
    )
    result = kbench_mod.check_reference("test_pref", "test-org/base-A", ["P", "R-text"])
    assert result is not None
    assert "R-text" in result


def test_base_mismatch_row_makes_emit_suppress_k_score_mean(kbench_mod, tmp_path):
    rows = [
        {
            "substrate": "P",
            "status": "base_mismatch",
            "detail": {"reason": "base mismatch: test-org/base-B != test-org/base-A"},
        },
        {
            "substrate": "C",
            "status": "base_mismatch",
            "detail": {"reason": "base mismatch: test-org/base-B != test-org/base-A"},
        },
    ]
    out = kbench_mod.emit("test_mismatch_method", rows, ["P", "C"], prefix="test_pref")
    saved = json.loads((tmp_path / "test_mismatch_method.kbench.json").read_text(encoding="utf-8"))

    assert out["k_score_mean"] is None
    assert out["k_score_mean_over"] is None
    assert out["k_score_mean_suppressed"] == "some requested substrate produced no scorable cell"
    assert out["coverage"]["invalid_or_incomplete"] == ["P", "C"]
    assert out["coverage"]["scored"] == []
    assert out["coverage"]["excluded_by_baseline_gate"] == []
    assert out == saved


def test_run_eval_preflight_refusal_exits_without_running_cells(kbench_mod, tmp_path, monkeypatch, capsys):
    _write_reference(
        tmp_path,
        prefix="test_pref",
        base_model="test-org/base-A",
        seeds=[0, 137, 271],
        substrates=["P"],
    )

    def fake_run(*args, **kwargs):
        raise AssertionError("subprocess.run must not be called when preflight fails!")

    monkeypatch.setattr(kbench_mod.subprocess, "run", fake_run)

    args = argparse.Namespace(
        model="test-org/candidate-path",
        api_model=None,
        method=None,
        name="test_candidate",
        substrate="P",
        prefix="test_pref",
        base="test-org/base-B",  # mismatch
        n=10,
    )

    with pytest.raises(SystemExit) as exc_info:
        kbench_mod.run_eval(args)

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "test-org/base-A" in captured.out
    assert "test-org/base-B" in captured.out


def test_run_score_base_mismatch_prints_reason_and_exits_nonzero(
    kbench_mod, tmp_path, capsys
):
    _write_reference(
        tmp_path,
        prefix="test_pref",
        base_model="test-org/base-A",
        seeds=[0, 137, 271],
        substrates=["P", "C"],
    )

    args = argparse.Namespace(
        cells=str(tmp_path),
        name="score_mismatch_candidate",
        substrate="P,C",
        prefix="test_pref",
        base="test-org/base-B",  # mismatch
    )

    with pytest.raises(SystemExit) as exc_info:
        kbench_mod.run_score(args)
    assert exc_info.value.code == 2
    output = capsys.readouterr().out
    assert "base_mismatch" in output
    assert "test-org/base-A" in output
    assert "test-org/base-B" in output

    out = json.loads((tmp_path / "score_mismatch_candidate.kbench.json").read_text())
    assert out["k_score_mean"] is None
    assert out["k_score_mean_suppressed"] == "some requested substrate produced no scorable cell"
    assert len(out["substrates"]) == 2
    for r in out["substrates"]:
        assert r["status"] == "base_mismatch"
        assert "test-org/base-A" in r["detail"]["reason"]
        assert "test-org/base-B" in r["detail"]["reason"]


def test_run_score_refuses_to_substitute_reference_cells_for_missing_candidate(
    kbench_mod, tmp_path, monkeypatch, capsys
):
    _write_reference(tmp_path, prefix="llama", base_model="test-org/base-A")
    (tmp_path / "llama_P_none_forget_seed0.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        kbench_mod,
        "score_substrate",
        lambda *args, **kwargs: pytest.fail("missing candidates must be rejected before scoring"),
    )

    args = argparse.Namespace(
        cells=str(tmp_path),
        name="MergeCheck",
        substrate="P",
        prefix="llama",
        base=None,
    )
    with pytest.raises(SystemExit) as exc_info:
        kbench_mod.run_score(args)

    assert exc_info.value.code == 2
    assert args.base == "test-org/base-A"
    output = capsys.readouterr().out
    assert "missing_candidate_cells" in output
    assert "prefix='llama'" in output
    assert "name='MergeCheck'" in output
    assert "substrates=['P']" in output


def test_p_eval_uses_reference_profile_without_stacking_injection_lora(
    kbench_mod, tmp_path, monkeypatch
):
    ref_path = _write_reference(
        tmp_path,
        prefix="test_pref",
        base_model="test-org/base-A",
        substrates=["P", "C", "R-text"],
    )
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    ref["evaluation"] = {
        "P": {
            "pii_in_weights": True,
            "inject_mode": "system_prompt",
            "lora_path": None,
            "allow_direct_answer": True,
            "n_incontext_bios": 50,
        }
    }
    ref_path.write_text(json.dumps(ref), encoding="utf-8")
    commands = []
    monkeypatch.setattr(kbench_mod.subprocess, "run", lambda cmd, check: commands.append(cmd))
    monkeypatch.setattr(
        kbench_mod,
        "score_substrate",
        lambda prefix, sub, name: {
            "substrate": sub,
            "status": "missing_candidate_cells",
        },
    )
    args = argparse.Namespace(
        model="test-org/unlearned",
        api_model=None,
        method=None,
        name="candidate",
        substrate="P,C,R-text",
        prefix="test_pref",
        base="test-org/base-A",
        n=10,
        resume=False,
    )

    kbench_mod.run_eval(args)

    p_commands = [cmd for cmd in commands if cmd[cmd.index("--substrate") + 1] == "P"]
    other_commands = [cmd for cmd in commands if cmd[cmd.index("--substrate") + 1] != "P"]
    assert len(p_commands) == 6
    assert len(other_commands) == 12
    for cmd in p_commands:
        assert "--pii-in-weights" in cmd
        assert cmd[cmd.index("--inject-mode") + 1] == "system_prompt"
        assert "--allow-direct-answer" in cmd
        assert cmd[cmd.index("--n-incontext-bios") + 1] == "50"
        assert "--lora-tplusd-path" not in cmd
        assert "--lora-path" not in cmd
    for cmd in other_commands:
        assert "--pii-in-weights" not in cmd
        assert "--inject-mode" not in cmd
        assert "--allow-direct-answer" not in cmd
        assert "--n-incontext-bios" not in cmd


def test_api_eval_without_matching_reference_is_forget_only_and_has_no_k_score(
    kbench_mod, tmp_path, monkeypatch
):
    _write_reference(
        tmp_path,
        prefix="llama",
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        substrates=["C"],
    )
    commands = []
    monkeypatch.setattr(kbench_mod.subprocess, "run", lambda cmd, check: commands.append(cmd))
    monkeypatch.setattr(
        kbench_mod,
        "score_forget_only",
        lambda prefix, sub, name: {
            "substrate": sub,
            "status": "no_reference_for_api_model",
            "or_forget": 0.25,
            "or_all": 0.25,
            "per_channel": {"Z_answer": 0.25},
            "worst_channel": "Z_answer",
            "seeds": [0, 137, 271],
            "seeds_complete": True,
            "n_forget": 200,
        },
    )
    monkeypatch.setattr(
        kbench_mod,
        "score_substrate",
        lambda *args, **kwargs: pytest.fail("API model must not receive a Llama K-Score"),
    )
    args = argparse.Namespace(
        model=None,
        api_model="x/y",
        method=None,
        name="api_candidate",
        substrate="C",
        prefix="llama",
        base=None,
        n=10,
        resume=False,
    )

    out = kbench_mod.run_eval(args)

    assert len(commands) == 3
    assert all(cmd[cmd.index("--query-subset") + 1] == "forget" for cmd in commands)
    assert not any("retain" in cmd for cmd in commands)
    assert out["substrates"][0]["status"] == "no_reference_for_api_model"
    assert out["k_score_mean"] is None
    assert "k_score" not in out["substrates"][0]


def test_make_reference_writes_expected_json(kbench_mod, tmp_path):
    args = argparse.Namespace(
        prefix="test_new_pref",
        base="test-org/base-A",
        substrates="P, C, R-text",
        cells=None,   # argparse always sets this; a hand-built namespace must too
    )
    kbench_mod.run_make_reference(args)

    ref_file = tmp_path / "test_new_pref.reference.json"
    assert ref_file.is_file()
    data = json.loads(ref_file.read_text(encoding="utf-8"))

    assert data["prefix"] == "test_new_pref"
    assert data["base_model"] == "test-org/base-A"
    assert data["seeds"] == sorted(kbench_mod.kscore.SEEDS)
    assert data["substrates"] == ["P", "C", "R-text"]
    assert data["protocol"] == "1"

    # Preflight check passes against the created reference
    assert kbench_mod.check_reference("test_new_pref", "test-org/base-A", ["P", "C"]) is None


def test_asset_tiers_globs_and_contents(kbench_mod):
    tiers = kbench_mod.ASSET_TIERS

    assert "mini" in tiers
    assert "full" in tiers

    assert tiers["mini"]["globs"] == [
        "llama_P_none_forget_seed*.jsonl",
        "llama_P_none_retain_seed*.jsonl",
    ]
    assert tiers["full"]["globs"] == [
        "llama_*_none_*_seed*.jsonl",
        "qwen_*_none_*_seed*.jsonl",
        "mistral_*_none_*_seed*.jsonl",
    ]


def test_cli_defaults_prefix_and_base(kbench_mod, monkeypatch):
    captured = []
    monkeypatch.setattr(kbench_mod, "run_eval", lambda args: captured.append(args))
    test_args = ["kbench", "eval", "--model", "test-org/candidate-model", "--name", "test"]
    monkeypatch.setattr(sys, "argv", test_args)
    kbench_mod.main()
    assert captured[-1].prefix == "llama"
    assert captured[-1].base is None

    monkeypatch.setattr(kbench_mod, "run_score", lambda args: captured.append(args))
    test_args_score = ["kbench", "score", "--cells", "some/dir", "--name", "test"]
    monkeypatch.setattr(sys, "argv", test_args_score)
    kbench_mod.main()
    assert captured[-1].prefix == "llama"
    assert captured[-1].base is None


def test_make_reference_path_traversal_prefix_refused(kbench_mod, tmp_path, capsys):
    args = argparse.Namespace(
        prefix="../escape",
        base="test-org/base-A",
        substrates="P, C",
        cells=None,
    )
    escape_file = tmp_path.parent / "escape.reference.json"
    assert not escape_file.exists()

    with pytest.raises(SystemExit) as exc_info:
        kbench_mod.run_make_reference(args)

    assert exc_info.value.code == 2
    assert list(tmp_path.rglob("*")) == []
    assert not escape_file.exists()
    captured = capsys.readouterr()
    assert "invalid prefix" in captured.out or "invalid prefix" in captured.err


def test_make_reference_unknown_substrate_refused(kbench_mod, tmp_path, capsys):
    args = argparse.Namespace(
        prefix="test_pref",
        base="test-org/base-A",
        substrates="P,Typo",
        cells=None,
    )
    with pytest.raises(SystemExit) as exc_info:
        kbench_mod.run_make_reference(args)

    assert exc_info.value.code == 2
    assert list(tmp_path.rglob("*")) == []
    captured = capsys.readouterr()
    assert "Typo" in captured.out


def test_make_reference_empty_substrates_refused(kbench_mod, tmp_path):
    args = argparse.Namespace(
        prefix="test_pref",
        base="test-org/base-A",
        substrates="",
        cells=None,
    )
    with pytest.raises(SystemExit) as exc_info:
        kbench_mod.run_make_reference(args)

    assert exc_info.value.code == 2
    assert list(tmp_path.rglob("*")) == []


def test_validate_prefix(kbench_mod):
    assert kbench_mod.validate_prefix("v77app") is None
    assert kbench_mod.validate_prefix("test-pref_1.0") is None
    assert kbench_mod.validate_prefix("none") is None
    assert kbench_mod.validate_prefix("../escape") is not None
    assert kbench_mod.validate_prefix("/absolute") is not None
    assert kbench_mod.validate_prefix("") is not None
    assert kbench_mod.validate_prefix("a" * 65) is not None


def test_run_eval_invalid_prefix_refused(kbench_mod):
    args = argparse.Namespace(
        model="test-org/candidate-path",
        api_model=None,
        method=None,
        name="test_candidate",
        substrate="P",
        prefix="../escape",
        base="test-org/base-A",
        n=10,
    )
    with pytest.raises(SystemExit) as exc_info:
        kbench_mod.run_eval(args)
    assert exc_info.value.code == 2


def test_run_score_invalid_prefix_refused(kbench_mod, tmp_path):
    args = argparse.Namespace(
        cells=str(tmp_path),
        name="score_candidate",
        substrate="P",
        prefix="../escape",
        base="test-org/base-A",
    )
    with pytest.raises(SystemExit) as exc_info:
        kbench_mod.run_score(args)
    assert exc_info.value.code == 2
