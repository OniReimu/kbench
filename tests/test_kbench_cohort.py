import importlib.util
import sys
import json
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
    load_module("kscore", KSCORE_PATH)   # bind kbench\'s `import kscore` to ours
    mod = load_module("kbench", KBENCH_PATH)
    monkeypatch.setattr(mod.kscore, "RES", tmp_path)
    return mod


def _metric(n=200):
    return {
        "n": n,
        "or_graded": 0.20,
        "or_binary": 0.50,
        "degen": 0.0,
        "chan_sev": {"Z_answer": 0.50, "Z_tool": 0.10},
        "n_rawfull_fallback": 0,
    }


def test_cohort_signature_contract(kscore_mod):
    # Valid rows
    rows_by_seed = {0: [{"query_id": f"q{i}"} for i in range(10)]}
    sig = kscore_mod.cohort_signature(rows_by_seed)
    assert 0 in sig
    assert sig[0]["n"] == 10
    assert sig[0]["unique"] == 10
    assert sig[0]["missing_ids"] == 0
    assert sig[0]["ids"] == frozenset(f"q{i}" for i in range(10))

    # Duplicate rows within a seed
    dup_by_seed = {137: [{"query_id": "q0"}, {"query_id": "q0"}]}
    sig_dup = kscore_mod.cohort_signature(dup_by_seed)
    assert sig_dup[137]["n"] == 2
    assert sig_dup[137]["unique"] == 1
    assert sig_dup[137]["missing_ids"] == 0

    # Missing query_id
    missing_by_seed = {271: [{"query_id": "q0"}, {"other": "val"}]}
    sig_missing = kscore_mod.cohort_signature(missing_by_seed)
    assert sig_missing[271]["n"] == 2
    assert sig_missing[271]["unique"] == 2
    assert sig_missing[271]["missing_ids"] == 1


def test_matching_cohorts_pass(kbench_mod, monkeypatch):
    base_by_seed = {
        s: [{"query_id": f"q{s}_{i}"} for i in range(200)] for s in [0, 137, 271]
    }
    cand_by_seed = {
        s: [{"query_id": f"q{s}_{i}"} for i in range(200)] for s in [0, 137, 271]
    }

    def mock_cell(prefix, sub_file, method, split):
        by_seed = cand_by_seed if method == "my_method" else base_by_seed
        all_rows = [r for rows in by_seed.values() for r in rows]
        return _metric(len(all_rows)), list(by_seed.keys()), by_seed

    monkeypatch.setattr(kbench_mod, "_cell", mock_cell)

    res = kbench_mod.score_substrate("v77app", "P", "my_method")
    assert res["status"] == "ok"
    assert res["substrate"] == "P"
    assert "k_score" in res
    assert res["seeds_complete"] is True


def test_cross_seed_id_overlap_is_not_a_duplicate(kbench_mod, monkeypatch):
    """Real situation: three seeds, 200 rows each, ids unique within each seed
    but overlapping across seeds so the pooled unique count is less than the pooled n,
    candidate and baseline identical per seed -> MUST pass (no cohort_mismatch)."""
    seeds = [0, 137, 271]
    id_ranges = {0: (0, 200), 137: (100, 300), 271: (150, 350)}
    shared_by_seed = {
        s: [{"query_id": f"q{i}"} for i in range(*id_ranges[s])]
        for s in seeds
    }

    # Verify our test setup matches the real phenomenon:
    all_ids = [r["query_id"] for rows in shared_by_seed.values() for r in rows]
    assert len(all_ids) == 600
    assert len(set(all_ids)) == 350  # 350 < 600 (cross-seed overlap)
    for s in seeds:
        assert len(shared_by_seed[s]) == 200
        assert len(set(r["query_id"] for r in shared_by_seed[s])) == 200

    def mock_cell(prefix, sub_file, method, split):
        all_rows = [r for rows in shared_by_seed.values() for r in rows]
        return _metric(len(all_rows)), seeds, shared_by_seed

    monkeypatch.setattr(kbench_mod, "_cell", mock_cell)

    res = kbench_mod.score_substrate("v77app", "P", "my_method")
    assert res["status"] == "ok"
    assert res["substrate"] == "P"
    assert "k_score" in res
    assert res["seeds_complete"] is True


def test_differing_n_fails(kbench_mod, monkeypatch):
    base_by_seed = {0: [{"query_id": f"q{i}"} for i in range(200)]}
    cand_by_seed = {0: [{"query_id": f"q{i}"} for i in range(150)]}

    def mock_cell(prefix, sub_file, method, split):
        by_seed = cand_by_seed if method == "my_method" else base_by_seed
        all_rows = [r for rows in by_seed.values() for r in rows]
        return _metric(len(all_rows)), list(by_seed.keys()), by_seed

    monkeypatch.setattr(kbench_mod, "_cell", mock_cell)

    res = kbench_mod.score_substrate("v77app", "P", "my_method")
    assert res["status"] == "cohort_mismatch"
    assert res["substrate"] == "P"
    detail = res["detail"]
    assert detail["split"] == "forget"
    assert detail["seed"] == 0
    assert detail["candidate_n"] == 150
    assert detail["baseline_n"] == 200
    assert detail["candidate_unique"] == 150
    assert detail["baseline_unique"] == 200
    assert detail["symmetric_difference_size"] == 50
    # Assert id sets themselves are NOT in detail
    assert not any(isinstance(v, (set, frozenset, list)) for v in detail.values())


def test_same_n_different_ids_fails(kbench_mod, monkeypatch):
    base_by_seed = {0: [{"query_id": f"base_{i}"} for i in range(200)]}
    cand_by_seed = {0: [{"query_id": f"cand_{i}"} for i in range(200)]}

    def mock_cell(prefix, sub_file, method, split):
        by_seed = cand_by_seed if method == "my_method" else base_by_seed
        all_rows = [r for rows in by_seed.values() for r in rows]
        return _metric(len(all_rows)), list(by_seed.keys()), by_seed

    monkeypatch.setattr(kbench_mod, "_cell", mock_cell)

    res = kbench_mod.score_substrate("v77app", "P", "my_method")
    assert res["status"] == "cohort_mismatch"
    detail = res["detail"]
    assert detail["split"] == "forget"
    assert detail["seed"] == 0
    assert detail["candidate_n"] == 200
    assert detail["baseline_n"] == 200
    assert detail["candidate_unique"] == 200
    assert detail["baseline_unique"] == 200
    assert detail["symmetric_difference_size"] == 400
    assert not any(isinstance(v, (set, frozenset, list)) for v in detail.values())


def test_duplicate_ids_fail(kbench_mod, monkeypatch):
    base_by_seed = {0: [{"query_id": f"q{i}"} for i in range(200)]}
    cand_by_seed = {0: [{"query_id": f"q{i % 50}"} for i in range(200)]}

    def mock_cell(prefix, sub_file, method, split):
        by_seed = cand_by_seed if method == "my_method" else base_by_seed
        all_rows = [r for rows in by_seed.values() for r in rows]
        return _metric(len(all_rows)), list(by_seed.keys()), by_seed

    monkeypatch.setattr(kbench_mod, "_cell", mock_cell)

    res = kbench_mod.score_substrate("v77app", "P", "my_method")
    assert res["status"] == "cohort_mismatch"
    detail = res["detail"]
    assert detail["split"] == "forget"
    assert detail["seed"] == 0
    assert detail["candidate_n"] == 200
    assert detail["candidate_unique"] == 50
    assert detail["baseline_unique"] == 200
    assert not any(isinstance(v, (set, frozenset, list)) for v in detail.values())


def test_missing_query_id_fails_closed(kbench_mod, monkeypatch):
    base_by_seed = {0: [{"query_id": f"q{i}"} for i in range(200)]}
    cand_by_seed = {
        0: [{"query_id": f"q{i}"} for i in range(199)] + [{"missing_id_field": "val"}]
    }

    def mock_cell(prefix, sub_file, method, split):
        by_seed = cand_by_seed if method == "my_method" else base_by_seed
        all_rows = [r for rows in by_seed.values() for r in rows]
        return _metric(len(all_rows)), list(by_seed.keys()), by_seed

    monkeypatch.setattr(kbench_mod, "_cell", mock_cell)

    res = kbench_mod.score_substrate("v77app", "P", "my_method")
    assert res["status"] == "cohort_mismatch"
    detail = res["detail"]
    assert detail["seed"] == 0
    assert detail["candidate_missing_ids"] == 1
    assert not any(isinstance(v, (set, frozenset, list)) for v in detail.values())


def test_baseline_extra_seeds_are_restricted_to_candidate_pool(kbench_mod, monkeypatch):
    base_by_seed = {
        s: [{"query_id": f"q{s}_{i}"} for i in range(200)] for s in [0, 137, 271]
    }
    cand_by_seed = {
        s: [{"query_id": f"q{s}_{i}"} for i in range(200)] for s in [0, 137]
    }

    def mock_cell(prefix, sub_file, method, split):
        by_seed = cand_by_seed if method == "my_method" else base_by_seed
        all_rows = [r for rows in by_seed.values() for r in rows]
        return _metric(len(all_rows)), list(by_seed.keys()), by_seed

    monkeypatch.setattr(kbench_mod, "_cell", mock_cell)
    monkeypatch.setattr(
        kbench_mod.kscore,
        "cell_metrics",
        lambda rows: _metric(len(rows)) if rows else None,
    )

    res = kbench_mod.score_substrate("v77app", "P", "my_method")
    assert res["status"] == "ok"
    assert res["seed_cohort"] == [0, 137]
    assert res["seeds"]["baseline_forget"] == [0, 137]
    assert res["seeds"]["baseline_retain"] == [0, 137]
    assert "baseline restricted to [0, 137]" in res["warning"]


def test_retain_split_mismatch(kbench_mod, monkeypatch):
    base_by_seed = {0: [{"query_id": f"q{i}"} for i in range(200)]}
    cand_f_by_seed = {0: [{"query_id": f"q{i}"} for i in range(200)]}
    cand_r_by_seed = {0: [{"query_id": f"q{i}"} for i in range(100)]}

    def mock_cell(prefix, sub_file, method, split):
        if method == "my_method":
            by_seed = cand_f_by_seed if split == "forget" else cand_r_by_seed
        else:
            by_seed = base_by_seed
        all_rows = [r for rows in by_seed.values() for r in rows]
        return _metric(len(all_rows)), list(by_seed.keys()), by_seed

    monkeypatch.setattr(kbench_mod, "_cell", mock_cell)

    res = kbench_mod.score_substrate("v77app", "P", "my_method")
    assert res["status"] == "cohort_mismatch"
    assert res["detail"]["split"] == "retain"
    assert res["detail"]["seed"] == 0


def test_cohort_mismatch_emit_lands_in_invalid_or_incomplete(kbench_mod, tmp_path):
    # Verify that cohort_mismatch status lands in invalid_or_incomplete and suppresses mean
    row = {
        "substrate": "P",
        "status": "cohort_mismatch",
        "detail": {
            "split": "forget",
            "seed": 0,
            "candidate_n": 100,
            "baseline_n": 200,
            "candidate_unique": 100,
            "baseline_unique": 200,
            "candidate_missing_ids": 0,
            "baseline_missing_ids": 0,
            "symmetric_difference_size": 100,
        },
    }

    out = kbench_mod.emit("method_mismatch", [row], ["P"], prefix="v77app")
    saved = json.loads((tmp_path / "method_mismatch.kbench.json").read_text())

    assert out["coverage"]["invalid_or_incomplete"] == ["P"]
    assert out["coverage"]["scored"] == []
    assert out["coverage"]["excluded_by_baseline_gate"] == []
    assert out["k_score_mean"] is None
    assert out["k_score_mean_suppressed"] == "some requested substrate produced no scorable cell"
    assert out == saved


def test_cell_returns_raw_rows(kbench_mod, tmp_path):
    # Verify _cell loads files and returns (metrics, seeds, by_seed)
    row = {
        "query_id": "q0",
        "raw_full": "Final Answer: apple",
        "answer": "apple",
        "ground_truth": "apple",
    }
    f = tmp_path / "testprefix_P_none_forget_seed0.jsonl"
    f.write_text(json.dumps(row) + "\n")

    metrics, seeds, by_seed = kbench_mod._cell("testprefix", "P", "none", "forget")
    assert seeds == [0]
    assert len(by_seed[0]) == 1
    assert by_seed[0][0]["query_id"] == "q0"
    assert metrics is not None
    assert metrics["n"] == 1
