import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

RELEASE_ROOT = Path(__file__).resolve().parents[1]
MONOREPO_ROOT = RELEASE_ROOT.parent
PUBLISHED_RESULTS = MONOREPO_ROOT / "results"
PAPER_SCORER = MONOREPO_ROOT / "paper_preprint_full" / "scripts" / "kscore.py"
RELEASE_SCORER = RELEASE_ROOT / "scripts" / "kscore.py"

# kscore.main's default method roster, evaluated for each published substrate.
PUBLISHED_METHODS = ("none", "noise", "eco", "star", "leace", "cha", "o3")
PUBLISHED_SUBSTRATES = ("P", "C", "R-text", "R-struct")
PUBLISHED_PREFIX = "v77app"


def _load_scorer(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _parse_rows(section: str) -> dict[str, tuple[str, ...]]:
    rows = {}
    for line in section.splitlines():
        fields = line.split()
        if fields and fields[0] in PUBLISHED_METHODS:
            method = fields[0]
            assert method not in rows, f"duplicate printed row for method {method!r}"
            rows[method] = tuple(fields[1:])
    return rows


def _run_scorer(scorer: ModuleType, substrate: str, scorer_version: str | None = None):
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        if scorer_version is None:
            scorer.main(substrate, PUBLISHED_PREFIX)
        else:
            scorer.main(substrate, PUBLISHED_PREFIX, scorer_version=scorer_version)
    output = stdout.getvalue()

    if "skipped" in output:
        status = "skipped"
    elif "SUBSTRATE-BROKEN" in output:
        status = "broken"
    else:
        status = "scored"

    leaderboard, separator, per_channel = output.partition("# per-channel")
    return status, _parse_rows(leaderboard), _parse_rows(per_channel if separator else "")


def _assert_rows_match(
    substrate: str,
    section: str,
    paper_rows: dict[str, tuple[str, ...]],
    release_rows: dict[str, tuple[str, ...]],
) -> None:
    for method in PUBLISHED_METHODS:
        paper_row = paper_rows.get(method)
        release_row = release_rows.get(method)
        assert paper_row == release_row, (
            f"{section} mismatch for substrate={substrate}, method={method}: "
            f"paper={paper_row!r}, release={release_row!r}"
        )


def test_published_k_scores_match_paper_scorer() -> None:
    if not PUBLISHED_RESULTS.is_dir():
        pytest.skip(f"monorepo-only: {PUBLISHED_RESULTS} absent")
    if not PAPER_SCORER.is_file():
        pytest.skip(f"monorepo-only: {PAPER_SCORER} absent")

    paper = _load_scorer("paper_preprint_full_kscore", PAPER_SCORER)
    release = _load_scorer("release_product_kscore", RELEASE_SCORER)
    paper.RES = PUBLISHED_RESULTS
    release.RES = PUBLISHED_RESULTS

    compared_leaderboard = 0
    compared_per_channel = 0
    for substrate in PUBLISHED_SUBSTRATES:
        paper_status, paper_leaderboard, paper_channels = _run_scorer(paper, substrate)
        release_status, release_leaderboard, release_channels = _run_scorer(
            release, substrate, scorer_version="v1"
        )

        assert paper_status == release_status, (
            f"status mismatch for substrate={substrate}: "
            f"paper={paper_status!r}, release={release_status!r}"
        )
        assert paper_status == release_status == "scored", (
            f"substrate={substrate} was not scored by both scorers: "
            f"paper={paper_status!r}, release={release_status!r}"
        )

        _assert_rows_match(substrate, "leaderboard", paper_leaderboard, release_leaderboard)
        _assert_rows_match(substrate, "per-channel", paper_channels, release_channels)
        compared_leaderboard += len(paper_leaderboard)
        compared_per_channel += len(paper_channels)

    assert compared_leaderboard > 0, "published results scan found no leaderboard rows"
    assert compared_per_channel > 0, "published results scan found no per-channel rows"
    print(f"compared leaderboard rows: {compared_leaderboard}")
    print(f"compared per-channel rows: {compared_per_channel}")
    print(f"scored substrates: {len(PUBLISHED_SUBSTRATES)}")

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        release.main("C", PUBLISHED_PREFIX, methods=["none"], scorer_version="v1")
    assert "scorer v1" not in stdout.getvalue()


EXPECTED_BINARY_FLIPS = {
    ("v67_gpt4omini_C_none_forget_seed137.jsonl", "pii-00989::occupation"),
    ("v67_gpt4omini_C_none_forget_seed271.jsonl", "pii-00581::occupation"),
}
EXPECTED_GRADED_CHANGES = EXPECTED_BINARY_FLIPS | {
    ("v67_gpt4omini_C_none_forget_seed137.jsonl", "pii-00877::occupation"),
}


def _v67_published_files() -> list[Path]:
    files = []
    for model in ("gpt4omini", "gemini20flash"):
        for substrate in ("C", "R-text", "R-struct"):
            for seed in (0, 137, 271):
                path = PUBLISHED_RESULTS / f"v67_{model}_{substrate}_none_forget_seed{seed}.jsonl"
                if path.is_file():
                    files.append(path)
    return files


def test_v2_published_diff_is_exactly_direct_answer_recovery() -> None:
    if not PUBLISHED_RESULTS.is_dir():
        pytest.skip(f"monorepo-only: {PUBLISHED_RESULTS} absent")
    release = _load_scorer("release_product_kscore_v2", RELEASE_SCORER)
    files = _v67_published_files()
    assert len(files) == 18, f"expected 18 published v67 files, found {len(files)}"

    rows = [
        (path.name, json.loads(line))
        for path in files
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 3600, f"expected 3600 published rows, found {len(rows)}"

    binary_changes = set()
    graded_changes = set()
    zanswer_row_changes = set()
    fallback_rows = set()
    for filename, row in rows:
        key = (filename, row["query_id"])
        v1 = release.cell_metrics([row], scorer_version="v1")
        v2 = release.cell_metrics([row], scorer_version="v2")
        assert v1["scorer_version"] == "v1"
        assert v2["scorer_version"] == "v2"
        assert set(v2) - set(v1) == {"n_rawfull_fallback"}
        assert "n_rawfull_fallback" not in v1
        assert v2["n_rawfull_fallback"] in (0, 1)
        if v2["n_rawfull_fallback"]:
            fallback_rows.add(key)
        if v1["or_binary"] != v2["or_binary"]:
            binary_changes.add(key)
            assert (v1["or_binary"], v2["or_binary"]) == (0.0, 1.0)
        if v1["or_graded"] != v2["or_graded"]:
            graded_changes.add(key)
        if v1["n_zanswer_rows"] != v2["n_zanswer_rows"]:
            zanswer_row_changes.add(key)

        for metric in set(v1) - {"scorer_version", "or_binary", "or_graded", "chan_sev", "n_zanswer_rows"}:
            assert v1[metric] == v2[metric], f"unexpected metric change for {key}: {metric}"
        for channel in release.CHANNELS:
            if channel == "Z_answer" and key in EXPECTED_GRADED_CHANGES:
                assert v1["chan_sev"][channel] != v2["chan_sev"][channel]
            else:
                assert v1["chan_sev"][channel] == v2["chan_sev"][channel], (
                    f"unexpected channel change for {key}: {channel}"
                )

    assert len(fallback_rows) == 46
    assert zanswer_row_changes == fallback_rows
    assert binary_changes == EXPECTED_BINARY_FLIPS
    assert graded_changes == EXPECTED_GRADED_CHANGES

    stdout = io.StringIO()
    release.RES = PUBLISHED_RESULTS
    with contextlib.redirect_stdout(stdout):
        release.main("C", PUBLISHED_PREFIX, methods=["none"], scorer_version="v2")
    assert "scorer v2" in stdout.getvalue()
