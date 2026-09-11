"""Cross-model reference identities and the external/ checkout location.

Both were found by a first-use GPU trial on the public repo: `kbench eval --prefix qwen`
exited with "no reference identity shipped for qwen", and the ECO adapter looked for
external/ one directory above the repository root.
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

RELEASE = Path(__file__).resolve().parents[1]
if str(RELEASE) not in sys.path:
    sys.path.insert(0, str(RELEASE))

BASES = {
    "qwen": ("Qwen/Qwen3.5-9B", ["P", "C", "R-text", "R-struct"]),
    "mistral": ("mistralai/Mistral-7B-Instruct-v0.3", ["P", "R-text", "R-struct"]),
}


def _load_kbench():
    spec = importlib.util.spec_from_file_location("kbench_xref", RELEASE / "scripts" / "kbench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _shipped_cells(mod) -> dict:
    """The public-name -> (internal name, base model) map for the published baseline cells."""
    for value in vars(mod).values():
        if isinstance(value, dict) and "qwen_C_none_forget_seed0.jsonl" in value:
            return value
    raise AssertionError("published baseline cell map not found in kbench.py")


@pytest.mark.parametrize("prefix", sorted(BASES))
def test_crossmodel_reference_is_shipped_and_accepts_its_base(prefix, monkeypatch):
    mod = _load_kbench()
    monkeypatch.setattr(mod.kscore, "SEEDS", [0])  # seed 0 is covered on every declared substrate
    base, subs = BASES[prefix]
    assert mod.check_reference(prefix, None, subs) is None
    assert mod.check_reference(prefix, base, subs) is None
    assert "base mismatch" in mod.check_reference(prefix, "meta-llama/Llama-3.1-8B-Instruct", subs[:1])


def test_mistral_c_is_refused_before_any_gpu_time():
    """No Mistral C retain baseline is published, so C cannot be scored for Mistral."""
    mod = _load_kbench()
    err = mod.check_reference("mistral", "mistralai/Mistral-7B-Instruct-v0.3", ["C"])
    assert err is not None and "unknown substrate 'C'" in err


@pytest.mark.parametrize("prefix", sorted(BASES))
def test_reference_matches_the_shipped_cells(prefix):
    """Every declared substrate has forget and retain baseline cells for every declared seed
    that the published set carries, and the cells were produced by the declared base."""
    mod = _load_kbench()
    ref = json.loads((RELEASE / "assets" / f"{prefix}.reference.json").read_text())
    cells = _shipped_cells(mod)
    for sub in ref["substrates"]:
        for split in ("forget", "retain"):
            name = f"{prefix}_{sub}_none_{split}_seed0.jsonl"
            assert name in cells, f"{name} is not in the published baseline set"
            assert cells[name][1] == ref["base_model"]
    for name, (_, base) in cells.items():
        if name.startswith(f"{prefix}_"):
            assert base == ref["base_model"]


def test_references_are_packaged_for_wheel_installs():
    text = (RELEASE / "pyproject.toml").read_text()
    manifest = (RELEASE / "PUBLISH_MANIFEST.txt").read_text().splitlines()
    for prefix in BASES:
        assert f'"assets/{prefix}.reference.json"' in text
        assert f"assets/{prefix}.reference.json" in manifest


def test_external_root_is_the_repository_root(monkeypatch):
    monkeypatch.delenv("KBENCH_EXTERNAL_DIR", raising=False)
    from chcons.methods import external_root
    assert external_root("eco-prompts") == RELEASE / "external" / "eco-prompts"


def test_external_root_honours_the_override(monkeypatch, tmp_path):
    monkeypatch.setenv("KBENCH_EXTERNAL_DIR", str(tmp_path))
    from chcons.methods import external_root
    assert external_root("depn") == tmp_path / "depn"


def test_no_adapter_resolves_external_above_the_repo():
    for path in (RELEASE / "chcons" / "methods").glob("*_adapter.py"):
        text = path.read_text()
        assert "parents[3]" not in text, f"{path.name} still resolves external/ above the repo root"


def test_qwen_r_refuses_seeds_its_baseline_lacks(monkeypatch):
    """Qwen R retain seed 271 is not published, so a default three-seed R run is refused
    before evaluation rather than ending in a cohort mismatch after it."""
    mod = _load_kbench()
    err = mod.check_reference("qwen", "Qwen/Qwen3.5-9B", ["R-text"])
    assert err is not None and "seed mismatch on R-text" in err and "KBENCH_SEEDS" in err
    assert mod.check_reference("qwen", "Qwen/Qwen3.5-9B", ["C"]) is None
    monkeypatch.setattr(mod.kscore, "SEEDS", [0])
    assert mod.check_reference("qwen", "Qwen/Qwen3.5-9B", ["R-text", "R-struct"]) is None


@pytest.mark.parametrize("prefix", sorted(BASES))
def test_declared_substrate_seeds_have_published_cells(prefix):
    mod = _load_kbench()
    ref = json.loads((RELEASE / "assets" / f"{prefix}.reference.json").read_text())
    cells = _shipped_cells(mod)
    for sub in ref["substrates"]:
        for seed in ref.get("substrate_seeds", {}).get(sub, ref["seeds"]):
            for split in ("forget", "retain"):
                assert f"{prefix}_{sub}_none_{split}_seed{seed}.jsonl" in cells, (prefix, sub, split, seed)


def _fake_hub(monkeypatch, cache, sha=None):
    hub = types.ModuleType("huggingface_hub")
    consts = types.ModuleType("huggingface_hub.constants")
    consts.HF_HUB_CACHE = str(cache)

    class HfApi:
        def model_info(self, repo, revision=None):
            if sha is None:
                raise OSError("offline")
            return types.SimpleNamespace(sha=sha)

    hub.HfApi = HfApi
    hub.constants = consts
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", consts)


def test_hub_fingerprint_pins_the_commit(monkeypatch, tmp_path):
    mod = _load_kbench()
    _fake_hub(monkeypatch, tmp_path, sha="a" * 40)
    assert mod.model_fingerprint("org/repo") == "hf:org/repo@" + "a" * 40
    _fake_hub(monkeypatch, tmp_path, sha="b" * 40)
    assert mod.model_fingerprint("org/repo") == "hf:org/repo@" + "b" * 40  # moved main -> new identity


def test_hub_fingerprint_offline_uses_the_cached_ref(monkeypatch, tmp_path):
    mod = _load_kbench()
    ref = tmp_path / "models--org--repo" / "refs" / "main"
    ref.parent.mkdir(parents=True)
    ref.write_text("c" * 40 + "\n")
    _fake_hub(monkeypatch, tmp_path, sha=None)
    assert mod.model_fingerprint("org/repo") == "hf:org/repo@" + "c" * 40


def test_hub_fingerprint_refuses_an_unresolvable_ref(monkeypatch, tmp_path):
    mod = _load_kbench()
    _fake_hub(monkeypatch, tmp_path, sha=None)
    with pytest.raises(ValueError, match="cannot resolve org/repo@main"):
        mod.model_fingerprint("org/repo")
