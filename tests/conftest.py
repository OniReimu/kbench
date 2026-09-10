"""Make the suite collectable both inside the full repository and in a standalone release.

Nine test modules here exercise code that lives at the project root (audit_checkpoint_effect,
score_checkpoint_method, prepare_tokenizer_compat, validate_tokenizer_vocab,
patch_open_unlearning_optional_deepspeed, and the *_live_contract driver helpers) rather than
under `release/`. Without this file every one of them raised ModuleNotFoundError at COLLECTION
time, which aborts the whole run: `pytest release/tests/` reported one error and zero tests, so
none of the 840 tests that do pass could be seen.

They stay where they are because they resolve fixture paths from their own location; moving them
changes `Path(__file__).parents[1]` and breaks them. Instead the project root is put on sys.path
when it is present, and when it is absent -- a release tarball shipped without it -- those
modules are skipped by name with a reason rather than taking the suite down.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_ROOT_ONLY = {
    "test_audit_checkpoint_effect": ('audit_checkpoint_effect',),
    "test_live_freeze_contract": ('live_freeze_contract',),
    "test_patch_open_unlearning_optional_deepspeed": ('patch_open_unlearning_optional_deepspeed',),
    "test_prepare_tokenizer_compat": ('prepare_tokenizer_compat',),
    "test_run_v88c_door_driver": ('door_live_contract', 'door_smoke_runner', 'flat_live_contract', 'live_freeze_contract'),
    "test_run_v88c_elm_driver": ('elm_live_contract', 'live_freeze_contract'),
    "test_run_v88c_flat_driver": ('flat_live_contract', 'live_freeze_contract'),
    "test_run_v88c_relearn_driver": ('live_freeze_contract',),
    "test_run_v93memflex_driver": ('memflex_live_contract', 'memflex_smoke_live_contract'),
    "test_score_checkpoint_method": ('score_checkpoint_method',),
    "test_v94loku_route": ('loku_kbench_current', 'loku_live_contract'),
    "test_v96lm_route": ('meap_kbench_current',),
    "test_v99rvs_gpu_route": ('revs_kbench_v99rvs',),
    "test_validate_tokenizer_vocab": ('validate_tokenizer_vocab',),
}

def _importable(module: str) -> bool:
    """Actually import it, do not merely locate it.

    `find_spec` answers whether the top module exists on the path. It says nothing about that
    module's OWN imports, and several helpers here import each other: `loku_live_contract` and
    `memflex_live_contract` both import `live_freeze_contract`. A static scan of the test file
    cannot see that, so removing one shared helper still aborted collection for two files whose
    dependency on it is transitive. Importing surfaces the whole chain in one question.
    """
    try:
        importlib.import_module(module)
    except ImportError:
        return False
    return True


collect_ignore = [
    f"{stem}.py"
    for stem, modules in _ROOT_ONLY.items()
    # ANY unimportable module makes the file uncollectable, directly or transitively.
    if not all(_importable(m) for m in modules)
]
