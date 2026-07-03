"""Common interface for the 5 channel-targeted unlearning methods we attack
to test the K conjecture.

Two integration patterns:
  - Inference-time:  install/remove a hook around each generation (ECO, O3)
  - Pre-eval:        modify the LoRA adapter once before evaluation (FALCON, Cha, DEPN)

Both fit the same lifecycle (setup once → optional per-query install/teardown →
final teardown), so 02_baseline_leakage.py dispatches uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chcons.agent import ReActAgent


def require_external(method: str, root: Path) -> None:
    """Fail loudly when an external unlearning library is not vendored.

    Several adapters reproduce / wrap third-party code that we do NOT
    redistribute (license + size). Those libraries are expected under
    `external/<x>` (see INSTALL.md). Without this guard the missing path was
    appended to sys.path silently and the eventual `import` failed deep inside
    `setup()` with an opaque ModuleNotFoundError. Call this at the top of an
    external-dependent `setup()` to raise a clear, actionable error instead.
    """
    if not root.exists():
        name = root.name
        raise RuntimeError(
            f"method {method!r} requires external library at '{root}' "
            f"which is absent. Clone it into 'external/{name}' "
            f"(see INSTALL.md for the repo + commit)."
        )


class UnlearnIntervention(ABC):
    """Channel-targeted unlearning method under K-test.

    Subclasses implement at least `setup`. Methods that only modify the LoRA
    once before evaluation (FALCON / Cha / DEPN) leave install_per_query +
    teardown_per_query as no-ops. Methods that hook generation per-query
    (ECO / O3) override them.
    """

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """CLI flag value for `--unlearn <name>`. Must be unique per method."""

    @abstractmethod
    def setup(
        self,
        agent: "ReActAgent",
        lora_path: Path | None,
        forget_ids: set[str],
        facts_path: Path,
    ) -> None:
        """Called once before evaluation begins.
        For pre-eval methods: edit the LoRA adapter or model weights here.
        For inference-time methods: cache forget-set encodings / hooks here.
        """

    def install_per_query(self, agent: "ReActAgent", query: dict) -> None:
        """Called once before each query's evaluation begins. Use this to cache
        per-query state (e.g. extract entity name from query). Do NOT install
        actual model hooks here — use before_generation instead, since hooks
        depending on tokenization of a specific prompt MUST match that prompt's
        length (otherwise CUDA index-out-of-bounds on the next generation with
        a different-length prompt)."""

    def before_generation(self, agent: "ReActAgent", prompt_text: str) -> None:
        """Called immediately before each model.generate() call inside agent
        (both _generate_block in the ReAct loop and elicit_summary). Install
        prompt-specific model hooks here using prompt_text as ground truth."""

    def after_generation(self, agent: "ReActAgent") -> None:
        """Called immediately after each model.generate() call. Remove the
        hooks installed by before_generation."""

    def teardown_per_query(self, agent: "ReActAgent") -> None:
        """Called immediately after each query's evaluation block."""

    def teardown(self) -> None:
        """Called once after all queries — release resources, restore weights."""

    def summary_dict(self) -> dict:
        """Optional method-specific provenance to record in the summary JSON."""
        return {}


def is_plugin_spec(name: str) -> bool:
    """True if `name` is an import-by-path plugin spec rather than a registered short name."""
    return ".py" in name


_PLUGIN_CACHE: dict = {}


def _resolve_plugin_class(spec: str) -> type:
    """Import a user adapter file and return its UnlearnIntervention subclass (WITHOUT
    instantiating it). `spec` is '<path>/adapter.py' (the file must define exactly one
    UnlearnIntervention subclass) or '<path>/adapter.py::ClassName'. Cached per spec so
    fail-fast validation and later dispatch import the file only once. Raises
    ValueError/TypeError on bad path / missing class / ambiguous class."""
    if spec in _PLUGIN_CACHE:
        return _PLUGIN_CACHE[spec]
    import importlib.util
    import inspect
    import sys

    path_str, _, clsname = spec.partition("::")
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"adapter file not found: {path}")
    mod_name = f"kbench_plugin_{path.stem}"
    mspec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(mspec)
    sys.modules[mod_name] = module  # register so dataclasses / relative refs resolve
    mspec.loader.exec_module(module)
    if clsname:
        cls = getattr(module, clsname, None)
        if cls is None:
            raise ValueError(f"class {clsname!r} not found in {path}")
    else:
        # De-dup by object identity: inspect.getmembers yields one entry per NAME, so
        # an alias (`Alias = MyUnlearn`) would otherwise look like two subclasses.
        by_id: dict = {}
        for _, c in inspect.getmembers(module, inspect.isclass):
            if (issubclass(c, UnlearnIntervention) and c is not UnlearnIntervention
                    and c.__module__ == mod_name):
                by_id[id(c)] = c
        cands = list(by_id.values())
        if len(cands) != 1:
            raise ValueError(
                f"expected exactly one UnlearnIntervention subclass in {path}, found "
                f"{[c.__name__ for c in cands]}; disambiguate with '{path}::ClassName'."
            )
        cls = cands[0]
    if not (isinstance(cls, type) and issubclass(cls, UnlearnIntervention)):
        raise TypeError(f"{cls!r} is not an UnlearnIntervention subclass")
    _PLUGIN_CACHE[spec] = cls
    return cls


def validate_plugin_spec(spec: str) -> None:
    """Resolve a plugin spec now so a bad path/class fails BEFORE the expensive model
    load (the result is cached, so the later dispatch reuses it)."""
    _resolve_plugin_class(spec)


def _load_plugin(spec: str) -> "UnlearnIntervention":
    """Instantiate the adapter class for a plugin spec -- drop in your own method with
    no repo edit (see `_resolve_plugin_class` for the spec format)."""
    return _resolve_plugin_class(spec)()


def get_intervention(name: str) -> UnlearnIntervention:
    """Factory: map `--unlearn <name>` to an adapter. `name` is either a registered
    short name (eco, o3, leace, ...) or an import-by-path plugin spec
    '<path>/adapter.py' / '<path>/adapter.py::ClassName' -- so an external user drops
    in their own method without editing this repo."""
    if is_plugin_spec(name):
        return _load_plugin(name)
    if name == "eco":
        from chcons.methods.eco_adapter import ECOPromptsIntervention
        return ECOPromptsIntervention()
    if name == "falcon":
        from chcons.methods.falcon_adapter import FALCONIntervention
        return FALCONIntervention()
    if name == "cha":
        from chcons.methods.cha_adapter import ChaIntervention
        return ChaIntervention()
    if name == "depn":
        from chcons.methods.depn_adapter import DEPNIntervention
        return DEPNIntervention()
    if name == "o3":
        from chcons.methods.o3_adapter import O3Intervention
        return O3Intervention()
    if name == "leace":
        from chcons.methods.leace_adapter import LEACEIntervention
        return LEACEIntervention()
    if name == "repe":
        from chcons.methods.repe_adapter import RepEIntervention
        return RepEIntervention()
    if name == "mlp_probe":
        from chcons.methods.mlp_probe_adapter import MLPProbeIntervention
        return MLPProbeIntervention()
    if name == "rlace":
        from chcons.methods.rlace_adapter import RLACEIntervention
        return RLACEIntervention()
    raise ValueError(f"unknown intervention: {name!r}")


REGISTERED = ("eco", "falcon", "cha", "depn", "o3", "leace", "repe", "mlp_probe", "rlace")
