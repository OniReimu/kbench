"""Adapter TEMPLATE — plug your own unlearning method into K-Bench.

Copy this file, fill in the two required methods (`name`, `setup`) plus whichever
optional hooks your method needs, and run WITHOUT editing the K-Bench repo:

    kbench eval --model <base> --method /path/to/my_adapter.py --name MyMethod

If your file defines more than one UnlearnIntervention subclass, disambiguate:

    --method /path/to/my_adapter.py::MyUnlearn

Two integration patterns fit the same lifecycle:
  * Pre-eval        — edit the LoRA adapter / weights ONCE in setup(); leave the
                      per-query / per-generation hooks as no-ops. (GA, NPO, RMU, ...)
  * Inference-time  — install a hook around each generation. (input corruption,
                      activation edits, prompt filters, ...)

Lifecycle order per evaluation cell:
    setup()                      # once, before any query
    for each query:
        install_per_query()      # cache per-query state (e.g. the entity name)
        for each generation:
            before_generation()  # install prompt-specific hooks
            <model.generate()>
            after_generation()   # remove those hooks
        teardown_per_query()
    teardown()                   # once, after all queries
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from chcons.methods import UnlearnIntervention

if TYPE_CHECKING:
    from chcons.agent import ReActAgent


class MyUnlearn(UnlearnIntervention):
    """Your method. Rename the class; keep it an UnlearnIntervention subclass."""

    @classmethod
    def name(cls) -> str:
        """Short identifier for logs / provenance. Does not need to match --name."""
        return "my_unlearn"

    def setup(
        self,
        agent: "ReActAgent",
        lora_path: Path | None,
        forget_ids: set[str],
        facts_path: Path,
    ) -> None:
        """Called ONCE before evaluation.
        - Pre-eval methods: edit the LoRA adapter at `lora_path` (or the merged
          weights in `agent`) here to remove `forget_ids`. `facts_path` holds the
          ground-truth records if you need them.
        - Inference-time methods: cache what you need (e.g. forget-set encodings).
        Store handles you will need later on `self`.
        """
        self._agent = agent
        # TODO: your setup. Example (inference-time): precompute forget-entity encodings.

    # --- optional per-query / per-generation hooks (delete if unused) ---

    def install_per_query(self, agent: "ReActAgent", query: dict) -> None:
        """Once before each query. Cache per-query state only (e.g. the target
        entity name from `query`). Do NOT install model hooks here — the prompt
        length is not fixed yet; use before_generation for that."""

    def before_generation(self, agent: "ReActAgent", prompt_text: str) -> None:
        """Immediately before each model.generate(). Install prompt-specific hooks
        using `prompt_text` (fires for both the ReAct loop and the summary channel).
        Hooks that index token positions MUST be sized to THIS prompt."""

    def after_generation(self, agent: "ReActAgent") -> None:
        """Immediately after each model.generate(). Remove the hooks you installed
        in before_generation (otherwise the next, differently-sized prompt crashes)."""

    def teardown_per_query(self, agent: "ReActAgent") -> None:
        """After each query's block. Release per-query state."""

    def teardown(self) -> None:
        """Once after all queries. Restore weights, free resources."""

    def summary_dict(self) -> dict:
        """Optional provenance recorded in the run summary JSON (hyperparameters, etc.)."""
        return {"method": self.name()}
