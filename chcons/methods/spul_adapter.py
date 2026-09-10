"""SPUL adapter — Soft-Prompt Unlearning (Bhaila et al., NAACL 2025).

karuna-bhaila/llm_unlearning. Rail-A inference-time method: a P-tuning soft
prompt (PromptEncoder, num_virtual_tokens=30) is learned offline while the
target weights stay FROZEN. At inference the soft prompt is prepended to every
forward by PEFT, steering the (otherwise untouched) merged target toward
refusing forget-set queries while preserving retain-set answers.

Integration for the K-test:
  - The soft prompt is trained offline (ports/llm_unlearning/spul_qa.py) from
    target_merged, producing a small PEFT P-tuning adapter per seed.
  - setup() wraps agent.model with PeftModel.from_pretrained(<adapter>), so the
    soft prompt is ALWAYS-ON for every model.generate() in the ReAct loop and
    elicit_summary. No per-query hooks: the prefix is independent of the prompt
    (P-tuning injects past_key_values, not text), so lifecycle hooks are no-ops.
  - The adapter path is read from env var SPUL_ADAPTER_PATH so the eval harness
    can select the per-seed adapter without a new CLI flag.

Unlike the activation-edit panel (LEACE/RepE/R-LACE/MLP-probe), SPUL never
touches the residual stream directly — it conditions the model via a learned
prefix. Unlike the weight-based recipes (GA/GD/NPO/...), the target weights are
unchanged; the intervention lives entirely in the prepended virtual tokens.
"""

from __future__ import annotations

import os
from pathlib import Path

from chcons.methods import UnlearnIntervention


class SPULIntervention(UnlearnIntervention):
    """SPUL (Bhaila NAACL'25): always-on learned soft prompt prepended at every forward."""

    @classmethod
    def name(cls) -> str:
        return "spul"

    def __init__(self) -> None:
        self._adapter_path: str | None = None
        self._wrapped = False

    def setup(self, agent, lora_path, forget_ids, facts_path):
        """Wrap agent.model with the trained P-tuning soft prompt (always-on).

        The adapter path comes from env SPUL_ADAPTER_PATH (per-seed selection).
        agent.model is the plain merged target (P substrate + --pii-in-weights
        => lora_path=None, so no LoRA is overlaid). PeftModel.from_pretrained
        attaches the PromptEncoder; the merged target weights stay frozen.
        """
        adapter_path = os.environ.get("SPUL_ADAPTER_PATH")
        if not adapter_path:
            raise SystemExit(
                "[spul] SPUL_ADAPTER_PATH env var is required (path to the trained "
                "P-tuning adapter dir, e.g. ports/llm_unlearning/spul_adapter_seed0)."
            )
        adapter_path = str(Path(adapter_path).expanduser())
        if not Path(adapter_path).is_dir():
            raise SystemExit(f"[spul] adapter dir not found: {adapter_path}")

        from peft import PeftModel

        # Wrap the (merged, frozen) target with the soft prompt. The PromptEncoder
        # prepends num_virtual_tokens learned vectors as past_key_values on every
        # forward — no text injection, so prompt length is unaffected.
        agent.model = PeftModel.from_pretrained(agent.model, adapter_path)
        agent.model.eval()
        self._adapter_path = adapter_path
        self._wrapped = True

        # Report soft-prompt size for the run log / provenance.
        try:
            cfg = agent.model.peft_config[agent.model.active_adapter]
            n_virtual = getattr(cfg, "num_virtual_tokens", "?")
        except Exception:
            n_virtual = "?"
        print(f"[spul] soft prompt loaded from {adapter_path} "
              f"(num_virtual_tokens={n_virtual}); always-on, target weights frozen.")

    # The soft prompt is always-on and prompt-independent — all per-query and
    # per-generation lifecycle hooks are no-ops (no position masks to rebuild).
    def install_per_query(self, agent, query):
        pass

    def before_generation(self, agent, prompt_text):
        pass

    def after_generation(self, agent):
        pass

    def teardown_per_query(self, agent):
        pass

    def teardown(self):
        self._adapter_path = None
        self._wrapped = False

    def summary_dict(self) -> dict:
        return {
            "method": "spul",
            "rail": "A_inference_time",
            "mechanism": "ptuning_soft_prompt_always_on",
            "adapter_path": self._adapter_path,
        }
