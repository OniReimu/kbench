"""FALCON adapter — activation-level intervention via contrastive orthogonal unalignment.

Per Hu et al. (NeurIPS 2025): CharlesJW222/FALCON.
Mechanism: trains a small representation-space orthogonal projection that
removes forget-set activations while preserving retain. Pre-eval pattern:
fine-tune the LoRA adapter (or a thin head on top of frozen base) using
FALCON's contrastive objective, then deploy.

Integration plan (pre-eval, no per-query work):
  setup() → run FALCON's training loop on (lora_path, forget_ids, retain_ids)
            using external/falcon/falcon/algorithms.py + unlearning.py;
            replace agent's loaded adapter with the FALCON-trained variant.
  install/teardown_per_query() → no-op.
  teardown() → restore original adapter (optional).
"""

from __future__ import annotations

import sys
from pathlib import Path

from chcons.methods import UnlearnIntervention, require_external

_FALCON_ROOT = Path(__file__).resolve().parents[3] / "external" / "falcon"
if str(_FALCON_ROOT) not in sys.path:
    sys.path.insert(0, str(_FALCON_ROOT))


class FALCONIntervention(UnlearnIntervention):
    """FALCON (Hu NeurIPS'25): contrastive orthogonal activation unalignment.

    NOT YET IMPLEMENTED — adapter scaffold only. Setup will train a FALCON-
    modified LoRA adapter using their contrastive objective on our forget/retain
    split, then swap it in.
    """

    @classmethod
    def name(cls) -> str:
        return "falcon"

    def __init__(self):
        self._original_lora = None

    def setup(self, agent, lora_path, forget_ids, facts_path):
        require_external("falcon", _FALCON_ROOT)
        # Build forget_set + retain_set sample loaders from facts_path, run the
        # FALCON training loop on agent.model + lora_path, then swap the trained
        # adapter in. The training loop itself is not vendored in this release.
        raise NotImplementedError(
            "FALCON adapter is a scaffold only — its training loop is not "
            "implemented/runnable in this release (see CONTRIBUTING.md)."
        )

    def teardown(self) -> None:
        pass

    def summary_dict(self) -> dict:
        return {"method": "falcon", "implementation_status": "scaffold-only"}
