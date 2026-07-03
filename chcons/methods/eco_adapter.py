"""ECO Prompts adapter — input-side intervention via embedding corruption.

Per Liu et al. (NeurIPS 2024): chrisliu298/llm-unlearn-eco.
Mechanism: install a forward hook on the embedding layer that adds random noise
to the token positions corresponding to forget-set entity names. The corrupted
embedding propagates through the transformer, yielding refusal-like output.

For our K-test:
  - Skip ECO's prompt-classifier step (we have ground-truth forget-set membership)
  - Per query: locate entity-name token positions, install corruption hook,
    run agent, remove hook (handled in 02_baseline_leakage.py via the
    UnlearnIntervention lifecycle).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from chcons.methods import UnlearnIntervention, require_external

# Make external ECO library importable
_ECO_ROOT = Path(__file__).resolve().parents[3] / "external" / "eco-prompts"
if str(_ECO_ROOT) not in sys.path:
    sys.path.insert(0, str(_ECO_ROOT))

_QUERY_NAME_RE = re.compile(r"^(?:What is|Who is|Where does)\s+(.+?)'s\s")


class ECOPromptsIntervention(UnlearnIntervention):
    """ECO Prompts (Liu NeurIPS'24): embedding-corruption hook on entity tokens."""

    @classmethod
    def name(cls) -> str:
        return "eco"

    def __init__(
        self,
        strength: float = 100.0,
        dims: int = 1,
        corrupt_method: str = "rand_noise_first_n",
    ):
        self.strength = strength
        self.dims = dims
        self.corrupt_method = corrupt_method
        self._attack_module = None
        self._forget_names: set[str] = set()

    def setup(self, agent, lora_path, forget_ids, facts_path):
        require_external("eco", _ECO_ROOT)
        # Use the canonical PyTorch API — works for both PeftModel-wrapped and
        # unwrapped HF models. Hardcoded paths like "model.embed_tokens" break
        # when LoRA wraps the base model (PeftModel→LoraModel→Llama→model→embed_tokens).
        self._attack_module = agent.model.get_input_embeddings()
        # Cache forget-set names for cheap membership checks
        from chcons.pii import read_jsonl
        self._forget_names = {
            r.name for r in read_jsonl(facts_path) if r.id in forget_ids
        }
        print(f"[eco] setup: cached {len(self._forget_names)} forget-set names; "
              f"hook target={type(self._attack_module).__name__}; strength={self.strength}")

    def install_per_query(self, agent, query):
        """Cache the entity name extracted from this query — but ONLY if it's
        in the forget set. ECO published method uses a learned prompt classifier
        to decide which queries to corrupt; we replace that with oracle ground-truth
        membership (skip classifier, assume perfect labeling). Without this filter,
        ECO would corrupt every query → destroys retain-set utility too →
        no selectivity measurable. The actual hook is installed per-generation
        (in before_generation) because each call has a different prompt length."""
        m = _QUERY_NAME_RE.match(query["query"])
        name = m.group(1) if m else None
        # Selectivity: only corrupt forget-set queries
        if name and name in self._forget_names:
            self._current_entity_name = name
        else:
            self._current_entity_name = None

    def before_generation(self, agent, prompt_text: str) -> None:
        """Install corruption hook with positions located in THIS prompt's tokens."""
        from eco.attack.utils import apply_corruption_hook
        if not getattr(self, "_current_entity_name", None):
            return
        prompt_ids = agent.tokenizer.encode(prompt_text, add_special_tokens=False)
        positions = self._find_positions_in_token_ids(
            agent.tokenizer, prompt_ids, self._current_entity_name
        )
        if not positions:
            return
        n_tokens = len(prompt_ids)
        pos_mask = [0] * n_tokens
        for p in positions:
            if 0 <= p < n_tokens:
                pos_mask[p] = 1
        apply_corruption_hook(
            self._attack_module,
            corrupt_method=self.corrupt_method,
            corrupt_args={"pos": [pos_mask], "dims": self.dims, "strength": self.strength},
        )

    def after_generation(self, agent) -> None:
        from eco.attack.utils import remove_hooks
        remove_hooks(agent.model)

    def teardown_per_query(self, agent):
        self._current_entity_name = None

    def teardown(self) -> None:
        self._attack_module = None
        self._forget_names.clear()
        self._current_entity_name = None

    def summary_dict(self) -> dict:
        return {
            "method": "eco_prompts",
            "attack_module": "get_input_embeddings()",
            "strength": self.strength,
            "dims": self.dims,
            "corrupt_method": self.corrupt_method,
        }

    # ---- internal helpers ----

    @staticmethod
    def _find_positions_in_token_ids(tokenizer, prompt_ids: list[int], entity_name: str) -> list[int]:
        """Locate entity-name token offsets within an already-tokenized prompt.
        Tries both leading-space and no-leading-space tokenizations because BPE
        produces different token sequences depending on the surrounding context."""
        positions: list[int] = []
        for variant in (" " + entity_name, entity_name):
            entity_ids = tokenizer.encode(variant, add_special_tokens=False)
            if not entity_ids:
                continue
            n = len(entity_ids)
            for i in range(len(prompt_ids) - n + 1):
                if prompt_ids[i : i + n] == entity_ids:
                    positions.extend(range(i, i + n))
            if positions:
                break
        return positions
