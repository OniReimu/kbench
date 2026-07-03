# Adding an unlearning method to K-Bench

K-Bench scores any method that conforms to a single interface. You implement one
class, register it under a CLI name, and the harness evaluates it across every
applicable substrate and channel, producing the OR(all) metric and K-class verdict.

## The contract

Every method subclasses `UnlearnIntervention` (`chcons/methods/__init__.py`). The
lifecycle covers both integration patterns with one ABC:

- **Inference-time** methods (e.g. ECO input corruption, O3 oracle gating) install
  and remove hooks around each `model.generate()` call.
- **Pre-evaluation** methods (e.g. gradient-based LoRA unlearning) edit the adapter
  or weights once in `setup()` and leave the per-query hooks as no-ops.

```python
class UnlearnIntervention(ABC):
    @classmethod
    @abstractmethod
    def name(cls) -> str:
        "CLI value for --unlearn <name>; unique per method."

    @abstractmethod
    def setup(self, agent, lora_path, forget_ids, facts_path) -> None:
        "Called once. Pre-eval methods edit weights here; inference-time methods cache state."

    def install_per_query(self, agent, query) -> None: ...     # per-query state (no model hooks)
    def before_generation(self, agent, prompt_text) -> None: ... # install prompt-specific hooks
    def after_generation(self, agent) -> None: ...             # remove those hooks
    def teardown_per_query(self, agent) -> None: ...
    def teardown(self) -> None: ...                            # restore weights / free resources
    def summary_dict(self) -> dict: return {}                  # method provenance in the result JSON
```

Only `name()` and `setup()` are mandatory.

## Minimal example

```python
# chcons/methods/mymethod_adapter.py
from chcons.methods import UnlearnIntervention

class MyMethodIntervention(UnlearnIntervention):
    @classmethod
    def name(cls): return "mymethod"

    def setup(self, agent, lora_path, forget_ids, facts_path):
        # e.g. load your edited adapter onto agent.model, or cache forget encodings
        self.forget_ids = forget_ids

    def before_generation(self, agent, prompt_text):
        # optional: install a logits processor / activation hook for this prompt
        ...

    def after_generation(self, agent):
        ...  # remove what before_generation installed
```

No repo edit is needed: point `--method` at your file (import-by-path plugin). If the
file defines exactly one `UnlearnIntervention` subclass it is auto-detected; otherwise
disambiguate with `<file>.py::ClassName`. Start from
[`chcons/methods/TEMPLATE_adapter.py`](chcons/methods/TEMPLATE_adapter.py). (Methods we
ship in-tree also keep a short name in `get_intervention()`, but external methods do
not need one.)

## Run the evaluation

One command runs every applicable substrate and writes the leaderboard row:

```bash
kbench eval --model meta-llama/Llama-3.1-8B-Instruct \
    --method /path/to/mymethod_adapter.py --name MyMethod
#  -> results/MyMethod.kbench.json  (K-Score, OR_forget, Δsel, degen, per-channel severity)
```

Under the hood this runs `scripts/02_baseline_leakage.py` per (substrate, split, seed)
and scores with `scripts/kscore.py` against the shipped `none` baseline. To drive a
single cell manually, `02_baseline_leakage.py --substrate C --unlearn
/path/to/mymethod_adapter.py --seed 0` also accepts the plugin path.

## Eligibility

Declare which substrates your method supports. Inference-time and activation-edit
methods are typically **portable** (P/C/R-text/R-struct). Methods needing weight or
LoRA-gradient access are **parametric-only** (P). The harness skips inadmissible
cells rather than scoring them as failures (see the eligibility table in the paper,
§4).

## Reporting requirements (so results are comparable)

A submission must report, per (substrate, method) cell, pooled over **three seeds**
`{0, 137, 271}` at `n=200` queries each:

1. per-channel CER for all six channels,
2. **OR(all)** with its std across seeds,
3. the **trajectory degeneration rate** (`parse_error`/`max_iters`) — this
   distinguishes genuine forgetting from agent collapse; an OR(all) of 0 at a 100%
   degeneration rate is collapse, not unlearning,
4. the retain-set OR(all) and its shift `Delta_sel` vs. the no-intervention baseline,
5. the BH-corrected paired-McNemar `p_adj` vs. that baseline.

See `docs/LEADERBOARD.md` for the submission format.
