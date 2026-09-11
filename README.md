# K-Bench

K-Bench measures whether information meant to be forgotten can still be recovered
from a deployed tool-using language-model agent. It observes six channels across
parametric (P), context (C), and retrieval (R-text and R-struct) substrates and
reports leakage together with retain-set damage and agent degeneration.

The public workflow is:

```bash
kbench fetch-assets --full --indexes
kbench eval --model <ckpt> --name MyMethod
# Or, for transcripts produced elsewhere:
kbench score --cells <dir> --name MyMethod
# `results` here is your evaluator output directory.
kbench bundle --cells results --out MyMethod.kbench-bundle
kbench report MyMethod.kbench-bundle
```

Submit the validated result bundle using the
[leaderboard protocol](docs/LEADERBOARD.md).

To regenerate the paper tables, use the separate [reproduction guide](reproduce/README.md).

## 1. 30-second smoke

This runs on CPU with no model, credentials, or asset download. From a clone:

```bash
bash reproduce.sh smoke
```

The shipped cells also exercise the documented offline bundle handoff:

```bash
kbench bundle --cells data/smoke/cells --out smoke.kbench-bundle
kbench report smoke.kbench-bundle
```

The Docker equivalent is:

```bash
docker build -f Dockerfile.cpu -t kbench:cpu .
docker run --rm --entrypoint bash kbench:cpu reproduce.sh smoke
```

The current `scripts/smoke_report.py` output is:

```text
K-Bench smoke — scoring path on a bundled fixture (CPU-only, no model, no download)
Fixture: data/smoke/cells (substrate P, 1 seed, 8 queries) — DEMONSTRATION ONLY

Method: demo
  per-channel CER (forget): Z_CoT 0.00  Z_tool 0.00  Z_tool_wide 0.75  Z_RAG 0.00  Z_answer 0.00  Z_summary 1.00
  OR(all) forget: 1.00   (no-intervention baseline: 1.00)
  retain shift Δsel: +0.00     collapse Δdegen: 0.00
  K-Score: 0.00     K-class: measured failure
  >> The answer channel reads FORGOTTEN (Z_answer 0.00), yet OR(all) stays 1.00 because the secret
     still surfaces via Z_summary / Z_tool_wide. A single-answer probe
     would certify this as unlearned; K-Bench does not.

This is the scoring half of the harness. To score a real method:
  kbench eval  --model <ckpt> --name MyMethod   # GPU
  kbench score --cells <dir> --name MyMethod   # offline transcripts
```

## 2. Bring your checkpoint

Install the packaged release with `pip install kbench`, or run the following from a
clone:

```bash
pip install -e .
kbench fetch-assets --full --indexes
kbench eval --model <ckpt> --name MyMethod
```

The final command evaluates the applicable substrates and writes
`results/MyMethod.kbench.json`. The K-Score combines forget-set observer rate,
retain-set selectivity, and degeneration; see [Metrics](docs/METRICS.md).

### Substrate-P contract and target

A weight-based method is comparable on P only when it was unlearned from K-Bench's
injected target. The target is a LoRA adapter merged into
`meta-llama/Llama-3.1-8B-Instruct` at Hugging Face snapshot
`0e9e39f249a16976918f6564b8830bc894c89659`. Re-merging that adapter with the
production procedure reproduced the paper target bit for bit (291 tensors, zero
differing elements). The SHA-256 of `adapter_model.safetensors` is
`185217937208be1398ba575ea9b0d95b44a107483e083f79497306a62ff60417`.

The untreated baseline cells, target adapter, and retrieval indexes are separate
download tiers hosted by the Hugging Face dataset `kbench/kbench-assets`:

```bash
kbench fetch-assets --target   # adapter, about 336 MB -> models/Llama-3.1-8B-kbench-target-adapter/
kbench fetch-assets --indexes  # two indexes, about 8.4 GB each -> data/wiki_index_v21_{target_in,distractor}/
```

The flags are additive: for example, `kbench fetch-assets --full --target --indexes`
also fetches the full untreated baseline. With no tier flag, `fetch-assets` keeps its
existing `--mini` default. In an installed, non-repository invocation, model and index
assets are written under the current directory, which is also where `kbench eval`
resolves the default `data/...` index paths. The adapter is distributed under the
Llama 3.1 Community License; review its downloaded `LICENSE` and `USE_POLICY.md`.

The merge below needs Hugging Face access to the gated
`meta-llama/Llama-3.1-8B-Instruct`, a roughly 16 GB model download, and enough RAM
to hold an fp32 8B model (32 GB or more). Merge the target adapter with the exact
fp32 CPU load, PEFT merge, and bf16 cast used for the production target:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "meta-llama/Llama-3.1-8B-Instruct"
revision = "0e9e39f249a16976918f6564b8830bc894c89659"
adapter_dir = "models/Llama-3.1-8B-kbench-target-adapter"
output_dir = "/path/to/kbench-injected-target"

model = AutoModelForCausalLM.from_pretrained(
    base_id, revision=revision, torch_dtype=torch.float32, device_map="cpu"
)
model = PeftModel.from_pretrained(model, adapter_dir)
model = model.merge_and_unload().to(torch.bfloat16)
model.save_pretrained(output_dir, safe_serialization=True)
AutoTokenizer.from_pretrained(base_id, revision=revision).save_pretrained(output_dir)
```

R-text and R-struct require two retrieval indexes shipped with the release assets:
`data/wiki_index_v21_target_in` and `data/wiki_index_v21_distractor`. Each is about
8.4 GB (approximately 6.1 GB of FAISS index plus 2.3 GB of passages), about 16.8 GB
in total. Fetch both with `kbench fetch-assets --indexes`.

For NPO training from the injected target with OpenUnlearning, follow the
[OpenUnlearning recipe](docs/OPENUNLEARNING.md).

## 3. Bring your adapter

Inference-time methods can be loaded by file without editing this repository. Copy
[`chcons/methods/TEMPLATE_adapter.py`](chcons/methods/TEMPLATE_adapter.py), implement
the `UnlearnIntervention` hooks, and run:

```bash
# all substrates: P needs the merged injected target (see the substrate-P contract above)
kbench eval --model /path/to/kbench-injected-target --method path/to/adapter.py --name MyMethod

# context and retrieval substrates only: the untreated base model is enough
kbench eval --model meta-llama/Llama-3.1-8B-Instruct --substrate C,R-text,R-struct \
  --method path/to/adapter.py --name MyMethod
```

If the file contains more than one intervention subclass, append `::ClassName` to
the method path. The full adapter contract is in [Contributing](CONTRIBUTING.md).

## Included method adapters

K-Bench v1.0 exposes **9 available built-in adapter short names**. Each is accepted
by the evaluator, constructs through `get_intervention`, and has no recorded failing
port-conformance result:

- `eco`
- `cha`
- `depn`
- `o3`
- `leace`
- `repe`
- `mlp_probe`
- `rlace`
- `uld`

Use one as `--method <short-name>`. Some require a trained method artifact or an
external upstream checkout; [Installation](INSTALL.md) records those prerequisites.
The harness controls `none`, `star`, `star_full`, and `noise` are accepted by the
low-level `--unlearn` interface but are not adapter-registry entries.

The following registered ports are experimental and are not counted as available:

- `falcon`: its harness-owned SophiaG optimizer has no upstream reference, so the
  conformance ledger cannot certify the complete runnable port.
- `spul`: the upstream sentiment-data builder cannot consume K-Bench QA data, so the
  substituted dataset builder has no conformance reference.
- `grun`: conformance still requires a separate activation-space harness for the
  gated-ReFT edit.

## Transcript bundles and offline reports

`kbench bundle` creates a directory using the candidate schema
`kbench-transcript-bundle@1-candidate`. Its `bundle.json` records the K-Bench
release and scoring semantics, creation time, per-cell run identity (`model`, `base_model`,
`method`, `substrate`, `split`, `seed`, and `n`), SHA-256, embedded evaluator
sidecar config, and explicit forget/retain/reference pairings. Each JSONL must
have its evaluator `.config.json` sidecar; substrate, split, seed, row count, and
API identity are derived from that sidecar and reconciled with the filename and
manifest. Non-API forget cells must declare a same-model retain cell and the
model/base-matched untreated `none` forget/retain references. Only API cells may
be forget-only. The bundle also contains a copy of every listed JSONL cell.

Every row must include `query_id`, `pii_id`, `field`, `ground_truth`, `raw_full`,
`halted_reason`, and `leakage`. API rows additionally require `api_incidents`,
`api_retries`, and `raw_reasoning`. `kbench report` validates those fields and
hashes before printing the K-Score and per-channel table. Reporting is offline
and deterministic for a fixed bundle.

```bash
# `results` here is your evaluator output directory, containing JSONL cells and sidecars.
kbench bundle --cells results --out MyMethod.kbench-bundle
kbench report MyMethod.kbench-bundle
```

## API models

The API path is C/R-only and requires `OPENROUTER_API_KEY`. A credential-only C
run needs no retrieval-index download, so start with:

```bash
export OPENROUTER_API_KEY=<your-key>
kbench eval --api-model openai/gpt-4o-mini --substrate C --name MyMethod
```

R-text and R-struct additionally need `data/wiki_index_v21_target_in` and
`data/wiki_index_v21_distractor`, about 8.4 GB each. With both indexes present:

```bash
kbench fetch-assets --indexes
kbench eval --api-model openai/gpt-4o-mini --substrate C,R-text,R-struct --name MyMethod
```

It reports leakage but cannot produce a K-Score from the shipped assets because the
API reference cells have no retain split. K-Bench does not substitute another
model's retain baseline.

## Repository layout

```text
chcons/                 agent harness, metrics, and method adapters
data/                   synthetic PII data and the bundled smoke fixture
docs/                   protocol, metrics, compute, and submission documentation
reproduce/              paper-table reproduction guide
openunlearn_configs/    public OpenUnlearning Hydra configuration
scripts/                evaluation, scoring, and reproduction pipeline
tests/                  release contract and regression tests
reproduce.sh            smoke and paper-table reproduction entry point
```

## Citation

```bibtex
@inproceedings{kbench,
  title  = {K-Bench: A Benchmark for LLM Unlearning in Agentic Deployments},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review}
}
```

## License

Code is MIT licensed ([LICENSE](LICENSE)). The Faker-generated synthetic corpus is
CC BY 4.0 licensed ([data/LICENSE](data/LICENSE)) and contains no real personal data;
see the [Datasheet](docs/DATASHEET.md).
