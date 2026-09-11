# Installing K-Bench prerequisites

`reproduce.sh` evaluates unlearning methods on pre-built artifacts. Build these
once, then run `reproduce.sh {topology|interfaces|substrate}`.

## 0. Environment

The Docker image freezes the exact CUDA + pinned-library stack (recommended):
```bash
docker build -t kbench .                       # see Dockerfile
docker run --gpus all -it -v "$PWD/results:/workspace/kbench/results" kbench bash
```
Or the local pinned GPU environment used for the Llama-3.1-8B and Qwen3.5-9B experiments:
```bash
uv venv --python 3.11 && uv pip install -r main_pinned_requirements.txt && uv pip install -e . --no-deps
```
The pinned `torch==2.11.0` wheel from PyPI bundles CUDA 13 runtime libraries, so the host needs an
NVIDIA driver that supports CUDA 13 (this applies to the Docker image as well).
For CPU-only users (macOS or Linux without CUDA; runs `kbench eval --api-model` on C/R substrates and offline `kbench score`):
```bash
uv venv --python 3.11 && uv pip install --torch-backend cpu -r cpu_pinned_requirements.txt && uv pip install -e . --no-deps
```
No lock file ships in the public repository, so `uv sync --extra infer --extra dev`
(inside a CUDA GPU session) resolves the ranges in `pyproject.toml` instead of the
maintainers' exact versions; use a pinned requirements file for reproduction.
For the same reason, the commands below use `uv run --no-sync`: a plain `uv run`
would re-resolve the `pyproject.toml` ranges over the pinned installation.

### Pinned requirement files

- `main_pinned_requirements.txt`: Default GPU stack for Llama-3.1-8B and Qwen3.5-9B, pinning `torch==2.11.0`, `transformers==5.7.0`, `peft==0.19.1`, and `accelerate==1.13.0` (matching `Dockerfile`).
- `cross_model_pinned_requirements.txt`: Separate Mistral-7B-v0.3 GPU stack (`transformers==4.51.3`); create a separate venv for it as described in `docs/COMPUTE.md`.
- `qwen_pinned_requirements.txt`: Legacy Qwen2.5-era compatibility stack (`transformers==4.47.1`, `peft==0.14.0`, `accelerate==1.2.1`).
- `cpu_pinned_requirements.txt`: CPU/API evaluation and offline-scoring stack, including the CPU retriever dependencies and omitting local weight-training dependencies such as `peft`, `accelerate`, and `vllm` (matching `Dockerfile.cpu`).

For the maintainers' record of which environment stack produced which published paper results, see [`docs/COMPUTE.md`](docs/COMPUTE.md#environments).

## 1. Synthetic PII corpus (deterministic, no GPU)
```bash
uv run --no-sync python scripts/01_generate_pii.py --n-facts 5000 --seed 0 --out-dir data/pii_facts --name v1
uv run --no-sync python scripts/00d_build_distractor_pool.py     # v2.1 distractor pool (startup-audit input)
```
Produces `data/pii_facts/` (the synthetic corpus and query set) and the v2.1
distractor pool. The PII is Faker-synthetic — no real persons (see `docs/DATASHEET.md`).
Fully regenerable from the seed. The LoRA "injection" (`scripts/03_inject_pii.py`)
is a GPU step covered in section 3, not here. The distractor pool (`data/v21/bios_distractor.jsonl`) is regenerated
as the retain-adapter split; the original composition was not preserved, so the
P-substrate headline is unaffected (P reads it neither at eval nor at LoRA training)
while C / R-substrate context padding may differ marginally from the published run.

## 2. RAG / Wiki index (GPU, ~hours for the production index)
```bash
uv run --no-sync python scripts/00_build_rag_index.py --config configs/rag_pilot.yaml
```
Produces `data/wiki_index*`. The R-text / R-struct substrates need the full
index; `data/wiki_index_smoke/` ships complete (passages + `index.faiss`) for a
quick CPU smoke run. It is built with `BAAI/bge-small-en-v1.5`, so pass
`--embed-model BAAI/bge-small-en-v1.5` when pointing the agent at it (the agent
default is `bge-base`, whose 768-dim vectors would not match this 384-dim index).
For evaluation with the fixed v2.1 assets, download the two production indexes
(about 8.4 GB each):

```bash
kbench fetch-assets --indexes
```

They are written to `data/wiki_index_v21_target_in/` and
`data/wiki_index_v21_distractor/`, the default paths read by `kbench eval`. You can
also download selectively: `kbench fetch-assets --indexes distractor` for P, C, and
R-struct, or `kbench fetch-assets --indexes target_in` for R-text. In an
installed, non-repository invocation, run the command from the directory where you
will run `kbench eval`. To rebuild the two substrate-isolation indexes instead
(injecting the PII into the wiki corpus, with the target absent from the distractor
index and present in the target-in index):
```bash
uv run --no-sync python scripts/00b_inject_pii_corpus.py \
  --orig-passages data/wiki_index/passages.jsonl --orig-embeddings-dir data/wiki_index/embeddings \
  --facts data/v21/bios_distractor.jsonl --n-forget 5000 --out-dir data/wiki_index_v21_distractor/
uv run --no-sync python scripts/00b_inject_pii_corpus.py \
  --orig-passages data/wiki_index/passages.jsonl --orig-embeddings-dir data/wiki_index/embeddings \
  --facts data/pii_facts/v1_facts.jsonl --n-forget 5000 --out-dir data/wiki_index_v21_target_in/
```

## 3. PII LoRA + merged target (substrate P)

For a paper-comparable substrate-P run, start from the injected target adapter
published as a v1.0 release asset on the Hugging Face dataset
`kbench/kbench-assets`:

```bash
kbench fetch-assets --target
```

The roughly 336 MB adapter is written to
`models/Llama-3.1-8B-kbench-target-adapter/` and is distributed under the Llama
3.1 Community License. It targets
`meta-llama/Llama-3.1-8B-Instruct` snapshot
`0e9e39f249a16976918f6564b8830bc894c89659`; its
`adapter_model.safetensors` SHA-256 is
`185217937208be1398ba575ea9b0d95b44a107483e083f79497306a62ff60417`.
The exact fp32-CPU merge procedure is in the README. The merged target is the
starting checkpoint for weight-based unlearning, including the
[OpenUnlearning NPO recipe](docs/OPENUNLEARNING.md).

Merge the published target adapter with the production procedure:
run this CPU-only command in the main pinned environment from Section 0
(`main_pinned_requirements.txt`), which includes `peft`:

```bash
uv run --no-sync python scripts/20_merge_target.py        # -> models/target_merged
```

To build a new, non-reference target instead, train a LoRA locally and pass its
adapter directory explicitly:

```bash
uv run --no-sync python scripts/03_inject_pii.py --facts data/pii_facts/v1_facts.jsonl \
  --out-dir models/lora_v1 --epochs 5 --r 32 --alpha 64 --seed 0
uv run --no-sync python scripts/20_merge_target.py --adapter models/lora_v1/final_adapter \
  --out models/target_merged
```

That locally trained target is useful for development but is not the fixed
paper target unless its adapter identity matches the published asset.

## 4. Method availability and external libraries

The **9 available adapter short names** are `eco`, `cha`, `depn`, `o3`, `leace`,
`repe`, `mlp_probe`, `rlace`, and `uld`. They are accepted by the evaluator,
construct through `get_intervention`, and have no recorded failing conformance
result. A trained artifact is still required where the method's setup demands
one (for example, `ULD_ASSISTANT_PATH` for `uld`).

`falcon`, `spul`, and `grun` remain registered experimental ports and are not
counted as available. FALCON's harness-owned SophiaG optimizer has no upstream
reference; SPUL replaces an upstream sentiment-only dataset builder; GRUN still
needs an activation-space conformance harness. `spul` and `grun` therefore remain
registered-but-rejected by the low-level evaluator instead of silently appearing
as runnable short names.

Third-party method code is **not** redistributed (license + size). Clone each
into `external/<dir>` before running that method — the adapters fail loudly
(`require_external`) when the path is missing:

| Method | `external/<dir>` |
|---|---|
| ECO    | `eco-prompts`  |
| Cha    | `cha-ihl-fila` |
| DEPN   | `depn`         |
| O3     | `o3-gao`       |
| LEACE  | `leace`        |

Pin the upstream commit you use; record it in your run notes.

The experimental FALCON adapter file is retained for auditability. If you work on
that port, its checkout belongs at `external/falcon`, but it is not an available
release method until complete conformance evidence exists.

## Run
```bash
bash reproduce.sh prep         # steps 1-2 (and prints the step 3-4 manual notes)
bash reproduce.sh topology     # baseline leak-topology diagnostic, Llama
bash reproduce.sh interfaces   # five-interface comparison, Llama P (tab:benchmark_compare)
bash reproduce.sh substrate    # beyond-weight panel, Llama (tab:substrate_blindness; cross-model: see docs/COMPUTE.md)
```
