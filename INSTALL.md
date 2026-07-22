# Installing K-Bench prerequisites

`reproduce.sh` evaluates unlearning methods on pre-built artifacts. Build these
once, then run `reproduce.sh {topology|interfaces|substrate}`.

## 0. Environment

The Docker image freezes the exact CUDA + pinned-library stack (recommended):
```bash
docker build -t kbench .                       # see Dockerfile
docker run --gpus all -it -v "$PWD/results:/workspace/kbench/results" kbench bash
```
Or a local environment with the pinned libraries (torch / transformers / peft / accelerate):
```bash
uv venv && uv pip install -r cross_model_pinned_requirements.txt && uv pip install -e . --no-deps
```
The unpinned `uv sync --extra infer --extra dev` (inside a CUDA GPU session) also works, but
resolves library ranges rather than the pinned versions, so it is less reproducible.

## 1. Synthetic PII corpus (deterministic, no GPU)
```bash
uv run python scripts/01_generate_pii.py --n-facts 5000 --seed 0 --out-dir data/pii_facts --name v1
uv run python scripts/00d_build_distractor_pool.py     # v2.1 distractor pool (startup-audit input)
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
uv run python scripts/00_build_rag_index.py --config configs/rag_pilot.yaml
```
Produces `data/wiki_index*`. The R-text / R-struct substrates need the full
index; `data/wiki_index_smoke/` ships complete (passages + `index.faiss`) for a
quick CPU smoke run. It is built with `BAAI/bge-small-en-v1.5`, so pass
`--embed-model BAAI/bge-small-en-v1.5` when pointing the agent at it (the agent
default is `bge-base`, whose 768-dim vectors would not match this 384-dim index).
Then build the two
v2.1 substrate-isolation indexes (inject the PII into the wiki corpus, with the
target absent from the distractor index and present in the target-in index):
```bash
uv run python scripts/00b_inject_pii_corpus.py \
  --orig-passages data/wiki_index/passages.jsonl --orig-embeddings-dir data/wiki_index/embeddings \
  --facts data/v21/bios_distractor.jsonl --n-forget 5000 --out-dir data/wiki_index_v21_distractor/
uv run python scripts/00b_inject_pii_corpus.py \
  --orig-passages data/wiki_index/passages.jsonl --orig-embeddings-dir data/wiki_index/embeddings \
  --facts data/pii_facts/v1_facts.jsonl --n-forget 5000 --out-dir data/wiki_index_v21_target_in/
```

## 3. PII LoRA + merged target (GPU; substrate P and the C/R distractor LoRA)
Train the target LoRA (memorizes the synthetic PII) and the distractor-only
LoRA-D (used by the C / R-text / R-struct cells), then merge the target:
```bash
uv run python scripts/03_inject_pii.py --facts data/pii_facts/v1_facts.jsonl \
  --out-dir models/lora_v1 --epochs 5 --r 32 --alpha 64 --seed 0
uv run python scripts/03_inject_pii.py --facts data/v21/bios_distractor.jsonl \
  --out-dir models/v21_lora_d --epochs 5 --r 32 --alpha 64 --seed 0
uv run python scripts/20_merge_target.py        # -> models/target_merged
```
`models/target_merged` is the substrate-P target (PII baked into weights) that
the weight-based methods unlearn from; `models/v21_lora_d/final_adapter` is the
always-loaded distractor LoRA for the context and retrieval substrates.

## 4. External unlearning libraries
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
| FALCON | `falcon`       |

Pin the upstream commit you use; record it in your run notes.

## Run
```bash
bash reproduce.sh prep         # steps 1-2 (and prints the step 3-4 manual notes)
bash reproduce.sh topology     # Fig. 2  (baseline leak topology, Llama)
bash reproduce.sh interfaces   # Table 4 (five-interface comparison, Llama P)
bash reproduce.sh substrate    # Table 5 (beyond-weight panel, Llama; cross-model: see docs/COMPUTE.md)
```
