# Compute & reproducibility

## Hardware

All open-weight experiments ran on a single NVIDIA H100 (80GB) per job. Total budget
≈ 500 GPU-hours covering: LoRA injection, adapter-based unlearning, the full
(substrate × method × seed) evaluation matrix across three model families, and the
faithful TOFU/MUSE/WMDP probes.

## Models

| Model | Source | Substrates |
|-------|--------|-----------|
| Llama-3.1-8B-Instruct | HuggingFace | P, C, R-text, R-struct |
| Qwen3.5-9B | HuggingFace | P, C, R-text, R-struct |
| Mistral-7B-Instruct-v0.3 | HuggingFace | P, C, R-text, R-struct |
| API frontier models (optional) | OpenRouter | C, R-text, R-struct (weights immutable, so no P / no weight-based defenses) |

## Determinism

- Greedy decoding (`T=0`) — no sampling variance.
- Three pre-registered seeds `{0, 137, 271}`; `n=200` queries per seed.
- The synthetic corpus is regenerable from a fixed seed (`scripts/01_generate_pii.py`).
- Aggregation (`scripts/09_k_verdict_v2.py`, `scripts/23_aggregate_benchmark.py`) is
  pure post-processing over the JSONL, so verdicts reproduce even when the underlying
  model is API-gated and not bit-reproducible.

## Environments

The maintainers' exact main-environment versions came from their monorepo lock and
are published as `main_pinned_requirements.txt`; `pyproject.toml` holds compatible
dependency ranges. The Llama-3.1-8B and Qwen3.5-9B experiments run in this main
environment. One additional pinned
requirement file ships for the Mistral-7B-v0.3 stack, whose transformers version
diverges: `cross_model_pinned_requirements.txt` (transformers 4.51.3). Build it as
a separate venv (e.g. `uv venv --python 3.11 .venv-mistral` then `uv pip install -r
cross_model_pinned_requirements.txt`); do not install it into the main project
venv. `qwen_pinned_requirements.txt` pins the older Qwen2.5-era stack (transformers
4.47.1) and is retained for reproducing that compatibility path. Weight-based
unlearning uses `locuslab/open-unlearning` commit
`4ad738aaf60f6a4385f6e2506d01da99e76c31f3` in its own isolated environment. The
framework's committed `requirements.txt` pins `transformers==4.51.3` and
`torch==2.4.1`. It does not declare `peft`; the maintainers' cluster-side
`.venv-openunlearn` was not available locally, so the installed PEFT version and a
complete offline `uv pip freeze` could not be determined. For that reason this
release does not claim a PEFT version or ship an OpenUnlearning freeze file.

## Cross-model block (`reproduce.sh substrate`)

Only the Llama rows in the substrate block are emitted under the public `llama_` naming
that `scripts/09_k_verdict_v2.py` discovers (the legacy `v77app_` alias is also
accepted; its `FILE_RE` / aggregation are
single-model by design — the nested substrate/method/seed dict has no model
axis). The cross-model rows (Qwen / Mistral) are scored under separate per-family
prefixes and aggregated independently per family; the `substrate` target prints the
commands rather than pooling them into the single-model Llama verdict. Run Mistral in its pinned
venv above and Qwen in the main environment.

## Wall-clock guidance for `reproduce.sh`

| Target | Scope | Approx. GPU-h |
|--------|-------|--------------|
| `prep` | RAG index build + PII gen/inject (one-time) | ~hours (index dominates) |
| `topology` | Llama baseline on P, C, R-text, and R-struct | ~4 |
| `interfaces` | Five-interface comparison table (Llama P) | ~6 |
| `substrate` | Weight-probe versus agentic-observer table (Llama C/R; cross-model rows separate) | ~8 |
| `all` | released Llama targets above; cross-model and faithful probes remain separate | ~18 after `prep` |

The approximately 500 GPU-hour total at the top of this page is the complete paper
study, including the cross-model and faithful-probe jobs that `reproduce.sh` prints
but does not launch automatically.
