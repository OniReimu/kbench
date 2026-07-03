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
| Qwen2.5-7B-Instruct | HuggingFace | P, C, R-text, R-struct |
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

The base stack is pinned in `pyproject.toml` (`uv sync`). Two additional pinned
requirement files ship for the model-family-specific stacks where transformers
versions diverge: `cross_model_pinned_requirements.txt` (Mistral / Gemma-3,
transformers 4.51.3) and `qwen_pinned_requirements.txt` (Qwen2.5, transformers
4.47.1). Build each as a separate venv (e.g. `uv venv .venv-qwen` then
`uv pip install -r qwen_pinned_requirements.txt`); never install them into the
main project venv. Weight-based unlearning uses the open-unlearning framework
(transformers 4.51) in its own isolated environment.

## Cross-model block (`reproduce.sh substrate`)

Only the Llama rows in the substrate block are emitted under the `v21B_` naming
that `scripts/09_k_verdict_v2.py` discovers (its `FILE_RE` / aggregation are
single-model by design — the nested substrate/method/seed dict has no model
axis). The cross-model rows (Qwen / Mistral) are scored under the separate paper
prefixes `v53_qwen`, `v26_mistral`, `v29_mistral` and aggregated independently
per family; the `substrate` target prints the commands rather than pooling them
into the v21B verdict. Run those families in their pinned venvs above.

## Wall-clock guidance for `reproduce.sh`

| Target | Scope | Approx. GPU-h |
|--------|-------|--------------|
| `prep` | RAG index build + PII gen/inject (one-time) | ~hours (index dominates) |
| `topology` | Llama P baseline | < 1 |
| `interfaces` | Table 7 block A (7 defenses, Llama P) | ~6 |
| `substrate` | Table 7 block B, Llama C/R (cross-model rows separate) | ~8 |
| `all` | full matrix | ~500 |
