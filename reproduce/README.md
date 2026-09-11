# Reproducing the paper tables

This page is only for regenerating the paper's released table inputs and reports.
The checkpoint-evaluation and submission lifecycle remains in the
[top-level README](../README.md).

The AAAI-2027 printed numbers are reproduced from the public repository's frozen
`aaai2027` branch (commit `bc52cd00`). The main branch scores a bare direct reply
as the answer, which is the scoring rule the arXiv preprint reports.

Run the targets below from the release root after completing the installation and
asset steps
in [INSTALL.md](../INSTALL.md). Runtime estimates are the single-H100 guidance in
[COMPUTE.md](../docs/COMPUTE.md); hardware and storage can change wall time.

| Target | Regenerates | Requirements | Approximate time |
|---|---|---|---|
| `bash reproduce.sh smoke` | Demonstration report only; no paper table | CPU, bundled smoke cells, one-time `numpy` environment resolution | about 30 seconds |
| `bash reproduce.sh prep` | Inputs used by all released tables | CPU for deterministic PII generation; GPU for embedding/index construction and the printed LoRA training commands; base-model and full corpus assets | hours |
| `bash reproduce.sh topology` | Baseline leak-topology diagnostic used for the topology analysis; it is not a final paper table | Llama injected target for P, both retrieval indexes for R, GPU | about 4 GPU-hours |
| `bash reproduce.sh interfaces` | K-Bench cells and P-substrate leaderboard for the five-interface comparison table (`tab:benchmark_compare`); the target prints, but does not launch, the separate TOFU/MUSE/WMDP probes needed to complete that table | Llama injected target, method artifacts and any external method checkouts listed in INSTALL.md, GPU | about 6 GPU-hours |
| `bash reproduce.sh substrate` | Llama C/R block for the weight-probe versus agentic-observer table (`tab:substrate_blindness`); it prints the separate Qwen/Mistral commands rather than pooling model families | both retrieval indexes, model downloads, GPU; separate pinned environments for cross-model rows | about 8 GPU-hours for the Llama block |
| `bash reproduce.sh all` | Runs the three released Llama targets above in order; cross-model and faithful-probe commands remain separate | union of the requirements above | about 18 GPU-hours after `prep`; the complete paper study is about 500 GPU-hours |

The target names describe report inputs, not a promise that every publication
figure can be regenerated from this release. In particular,
`panel_collapse.pdf` and `panel_selectivity.pdf` are shipped with the paper as
pre-rendered figures. Their paper-only `gen_panels.py` needs experiment prefixes
that none of the public targets above generate, so this release cannot rebuild
those panels and does not imply otherwise.

## CPU transcript-bundle check

The offline handoff operates without a model or network access. The shipped smoke
fixture includes each JSONL together with its evaluator-generated `.config.json`
sidecar:

```bash
kbench bundle --cells data/smoke/cells --out smoke.kbench-bundle
kbench report smoke.kbench-bundle
```

For your own results, point it at the evaluator output directory:

```bash
kbench bundle --cells <evaluator-output> --out method.kbench-bundle
kbench report method.kbench-bundle
```
