# K-Bench Leaderboard Submissions

This directory contains community submissions to the K-Bench leaderboard.

## How to Submit

To submit a method to the leaderboard, open a pull request against the repository at:
**https://github.com/OniReimu/kbench**

Your pull request must add a new directory under `leaderboard/submissions/<method-name>/`:

```text
leaderboard/submissions/<method-name>/
├── <method-name>.kbench-bundle/      # Self-contained bundle built by `kbench bundle`
│   ├── bundle.json                   # Bundle manifest with hashes, configs, and pairings
│   └── cells/                        # Shipped JSONL cells and .config.json sidecars
│       ├── llama_P_<method>_forget_seed0.jsonl
│       ├── llama_P_<method>_forget_seed0.config.json
│       ├── llama_P_<method>_retain_seed0.jsonl
│       ├── llama_P_<method>_retain_seed0.config.json
│       ├── llama_P_none_forget_seed0.jsonl
│       ├── llama_P_none_forget_seed0.config.json
│       ├── llama_P_none_retain_seed0.jsonl
│       └── llama_P_none_retain_seed0.config.json
├── metadata.yaml                     # Submission metadata (see TEMPLATE/metadata.yaml)
└── README.md                         # Description of method, environment, and reproduction
```

### 1. Generate transcripts and score

Run your candidate method using `kbench eval` (or evaluate offline and score with `kbench score`):
- A leaderboard submission minimally requires **seed 0 with n = 200 forget and 200 retain queries per cell** (matching the setting of the published twenty-method leaderboard in Table 15 of the paper).
- Seeds `{0, 137, 271}` are optional additional evidence.
- `kbench eval` runs all three seeds by default with `n = 200`; set `KBENCH_SEEDS=0` to run only seed 0. Externally produced cells may also use the seed-0 minimum.
- Both candidate and reference `none` cells must be present for each evaluated substrate.
- K-Score is evaluated and reported per substrate; K-Scores are never averaged across substrates.

### 2. Build the transcript bundle

Package your evaluated transcripts into a self-contained bundle:

```bash
kbench bundle --cells results --out leaderboard/submissions/<method-name>/<method-name>.kbench-bundle
```

Validate the bundle locally before submitting:

```bash
kbench report leaderboard/submissions/<method-name>/<method-name>.kbench-bundle
```

### 3. Fill in metadata and documentation

- Copy `leaderboard/submissions/TEMPLATE/metadata.yaml` to `leaderboard/submissions/<method-name>/metadata.yaml` and complete all fields:
  - `method_name`: canonical short name.
  - `paper`: title and URL/arXiv link.
  - `code`: public code repository URL.
  - `contact`: author name and email.
  - `base_model`: base model identifier (default `meta-llama/Llama-3.1-8B-Instruct`).
  - `substrates`: list of evaluated substrates (`P`, `C`, `R-text`, `R-struct`).
  - `seeds`: list of seeds evaluated (minimally `[0]`).
  - `n`: number of queries per seed per split (typically `200`).
  - `kbench_commit`: git commit SHA of the K-Bench version used.
  - `checkpoint_or_adapter`: public status and Hugging Face or download URL.
  - `deviations_from_published_method`: note any hyperparameter modifications.
- Write `leaderboard/submissions/<method-name>/README.md` explaining the method, runtime requirements, training procedure, and reproduction commands.

---

## What Reviewers Check

Before merging a submission PR, maintainers and automated checks verify:

1. **Bundle integrity and report validation**:
   - `kbench report` must run cleanly on the bundle directory, verifying that all SHA-256 digests in `bundle.json` match the files under `cells/`.
   - Every candidate forget cell must declare a matching retain cell and corresponding baseline `none` forget/retain reference cells.
2. **Deterministic re-aggregation**:
   - Maintainers recompute the scores from the bundle with the same code path the submitter ran (the aggregation, not the model weights):
     ```bash
     kbench report leaderboard/submissions/<method-name>/<method-name>.kbench-bundle
     ```
   - The recomputed K-Score, graded observer rate, Δsel, degeneration, per-channel severity, eligibility verdict, K-class and BH-adjusted p-value must reproduce the reported results bit for bit. `kbench report` scores a seed-0 submission against the baseline restricted to seed 0. `scripts/09_k_verdict_v2.py` pools all three seeds, so it applies to three-seed submissions only.
3. **Model provenance**:
   - Evaluator sidecars must verify that architecture parameters (`model_type`, `vocab_size`, `hidden_size`, `num_hidden_layers`) match the declared reference base model.
   - Runs with modified architectures must be explicitly declared and marked with `"comparable": false`.
4. **Substrate-P target contract**:
   - Parametric-substrate (P) methods must be unlearned from K-Bench's injected target adapter merged into the base model (not the bare instruct base).
5. **Eligibility gate**:
   - The method is checked against the primary eligibility criteria (paper Section 5.6):
     - K-Score factors and retain preservation use the graded observer rate (token-recall severity), not binary per-query OR(all), which is reported alongside.
     - **Retain preservation $\ge 0.80$**: $\overline{\text{OR}}_{\text{retain}}^{\text{method}} / \overline{\text{OR}}_{\text{retain}}^{\text{none}} \ge 0.80$.
     - **Added degeneration $\le 0.20$**: $\Delta\text{degen} \le 0.20$ over baseline.
     - **No terminal agent collapse**: candidate absolute degeneration rate $< 0.50$ on both splits; the untreated baseline is exempt. On Llama substrate P, a baseline-like candidate can therefore carry this label because the baseline itself degenerates on roughly half its queries; read `Δdeg` beside it.
   - Methods passing this gate are ranked by leakage suppression alone. Ineligible entries are cataloged under the specific condition failed (retain damage or agent collapse).
6. **Integrity and data-split rules**:
   - Evaluation queries must come solely from the disjoint evaluation entities (800 forget and 3,800 retain IDs).
   - Components fitted from examples (detectors, directions, or probes) must fit strictly on the disjoint adapter pools (`forget_ids_adapter.txt` and `retain_ids_adapter.txt`, 200 + 200 IDs).
   - Weight updates may use all 1,000 forget entities and, for methods with a retain term, all 4,000 retain entities; the latter include the 3,800 retain evaluation entities.
   - Hybrid methods apply the adapter-pool rule to fitted components and the full-set rule to weight-update terms.
