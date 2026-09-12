# Weight-Editing Method Quickstart (Substrate P)

This guide walks through the end-to-end evaluation of a weight-editing or gradient-based
unlearning method on K-Bench substrate P (the parametric memory substrate).

---

## Workflow Overview

1. **Fetch assets**: download the injected target adapter, the distractor retrieval index, and baseline reference cells.
2. **Merge target adapter**: merge the LoRA adapter into the base model on CPU to obtain the target checkpoint.
3. **Unlearn with your method**: run your weight unlearning algorithm to produce an edited checkpoint.
4. **Evaluate with K-Bench**: run `kbench eval --substrate P --model <edited checkpoint> --method none`.
5. **Score**: run `kbench score --cells results --name <method-name>`.
6. **Bundle and report**: package with `kbench bundle` and verify with `kbench report`.
7. **Submit PR**: create a pull request to the leaderboard repository.

### Environment map

| Step | Environment |
|---|---|
| 1, fetch assets | Any K-Bench environment, including the CPU-only environment |
| 2, `scripts/20_merge_target.py` | Main pinned K-Bench environment (`main_pinned_requirements.txt`) |
| 3, data conversion | Any K-Bench environment; run the unlearning method in its own required environment. OpenUnlearning training uses its isolated OpenUnlearning environment |
| 4, `kbench eval` | Main pinned K-Bench environment; the command launches each evaluator with that environment's `sys.executable` |
| 5, inspect generated files | No command-specific environment; these are outputs from step 4 |
| 6, `kbench score` | Any K-Bench environment, including the CPU-only environment |
| 7, `kbench bundle` and `kbench report` | Any K-Bench environment, including the CPU-only environment |
| 8, prepare the submission | No Python environment required |

---

## Step 1: Fetch Reference Assets

Substrate P requires the injected target adapter, the distractor retrieval index (distractor Wikipedia corpus passages), and the baseline reference cells (`none`).

```bash
# Download the target LoRA adapter (~336 MB)
kbench fetch-assets --target

# Download the distractor retrieval index (~8.4 GB; used for P, C, and R-struct)
kbench fetch-assets --indexes distractor

# Download the untreated baseline cells (~0.4 MB for substrate P)
kbench fetch-assets --mini
```

`--mini` fetches the substrate-P Llama baseline, which is all this P-only weight
method workflow needs. `--full` is for workflows that score every substrate and base.

**Hardware requirement:** CPU only (network download and disk storage).
- Target adapter: `models/Llama-3.1-8B-kbench-target-adapter/` (~336 MB)
- Distractor index: `data/wiki_index_v21_distractor/` (~8.4 GB)
- Shipped baseline cells: `results/llama_P_none_{forget,retain}_seed*.jsonl` (~0.4 MB)

---

## Step 2: Merge the Target Adapter into the Base Model

Weight-based unlearning on substrate P must start from K-Bench's injected target model (so the forget PII was present in the weights before unlearning). The target is distributed as a PEFT LoRA adapter targeting `meta-llama/Llama-3.1-8B-Instruct` at Hugging Face snapshot `0e9e39f249a16976918f6564b8830bc894c89659`.

In the main pinned environment (`main_pinned_requirements.txt`, which includes
`peft`), merge the adapter on CPU in full precision (fp32) and cast to bfloat16,
matching the exact production procedure. No GPU is needed for this step:

```bash
uv run --no-sync python scripts/20_merge_target.py   # -> models/target_merged
```

The equivalent inline Python is:

```python
import torch
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "meta-llama/Llama-3.1-8B-Instruct"
revision = "0e9e39f249a16976918f6564b8830bc894c89659"
adapter_dir = "models/Llama-3.1-8B-kbench-target-adapter"
output_dir = "models/target_merged"

print("Loading base model in fp32 on CPU...")
model = AutoModelForCausalLM.from_pretrained(
    base_id, revision=revision, torch_dtype=torch.float32, device_map="cpu"
)
print("Applying target adapter...")
model = PeftModel.from_pretrained(model, adapter_dir)
print("Merging weights and casting to bfloat16...")
model = model.merge_and_unload().to(torch.bfloat16)
model.save_pretrained(output_dir, safe_serialization=True)
AutoTokenizer.from_pretrained(base_id, revision=revision).save_pretrained(output_dir)

# transformers 5.x records `tokenizer_class: "TokenizersBackend"`, a name only 5.x knows.
# Training recipes that run in the transformers-4.51 environment (see docs/OPENUNLEARNING.md)
# refuse it with `ValueError: Tokenizer class TokenizersBackend does not exist`, so write the
# base model's own class back and keep one checkpoint both environments can read.
import json
from huggingface_hub import hf_hub_download

saved = Path(output_dir) / "tokenizer_config.json"
base_class = json.loads(
    Path(hf_hub_download(base_id, "tokenizer_config.json", revision=revision)).read_text()
)["tokenizer_class"]
config = json.loads(saved.read_text())
config["tokenizer_class"] = base_class
saved.write_text(json.dumps(config, indent=2) + "\n")

print(f"Merged target saved to {output_dir}")
```

`scripts/20_merge_target.py` already does that last step for you.

**Hardware requirement:** CPU only. Requires approximately 32 GB of system RAM to hold the model in fp32 during merging, plus Hugging Face access to the gated repository `meta-llama/Llama-3.1-8B-Instruct`.

---

## Step 3: Produce an Edited Checkpoint with Your Method

Train your unlearning algorithm starting from `models/target_merged`:

- **Example-fitted components:** Detectors, directions, and probes fit only on `data/pii_facts/{forget,retain}_ids_adapter.txt` (200 forget + 200 retain entities), which are disjoint from evaluation.
- **Weight-update forget data:** Weight updates may use all 1,000 forget entities (`pii-00000`..`pii-00999`).
- **Weight-update retain data:** When the method has a retain term, weight updates may use all 4,000 retain entities (`pii-01000`..`pii-04999`). This includes the 3,800 retain evaluation entities, so retain weight training overlaps retain evaluation by design.
- **Hybrid methods:** Apply the corresponding rule to each component: fit detectors/directions/probes only on adapter pools, while weight-update terms may use the full forget and retain sets above.

To convert K-Bench synthetic records into QA formats used by standard unlearning libraries (such as OpenUnlearning):

```bash
python3 scripts/19_convert_pii_to_openunlearn.py
```

This writes:
- `data/openunlearn/v1_qa_forget.jsonl` (4,000 QA pairs across 1,000 forget entities)
- `data/openunlearn/v1_qa_retain.jsonl` (16,000 QA pairs across 4,000 retain entities)

For a detailed walkthrough using OpenUnlearning (e.g. NPO), see [docs/OPENUNLEARNING.md](OPENUNLEARNING.md).

That converter emits question-answer pairs, the shape fine-tuning libraries expect. A method
that locates and rewrites a stored fact needs the record instead, so K-Bench ships the two
files it was built from:

- `data/pii_facts/v1_facts.jsonl`: one row per entity, carrying `id`, `name`, the attribute
  fields `date_of_birth`, `address`, `occupation` and `employer`, and the `bio` sentence the
  injection was trained on.
- `data/pii_facts/v1_queries.jsonl`: one row per evaluation query, carrying `query_id`,
  `pii_id`, `field`, the `query` as the agent asks it, and the `ground_truth` value that
  counts as leaked.

From those two files a locate-and-edit method can assemble its own input: the subject is
`name`, the relation is `field`, the value to remove is the attribute or `ground_truth`, and
the prompt is `query`. The four id lists in the same directory give the splits.

**Hardware requirement:** GPU required; see [COMPUTE.md](COMPUTE.md) for the hardware used in the paper.

Save your unlearned checkpoint to a directory, e.g. `models/my_edited_checkpoint`.

---

## Step 4: Evaluate with K-Bench

In the main pinned K-Bench environment, run `kbench eval` passing your edited
checkpoint via `--model` and specifying `--method none` (since the unlearning has
already been performed in the weights). This environment matters because `kbench
eval` spawns each evaluator with `sys.executable`:

```bash
kbench eval --substrate P --model models/my_edited_checkpoint --method none --name MyMethod
```

By default this runs seeds `{0, 137, 271}` with 200 forget and 200 retain queries
per seed. Set `KBENCH_SEEDS=0` in the environment to run only seed 0. The leaderboard minimum for
externally produced transcripts is seed 0 with 200 queries per split.

`kbench eval` refuses to start when transcripts for the same `--name` already exist. It lists
them and exits with status 2. Pass `--resume` to keep the finished cells and run only the rest,
which is what a job cut short by a walltime limit needs, or pick a different `--name` to start
clean.

**Hardware requirement:** GPU required; see [COMPUTE.md](COMPUTE.md). Each evaluation
subprocess also holds the distractor index in host memory (see the retrieval-index notes in
the [README](../README.md)).

### Process Architecture Note

`kbench eval` orchestrates evaluation by spawning an isolated subprocess for every `(substrate, split, seed)` tuple.
- For substrate P and seed 0, it launches two separate processes:
  1. Split `forget`, seed `0`
  2. Split `retain`, seed `0`
- **Important:** If weight editing were implemented inside an adapter's `setup()` method, that training/editing code would execute repeatedly in every spawned subprocess. By contrast, passing an offline edited checkpoint with `--model <path> --method none` loads the pre-edited model directly, ensuring exact reproducibility and avoiding redundant computation.

### Model Provenance Verification

`kbench eval` inspects `config.json` before generation to ensure `model_type`, `vocab_size`, `hidden_size`, and `num_hidden_layers` match the declared base model (`meta-llama/Llama-3.1-8B-Instruct`). If a non-matching model is passed, `kbench eval` refuses to proceed unless `--allow-nonstandard-model` is given:

```text
model provenance mismatch: --model does not match the declared base 'meta-llama/Llama-3.1-8B-Instruct' (...). Pass --allow-nonstandard-model to continue as a non-comparable run.
```

Furthermore, passing the bare instruct base model for substrate P prints:
```text
WARNING: substrate P requires the K-Bench injected target; --model is the bare instruct base 'meta-llama/Llama-3.1-8B-Instruct', so this run does not satisfy the Substrate-P contract.
```

---

## Step 5: Cell Naming Convention and Locations

`kbench eval` writes per-query JSONL transcript cells and corresponding `.config.json` sidecar files into `results/`:

```text
results/
├── llama_P_MyMethod_forget_seed0.jsonl
├── llama_P_MyMethod_forget_seed0.config.json
├── llama_P_MyMethod_retain_seed0.jsonl
├── llama_P_MyMethod_retain_seed0.config.json
├── llama_P_none_forget_seed0.jsonl
├── llama_P_none_forget_seed0.config.json
├── llama_P_none_retain_seed0.jsonl
└── llama_P_none_retain_seed0.config.json
```

The canonical file naming format is:
`<prefix>_<substrate>_<method>_<split>_seed<seed>.jsonl`
- `<prefix>`: default `llama`
- `<substrate>`: `P` (or `C`, `R-text`, `R-struct`)
- `<method>`: candidate method label (e.g. `MyMethod`) or `none` for untreated baseline
- `<split>`: `forget` or `retain`
- `<seed>`: integer seed, e.g. `seed0` (seeds `{0, 137, 271}` for full 3-seed replication)

---

## Step 6: Score Candidate Transcripts

Run `kbench score` over your cells directory (CPU-only):

```bash
kbench score --cells results --name MyMethod
```

`kbench score` automatically intersects available candidate and baseline cells, explains skipped substrates, and prints the full score card and eligibility verdict (layout shown with placeholder values):

```text
[substrates] default local intersection (reference + candidate forget/retain): P
[substrates] skipped C: missing local reference forget, reference retain, candidate forget, candidate retain cells
[substrates] skipped R-text: missing local reference forget, reference retain, candidate forget, candidate retain cells
[substrates] skipped R-struct: missing local reference forget, reference retain, candidate forget, candidate retain cells

== MyMethod -- K-Bench ==
  P         : K-Score <k> (untreated baseline <k_none>) | graded observer rate (K-Score input) forget <or>  Δsel <d_sel>  degen <degen>% | worst: <channel>
    binary per-query OR(all): forget <or_bin> | absolute retain <or_retain>
    K-class: <K-REF α× | K-SUP | measured failure> | BH-adjusted McNemar p_adj: <p>
    eligibility: <PASS | FAIL> (retain preservation ratio <r>; added degeneration Δdeg <a>; ...)
    seed coverage: seed-0 leaderboard minimum
    raw_full fallbacks: <n> (bare direct replies scored as Z_answer when no parsed answer, tool call, or thought was recorded)
  coverage  : 1/1 scored
  aggregate : not reported; K-Score is read separately for each substrate
  note      : two runs may be compared ONLY when their coverage_signature values are equal
  -> leaderboard row: results/MyMethod.kbench.json
```

**Hardware requirement:** CPU only (fast evaluation over saved JSONL files).

---

## Step 7: Build and Validate the Transcript Bundle

Validate the source sidecars and bundle the evaluated JSONL cells into a single
self-contained artifact:

```bash
kbench bundle --cells results --out MyMethod.kbench-bundle
kbench report MyMethod.kbench-bundle
```

`kbench bundle` validates the cell pairings and sidecar metadata, hashes each JSONL,
embeds the sidecar contents in `bundle.json`, and copies the candidate and baseline
JSONL cells under `cells/`. It does not copy separate `.config.json` files.
`kbench report` verifies the bundle offline and outputs the official report (placeholder values):

```text
built MyMethod.kbench-bundle (MyMethod.kbench-bundle/bundle.json)
K-Bench offline report
Schema: kbench-transcript-bundle@1-candidate
K-Bench version: 1.0.0
Scorer: v2

Method: MyMethod
  model: models/my_edited_checkpoint
  base_model: meta-llama/Llama-3.1-8B-Instruct
  candidate name: MyMethod
  model fingerprint: sha256:<digest>
  P         : K-Score <k> | OR_forget <or>  Δsel <d_sel>  degen <degen>% | worst: <channel>
    eligibility: <PASS | FAIL> (retain preservation ratio <r>; added degeneration Δdeg <a>; <details>)
    K-class: <K-REF α× | K-SUP | measured failure> | BH-adjusted McNemar p_adj: <p>
    untreated baseline K-Score: <k_none>
    seeds covered: [0] (seed-0 leaderboard minimum)
    per-channel severity:
    channel       severity
    Z_CoT         <severity>
    Z_tool        <severity>
    Z_tool_wide   <severity>
    Z_RAG         <severity>
    Z_answer      <severity>
    Z_summary     <severity>
```

**Hardware requirement:** CPU only.

---

## Step 8: Submit to the Leaderboard

To publish your result:
1. Fork `https://github.com/OniReimu/kbench`.
2. Create `leaderboard/submissions/MyMethod/`.
3. Copy `MyMethod.kbench-bundle` into that directory.
4. Copy `leaderboard/submissions/TEMPLATE/metadata.yaml` to `leaderboard/submissions/MyMethod/metadata.yaml` and fill in your details.
5. Add a `README.md` describing your method and reproduction commands.
6. Open a Pull Request following [leaderboard/submissions/README.md](../leaderboard/submissions/README.md).
