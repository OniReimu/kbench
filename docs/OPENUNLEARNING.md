# Train NPO with OpenUnlearning, then score it with K-Bench

This recipe trains OpenUnlearning's NPO implementation from K-Bench's injected
Llama target and passes the resulting checkpoint to K-Bench. Starting from the
bare `meta-llama/Llama-3.1-8B-Instruct` base would not unlearn the injected target
and is not a valid substrate-P comparison.

This recipe pins `locuslab/open-unlearning` at commit
`4ad738aaf60f6a4385f6e2506d01da99e76c31f3`. Its committed
`requirements.txt` pins `transformers==4.51.3` and `torch==2.4.1`. It does not list
`peft`, and the maintainers' `.venv-openunlearn` is not available in this local
checkout, so its installed PEFT version could not be determined and no environment
freeze file is published here.

## 1. Prepare the injected target and QA data

Download and merge the v1.0 target adapter using the fp32 CPU procedure in the
[README](../README.md#substrate-p-contract-and-target). Set the result path to a
clear environment variable:

```bash
export KBENCH_INJECTED_TARGET=/path/to/kbench-injected-target
```

From the K-Bench checkout, convert the bundled synthetic records into the QA files
that OpenUnlearning reads:

```bash
python3 scripts/19_convert_pii_to_openunlearn.py
```

This writes `data/openunlearn/v1_qa_forget.jsonl` and
`data/openunlearn/v1_qa_retain.jsonl`.

## 2. Install the public Hydra configs

Assuming the K-Bench and OpenUnlearning checkouts are sibling directories, copy the
experiment and dataset configs into OpenUnlearning:

```bash
mkdir -p ../open-unlearning/configs/experiment/unlearn/kbench
mkdir -p ../open-unlearning/configs/data/datasets
cp openunlearn_configs/experiment/unlearn/kbench/default.yaml ../open-unlearning/configs/experiment/unlearn/kbench/default.yaml
cp openunlearn_configs/data/datasets/KBench_PII_forget.yaml ../open-unlearning/configs/data/datasets/KBench_PII_forget.yaml
cp openunlearn_configs/data/datasets/KBench_PII_retain.yaml ../open-unlearning/configs/data/datasets/KBench_PII_retain.yaml
```

The public experiment config resolves both the model and tokenizer from
`KBENCH_INJECTED_TARGET`. Its five-epoch schedule and method overrides match the
K-Bench run configuration; the monorepo's older bare-base config is not used.

## 3. Train NPO

Run OpenUnlearning's Hydra entry point from its own isolated OpenUnlearning
environment (the transformers-4.51 environment described in
[COMPUTE.md](COMPUTE.md#environments)), not the K-Bench main environment. This is
a GPU command; replace the K-Bench checkout path and output directory for your machine:

```bash
python3 src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/kbench/default \
  trainer=NPO \
  task_name=kbench_npo \
  paths.data_dir=/path/to/kbench/data \
  paths.output_dir=/path/to/openunlearning-output \
  trainer.args.eval_on_start=False \
  trainer.args.eval_strategy=no \
  trainer.args.do_eval=False \
  trainer.args.gradient_accumulation_steps=8 \
  model.model_args.attn_implementation=eager
```

Do not override `model.model_args.pretrained_model_name_or_path` with the bare base.
Hydra reads it from `KBENCH_INJECTED_TARGET` in the public config.

Although the shipped experiment config sets `save_strategy: 'no'`, that disables
intermediate Trainer checkpoints only. At the end of training, OpenUnlearning's
`src/train.py` calls `trainer.save_model(trainer_args.output_dir)`. The inherited
trainer config maps that directory to `paths.output_dir`, so the command above writes
the final Hugging Face checkpoint directly to `/path/to/openunlearning-output` (not
to a `checkpoint-*` child). Pass that exact directory to `kbench eval --model`.

## 4. Evaluate and obtain the K-Score

Back in the K-Bench checkout, fetch the fixed reference cells and evaluate the
saved OpenUnlearning model:

```bash
kbench fetch-assets --mini
kbench eval --model /path/to/openunlearning-output --substrate P --method none --name NPO
```

`--mini` is sufficient here because this recipe scores only the substrate-P Llama
cell. Use `--full` only when evaluating every substrate and base. Fetching can run in
any K-Bench environment, including the CPU-only one. Run `kbench eval` in the main
pinned K-Bench environment because it spawns the evaluator with `sys.executable`.

The evaluator prints the per-substrate result and writes `results/NPO.kbench.json`,
including the K-Score. Substrate P is comparable only if the trained checkpoint in
the preceding command descends from `KBENCH_INJECTED_TARGET`. The P reference
profile makes the evaluator pass `--pii-in-weights`, so it evaluates that checkpoint
directly and does not stack the original injection LoRA on top.

## Expected Llama NPO row

The paper's benchmark comparison table (Table 10,
Llama-3.1-8B block, NPO row) reports:

| Forget observer rate | Degeneration | K-Score |
|---:|---:|---:|
| 0.368 | 31% | 0.446 |

These are the paper run's reference values, not acceptance tolerances for a newly
trained checkpoint. Hardware, dependency, or training differences should be
reported with the resulting row rather than adjusted to force a match.
