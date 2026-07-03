"""Faithful TOFU-style evaluation (Maini et al. 2024).

For each (entity, field) this computes the three TOFU building blocks:
  - Probability : length-normalized conditional prob P(answer | question)^(1/|answer|)
  - ROUGE-L     : ROUGE-L recall of the greedy answer against the ground truth
  - Truth Ratio : mean prob of five perturbed (wrong) answers over the prob of the
                  correct answer  (R_truth = mean_p(pert) / p(correct));
                  R_truth > 1 means the model prefers wrong answers (forgotten-like).

These per-(entity,field) quantities are dumped to JSONL. The cross-model TOFU
metrics (Forget Quality = KS test of the Truth-Ratio distribution against the gold
retain model; Model Utility = harmonic mean over the retain set) are computed by
23_aggregate_benchmark.py, which needs all models' outputs together.

Prompt format matches the LoRA training distribution (chcons.pii.render_qa_pairs):
"Q: {query}\\nA: {value}". Resume-safe: skips completed (id, field) pairs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

from chcons.pii import QUERY_TEMPLATES

FIELDS = ("date_of_birth", "address", "occupation", "employer")
TR_CAP = 1.0e4  # cap Truth Ratio when p(correct) underflows to 0


def load_model(model_path: str, lora: str | None):
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora)
    model.eval()
    return model, tok


def cond_prob(model, tok, prompt: str, answer: str, device) -> float:
    """Length-normalized P(answer | prompt)^(1/|answer|) = exp(-avg_nll)."""
    full = tok(prompt + answer, truncation=True, max_length=512,
               add_special_tokens=True, return_tensors="pt").to(device)
    n_prompt = len(tok(prompt, add_special_tokens=True).input_ids)
    labels = full.input_ids.clone()
    labels[:, :n_prompt] = -100
    with torch.no_grad():
        out = model(**full)
    shift_logits = out.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),
        ignore_index=-100, reduction="sum",
    )
    n_ans = int((shift_labels != -100).sum().item())
    if n_ans == 0:
        return 0.0
    return float(math.exp(-loss.item() / n_ans))


def greedy(model, tok, prompt: str, device, max_new_tokens: int = 48) -> str:
    inp = tok(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()


def load_subset(subset: str, facts_path: Path, split_dir: Path, holdout_path: Path) -> list[dict]:
    if subset == "holdout":
        return [json.loads(l) for l in holdout_path.read_text().splitlines() if l.strip()]
    ids_file = split_dir / f"{subset}_ids_eval.txt"
    keep = {x.strip() for x in ids_file.read_text().splitlines() if x.strip()}
    return [json.loads(l) for l in facts_path.read_text().splitlines()
            if l.strip() and json.loads(l)["id"] in keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF name or local checkpoint dir")
    ap.add_argument("--model-name", required=True, help="row label (base/target/GA/.../gold)")
    ap.add_argument("--lora", default=None, help="optional LoRA adapter to load on top")
    ap.add_argument("--subset", choices=["forget", "retain", "holdout"], default="forget")
    ap.add_argument("--facts", type=Path, default=Path("data/pii_facts/v1_facts.jsonl"))
    ap.add_argument("--split-dir", type=Path, default=Path("data/pii_facts"))
    ap.add_argument("--holdout", type=Path, default=Path("data/faithful/holdout_facts.jsonl"))
    ap.add_argument("--perturb", type=Path, default=Path("data/faithful/perturbations.jsonl"))
    ap.add_argument("--n-sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-jsonl", type=Path, default=None)
    args = ap.parse_args()

    if args.out_jsonl is None:
        args.out_jsonl = Path(
            f"results/faithful_tofu_{args.model_name}_{args.subset}"
            f"_s{args.seed}_n{args.n_sample}.jsonl")
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # perturbation lookup keyed by (id, field)
    pert: dict[tuple[str, str], list[str]] = {}
    for line in args.perturb.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        pert[(r["id"], r["field"])] = r["perturbed"]

    pool = load_subset(args.subset, args.facts, args.split_dir, args.holdout)
    rng = random.Random(args.seed)
    sample = rng.sample(pool, min(args.n_sample, len(pool)))

    done = set()
    if args.out_jsonl.exists():
        for line in args.out_jsonl.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["id"], r["field"]))

    todo = [(e, f) for e in sample for f in FIELDS
            if e.get(f) and (e["id"], f) not in done]
    if not todo:
        print(f"[skip] {args.model_name}/{args.subset}: all done")
        return

    print(f"[load] {args.model}" + (f" + LoRA {args.lora}" if args.lora else ""))
    model, tok = load_model(args.model, args.lora)
    device = model.device
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    with args.out_jsonl.open("a") as fout:
        for i, (entity, field) in enumerate(todo):
            name = entity.get("name", "")
            gt = entity[field]
            prompt = f"Q: {QUERY_TEMPLATES[field].format(name=name)}\nA:"
            p_gt = cond_prob(model, tok, prompt, f" {gt}", device)
            gen = greedy(model, tok, prompt, device)
            rl = scorer.score(gt, gen)["rougeL"].recall  # TOFU uses ROUGE-L recall
            pkey = (entity["id"], field)
            if pkey not in pert:
                raise KeyError(f"missing perturbations for {pkey}; regenerate {args.perturb}")
            perturbed = pert[pkey]
            p_pert = [cond_prob(model, tok, prompt, f" {w}", device) for w in perturbed]
            mean_pert = sum(p_pert) / len(p_pert) if p_pert else 0.0
            truth_ratio = TR_CAP if p_gt <= 0 else min(mean_pert / p_gt, TR_CAP)
            fout.write(json.dumps({
                "id": entity["id"], "field": field, "model": args.model_name,
                "subset": args.subset, "prob": p_gt, "rouge_l": rl,
                "truth_ratio": truth_ratio,
            }) + "\n")
            fout.flush()
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(todo)}] {entity['id']}::{field} "
                      f"prob={p_gt:.4f} R_truth={truth_ratio:.3f}")

    print(f"[done] {args.out_jsonl}")


if __name__ == "__main__":
    main()
