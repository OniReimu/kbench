"""Faithful MUSE-style evaluation (Shi et al. 2024).

Per entity this computes the building blocks of MUSE's four model-side criteria:
  - KnowMem  : mean ROUGE-L of greedy QA answers vs ground truth over the 4 fields
               (C2 on forget, C4/utility on retain).
  - VerbMem  : ROUGE-L of a greedy continuation of the bio prefix vs the true suffix
               (C1 verbatim memorization).
  - MinKProb : Min-K%% Prob membership signal on the bio text (mean log-prob of the
               lowest-K%% tokens). Higher = more member-like. PrivLeak (C3) is the
               AUC of forget(members) vs holdout(non-members) on this signal,
               normalized by the gold retain model; computed in 23_aggregate_benchmark.py.

Prompt format matches the LoRA training distribution ("Q: {query}\\nA: {value}").
Resume-safe: skips completed entity ids.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

from chcons.pii import QUERY_TEMPLATES

FIELDS = ("date_of_birth", "address", "occupation", "employer")


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


def greedy(model, tok, prompt: str, device, max_new_tokens: int = 64) -> str:
    inp = tok(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()


def min_k_logprob(model, tok, text: str, device, k_frac: float = 0.2) -> float:
    """Mean log-prob of the lowest-k_frac fraction of token log-probs (Min-K% Prob)."""
    ids = tok(text, truncation=True, max_length=512, add_special_tokens=True,
              return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**ids).logits
    logp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    tgt = ids.input_ids[:, 1:]
    tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).squeeze(0)
    n = tok_logp.numel()
    if n == 0:
        return 0.0
    k = max(1, int(n * k_frac))
    lowest = torch.topk(tok_logp, k, largest=False).values
    return float(lowest.mean().item())


def load_subset(subset: str, facts_path: Path, split_dir: Path, holdout_path: Path) -> list[dict]:
    if subset == "holdout":
        return [json.loads(l) for l in holdout_path.read_text().splitlines() if l.strip()]
    ids_file = split_dir / f"{subset}_ids_eval.txt"
    keep = {x.strip() for x in ids_file.read_text().splitlines() if x.strip()}
    return [json.loads(l) for l in facts_path.read_text().splitlines()
            if l.strip() and json.loads(l)["id"] in keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--subset", choices=["forget", "retain", "holdout"], default="forget")
    ap.add_argument("--facts", type=Path, default=Path("data/pii_facts/v1_facts.jsonl"))
    ap.add_argument("--split-dir", type=Path, default=Path("data/pii_facts"))
    ap.add_argument("--holdout", type=Path, default=Path("data/faithful/holdout_facts.jsonl"))
    ap.add_argument("--n-sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-jsonl", type=Path, default=None)
    args = ap.parse_args()

    if args.out_jsonl is None:
        args.out_jsonl = Path(
            f"results/faithful_muse_{args.model_name}_{args.subset}"
            f"_s{args.seed}_n{args.n_sample}.jsonl")
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    pool = load_subset(args.subset, args.facts, args.split_dir, args.holdout)
    rng = random.Random(args.seed)
    sample = rng.sample(pool, min(args.n_sample, len(pool)))

    done = set()
    if args.out_jsonl.exists():
        for line in args.out_jsonl.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    todo = [e for e in sample if e["id"] not in done]
    if not todo:
        print(f"[skip] {args.model_name}/{args.subset}: all done")
        return

    print(f"[load] {args.model}" + (f" + LoRA {args.lora}" if args.lora else ""))
    model, tok = load_model(args.model, args.lora)
    device = model.device
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    with args.out_jsonl.open("a") as fout:
        for i, entity in enumerate(todo):
            name = entity.get("name", "")
            # KnowMem: ROUGE of QA answers over fields
            rouges = []
            for field in FIELDS:
                gt = entity.get(field, "")
                if not gt:
                    continue
                prompt = f"Q: {QUERY_TEMPLATES[field].format(name=name)}\nA:"
                gen = greedy(model, tok, prompt, device)
                rouges.append(scorer.score(gt, gen)["rougeL"].fmeasure)
            knowmem = sum(rouges) / len(rouges) if rouges else 0.0

            # VerbMem: bio prefix -> continuation
            bio = entity.get("bio", "")
            if bio and len(bio) >= 60:
                cut = len(bio) // 3
                gen = greedy(model, tok, bio[:cut], device, max_new_tokens=256)
                verbmem = scorer.score(bio[cut:], gen)["rougeL"].fmeasure
            else:
                verbmem = 0.0

            mink = min_k_logprob(model, tok, bio, device) if bio else 0.0

            fout.write(json.dumps({
                "id": entity["id"], "model": args.model_name, "subset": args.subset,
                "knowmem": knowmem, "verbmem": verbmem, "mink": mink,
            }) + "\n")
            fout.flush()
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(todo)}] {entity['id']} "
                      f"KnowMem={knowmem:.3f} VerbMem={verbmem:.3f} MinK={mink:.2f}")

    print(f"[done] {args.out_jsonl}")


if __name__ == "__main__":
    main()
