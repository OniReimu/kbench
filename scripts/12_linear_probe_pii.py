"""Layer-wise linear probe: can a logistic regression distinguish forget vs
retain query hidden states?

Purpose: validate whether LEACE's null result on PII unlearning is a
*legitimate ontology finding* (PII concept not linearly separable in hidden
states → LEACE has no linear direction to erase) vs a *hyperparameter
artifact* (we picked the wrong layer / token / sample size).

Setup mirrors LEACE adapter:
  - Llama-3.1-8B-Instruct + (optional) deployed LoRA
  - For each layer ℓ in stride: collect hidden state at last token across
    N forget queries (label=1) and N retain queries (label=0)
  - Train sklearn LogisticRegression with 5-fold CV; report accuracy
  - Chance = 0.5; if probe accuracy ≈ 0.5 → linearly inseparable
                  if probe accuracy >> 0.5 → LEACE hyperparams were wrong

Output: results/probe_pii_layerwise.json with {layer_idx: cv_accuracy_mean}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from transformers import AutoModelForCausalLM, AutoTokenizer

# chcons is installed as a package (see pyproject.toml); import it directly.
from chcons.pii import QUERY_TEMPLATES, read_jsonl


def get_decoder_layers(model):
    m = model
    for attr in ("base_model", "model", "model"):
        if hasattr(m, attr):
            inner = getattr(m, attr)
            if hasattr(inner, "layers"):
                return inner.layers
            m = inner
    if hasattr(model, "get_decoder"):
        dec = model.get_decoder()
        if hasattr(dec, "layers"):
            return dec.layers
    raise RuntimeError("Could not locate decoder layers")


def build_query_prompts(records, n_per_class):
    """Build (prompt, name) pairs — first available PII field per record."""
    out = []
    for r in records[:n_per_class]:
        d = r.to_dict()
        for field in ("date_of_birth", "address", "occupation", "employer"):
            if not d.get(field):
                continue
            q = QUERY_TEMPLATES[field].format(name=r.name)
            out.append(f"Q: {q}\nA:")
            break
    return out


def collect_hidden_states(model, tokenizer, prompts, layer):
    """For each prompt, run forward and grab last-token hidden state at
    decoder layer `layer`. Returns [N, hidden_dim] tensor on CPU."""
    captured = []

    def hook(_m, _inp, output):
        h = output[0] if isinstance(output, tuple) else output
        captured.append(h[:, -1, :].detach().cpu().float())

    handle = layer.register_forward_hook(hook)
    try:
        model.eval()
        with torch.no_grad():
            for prompt in prompts:
                ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
                _ = model(input_ids=ids["input_ids"], attention_mask=ids["attention_mask"])
    finally:
        handle.remove()
    return torch.cat(captured, dim=0).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--lora-path", type=Path, default=None,
                   help="Optional PEFT adapter (LoRA regime). Omit for vanilla model.")
    p.add_argument("--facts", type=Path, default=Path("data/pii_facts/v1_facts.jsonl"))
    p.add_argument("--n-per-class", type=int, default=200)
    p.add_argument("--layer-stride", type=int, default=4,
                   help="Probe every Nth layer (Llama 32 layers; stride 4 → 8 probes)")
    p.add_argument("--out", type=Path, default=Path("results/probe_pii_layerwise.json"))
    args = p.parse_args()

    print(f"[probe] loading {args.model_name} (lora_path={args.lora_path})")
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if args.lora_path is not None:
        from peft import PeftModel
        print(f"[probe] loading LoRA: {args.lora_path}")
        model = PeftModel.from_pretrained(model, str(args.lora_path))
    model.eval()

    print(f"[probe] reading PII from {args.facts}")
    all_recs = read_jsonl(args.facts)
    forget_ids = {f"pii-{i:05d}" for i in range(1000)}
    forget_recs = [r for r in all_recs if r.id in forget_ids]
    retain_recs = [r for r in all_recs if r.id not in forget_ids]
    print(f"[probe] forget={len(forget_recs)}, retain={len(retain_recs)}")

    rng = np.random.default_rng(0)
    f_idx = rng.choice(len(forget_recs), size=args.n_per_class, replace=False)
    r_idx = rng.choice(len(retain_recs), size=args.n_per_class, replace=False)
    forget_prompts = build_query_prompts([forget_recs[i] for i in f_idx], args.n_per_class)
    retain_prompts = build_query_prompts([retain_recs[i] for i in r_idx], args.n_per_class)
    print(f"[probe] built {len(forget_prompts)} forget + {len(retain_prompts)} retain prompts")

    layers = get_decoder_layers(model)
    n_layers = len(layers)
    layer_indices = list(range(0, n_layers, args.layer_stride))
    if (n_layers - 1) not in layer_indices:
        layer_indices.append(n_layers - 1)
    print(f"[probe] probing layers: {layer_indices} (of {n_layers} total)")

    results = {
        "model": args.model_name,
        "lora_path": str(args.lora_path) if args.lora_path else None,
        "n_per_class": args.n_per_class,
        "n_layers_total": n_layers,
        "layers_probed": layer_indices,
        "per_layer": {},
    }

    for layer_idx in layer_indices:
        print(f"\n[probe] === layer {layer_idx} ===")
        X_f = collect_hidden_states(model, tok, forget_prompts, layers[layer_idx])
        X_r = collect_hidden_states(model, tok, retain_prompts, layers[layer_idx])
        X = np.concatenate([X_f, X_r], axis=0)
        y = np.concatenate([np.ones(len(X_f)), np.zeros(len(X_r))], axis=0)
        print(f"[probe]   X.shape={X.shape}, y mean={y.mean():.3f}")

        clf = LogisticRegression(max_iter=2000, C=1.0)
        scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
        acc_mean = float(scores.mean())
        acc_std = float(scores.std())
        print(f"[probe]   5-fold CV accuracy = {acc_mean:.4f} ± {acc_std:.4f}")
        results["per_layer"][str(layer_idx)] = {
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "scores": scores.tolist(),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\n[probe] wrote {args.out}")
    print("\n=== SUMMARY ===")
    print(f"{'layer':<8} {'acc':<8} {'std':<8}")
    for li, m in results["per_layer"].items():
        print(f"{li:<8} {m['accuracy_mean']:<8.4f} {m['accuracy_std']:<8.4f}")
    chance = 0.5
    max_acc = max(m["accuracy_mean"] for m in results["per_layer"].values())
    print(f"\nchance={chance:.3f}; max layer accuracy={max_acc:.4f}")
    if max_acc < 0.60:
        print("→ Linearly inseparable; LEACE null is LEGITIMATE ontology finding.")
    elif max_acc < 0.75:
        print("→ Marginal separability; LEACE null may be hyperparameter-sensitive.")
    else:
        print("→ STRONG linear separability; LEACE hyperparameters likely wrong.")


if __name__ == "__main__":
    main()
