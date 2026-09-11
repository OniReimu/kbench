#!/usr/bin/env bash
# One-command reproduction of the K-Bench paper tables.
# Usage: bash reproduce.sh {smoke|prep|topology|interfaces|substrate|all}
#   smoke : 30-second CPU-only check — no model, no dataset download (one-time numpy
#           fetch on first run). Scores a tiny bundled
#           fixture through the real scorer and prints the multi-channel report card.
# Requires: the kbench environment -- the Docker image, or a local pinned env
#   (`uv pip install -r cross_model_pinned_requirements.txt`); see INSTALL.md.
# See INSTALL.md for the one-time prerequisites (RAG index, PII corpus + LoRA,
# external unlearning libs). Run `reproduce.sh prep` first.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_LLAMA="meta-llama/Llama-3.1-8B-Instruct"
MODEL_QWEN="Qwen/Qwen2.5-7B-Instruct"
MODEL_MISTRAL="mistralai/Mistral-7B-Instruct-v0.3"
SEEDS=(0 137 271)
N=200
TARGET="${1:-help}"

result_tag () {  # substrate method subset seed -> public result filename stem
  python3 "$SCRIPT_DIR/scripts/reproduction_names.py" "$1" "$2" "$3" "$4"
}

run_cell () {  # model substrate method
  # Emits results/llama_<substrate>_<method>_<subset>_seed<s>.jsonl for BOTH the
  # forget and retain subsets, matching FILE_RE in scripts/09_k_verdict_v2.py.
  local model="$1" sub="$2" method="$3"
  for subset in forget retain; do
    for s in "${SEEDS[@]}"; do
      local tag; tag="$(result_tag "$sub" "$method" "$subset" "$s")"
      uv run python scripts/02_baseline_leakage.py \
        --model "$model" --substrate "$sub" --unlearn "$method" \
        --query-subset "$subset" --n-sample "$N" --seed "$s" \
        --out-jsonl "results/${tag}.jsonl" \
        --out-summary "results/${tag}.json"
    done
  done
}

# Tests source the exact public naming function without running a target.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  return 0
fi

case "$TARGET" in
  smoke)      # 30-second CPU-only showcase — no model, no credentials, no heavy env.
              # Reuse an installed numpy when available; otherwise fetch only numpy in
              # an isolated uv env, skipping the torch/transformers/faiss build.
    if python3 -c "import numpy" >/dev/null 2>&1; then
      python3 scripts/smoke_report.py
    else
      uv run --no-project --with numpy python scripts/smoke_report.py
    fi
    ;;
  prep)       # one-time prerequisites — see INSTALL.md for the full manual steps
    echo ">> [prep] build the production Wiki RAG index (HPC GPU, ~hours)"
    uv run python scripts/00_build_rag_index.py --config configs/rag_pilot.yaml
    echo ">> [prep] generate the synthetic PII corpus + query set (deterministic)"
    uv run python scripts/01_generate_pii.py \
      --n-facts 5000 --seed 0 --out-dir data/pii_facts --name v1
    echo ">> [prep] regenerate the v2.1 distractor pool (retain-adapter split) so"
    echo "   startup_audit passes -- see scripts/00d_build_distractor_pool.py"
    uv run python scripts/00d_build_distractor_pool.py
    echo ">> [prep] build the v2.1 substrate-isolation RAG indexes for C / R cells"
    echo "   (GPU embedding; injects PII into the production wiki corpus)"
    uv run python scripts/00b_inject_pii_corpus.py \
      --orig-passages data/wiki_index/passages.jsonl \
      --orig-embeddings-dir data/wiki_index/embeddings \
      --facts data/v21/bios_distractor.jsonl --n-forget 5000 \
      --out-dir data/wiki_index_v21_distractor/
    uv run python scripts/00b_inject_pii_corpus.py \
      --orig-passages data/wiki_index/passages.jsonl \
      --orig-embeddings-dir data/wiki_index/embeddings \
      --facts data/pii_facts/v1_facts.jsonl --n-forget 5000 \
      --out-dir data/wiki_index_v21_target_in/
    echo ">> [prep] LoRA targets (GPU + base-model download). Train the P-substrate"
    echo "   target LoRA and the distractor-only LoRA-D, then merge the target:"
    echo "     uv run python scripts/03_inject_pii.py --facts data/pii_facts/v1_facts.jsonl \\"
    echo "        --out-dir models/lora_v1 --epochs 5 --r 32 --alpha 64 --seed 0"
    echo "     uv run python scripts/03_inject_pii.py --facts data/v21/bios_distractor.jsonl \\"
    echo "        --out-dir models/v21_lora_d --epochs 5 --r 32 --alpha 64 --seed 0"
    echo "     uv run python scripts/20_merge_target.py   # -> models/target_merged (P substrate)"
    echo ">> [prep] external unlearning libs are NOT vendored. Clone them into"
    echo "   external/<x> as listed in INSTALL.md before running weight-based methods."
    echo ">> [prep] NOTE: all substrate assets are now buildable from this tree --"
    echo "   the P headline ('interfaces') and the C / R indexes + LoRA-D above."
    echo "   The two LoRA trains need a GPU and a base-model download, so they are"
    echo "   printed as commands rather than auto-run."
    ;;
  topology)   # baseline leak-topology table, Llama, all substrates
    for sub in P C R-text R-struct; do run_cell "$MODEL_LLAMA" "$sub" none; done
    uv run python scripts/09_k_verdict_v2.py --results-dir results --out docs/verdict_topology.md
    ;;
  interfaces) # five-interface comparison on Llama P (tab:benchmark_compare)
    echo ">> agentic K-Bench on weight-based checkpoints"
    for unl in none eco star leace cha o3; do run_cell "$MODEL_LLAMA" P "$unl"; done
    uv run python scripts/09_k_verdict_v2.py --results-dir results --out docs/verdict_interfaces.md
    echo ">> headline K-Score leaderboard (ECO reference vs the five defenses; App. table)"
    uv run python scripts/kscore.py P llama
    echo ">> faithful TOFU / MUSE / WMDP probes (see scripts 21/22/24)"
    echo "   run: scripts/21_tofu_faithful.py, 22_muse_faithful.py, 24_wmdp_mcq.py per checkpoint"
    echo "   then: scripts/23_aggregate_benchmark.py (merges the probe shards into the table;"
    echo "   needs the faithful_{tofu,muse}_* shards from the probes above)"
    ;;
  substrate)  # substrate blindness across families (tab:substrate_blindness)
    # Only the Llama cells use the llama_ naming that 09_k_verdict_v2.py discovers.
    # The cross-model rows (Qwen / Mistral) are scored by kscore_crossmodel.py against per-arch baselines
    # (v77app_P_none_qwen / v77app_P_none_mistral). FILE_RE in
    # 09_k_verdict_v2.py is single-model by design (its nested
    # substrate/method/seed dict has no model axis), so those families are
    # aggregated independently. See docs/COMPUTE.md ("Cross-model block").
    for sub in C R-text R-struct; do run_cell "$MODEL_LLAMA" "$sub" none; done
    uv run python scripts/09_k_verdict_v2.py --results-dir results --out docs/verdict_substrate.md
    echo ">> cross-model rows (Qwen / Mistral) — emit under their own prefixes"
    echo "   and aggregate separately; see docs/COMPUTE.md (Cross-model block)."
    for model in "$MODEL_QWEN" "$MODEL_MISTRAL"; do
      for sub in C R-text R-struct; do
        echo "   would run: $model / $sub (prefix per docs/COMPUTE.md)"
      done
    done
    ;;
  all)
    bash "$0" topology; bash "$0" interfaces; bash "$0" substrate
    echo ">> released Llama targets done; cross-model and faithful-probe commands"
    echo "   printed above remain separate (full paper study: ~500 GPU-h; see docs/COMPUTE.md)"
    ;;
  *)
    echo "Usage: bash reproduce.sh {smoke|prep|topology|interfaces|substrate|all}"; exit 1
    ;;
esac
