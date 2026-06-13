#!/usr/bin/env bash
# Reviewer M2 mitigation: LongBench narrative_qa as a prefix-share-WEAK
# workload. Mooncake's prefix-share invariant is what lets vLLM's
# prefix cache hide most of SEER's prefetch advantage; a fresh-prompt
# workload (each request unique, no shared prefix) is the regime
# where SEER should win even with prefix cache ON.
#
# Used in §6.8 to argue: "SEER's gain on real vLLM is conditional
# on a workload where the prefix cache *cannot* absorb our prefetch
# decision — narrative_qa is that workload."
#
#   CUDA_VISIBLE_DEVICES=1 bash experiments/eA_tail_latency/run_longbench_freshprompts.sh
#
# Per-policy wall time ≈ 8-10 min (50 prompts × 64 tokens × Llama-2-7B).
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL=${MODEL:-meta-llama/Llama-2-7b-hf}
LAP=${LAP:-checkpoints/lap_prod_dyn.onnx}
WORKLOAD=${WORKLOAD:-longbench:narrativeqa}
CONTEXT=${CONTEXT_LENGTH:-8192}
NUM=${NUM_REQUESTS:-50}
MAX_NEW=${MAX_NEW_TOKENS:-64}
BUDGET=${BUDGET:-0.20}
OUT_DIR=${OUT_DIR:-experiments/eA_tail_latency/results_longbench_fresh}
POLICIES=${POLICIES:-"full streaming h2o seer"}
SEEDS=${SEEDS:-"0 1 2"}

mkdir -p "$OUT_DIR"

for seed in $SEEDS; do
  for policy in $POLICIES; do
    tag="${policy}_b020_s${seed}"
    out="$OUT_DIR/$tag.json"
    if [ -f "$out" ]; then
      echo "[lb-fresh] skip $tag"; continue
    fi
    extra=()
    [ "$policy" = "seer" ] && extra=(--lap "$LAP")
    echo "[lb-fresh] $tag (n=$NUM, max_new=$MAX_NEW, workload=$WORKLOAD)"
    python -m seer.eval.runner \
        --model "$MODEL" \
        --policy "$policy" \
        --workload "$WORKLOAD" \
        --context_length "$CONTEXT" \
        --num_requests "$NUM" \
        --max_new_tokens "$MAX_NEW" \
        --slo "P99=50ms" \
        --hbm_budget "$BUDGET" \
        --seed "$seed" \
        --metrics tpot_p50,tpot_p99,tpot_p999,miss_ratio,quality \
        --out "$out" \
        "${extra[@]}"
  done
done

echo "[lb-fresh] done. Bootstrap CIs:"
echo "  PYTHONPATH=. python seer/eval/bootstrap.py "
echo "      experiments/eA_tail_latency/results_longbench_fresh/*_b020_s*.json "
echo "      --n_resamples 3000 "
echo "      --out_csv experiments/eA_tail_latency/results_longbench_fresh/bootstrap_ci.csv"
