#!/usr/bin/env bash
# P2: J.7-grade scale-up (review C3 / R5: "P999 unstable / sample size").
#
# Goal: tighten the P999 confidence interval from ±2 ms (n=12.6K, J.7) to
# ±0.5 ms. The 95% bootstrap CI half-width scales as ~σ/sqrt(n_tail).
# For chat P999 the tail population is 0.1% of n, so:
#   J.7:    n=12.6K → n_tail=12.6 → CI half-width ≈ 1.5σ (≈2 ms)
#   P2:     n=192K → n_tail=192   → CI half-width ≈ 0.12σ (≈0.16 ms)
#
# Grid: 4 policies × 3 seeds × 1000 prompts × 64 tokens = 192K steps each.
# Per cell: ~50 min on A100. 12 cells × 50 min = ~10 hours.
#
# Run:
#     CUDA_VISIBLE_DEVICES=0 bash experiments/eA_tail_latency/run_p2_scaleup.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL=${MODEL:-meta-llama/Llama-2-7b-hf}
LAP=${LAP:-checkpoints/lap_prod_tiny.pt}
WORKLOAD=${WORKLOAD:-mooncake}
CONTEXT=${CONTEXT_LENGTH:-2048}
NUM=${NUM_REQUESTS:-1000}
MAX_NEW=${MAX_NEW_TOKENS:-64}
BUDGET=${BUDGET:-0.20}
OUT_DIR=${OUT_DIR:-experiments/eA_tail_latency/results_p2_scaleup}
POLICIES=${POLICIES:-"full streaming h2o seer"}
SEEDS=${SEEDS:-"0 1 2"}

mkdir -p "$OUT_DIR"

total=0
for seed in $SEEDS; do for p in $POLICIES; do total=$((total + 1)); done; done
echo "[P2] grid size: $total cells (target ~10 h)"

done_n=0
for seed in $SEEDS; do
  for policy in $POLICIES; do
    done_n=$((done_n + 1))
    tag="${policy}_b020_s${seed}"
    out="$OUT_DIR/$tag.json"
    if [ -f "$out" ]; then
      echo "[P2][$done_n/$total] skip $tag"
      continue
    fi
    extra=()
    [ "$policy" = "seer" ] && extra=(--lap "$LAP")
    echo "[P2][$done_n/$total] $tag (n=$NUM)"
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
        --skip_prewarm \
        --metrics tpot_p50,tpot_p99,tpot_p999,miss_ratio,quality \
        --out "$out" \
        "${extra[@]}" 2>&1 | tail -1
  done
done

echo "[P2] done."
