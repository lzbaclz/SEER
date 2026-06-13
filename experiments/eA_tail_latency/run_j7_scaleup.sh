#!/usr/bin/env bash
# eA J.7 scale-up: 200 prompts × 64 max-tokens × 3 seeds.
#
# The published eA table was n_requests=20. Reviewers asked for
# tighter confidence intervals; this script re-runs the headline
# (b=0.20) point at 10× the prompt count and 3 seeds, on the four
# policies that actually compete (full / streaming / h2o / seer).
# Quest, snapkv, recency are skipped here — eA's narrative compares
# SEER against the strongest tail-latency baseline (h2o) and a
# heuristic floor (streaming).
#
# Per-run wall time on Llama-2-7B fp16 + A100: ~5 min × 12 runs
# = ~1 h.
#
# Run:
#     CUDA_VISIBLE_DEVICES=1 bash experiments/eA_tail_latency/run_j7_scaleup.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL=${MODEL:-meta-llama/Llama-2-7b-hf}
LAP=${LAP:-checkpoints/lap_prod_dyn.onnx}
WORKLOAD=${WORKLOAD:-mooncake}
CONTEXT=${CONTEXT_LENGTH:-2048}
NUM=${NUM_REQUESTS:-200}
MAX_NEW=${MAX_NEW_TOKENS:-64}
BUDGET=${BUDGET:-0.20}
OUT_DIR=${OUT_DIR:-experiments/eA_tail_latency/results_j7_scaleup}
POLICIES=${POLICIES:-"full streaming h2o seer"}
SEEDS=${SEEDS:-"0 1 2"}

mkdir -p "$OUT_DIR"

for seed in $SEEDS; do
  for policy in $POLICIES; do
    tag="${policy}_b020_s${seed}"
    out="$OUT_DIR/$tag.json"
    if [ -f "$out" ]; then
      echo "[J.7] skip $tag"
      continue
    fi
    extra=()
    [ "$policy" = "seer" ] && extra=(--lap "$LAP")
    echo "[J.7] $tag (n=$NUM, max_new=$MAX_NEW)"
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

echo "[J.7] done. Aggregate with experiments/eA_tail_latency/aggregate_j7.py"
