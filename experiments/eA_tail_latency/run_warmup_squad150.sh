#!/usr/bin/env bash
# Q9: SQuAD150 baseline vs SEER+speculative-warmup comparison.
#
# Claim: SEER P999 = 179 ms (3.6× SLO) is the worst-case head, driven
# by hard semantic shifts. The speculative-warmup wrapper detects
# LAP confidence collapse + miss-rate spike and forces a 1-step
# pre-fetch of the top-(budget + N_warmup) raw-LAP blocks. This
# replaces the unbounded miss step with a bounded warmup step
# (C_base + N_warmup × ℓ̄).
#
# Grid: 2 policies (seer, seer-warmup) × 3 seeds = 6 cells.
# Per-cell ~10 min on A100 (150 prompts × 64 tokens).
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL=${MODEL:-meta-llama/Llama-2-7b-hf}
LAP=${LAP:-checkpoints/lap_prod_tiny.pt}
NUM=${NUM_REQUESTS:-150}
MAX_NEW=${MAX_NEW_TOKENS:-64}
BUDGET=${BUDGET:-0.20}
SEEDS=${SEEDS:-"0 1 2"}
OUT_DIR=${OUT_DIR:-experiments/eA_tail_latency/results_q9_warmup}

mkdir -p "$OUT_DIR"
echo "[Q9-warmup] grid: 2 policies × 3 seeds (SQuAD150, hbm=$BUDGET)"

export SEER_MOONCAKE_SYNTHESISER=squad
for seed in $SEEDS; do
    for policy in seer seer-warmup; do
        tag="${policy}_s${seed}"
        out="$OUT_DIR/$tag.json"
        if [ -f "$out" ]; then echo "[Q9-warmup] skip $tag"; continue; fi
        echo "[Q9-warmup] $tag"
        python -m seer.eval.runner \
            --model "$MODEL" \
            --policy "$policy" \
            --lap "$LAP" \
            --workload mooncake \
            --num_requests "$NUM" \
            --max_new_tokens "$MAX_NEW" \
            --slo "P99=50ms" \
            --hbm_budget "$BUDGET" \
            --seed "$seed" \
            --skip_prewarm \
            --metrics tpot_p50,tpot_p99,tpot_p999,miss_ratio,quality \
            --out "$out" 2>&1 | tail -2
    done
done
echo "[Q9-warmup] done. Aggregate with aggregate_q9_warmup.py"
