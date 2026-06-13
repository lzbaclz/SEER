#!/usr/bin/env bash
# P3-close: production-default HBM savings sweep.
#
# Claim to validate: on the production-default vLLM config (where SEER /
# H2O / Streaming / Full all tie at chat-miss = 0.0070 because vLLM's
# prefix cache absorbs the policy gap), SEER reaches the same SLO at a
# *smaller* HBM budget. If `full` at b=1.0 gives 0.0070 and SEER at b=φ_*
# gives 0.0070, the operator saves (1 - φ_*) × HBM bytes.
#
# We sweep $hbm \in \{0.10, 0.15, 0.20, 0.30, 0.50, 1.0\}$ for SEER and
# Full, all other policies for completeness. The Pareto break-even point
# is the smallest φ at which SEER's chat-miss ≤ Full's at φ=1.0 within
# Wilson 95% CI.
#
# Wall time: 6 budgets × 4 policies × 3 seeds = 72 cells. At ~3 min/cell
# (analytical mode, 50 prompts × 64 tokens) = ~3.6 h on A100.
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL=${MODEL:-meta-llama/Llama-2-7b-hf}
LAP=${LAP:-checkpoints/lap_prod_tiny.pt}
WORKLOAD=${WORKLOAD:-mooncake}
CONTEXT=${CONTEXT_LENGTH:-2048}
NUM=${NUM_REQUESTS:-50}
MAX_NEW=${MAX_NEW_TOKENS:-64}
OUT_DIR=${OUT_DIR:-experiments/eA_tail_latency/results_p3_hbm_savings}
POLICIES=${POLICIES:-"full streaming h2o seer"}
BUDGETS=${BUDGETS:-"0.10 0.15 0.20 0.30 0.50 1.00"}
SEEDS=${SEEDS:-"0 1 2"}

mkdir -p "$OUT_DIR"

total=0
for s in $SEEDS; do for b in $BUDGETS; do for p in $POLICIES; do total=$((total + 1)); done; done; done
echo "[P3-savings] grid size: $total cells (~$((total * 3 / 60)) h)"

n=0
for seed in $SEEDS; do
  for budget in $BUDGETS; do
    for policy in $POLICIES; do
      n=$((n + 1))
      # Tag without %03d to avoid bash interpreting "010" as octal.
      b_tag=$(printf "%.2f" "$budget" | tr -d '.')
      tag="${policy}_b${b_tag}_s${seed}"
      out="$OUT_DIR/$tag.json"
      if [ -f "$out" ]; then
        echo "[P3-savings][$n/$total] skip $tag"
        continue
      fi
      extra=()
      [ "$policy" = "seer" ] && extra=(--lap "$LAP")
      echo "[P3-savings][$n/$total] $tag"
      python -m seer.eval.runner \
          --model "$MODEL" \
          --policy "$policy" \
          --workload "$WORKLOAD" \
          --context_length "$CONTEXT" \
          --num_requests "$NUM" \
          --max_new_tokens "$MAX_NEW" \
          --slo "P99=50ms" \
          --hbm_budget "$budget" \
          --seed "$seed" \
          --skip_prewarm \
          --metrics tpot_p50,tpot_p99,tpot_p999,miss_ratio,quality \
          --out "$out" \
          "${extra[@]}" 2>&1 | tail -1
    done
  done
done

echo "[P3-savings] done. Aggregate with aggregate_p3.py"
