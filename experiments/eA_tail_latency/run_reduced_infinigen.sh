#!/usr/bin/env bash
# Reduced InfiniGen h2h vs SEER on the J.7 measured-DMA workload.
# Full sweep is ~5 GPU hours (3 seeds × 200 prompts × 64 tokens × 2 policies × 4 budgets);
# this driver runs 1 seed × 50 prompts × 32 tokens (≈30 min) so the
# head-to-head closes the M12 reviewer concern within one session.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=${SEER_PY:-/home/lzq/miniconda3/envs/seer/bin/python}
MODEL=${SEER_MODEL:-meta-llama/Llama-2-7b-hf}
LAP=${SEER_LAP:-checkpoints/lap_prod_tiny.pt}
OUT_DIR=experiments/eA_tail_latency/results_infinigen_h2h_reduced
mkdir -p "$OUT_DIR"

POLICIES=(seer infinigen)
BUDGETS=(0.10 0.20 0.40 0.80)
SEED=0
NUM_REQ=${SEER_NUM_REQ:-50}
MAX_NEW=${SEER_MAX_NEW:-32}

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for pol in "${POLICIES[@]}"; do
  for budget in "${BUDGETS[@]}"; do
    btag=$(echo "$budget" | tr -d '.')
    out="$OUT_DIR/${pol}_b${btag}_s${SEED}.json"
    if [ -f "$out" ]; then
      echo "[skip] $out"
      continue
    fi
    echo "[run]  pol=$pol budget=$budget"
    PYTHONPATH=. "$PY" -m seer.eval.runner \
      --model "$MODEL" \
      --workload mooncake \
      --num_requests "$NUM_REQ" \
      --max_new_tokens "$MAX_NEW" \
      --policy "$pol" \
      --hbm_budget "$budget" \
      --io_mode measured-dma \
      --slo "P99=50ms" \
      --seed "$SEED" \
      ${LAP:+--lap "$LAP"} \
      --out "$out"
  done
done

PYTHONPATH=. "$PY" experiments/eA_tail_latency/aggregate_j7.py \
  --in_dir "$OUT_DIR" \
  --out_tex "$OUT_DIR/table.tex" \
  --out_json "$OUT_DIR/summary.json" 2>&1 || true
echo "[done] $OUT_DIR/table.tex"
