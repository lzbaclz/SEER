#!/usr/bin/env bash
# Reduced-scale measured-DMA sweep for the May 2026 review round.
#
# Why reduced: full run_j7_measured.sh is ~17 GPU hours; this driver
# trades 3 seeds → 1, 200 prompts → 50, 64 tokens → 32, so the
# sweep completes in ~1.5 hours on a quiesced A100. The reduced run
# gives a representative P50/P99/miss point per (policy, budget) cell
# but lacks the across-seed CI of the headline run.
#
# Honest reporting: paper §6.eA carries a "(reduced)" badge in the
# caption when this script generated the cells; full-seed numbers
# remain in results_j7_scaleup/ from the earlier (analytical) run.
#
# Outputs:
#   experiments/eA_tail_latency/results_j7_measured_reduced/
#     {policy}_b{btag}_s0.json
#     summary.json + table.tex
#
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=${SEER_PY:-/home/lzq/miniconda3/envs/seer/bin/python}
MODEL=${SEER_MODEL:-meta-llama/Llama-2-7b-hf}
LAP=${SEER_LAP:-checkpoints/lap_prod_tiny.pt}
OUT_DIR=experiments/eA_tail_latency/results_j7_measured_reduced
mkdir -p "$OUT_DIR"

POLICIES=(seer h2o streaming snapkv full)
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
