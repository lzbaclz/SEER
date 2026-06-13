#!/usr/bin/env bash
#
# eA J.7 scaleup re-run under measured-DMA IO mode (P1-1, M1/M3 fix).
#
# Reviewer feedback (May 2026): the analytical IO penalty
# (Lemma-1 first-moment form injected into per-step latency)
# is circular validation — the bound being checked drives the
# very numbers the bound is fit on. This driver re-runs the eA
# scaleup with --io_mode measured-dma so per-step IO time is
# read off a real cudaMemcpyAsync HtoD probe instead.
#
# Cost: ~2x analytical wall-clock per request. With 200 prompts ×
# 64 max-new-tokens × 3 seeds × 7 policies × 4 budgets = ~17 GPU
# hours on a quiesced A100. Run in screen / tmux.
#
# Outputs:
#   experiments/eA_tail_latency/results_j7_measured/
#     {policy}_{budget}_s{seed}.json
#     summary.json (aggregated post-run)
#
# Reproduce:
#   bash experiments/eA_tail_latency/run_j7_measured.sh
#
set -euo pipefail

cd "$(dirname "$0")/../.."  # repo root

MODEL=${SEER_MODEL:-meta-llama/Llama-2-7b-hf}
LAP=${SEER_LAP:-checkpoints/lap_tinymlp_a100.pt}
OUT_DIR=experiments/eA_tail_latency/results_j7_measured
mkdir -p "$OUT_DIR"

POLICIES=(seer h2o streaming snapkv quest recency full)
BUDGETS=(0.10 0.20 0.40 0.80)
SEEDS=(0 1 2)
NUM_REQ=${SEER_NUM_REQ:-200}
MAX_NEW=${SEER_MAX_NEW:-64}

for seed in "${SEEDS[@]}"; do
  for pol in "${POLICIES[@]}"; do
    for budget in "${BUDGETS[@]}"; do
      btag=$(echo "$budget" | tr -d '.')
      out="$OUT_DIR/${pol}_b${btag}_s${seed}.json"
      if [ -f "$out" ]; then
        echo "[skip] $out already exists"
        continue
      fi
      echo "[run]  pol=$pol budget=$budget seed=$seed"
      PYTHONPATH=. python -m seer.eval.runner \
        --model "$MODEL" \
        --workload mooncake \
        --num_requests "$NUM_REQ" \
        --max_new_tokens "$MAX_NEW" \
        --policy "$pol" \
        --hbm_budget "$budget" \
        --io_mode measured-dma \
        --slo "P99=50ms" \
        --seed "$seed" \
        ${LAP:+--lap "$LAP"} \
        --out "$out"
    done
  done
done

# Aggregate (this matches the existing aggregate_j7.py contract).
echo "[done] all cells written, running aggregator..."
PYTHONPATH=. python experiments/eA_tail_latency/aggregate_j7.py \
  --in_dir "$OUT_DIR" \
  --out_tex "$OUT_DIR/table.tex" \
  --out_json "$OUT_DIR/summary.json"
echo "[done] $OUT_DIR/table.tex"
