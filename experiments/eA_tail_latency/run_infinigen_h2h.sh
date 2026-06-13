#!/usr/bin/env bash
#
# Head-to-head SEER vs InfiniGen on eA J.7 scaleup
# (P1-7, addresses R1's recommendation (2) and R3's O6).
#
# Reviewer feedback (May 2026): the paper lists InfiniGen as
# "closest prior work" but provides no direct head-to-head data on
# the same workload / simulator / IO model. This driver runs both at
# every active-mask fraction φ ∈ {0.10, 0.20, 0.40, 0.80} with all
# other knobs held identical (same trace, same measured-DMA IO mode,
# same n=200 prompts × 64 max-new × 3 seeds).
#
# Outputs:
#   experiments/eA_tail_latency/results_infinigen_h2h/
#     {policy}_b{btag}_s{seed}.json   (policy ∈ {seer, infinigen})
#     summary.json
#     table.tex
#
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL=${SEER_MODEL:-meta-llama/Llama-2-7b-hf}
LAP=${SEER_LAP:-checkpoints/lap_tinymlp_a100.pt}
OUT_DIR=experiments/eA_tail_latency/results_infinigen_h2h
mkdir -p "$OUT_DIR"

POLICIES=(seer infinigen)
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
        echo "[skip] $out exists"
        continue
      fi
      echo "[run]  pol=$pol budget=$budget seed=$seed"
      # InfiniGen does not use the LAP, but we still pass --lap to
      # keep the runner contract identical (the policy itself ignores it).
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

PYTHONPATH=. python experiments/eA_tail_latency/aggregate_j7.py \
  --in_dir "$OUT_DIR" \
  --out_tex "$OUT_DIR/table.tex" \
  --out_json "$OUT_DIR/summary.json"
echo "[done] $OUT_DIR/table.tex"
