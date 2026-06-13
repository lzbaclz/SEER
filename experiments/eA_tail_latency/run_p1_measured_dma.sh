#!/usr/bin/env bash
# P1: measured-DMA at scale — RTSS reviewer R5/O1/M3 ("circular validation").
#
# Compares analytical Lemma-1 IO penalty against real cudaMemcpyAsync HtoD
# timing on a pinned-host scratch (J.6 transport). If the policy ranking
# is invariant across modes, the bound's IO term is validated against the
# production substrate rather than against its own analytical form.
#
# Grid: 3 policies × 3 budgets × 2 io_modes × 3 seeds = 54 cells.
# Per-cell: 100 prompts × 64 tokens ≈ 6.4K decode steps.
# Wall time: ~5 h on a single A100 (measured-dma path is ~2× slower than
# analytical due to the synchronous DMA timing in the IO probe).
#
# Run:
#     CUDA_VISIBLE_DEVICES=0 bash experiments/eA_tail_latency/run_p1_measured_dma.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL=${MODEL:-meta-llama/Llama-2-7b-hf}
LAP=${LAP:-checkpoints/lap_prod_tiny.pt}
WORKLOAD=${WORKLOAD:-mooncake}
CONTEXT=${CONTEXT_LENGTH:-2048}
NUM=${NUM_REQUESTS:-50}
MAX_NEW=${MAX_NEW_TOKENS:-64}
OUT_DIR=${OUT_DIR:-experiments/eA_tail_latency/results_p1_measured_dma}
POLICIES=${POLICIES:-"seer h2o streaming"}
BUDGETS=${BUDGETS:-"0.10 0.20 0.30"}
SEEDS=${SEEDS:-"0 1 2"}
IO_MODES=${IO_MODES:-"analytical measured-dma"}

mkdir -p "$OUT_DIR"

total=0
done_n=0
for seed in $SEEDS; do
  for budget in $BUDGETS; do
    for mode in $IO_MODES; do
      for policy in $POLICIES; do
        total=$((total + 1))
      done
    done
  done
done
echo "[P1] grid size: $total cells"

for seed in $SEEDS; do
  for budget in $BUDGETS; do
    for mode in $IO_MODES; do
      for policy in $POLICIES; do
        done_n=$((done_n + 1))
        # mode suffix: 'a' for analytical, 'd' for measured-dma
        msuf="a"; [ "$mode" = "measured-dma" ] && msuf="d"
        b_tag=$(printf "%.2f" "$budget" | sed 's/\.//')
        tag="${policy}_b${b_tag}_${msuf}_s${seed}"
        out="$OUT_DIR/$tag.json"
        if [ -f "$out" ]; then
          echo "[P1][$done_n/$total] skip $tag (exists)"
          continue
        fi
        extra=()
        [ "$policy" = "seer" ] && extra=(--lap "$LAP")
        echo "[P1][$done_n/$total] $tag"
        python -m seer.eval.runner \
            --model "$MODEL" \
            --policy "$policy" \
            --workload "$WORKLOAD" \
            --context_length "$CONTEXT" \
            --num_requests "$NUM" \
            --max_new_tokens "$MAX_NEW" \
            --slo "P99=50ms" \
            --hbm_budget "$budget" \
            --io_mode "$mode" \
            --seed "$seed" \
            --skip_prewarm \
            --metrics tpot_p50,tpot_p99,tpot_p999,miss_ratio,quality \
            --out "$out" \
            "${extra[@]}" 2>&1 | tail -1
      done
    done
  done
done

echo "[P1] done. Aggregate with experiments/eA_tail_latency/aggregate_p1.py"
