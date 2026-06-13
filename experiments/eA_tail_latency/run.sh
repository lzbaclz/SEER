#!/usr/bin/env bash
# eA: tail-latency main figure -- EXTENDED-SCOPE configuration.
#
# IMPORTANT: this script's defaults (Llama-3-8B + ShareGPT-200) are
# NOT the paper-headline configuration. The paper §VI headline runs
# Llama-2-7B + Mooncake-24 -- driven from the top-level Makefile
# (PROD_MODEL / EA_WORKLOAD / EA_NUM in `make eA`). The extended
# Llama-3 + ShareGPT-200 sweep is queued as future work; see
# todo_atc.md~B for the larger workload run.
#
# To reproduce the paper-headline cells: `make eA` (uses Llama-2-7B +
# mooncake + 24 prompts).
# To reproduce the extended sweep: invoke this script directly.
#
# For local smoke runs use POLICIES="h2o seer" BUDGETS="0.2".
set -euo pipefail
cd "$(dirname "$0")"

MODEL=${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}
LAP=${LAP:-../../checkpoints/lap_llama3_8b.onnx}
WORKLOAD=${WORKLOAD:-sharegpt}
CONTEXT=${CONTEXT_LENGTH:-8192}
SLO=${SLO:-P99=50ms}
NUM=${NUM_REQUESTS:-200}
OUT_DIR=${OUT_DIR:-results}
POLICIES=${POLICIES:-"full streaming h2o snapkv quest recency seer"}
BUDGETS=${BUDGETS:-"0.10 0.20 0.40 0.80"}

mkdir -p "$OUT_DIR"

for budget in $BUDGETS; do
  for policy in $POLICIES; do
    # full cache is budget-invariant; only run once
    if [ "$policy" = "full" ] && [ "$budget" != "0.20" ]; then
      continue
    fi
    tag="${policy}_b$(echo "$budget" | tr -d .)"
    out="$OUT_DIR/$tag.json"
    if [ -f "$out" ]; then
      echo "[eA] skip $tag"
      continue
    fi
    extra=()
    [ "$policy" = "seer" ] && extra=(--lap "$LAP")
    echo "[eA] $tag"
    python -m seer.eval.runner \
        --model "$MODEL" \
        --policy "$policy" \
        --workload "$WORKLOAD" \
        --context_length "$CONTEXT" \
        --num_requests "$NUM" \
        --slo "$SLO" \
        --hbm_budget "$budget" \
        --metrics tpot_p50,tpot_p99,tpot_p999,miss_ratio,quality \
        --out "$out" \
        "${extra[@]}"
  done
done

python analyze.py "$OUT_DIR" || true
echo "[eA] sweep done."
