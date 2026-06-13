#!/usr/bin/env bash
# Parallel multi-GPU sweep driver for eA.
#
# Distributes (policy, budget) work across $NGPUS GPUs by setting
# CUDA_VISIBLE_DEVICES per worker. Each worker is sequential within
# itself; concurrency comes from the GPU partitioning.
set -euo pipefail
cd "$(dirname "$0")"

MODEL=${MODEL:-meta-llama/Llama-2-7b-hf}
LAP=${LAP:-../../checkpoints/lap_prod_dyn.onnx}
WORKLOAD=${WORKLOAD:-ruler}
CONTEXT=${CONTEXT_LENGTH:-2048}
SLO=${SLO:-P99=100ms}
NUM=${NUM_REQUESTS:-30}
MAX_NEW=${MAX_NEW_TOKENS:-32}
OUT_DIR=${OUT_DIR:-results}
POLICIES=${POLICIES:-"full streaming h2o snapkv quest recency seer"}
BUDGETS=${BUDGETS:-"0.10 0.20 0.40 0.80"}
NGPUS=${NGPUS:-2}

mkdir -p "$OUT_DIR"

# Build the full task list (policy, budget) skipping budget-invariant 'full'
TASKS=()
for budget in $BUDGETS; do
  for policy in $POLICIES; do
    if [ "$policy" = "full" ] && [ "$budget" != "0.20" ]; then continue; fi
    TASKS+=("$policy:$budget")
  done
done

echo "[eA-par] $((${#TASKS[@]})) tasks across $NGPUS GPUs"

# Distribute task indices to GPUs round-robin via subshells.
run_one() {
  local gpu=$1; shift
  local task=$1; shift
  local policy=${task%%:*}
  local budget=${task##*:}
  local tag="${policy}_b$(echo "$budget" | tr -d .)"
  local out="$OUT_DIR/$tag.json"
  if [ -f "$out" ]; then echo "[eA-par gpu$gpu] skip $tag"; return; fi
  local extra=()
  [ "$policy" = "seer" ] && extra=(--lap "$LAP")
  echo "[eA-par gpu$gpu] $tag"
  CUDA_VISIBLE_DEVICES=$gpu python -m seer.eval.runner \
      --model "$MODEL" \
      --policy "$policy" \
      --workload "$WORKLOAD" \
      --context_length "$CONTEXT" \
      --num_requests "$NUM" \
      --slo "$SLO" \
      --hbm_budget "$budget" \
      --max_new_tokens "$MAX_NEW" \
      --metrics tpot_p50,tpot_p99,tpot_p999,miss_ratio,quality \
      --out "$out" \
      "${extra[@]}" 2>&1 | grep -E "(saved →|F1=|Error|Traceback)" | tail -2
}

# Round-robin partition: GPU $i takes tasks $i, $i+NGPUS, $i+2*NGPUS, ...
pids=()
for ((i=0; i<NGPUS; i++)); do
  (
    for ((j=i; j<${#TASKS[@]}; j+=NGPUS)); do
      run_one "$i" "${TASKS[$j]}"
    done
  ) &
  pids+=($!)
done

# Wait for all GPU workers
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "[eA-par] sweep done."
python analyze.py "$OUT_DIR" || true
