#!/usr/bin/env bash
# eB: SLO sweep across deadlines D for the §6.4 plot.
set -euo pipefail
cd "$(dirname "$0")"

MODEL=${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}
LAP=${LAP:-../../checkpoints/lap_llama3_8b.onnx}
WORKLOAD=${WORKLOAD:-sharegpt}
NUM=${NUM_REQUESTS:-100}
BUDGET=${BUDGET:-0.2}
OUT_DIR=${OUT_DIR:-results}
mkdir -p "$OUT_DIR"

SLOS=(P99=25ms P99=35ms P99=50ms P99=75ms P99=100ms)
POLICIES=(full streaming h2o snapkv quest recency seer)

for slo in "${SLOS[@]}"; do
  for policy in "${POLICIES[@]}"; do
    tag="${policy}_$(echo "$slo" | tr '=:' '__')"
    out="$OUT_DIR/$tag.json"
    [ -f "$out" ] && { echo "[eB] skip $tag"; continue; }
    extra=()
    [ "$policy" = "seer" ] && extra=(--lap "$LAP")
    echo "[eB] $tag"
    python -m seer.eval.runner \
        --model "$MODEL" --policy "$policy" \
        --workload "$WORKLOAD" --num_requests "$NUM" \
        --slo "$slo" --hbm_budget "$BUDGET" \
        --out "$out" "${extra[@]}"
  done
done

echo "[eB] sweep done."
