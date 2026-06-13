#!/usr/bin/env bash
# R17 ShareGPT-200 q_base refit runner.
#
# Usage:
#     bash experiments/eA_tail_latency/run_eA_sharegpt.sh \
#         --num_prompts 200
#
# Pulls $num_prompts ShareGPT first-turn conversations via the
# in-tree seer.trace.datasets._load_sharegpt loader, runs the
# eA tail-latency pipeline with --disable_prefix_cache (IO-free
# regime), and emits the per-step latency cache used by
# q_base_refit_sharegpt.py.
#
# Requires: 1x A100, Llama-2-7B weights, datasets>=2.19.
set -euo pipefail

NUM_PROMPTS="${1:-200}"
WORKLOAD="${WORKLOAD:-sharegpt}"
SEEDS="${SEEDS:-0 1 2}"
OUT_DIR="experiments/eC_bound_tightness/results"
CACHE="$OUT_DIR/per_step_${WORKLOAD}_n${NUM_PROMPTS}.json"

mkdir -p "$OUT_DIR"

echo "[run_eA_sharegpt] workload=$WORKLOAD n=$NUM_PROMPTS seeds=$SEEDS"
echo "[run_eA_sharegpt] cache target: $CACHE"

# Step 1: run eA on ShareGPT-200, IO-free; output goes to a sibling
# results_sharegpt_n${NUM_PROMPTS}/ directory.
RES_RAW="experiments/eA_tail_latency/results_${WORKLOAD}_n${NUM_PROMPTS}"
mkdir -p "$RES_RAW"

for seed in $SEEDS; do
    python -m experiments.eA_tail_latency.run \
        --workload "$WORKLOAD" \
        --num_prompts "$NUM_PROMPTS" \
        --seed "$seed" \
        --budget 0.20 \
        --disable_prefix_cache \
        --out_dir "$RES_RAW"
done

# Step 2: aggregate per-step latencies across seeds into the cache
# expected by q_base_refit_sharegpt.py.
python - <<PY
import json, pathlib
res_dir = pathlib.Path("$RES_RAW")
all_per_step = []
for p in sorted(res_dir.glob("*.json")):
    d = json.loads(p.read_text())
    for rec in d.get("records", []) or [d]:
        ps = rec.get("per_step_us") or rec.get("per_step_base_us") or []
        all_per_step.extend(float(x) for x in ps if x == x)
out = {
    "workload": "$WORKLOAD",
    "num_prompts": $NUM_PROMPTS,
    "n_per_step_samples": len(all_per_step),
    "per_step_us": all_per_step,
}
pathlib.Path("$CACHE").write_text(json.dumps(out))
print(f"[run_eA_sharegpt] wrote $CACHE with {len(all_per_step)} per-step samples")
PY

# Step 3: invoke the q_base refit estimator on the cache.
python -m experiments.eC_bound_tightness.q_base_refit_sharegpt \
    --num_prompts "$NUM_PROMPTS" --workload "$WORKLOAD"

echo "[run_eA_sharegpt] complete; see $CACHE and $OUT_DIR/q_base_refit_sharegpt.{json,tex}"
