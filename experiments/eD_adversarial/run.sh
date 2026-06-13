#!/usr/bin/env bash
# eD: run all 7 policies across the 5 adversarial-shift families.
# Each (policy, family) combo produces one JSON.
set -euo pipefail
cd "$(dirname "$0")"

MODEL=${MODEL:-Qwen/Qwen2.5-7B}
LAP=${LAP:-../../checkpoints/lap_smoke.onnx}
SLO=${SLO:-P99=50ms}
BUDGET=${BUDGET:-0.2}
N_PER_FAMILY=${N_PER_FAMILY:-50}
OUT_DIR=${OUT_DIR:-results}
PROMPTS_DIR=${PROMPTS_DIR:-prompts}
POLICIES=${POLICIES:-"streaming h2o snapkv quest recency seer"}

mkdir -p "$OUT_DIR" "$PROMPTS_DIR"

# 1) Generate prompts (idempotent if already present)
if [ ! -f "$PROMPTS_DIR/topic_switch.jsonl" ]; then
  python generate_prompts.py --out_dir "$PROMPTS_DIR" --n "$N_PER_FAMILY"
fi

# 2) For each (family, policy) — write a tiny custom driver that loads
# the JSONL file and feeds each prompt through seer.eval.runner.
# We use the runner's --workload synthetic + a side-channel by
# concatenating prompts; the simpler path is to feed prompts via a
# small script. For RTSS 2027 we add a proper --prompts_file flag to
# the runner; until then this skeleton is the entry point for that
# work.
echo "[eD] skeleton: per-family policy sweep is RTSS 2027 work."
echo "    Each (family, policy) pair will produce a JSON in $OUT_DIR."
echo "    Implementation note: extend seer.eval.runner with"
echo "    --prompts_file <jsonl> + per-prompt question/answer plumbing."
