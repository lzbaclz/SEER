#!/usr/bin/env bash
# 修改 7 — RULER 5.4× claim source: re-run eA on the RULER 8K-needle
# workload so the abstract / §6.3 SnapKV P999=263 / SEER 48 claim has
# a committed source JSON. Reviewer flagged this as m6.
#
# Time: ~30 min on A100 (50 prompts × 5 policies × 2 budgets).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
LAP=${LAP:-../../checkpoints/lap_prod_dyn.onnx}
mkdir -p results_ruler
for budget in 0.10 0.20; do
  for policy in full streaming snapkv quest h2o seer; do
    out="results_ruler/${policy}_b$(echo "$budget" | tr -d '.').json"
    [ -f "$out" ] && { echo "[ruler] skip $policy b=$budget"; continue; }
    extra=""
    [ "$policy" = "seer" ] && extra="--lap $LAP"
    echo "[ruler] $policy b=$budget"
    $PY -m seer.eval.runner \
      --model meta-llama/Llama-2-7b-hf \
      --workload ruler --context_length 8192 \
      --policy "$policy" $extra \
      --hbm_budget "$budget" \
      --num_requests 50 --max_new_tokens 64 \
      --slo "P99=50ms" --out "$out"
  done
done
echo "[ruler] done — generate paper table:"
echo "  python aggregate_j7.py --in_dir results_ruler --out_tex results_ruler/table.tex"
