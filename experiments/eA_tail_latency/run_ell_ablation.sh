#!/usr/bin/env bash
# 修改 2.2 — ℓ̄ ablation: re-run eA at the headline budget with measured
# pinned-host ℓ̄=34µs (J.6 probe) instead of the vendor-DRAM 200µs proxy.
# Goal: show the policy ranking is preserved when the analytical IO
# penalty is calibrated against real DMA latency, addressing reviewer M3
# (circular-validation concern on the synthetic IO penalty).
#
# Time: ~30 min on A100 (200 prompts × 6 policies × 2 budgets).
# Run from repo root: bash experiments/eA_tail_latency/run_ell_ablation.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
LAP=${LAP:-../../checkpoints/lap_prod_dyn.onnx}
mkdir -p results_ell34
for budget in 0.10 0.20 0.40 0.80; do
  for policy in full streaming snapkv quest h2o seer; do
    out="results_ell34/${policy}_b$(echo "$budget" | tr -d '.').json"
    [ -f "$out" ] && { echo "[ell34] skip $policy b=$budget"; continue; }
    extra=""
    [ "$policy" = "seer" ] && extra="--lap $LAP"
    echo "[ell34] $policy b=$budget"
    $PY -m seer.eval.runner \
      --model meta-llama/Llama-2-7b-hf \
      --workload mooncake \
      --policy "$policy" $extra \
      --hbm_budget "$budget" --ell_bar_us 34 \
      --num_requests 200 --max_new_tokens 64 \
      --slo "P99=50ms" --out "$out"
  done
done
echo "[ell34] done; aggregate via python aggregate_j7.py --in_dir results_ell34 --out_tex results_ell34/table.tex"
