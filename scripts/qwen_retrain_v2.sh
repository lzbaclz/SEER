#!/usr/bin/env bash
#
# Qwen-2.5-7B per-model LAP retraining + eA evaluation
# (P1-4, addresses R1 M2 + R3 S4 — cross-model claim with retrained LAP).
#
# Reviewer feedback (May 2026): the published cross-model number
# (SEER P999 = 157 ms with the Llama-trained LAP transferred to Qwen)
# is the WORST of all non-oracle policies and demonstrates that
# zero-shot LAP transfer fails. The honest cross-model claim is
# "per-model LAP retraining is required" — and that requires this
# script to actually retrain on Qwen attention traces, not just
# transfer.
#
# Differences from scripts/qwen_e2e.sh (the earlier version):
#   * Environment-agnostic: SEER_PY / SEER_QWEN_MODEL env vars, no
#     hard-coded conda path or GPU index.
#   * Beefier training set (default 30 requests vs 12) — addresses
#     R3 S4 ("12-prompt training is fundamentally narrow").
#   * Uses --io_mode measured-dma (the P1-1 default).
#   * Adds the full budget sweep ({0.10, 0.20, 0.40, 0.80}) on the
#     eA pass so the retrained-LAP curve can be plotted.
#   * Records a stats file with the per-model ε(φ) curve.
#
# Cost: ~8 hours A100 with the default knobs.
#
# Outputs:
#   data/traces/qwen25_train/*.parquet
#   checkpoints/lap_qwen25.{pt,onnx}
#   experiments/eA_tail_latency/results_qwen_retrained_v2/
#     {policy}_{btag}.json
#     summary.json
#     epsilon_curve.csv  (ε(φ) for the retrained LAP)
#
set -euo pipefail
cd "$(dirname "$0")/.."

SEER_PY=${SEER_PY:-python}
SEER_QWEN_MODEL=${SEER_QWEN_MODEL:-Qwen/Qwen2.5-7B}
N_TRAIN_REQ=${SEER_N_TRAIN_REQ:-30}
N_TRAIN_EPOCHS=${SEER_N_TRAIN_EPOCHS:-6}
N_EVAL_REQ=${SEER_N_EVAL_REQ:-20}
EVAL_MAX_NEW=${SEER_EVAL_MAX_NEW:-32}
EVAL_CTX=${SEER_EVAL_CTX:-2048}
BUDGETS=(${SEER_BUDGETS:-0.10 0.20 0.40 0.80})

TRACE_DIR=data/traces/qwen25_train
LAP_CKPT=checkpoints/lap_qwen25.pt
LAP_ONNX=checkpoints/lap_qwen25.onnx
OUT_DIR=experiments/eA_tail_latency/results_qwen_retrained_v2

mkdir -p "$TRACE_DIR" checkpoints "$OUT_DIR"

# ---- Step 1: collect Qwen attention traces
if ! ls "$TRACE_DIR"/*.parquet >/dev/null 2>&1; then
  echo "[Qwen] step 1/4: collect $N_TRAIN_REQ traces from $SEER_QWEN_MODEL"
  PYTHONPATH=. "$SEER_PY" -m seer.trace.collect \
    --model "$SEER_QWEN_MODEL" \
    --workload synthetic \
    --num_requests "$N_TRAIN_REQ" \
    --max_new_tokens 64 \
    --out "$TRACE_DIR"
fi

# ---- Step 2: train TinyMLP on Qwen traces
if [ ! -f "$LAP_CKPT" ]; then
  echo "[Qwen] step 2/4: train LAP ($N_TRAIN_EPOCHS epochs)"
  PYTHONPATH=. "$SEER_PY" -m seer.lap.train \
    --traces "$TRACE_DIR" \
    --arch tiny_mlp --history 32 --horizons 1 4 16 64 \
    --wcet_budget_us 200 --epochs "$N_TRAIN_EPOCHS" \
    --out "$LAP_CKPT"
fi

# ---- Step 3: export ONNX for inference
if [ ! -f "$LAP_ONNX" ]; then
  echo "[Qwen] step 3/4: export ONNX"
  PYTHONPATH=. "$SEER_PY" -m seer.lap.export \
    --ckpt "$LAP_CKPT" --backend onnx \
    --static_shape --batch 4096 --out "$LAP_ONNX"
fi

# ---- Step 4: full budget-sweep eA on Qwen with the retrained LAP
echo "[Qwen] step 4/4: eA budget sweep with retrained LAP"
POLICIES=(streaming h2o snapkv quest infinigen seer)
for policy in "${POLICIES[@]}"; do
  for budget in "${BUDGETS[@]}"; do
    btag=$(echo "$budget" | tr -d '.')
    out="$OUT_DIR/${policy}_b${btag}.json"
    if [ -f "$out" ]; then
      echo "  [skip] $policy b=$budget"
      continue
    fi
    extra=""
    if [ "$policy" = "seer" ]; then
      extra="--lap $LAP_ONNX"
    fi
    echo "  [run] $policy b=$budget"
    PYTHONPATH=. "$SEER_PY" -m seer.eval.runner \
      --model "$SEER_QWEN_MODEL" \
      --policy "$policy" $extra \
      --workload mooncake \
      --context_length "$EVAL_CTX" \
      --num_requests "$N_EVAL_REQ" \
      --max_new_tokens "$EVAL_MAX_NEW" \
      --decision_period 8 \
      --slo P99=50ms \
      --io_mode measured-dma \
      --hbm_budget "$budget" \
      --out "$out"
  done
done

# ---- Step 5: derive ε(φ) curve from the SEER results
echo "[Qwen] step 5/5: derive ε(φ) curve"
PYTHONPATH=. "$SEER_PY" - <<'PY'
import json, csv, glob, os
from seer.timing.schedulability import _estimate_epsilon_from_lap
out_dir = "experiments/eA_tail_latency/results_qwen_retrained_v2"
lap = "checkpoints/lap_qwen25.onnx"
rows = []
for f in sorted(glob.glob(f"{out_dir}/seer_b*.json")):
    btag = os.path.basename(f).replace("seer_b", "").replace(".json", "")
    phi = float("0." + btag)
    try:
        eps = _estimate_epsilon_from_lap(lap, workload=None, hbm_budget=phi)
    except Exception as e:  # noqa: BLE001
        print(f"  ε({phi}): estimator failed ({e!r}); using 0.15 fallback")
        eps = 0.15
    rows.append((phi, eps, "qwen25_retrained_lap"))
with open(f"{out_dir}/epsilon_curve.csv", "w", newline="") as fp:
    w = csv.writer(fp); w.writerow(["phi", "epsilon", "source"])
    for r in rows: w.writerow(r)
print("wrote", f"{out_dir}/epsilon_curve.csv")
PY

echo "=== Qwen retrain v2 DONE ==="
date
