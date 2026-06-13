#!/usr/bin/env bash
# Smoke test for the TensorRT export + WCET path.
#
# Pre-condition: a trained LAP checkpoint + tensorrt installed. Default
# paths align with `make export-lap` (see Makefile: PROD_LAP_PT,
# PROD_LAP_PLAN). Override via env vars to point at alternate ckpts:
#
#   CKPT=checkpoints/lap_smoke.pt PLAN=/tmp/x.plan bash scripts/smoke_tensorrt.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=${CKPT:-checkpoints/lap_prod_tiny.pt}
PLAN=${PLAN:-checkpoints/lap_prod_b4096.plan}
BATCH=${BATCH:-4096}
REPS=${REPS:-3000}
GPU_TAG=${GPU_TAG:-A100}

if [ ! -f "$CKPT" ]; then
    echo "missing $CKPT — train first with seer.lap.train"
    exit 1
fi

echo "=== build TRT plan from $CKPT ==="
python -m seer.lap.export \
    --ckpt "$CKPT" \
    --backend tensorrt \
    --batch "$BATCH" \
    --out "$PLAN"

echo "=== TRT WCET (graphed replay) ==="
SAMPLES_FLAG=()
if [ -n "${SAVE_SAMPLES:-}" ]; then
    SAMPLES_FLAG=(--samples_out "experiments/eE_lap_wcet/results/lap_wcet_${GPU_TAG}_tiny_mlp_trt_b${BATCH}_samples.json")
fi
python -m seer.lap.wcet \
    --plan "$PLAN" \
    --backend tensorrt \
    --device cuda \
    --batch "$BATCH" \
    --reps "$REPS" \
    --warmup 200 \
    --out "experiments/eE_lap_wcet/results/lap_wcet_${GPU_TAG}_tiny_mlp_trt_b${BATCH}.json" \
    "${SAMPLES_FLAG[@]}"

echo "=== predictor smoke (round-trip np input) ==="
python <<EOF
import numpy as np
from seer.lap.infer import LAPPredictor

pred = LAPPredictor.from_tensorrt("$PLAN")
print(f"backend: {pred.backend}")
X = np.random.randn(128, 37).astype(np.float32)
y = pred(X)
print(f"shape: {y.shape}  dtype: {y.dtype}  range: [{y.min():.3f}, {y.max():.3f}]")
assert y.shape == (128, 4), f"unexpected output shape {y.shape}"
print("OK")
EOF
