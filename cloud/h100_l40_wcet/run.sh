#!/usr/bin/env bash
# cloud/h100_l40_wcet/run.sh
#
# Sweeps the eE LAP-WCET grid on the present GPU and writes one
# JSON per combination into experiments/eE_lap_wcet/results/, tagged
# with the GPU SKU so the new files don't collide with the A100
# reference numbers already in the repo.
#
# Use this *after* setup_env.sh has succeeded and (optionally)
# lock_clocks.sh has run.
#
# Usage:
#   bash cloud/h100_l40_wcet/run.sh                   # GPU 0, default grid
#   GPU_INDEX=1 bash cloud/h100_l40_wcet/run.sh       # GPU 1
#   REPS=10000 bash cloud/h100_l40_wcet/run.sh        # tighter percentiles
#   ARCHS="tiny_mlp" bash cloud/h100_l40_wcet/run.sh  # smoke-only

set -euo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_ROOT"

# Activate venv if setup_env.sh created one.
if [[ -z "${VIRTUAL_ENV:-}" && -f "$REPO_ROOT/.venv-cloud/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv-cloud/bin/activate"
fi

GPU_INDEX="${GPU_INDEX:-0}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/experiments/eE_lap_wcet/results}"
REPS="${REPS:-3000}"          # match A100 measurement budget (3000 reps)
WARMUP="${WARMUP:-200}"
ARCHS="${ARCHS:-tiny_mlp block_rnn block_xfmr}"
# Backend tags match the existing A100 filename schema in this repo
# (trt = tensorrt fp32, trt_fp16 = tensorrt fp16, onnx, torch).
BACKENDS="${BACKENDS:-trt trt_fp16 onnx torch}"
BATCHES="${BATCHES:-256 1024 4096 16384}"

mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------- GPU tag
GPU_NAME=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=name --format=csv,noheader \
    | tr ' ' '_' | tr -cd '[:alnum:]_-')
echo "[wcet] GPU $GPU_INDEX = $GPU_NAME"
echo "[wcet] writing to $OUT_DIR"

# ---------------------------------------------------------------- ckpt mapping
# checkpoints/ ships these for the deployment LAP. We pick the right
# file per (arch, backend, batch).
ckpt_for() {
    local arch="$1" backend="$2" batch="$3"
    case "$arch.$backend" in
        tiny_mlp.trt)
            echo "checkpoints/lap_prod_b${batch}.plan" ;;
        tiny_mlp.trt_fp16)
            # The fp16 plan only exists for batch 4096 in the repo.
            if [[ "$batch" == "4096" ]]; then
                echo "checkpoints/lap_prod_b4096_fp16.plan"
            else
                echo "SKIP_NO_FP16_PLAN"
            fi ;;
        tiny_mlp.onnx)
            echo "checkpoints/lap_prod_b${batch}.onnx" ;;
        tiny_mlp.torch)
            echo "checkpoints/lap_prod_tiny.pt" ;;
        block_rnn.torch)
            echo "checkpoints/lap_prod_rnn.pt" ;;
        block_xfmr.torch)
            echo "checkpoints/lap_prod_xfmr.pt" ;;
        *)
            echo "SKIP_NOT_PORTED" ;;
    esac
}

# ---------------------------------------------------------------- map backend → CLI flag
backend_flag() {
    case "$1" in
        trt|trt_fp16) echo "tensorrt" ;;
        onnx)         echo "onnx" ;;
        torch)        echo "torch" ;;
    esac
}

# ---------------------------------------------------------------- pick CLI flag for the model file
model_flag() {
    case "$1" in
        trt|trt_fp16) echo "--plan" ;;
        onnx)         echo "--onnx" ;;
        torch)        echo "--ckpt" ;;
    esac
}

# ---------------------------------------------------------------- run grid
N_OK=0; N_SKIP=0; N_FAIL=0
SUMMARY_FILE="$OUT_DIR/_run_summary_${GPU_NAME}.txt"
: > "$SUMMARY_FILE"

for arch in $ARCHS; do
  for backend in $BACKENDS; do
    for batch in $BATCHES; do
        ckpt=$(ckpt_for "$arch" "$backend" "$batch")
        case "$ckpt" in
            SKIP_NO_FP16_PLAN)
                echo "[wcet] skip $arch/$backend/b$batch  (no fp16 plan at this batch)"
                N_SKIP=$((N_SKIP+1)); continue ;;
            SKIP_NOT_PORTED)
                # BlockRNN and BlockXfmr only have torch checkpoints.
                echo "[wcet] skip $arch/$backend/b$batch  (no $backend plan for $arch)"
                N_SKIP=$((N_SKIP+1)); continue ;;
        esac

        if [[ ! -f "$REPO_ROOT/$ckpt" ]]; then
            echo "[wcet] skip $arch/$backend/b$batch  (missing checkpoint $ckpt)"
            N_SKIP=$((N_SKIP+1))
            echo "MISSING_CKPT $arch/$backend/b$batch  $ckpt" >> "$SUMMARY_FILE"
            continue
        fi

        out_file="$OUT_DIR/lap_wcet_${GPU_NAME}_${arch}_${backend}_b${batch}.json"
        if [[ -f "$out_file" ]]; then
            echo "[wcet] reuse $out_file"
            N_SKIP=$((N_SKIP+1)); continue
        fi

        flag=$(backend_flag "$backend")
        mflag=$(model_flag "$backend")
        echo "[wcet] $arch/$backend/b$batch on $GPU_NAME (reps=$REPS)"
        if CUDA_VISIBLE_DEVICES="$GPU_INDEX" python -m seer.lap.wcet \
                --backend "$flag" \
                "$mflag" "$ckpt" \
                --device cuda \
                --batch "$batch" \
                --reps "$REPS" \
                --warmup "$WARMUP" \
                --out "$out_file" >/dev/null 2>>"$SUMMARY_FILE"; then
            N_OK=$((N_OK+1))
            python -c "
import json, sys
d = json.load(open('$out_file'))
print(f'  P50={d[\"p50_us\"]:.1f}µs  P99={d[\"p99_us\"]:.1f}µs  P99.9={d[\"p999_us\"]:.1f}µs')
" || true
        else
            echo "[wcet] FAIL $arch/$backend/b$batch (see $SUMMARY_FILE)"
            N_FAIL=$((N_FAIL+1))
            echo "FAILED $arch/$backend/b$batch" >> "$SUMMARY_FILE"
        fi
    done
  done
done

# ---------------------------------------------------------------- summary
printf "\n[wcet] sweep complete on %s\n" "$GPU_NAME"
printf "[wcet]   succeeded: %d\n" "$N_OK"
printf "[wcet]   skipped  : %d\n" "$N_SKIP"
printf "[wcet]   failed   : %d  (see %s)\n" "$N_FAIL" "$SUMMARY_FILE"

cat <<EOF

[wcet] next steps:
   1. python cloud/h100_l40_wcet/verify.py --results_dir "$OUT_DIR" --sku "$GPU_NAME"
   2. on your laptop:
        scp <node>:$OUT_DIR/lap_wcet_${GPU_NAME}_*.json experiments/eE_lap_wcet/results/
        bash cloud/h100_l40_wcet/integrate.sh
EOF
