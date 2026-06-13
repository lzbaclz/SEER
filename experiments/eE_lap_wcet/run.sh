#!/usr/bin/env bash
# eE: LAP WCET measurement on this host. The §6.7 main-paper claim is
# A100 only; the H100 / L40 cross-hardware sweep is RTSS-2028 follow-up
# (see paper §8 "H100 / L40 cross-hardware").
#
# Set GPU_INDEX=N to pin to a specific card (the second A100 picks up
# 80% slower numbers in the uncontrolled run; see §6.7 "Cross-card
# asymmetry"). Set REQUIRE_QUIET=1 to abort if the chosen card has
# >5% utilisation when the script starts (so background workload
# can't silently double the WCET tail). Set CHECK_LOCKED=1 to abort
# unless `nvidia-smi -q -d CLOCK` reports the SM clock as locked
# (apply `sudo bash scripts/lock_gpu_clocks.sh` first).
set -euo pipefail
cd "$(dirname "$0")"

CKPT=${CKPT:-../../checkpoints/lap_llama3_8b.pt}
DEVICE=${DEVICE:-cuda}
BATCH=${BATCH:-4096}
REPS=${REPS:-10000}
OUT_DIR=${OUT_DIR:-results}
GPU_INDEX=${GPU_INDEX:-}
TAG_SUFFIX=${TAG_SUFFIX:-}
REQUIRE_QUIET=${REQUIRE_QUIET:-0}
CHECK_LOCKED=${CHECK_LOCKED:-0}

if [[ -n "$GPU_INDEX" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
    GPU_TAG=${GPU_TAG:-$(nvidia-smi -i "$GPU_INDEX" --query-gpu=name --format=csv,noheader 2>/dev/null | tr ' ' '_' || echo unknown)}
else
    GPU_TAG=${GPU_TAG:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_' || echo unknown)}
fi
GPU_TAG="${GPU_TAG}${TAG_SUFFIX}"

if [[ "$REQUIRE_QUIET" == "1" ]]; then
    QUERY_IDX=${GPU_INDEX:-0}
    UTIL=$(nvidia-smi -i "$QUERY_IDX" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    if [[ "$UTIL" -gt 5 ]]; then
        echo "ERROR: GPU $QUERY_IDX is at ${UTIL}% utilisation; refuse to measure under contention" >&2
        echo "       (set REQUIRE_QUIET=0 to override; see paper §6.7 Cross-card asymmetry)" >&2
        exit 1
    fi
fi

if [[ "$CHECK_LOCKED" == "1" ]]; then
    QUERY_IDX=${GPU_INDEX:-0}
    LOCKED=$(nvidia-smi -i "$QUERY_IDX" -q -d CLOCK 2>/dev/null | grep -c "SM Clock Lock" || true)
    if [[ "$LOCKED" -lt 1 ]]; then
        echo "WARN: SM clock not reported as locked on GPU $QUERY_IDX." >&2
        echo "      Run 'sudo bash scripts/lock_gpu_clocks.sh' for §6.7-grade controlled numbers." >&2
    fi
fi

mkdir -p "$OUT_DIR"

python -m seer.lap.wcet \
    --ckpt "$CKPT" \
    --backend torch \
    --device "$DEVICE" \
    --batch "$BATCH" \
    --reps "$REPS" \
    --out "$OUT_DIR/lap_wcet_${GPU_TAG}.json"

echo "[eE] WCET data -> $OUT_DIR/lap_wcet_${GPU_TAG}.json"
