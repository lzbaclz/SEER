#!/usr/bin/env bash
# When GPU 1 has been quiet for 3 x 30 s, re-run the eE smoke with
# SAVE_SAMPLES=1 to emit a *_samples.json sidecar of the raw per-rep
# latencies, then render the true cross-card histogram. Idempotent
# via marker file.
#
# Authorised by user 2026-05-09 (evening). Logs to
# /tmp/seer_wcet_samples.log.
set -uo pipefail
cd /home/lzq/codes/SEER

LOG=/tmp/seer_wcet_samples.log
exec >> "$LOG" 2>&1

MARKER=/tmp/seer_wcet_samples.done
if [ -f "$MARKER" ]; then
    echo "[$(date)] already done"
    exit 0
fi

QUIET_UTIL_PCT=${QUIET_UTIL_PCT:-10}
QUIET_FREE_MIB=${QUIET_FREE_MIB:-60000}
QUIET_STREAK=${QUIET_STREAK:-3}
SLEEP_S=${SLEEP_S:-30}
GPU_INDEX=${GPU_INDEX:-1}

source ~/miniconda3/bin/activate seer
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

streak=0
while true; do
    util=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    free=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    nproc=$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader | wc -l)
    if [[ "$util" -lt "$QUIET_UTIL_PCT" && "$free" -gt "$QUIET_FREE_MIB" && "$nproc" -eq 0 ]]; then
        streak=$((streak + 1))
        echo "$(date +%H:%M:%S) gpu${GPU_INDEX} quiet (streak=$streak)"
        if [[ "$streak" -ge "$QUIET_STREAK" ]]; then
            break
        fi
    else
        streak=0
        echo "$(date +%H:%M:%S) gpu${GPU_INDEX} busy util=${util}% free=${free} procs=${nproc}"
    fi
    sleep "$SLEEP_S"
done

echo "===== [$(date)] launching SAVE_SAMPLES=1 smoke on Card $GPU_INDEX ====="
CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
    GPU_TAG="A100_card${GPU_INDEX}_unlocked_$(date +%Y%m%d)_samples" \
    PLAN=checkpoints/lap_prod_b4096.plan BATCH=4096 REPS=10000 \
    SAVE_SAMPLES=1 \
    bash scripts/smoke_tensorrt.sh

echo "===== [$(date)] launching SAVE_SAMPLES=1 smoke on Card 0 ====="
CUDA_VISIBLE_DEVICES=0 \
    GPU_TAG="A100_card0_$(date +%Y%m%d)_samples" \
    PLAN=checkpoints/lap_prod_b4096.plan BATCH=4096 REPS=10000 \
    SAVE_SAMPLES=1 \
    bash scripts/smoke_tensorrt.sh

echo "===== [$(date)] rendering histogram ====="
python experiments/eE_lap_wcet/render_cross_card_hist.py

touch "$MARKER"
echo "===== [$(date)] DONE ====="
