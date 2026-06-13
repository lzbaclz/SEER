#!/bin/bash
# Watcher daemon: poll every 5 min for the contending FlexGen run to
# finish, then launch the J.5 vLLM separation-regime rerun (B1).
#
# Why this exists: the J.5 4-policy cell needs a quiet 2x A100 host
# (vLLM batching collapses to chat_miss=1.0 under heavy host-CPU
# contention). FlexGen on Llama-3.2-3B with 131K-context is currently
# pinning ~325% CPU continuously; this watcher waits for it to exit
# AND for 1-min load average to fall below the QUIET threshold, then
# runs scripts/run_separation_regime.sh and writes the four updated
# results JSONs into experiments/eF_mixed_slo/results_vllm_separation/.
#
# Detached: launched via `nohup ... &` so the session exiting does
# not kill the watcher. Markers:
#   logs/b1_auto.log          — combined stdout+stderr
#   /tmp/seer_b1_auto.done    — touched when run_separation_regime succeeds
#   /tmp/seer_b1_auto.fail    — touched on non-zero exit
set -euo pipefail

# ---------- config ----------
FLEXGEN_PID="${FLEXGEN_PID:-3765536}"
SLEEP_S="${SLEEP_S:-300}"           # poll every 5 min
QUIET_LOAD1="${QUIET_LOAD1:-10}"    # 1-min load average threshold
QUIET_GPU_UTIL="${QUIET_GPU_UTIL:-20}"  # both GPUs must be below this %
QUIET_STREAK="${QUIET_STREAK:-3}"   # need this many consecutive quiet samples
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-72}"

REPO=/home/lzq/codes/SEER
LOG="$REPO/logs/b1_auto.log"
DONE_MARK=/tmp/seer_b1_auto.done
FAIL_MARK=/tmp/seer_b1_auto.fail
SCRIPT="$REPO/scripts/run_separation_regime.sh"

mkdir -p "$REPO/logs"
: > "$LOG"

log() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG" >&2; }

log "watcher start: pid=$$ flexgen_pid=$FLEXGEN_PID sleep=${SLEEP_S}s load1<=$QUIET_LOAD1 streak=$QUIET_STREAK"
log "watcher log file: $LOG"

flexgen_alive() {
    [ -d "/proc/$FLEXGEN_PID" ]
}

load1_below() {
    local thr="$1"
    local load1
    load1="$(awk '{print $1}' /proc/loadavg)"
    # bash int compare is fine — load1 is "12.34" -> truncate
    local lo="${load1%%.*}"
    [ "$lo" -lt "$thr" ]
}

# All visible GPUs must be below the util threshold. This is the gate
# we missed on the first watcher fire (2026-05-13T03:09): host load1
# dropped under 10 but a competing HALO job was at 100% on GPU 1,
# producing per-step latency ~2.4s and chat_miss=1.0.
gpu_util_below() {
    local thr="$1"
    local max_util
    max_util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
        | tr -d ' ' | sort -n | tail -1)"
    if [ -z "$max_util" ]; then
        # nvidia-smi failed; treat as not-quiet to be safe.
        return 1
    fi
    [ "$max_util" -lt "$thr" ]
}

streak=0
deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))
while true; do
    if [ "$(date +%s)" -gt "$deadline" ]; then
        log "watcher: max wait exceeded (${MAX_WAIT_HOURS}h), giving up."
        touch "$FAIL_MARK"
        exit 2
    fi

    if flexgen_alive; then
        load1="$(awk '{print $1}' /proc/loadavg)"
        log "flexgen PID $FLEXGEN_PID still alive; load1=$load1 streak=$streak"
        streak=0
        sleep "$SLEEP_S"
        continue
    fi

    # FlexGen gone — additionally require load1 below threshold for QUIET_STREAK
    # consecutive samples so other tenants' bursts don't cause vLLM to
    # collapse mid-run.
    if load1_below "$QUIET_LOAD1" && gpu_util_below "$QUIET_GPU_UTIL"; then
        streak=$((streak + 1))
        local_load1="$(awk '{print $1}' /proc/loadavg)"
        local_gpu="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | paste -sd, -)"
        log "quiet sample $streak/$QUIET_STREAK (load1=$local_load1, gpu util=$local_gpu %)"
        if [ "$streak" -ge "$QUIET_STREAK" ]; then
            break
        fi
    else
        load1="$(awk '{print $1}' /proc/loadavg)"
        gpu_util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | paste -sd, -)"
        log "not quiet: load1=$load1 (thr<$QUIET_LOAD1) gpu=$gpu_util % (thr<$QUIET_GPU_UTIL); reset streak"
        streak=0
    fi
    sleep "$SLEEP_S"
done

log "quiet conditions met; launching $SCRIPT"

# Activate the seer conda env's PATH and force offline HF so the
# rerun does not stall on the proxy probe.
export PATH=/home/lzq/miniconda3/envs/seer/bin:$PATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SEER_CONNECTOR_STATS_PATH="$REPO/experiments/eF_mixed_slo/results_vllm_separation/connector_stats.json"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$REPO"
if bash "$SCRIPT" >>"$LOG" 2>&1; then
    log "run_separation_regime.sh OK"
    touch "$DONE_MARK"
    exit 0
else
    rc=$?
    log "run_separation_regime.sh FAILED rc=$rc"
    touch "$FAIL_MARK"
    exit "$rc"
fi
