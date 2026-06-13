#!/usr/bin/env bash
# Lock GPU SM clocks to a fixed frequency on every visible A100 so that
# the eE WCET measurement (paper §6.7) is repeatable across cards and
# across runs. The defaults match the boost clock (1410 MHz) on A100
# SXM4-80GB, but any value within [`clocks.min.sm`, `clocks.max.sm`] is
# accepted.
#
# Usage:
#   sudo bash scripts/lock_gpu_clocks.sh           # lock to 1410 MHz
#   sudo bash scripts/lock_gpu_clocks.sh 1200      # lock to 1200 MHz
#   sudo bash scripts/lock_gpu_clocks.sh reset     # release the lock
#
# This script intentionally requires sudo: `nvidia-smi --lock-gpu-clocks`
# and `nvidia-smi -c EXCLUSIVE_PROCESS` modify driver state visible to
# every user on the box. Re-run with `reset` before exiting the session
# so other workloads aren't affected.
#
# Cross-references:
#   * paper §6.7 "Cross-card asymmetry" — discusses the 26.6 vs 47.1 µs
#     gap measured *without* this lock applied.
#   * paper §8 "Cross-card asymmetry" — recommends running this script
#     on every card before the runtime measures C_LAP.
#   * Makefile target `eE-card1-controlled` — convenience wrapper that
#     calls this script then re-runs `seer.lap.wcet` on GPU 1.

set -euo pipefail

LOCK_MHZ=${1:-1410}

if [[ "$EUID" -ne 0 ]]; then
    echo "ERROR: this script needs root (try: sudo $0 $@)"
    exit 1
fi

if ! command -v nvidia-smi >/dev/null; then
    echo "ERROR: nvidia-smi not found"
    exit 1
fi

if [[ "$LOCK_MHZ" == "reset" ]]; then
    echo "[lock_gpu_clocks] resetting clocks + compute mode on all visible GPUs"
    nvidia-smi --reset-gpu-clocks
    nvidia-smi -c DEFAULT
    nvidia-smi --query-gpu=index,clocks.sm,compute_mode,persistence_mode --format=csv
    exit 0
fi

echo "[lock_gpu_clocks] locking SM clock to ${LOCK_MHZ} MHz on all visible GPUs"
nvidia-smi -pm 1                                # persistence mode on
nvidia-smi --lock-gpu-clocks=${LOCK_MHZ},${LOCK_MHZ}
nvidia-smi -c EXCLUSIVE_PROCESS                 # one CUDA process per card

# Sanity report
nvidia-smi --query-gpu=index,name,clocks.sm,clocks.max.sm,compute_mode,persistence_mode \
           --format=csv
echo "[lock_gpu_clocks] DONE -- remember to run 'sudo $0 reset' afterwards"
