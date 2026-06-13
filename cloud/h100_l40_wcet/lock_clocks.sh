#!/usr/bin/env bash
# cloud/h100_l40_wcet/lock_clocks.sh
#
# Locks SM clocks to the second-highest supported value for the
# present GPU SKU. Determinism is critical for WCET measurement —
# without it the P99.9 spread can double from ~5 % to ~10 %.
#
# Usage:
#   sudo bash cloud/h100_l40_wcet/lock_clocks.sh         # all GPUs
#   sudo GPU_INDEX=0 bash cloud/h100_l40_wcet/lock_clocks.sh   # single GPU
#
# This requires `nvidia-smi -lgc` permission, which on most cloud
# providers needs root. If you're on a sandbox without root, skip
# this — WCET numbers will be ~5% noisier but still publishable.
#
# Reset to dynamic clocks when done:
#   sudo nvidia-smi -rgc

set -euo pipefail
GPU_INDEX="${GPU_INDEX:-all}"

if [[ "$EUID" -ne 0 ]]; then
    echo "[lock-clocks] not running as root; skipping (clocks remain dynamic)"
    echo "[lock-clocks] WCET measurement will be ~5% more variable but still usable"
    exit 0
fi

# Discover SM clocks supported by this SKU.
if [[ "$GPU_INDEX" == "all" ]]; then
    QUERY=("nvidia-smi" "-q" "-d" "SUPPORTED_CLOCKS")
else
    QUERY=("nvidia-smi" "-i" "$GPU_INDEX" "-q" "-d" "SUPPORTED_CLOCKS")
fi

# Pull the SM (graphics) clocks; pick the second-highest to dodge
# thermal-binned top bins on some SKUs (e.g. some H100 PCIe cards).
SM_CLOCKS=$( "${QUERY[@]}" \
    | awk '/Graphics +:/ { print $3 }' \
    | sort -n -u )
N_CLOCKS=$( echo "$SM_CLOCKS" | wc -l )

if (( N_CLOCKS < 2 )); then
    echo "[lock-clocks] only one SM clock reported; locking to it"
    LOCK=$( echo "$SM_CLOCKS" | tail -n1 )
else
    LOCK=$( echo "$SM_CLOCKS" | tail -n2 | head -n1 )
fi

if [[ -z "$LOCK" ]]; then
    echo "[lock-clocks] could not determine SM clock from nvidia-smi; aborting" >&2
    exit 1
fi

echo "[lock-clocks] locking SM clock to $LOCK MHz on GPU $GPU_INDEX"
if [[ "$GPU_INDEX" == "all" ]]; then
    nvidia-smi -lgc "$LOCK,$LOCK"
else
    nvidia-smi -i "$GPU_INDEX" -lgc "$LOCK,$LOCK"
fi

echo "[lock-clocks] done. Reset with: sudo nvidia-smi -rgc"
