#!/usr/bin/env bash
# R36e operator helper: reload NVIDIA kernel modules to match
# user-space NVML library.
#
# Symptom: `nvidia-smi` fails with "Failed to initialize NVML:
# Driver/library version mismatch".
#
# Root cause: unattended-upgrades pushed a new libnvidia-ml.so
# (e.g. 535.309.01) but the kernel still has the old nvidia.ko
# loaded (e.g. 535.288.01).  No reboot needed — module reload is
# sufficient when no CUDA processes are active.
#
# Usage:
#   sudo bash scripts/fix_nvidia_driver_mismatch.sh
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "FATAL: needs sudo (rmmod/modprobe)."
  echo "       Re-run: sudo bash $0"
  exit 1
fi

echo "[fix-nvidia] before:"
echo "  NVML kernel: $(cat /proc/driver/nvidia/version 2>/dev/null | head -1 | grep -oE '[0-9.]+' | head -1)"
echo "  NVML lib:    $(ls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.* 2>/dev/null \
                       | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"

echo "[fix-nvidia] checking for active CUDA holders..."
# fuser may print to stderr; capture both
holders=$(fuser /dev/nvidia* 2>&1 | tr -s ' ' | grep -oE '[0-9]+' || true)
if [[ -n "$holders" ]]; then
  echo "FATAL: /dev/nvidia* still in use by PIDs: $holders"
  echo "       Identify and stop them (own them with: ps -p PID -o user=,cmd=)"
  echo "       Common: nvitop, gpustat, dcgm-exporter, a stuck python."
  exit 1
fi

echo "[fix-nvidia] unloading old modules..."
rmmod nvidia_uvm     2>/dev/null || true
rmmod nvidia_drm     2>/dev/null || true
rmmod nvidia_modeset 2>/dev/null || true
rmmod nvidia        || { echo "FATAL: rmmod nvidia failed (still in use)"; exit 1; }

echo "[fix-nvidia] loading new modules..."
modprobe nvidia
modprobe nvidia_uvm
modprobe nvidia_modeset || true   # only needed if X server present
modprobe nvidia_drm     || true

echo "[fix-nvidia] after:"
echo "  NVML kernel: $(cat /proc/driver/nvidia/version 2>/dev/null | head -1 | grep -oE '[0-9.]+' | head -1)"
echo "  nvidia-smi:"
nvidia-smi --query-gpu=index,name --format=csv 2>&1 | head -5

echo "[fix-nvidia] done.  Re-run:"
echo "  sudo -E bash scripts/run_mig_a7.sh"
