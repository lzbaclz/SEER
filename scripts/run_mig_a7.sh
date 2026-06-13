#!/usr/bin/env bash
# R36d MIG hard-partition A7 probe — executable companion to
# docs/mig_a7_runbook.md (which is the human-readable explanation).
#
# Usage:
#   sudo -E bash scripts/run_mig_a7.sh                # 3g.40gb on GPU 1
#   TARGET_GPU=0 PROFILE=19 sudo -E bash scripts/run_mig_a7.sh  # 1g.10gb on GPU 0
#
# Profile IDs (verify on your card with: nvidia-smi mig -lgip):
#   9  = 3g.40gb (half partition)
#   19 = 1g.10gb (1/7 partition — tightest)
#
# Requires:
#   - sudo (one-time MIG enable + GI/CI create + cleanup)
#   - conda env "seer" available
#   - No active CUDA processes on TARGET_GPU
#
# Outputs:
#   experiments/eC_bound_tightness/results/substrate_a7_mig.{json,tex}
#
# Reviewer R36 RTSS-push item #1: "做一个 'pass A7 的真实隔离部署'
# 完整闭环：MIG 或 cgroup-PCIe, substrate + integrated 全部 HOLDS".
set -euo pipefail

TARGET_GPU=${TARGET_GPU:-1}
PROFILE=${PROFILE:-9}            # 3g.40gb
CONTENDER_MB=${CONTENDER_MB:-1.0}
N_REPS=${N_REPS:-5000}
OUT_STEM=${OUT_STEM:-substrate_a7_mig}
KEEP_MIG=${KEEP_MIG:-0}          # set 1 to leave MIG enabled at the end

if [[ $EUID -ne 0 ]]; then
  echo "FATAL: this script needs sudo (MIG enable / GI / CI create)."
  echo "       Re-run with: sudo -E bash $0"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Pick up the calling user's conda. sudo -E preserves $PATH and $HOME.
CONDA_SH="${CONDA_SH:-/home/lzq/miniconda3/etc/profile.d/conda.sh}"
if [[ ! -f "$CONDA_SH" ]]; then
  echo "FATAL: conda.sh not at $CONDA_SH. Set CONDA_SH=... and retry."
  exit 1
fi
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate seer
echo "[mig-a7] python=$(which python)  conda env=${CONDA_DEFAULT_ENV}"

# Workaround for stale MPS pipe dir (see
# experiments/eA_tail_latency/run_baseline_parity.sh comment).
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-$(mktemp -d -t seer-no-mps-XXXX)}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-$CUDA_MPS_PIPE_DIRECTORY}"

echo "[mig-a7] step 1/6: confirm GPU $TARGET_GPU is idle"
busy=$(nvidia-smi --query-compute-apps=pid \
                  --format=csv,noheader --id="$TARGET_GPU" | wc -l)
if [[ $busy -gt 0 ]]; then
  echo "FATAL: GPU $TARGET_GPU has $busy active CUDA processes. "
  echo "       Stop them first (nvidia-smi --query-compute-apps=pid,process_name --format=csv --id=$TARGET_GPU)."
  exit 1
fi
# Non-fatal advisory: surface monitoring/non-compute holders.
# Targeted check: monitoring tools (nvitop, nvtop, gpustat, py-spy,
# dcgm-exporter) ALWAYS prevent MIG enable because they poll
# /dev/nvidiaN every 0.5s.  A snapshot lsof may miss them between
# polls (false-negative advisory), but they will race the actual
# nvidia-smi -mig 1 + -r calls below and cause "In use by another
# client" 100% of the time.  Hard-FATAL on these by name, NOT by
# lsof snapshot.
mon_pids=$(pgrep -af 'nvitop|nvtop|gpustat|dcgm-exporter|nv-hostengine|py-spy' \
           2>/dev/null | awk '{print $1}' | tr '\n' ' ' || true)
if [[ -n "$mon_pids" ]]; then
  echo "[mig-a7] FATAL: monitoring tool(s) running that race MIG enable:"
  pgrep -af 'nvitop|nvtop|gpustat|dcgm-exporter|nv-hostengine|py-spy' 2>/dev/null \
    | sed 's/^/  /'
  echo ""
  echo "  These polling tools take NVML handles on /dev/nvidia$TARGET_GPU"
  echo "  and 'nvidia-smi -mig 1' refuses while the device is in use."
  echo "  An lsof snapshot may say 0 holders (NVML handles don't always"
  echo "  show up in /dev/nvidia* lsof), but the kernel ref-count"
  echo "  blocks the actual MIG operation."
  echo ""
  echo "  Stop them and re-run:"
  echo "    sudo pkill -9 -f 'nvitop|nvtop|gpustat|dcgm-exporter|nv-hostengine|py-spy'"
  echo "    sudo -E bash $0"
  exit 1
fi
# nvidia-dcgm.service uses nv-hostengine; also check via systemctl
# for the cases where the daemon was launched via systemd and not
# visible via pgrep -f.
if systemctl is-active --quiet nvidia-dcgm 2>/dev/null \
   || systemctl is-active --quiet dcgm 2>/dev/null; then
  echo "[mig-a7] FATAL: nvidia-dcgm (DCGM) service is running."
  echo "  DCGM holds NVML handles on every GPU 24/7; MIG enable will"
  echo "  fail with 'In use by another client' every time."
  echo ""
  echo "  Stop (and mask to prevent auto-restart) before MIG enable:"
  echo "    sudo systemctl stop nvidia-dcgm"
  echo "    sudo systemctl mask nvidia-dcgm    # optional: stay down"
  echo "    sudo -E bash $0"
  echo ""
  echo "  After your MIG work, restore with:"
  echo "    sudo systemctl unmask nvidia-dcgm"
  echo "    sudo systemctl start nvidia-dcgm"
  exit 1
fi
# Non-blocking advisory for other holders (lsof snapshot).
dev="/dev/nvidia$TARGET_GPU"
if command -v lsof >/dev/null 2>&1; then
  holders=$(lsof "$dev" 2>/dev/null | awk 'NR>1{print $1"("$2","$3")"}' \
                                   | sort -u | tr '\n' ' ' || true)
  if [[ -n "$holders" ]]; then
    echo "[mig-a7]   advisory: holders of $dev: $holders"
    echo "[mig-a7]   (these may interfere; continuing with retry chain)"
  fi
fi

echo "[mig-a7] step 2/6: enable MIG on GPU $TARGET_GPU"
# Self-heal pending-enable state: a prior enable attempt may have
# left current=Disabled / pending=Enabled because the GPU had an
# active client at enable time.  Clear it before re-enabling.
pending=$(nvidia-smi --query-gpu=mig.mode.pending --format=csv,noheader \
                   --id="$TARGET_GPU" | tr -d ' ')
current=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader \
                   --id="$TARGET_GPU" | tr -d ' ')
if [[ "$current" == "Disabled" && "$pending" == "Enabled" ]]; then
  echo "[mig-a7]   detected stale pending=Enabled; clearing it via -mig 0 + -r"
  nvidia-smi -i "$TARGET_GPU" -mig 0 || true
  sleep 1
  # Reset is needed to actually flip the live state on A100 SXM4
  # even when no client holds the device.
  nvidia-smi -i "$TARGET_GPU" -r || true
  sleep 2
fi
echo "[mig-a7]   enabling MIG (-mig 1)..."
nvidia-smi -i "$TARGET_GPU" -mig 1 || true
sleep 2
current=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader \
                   --id="$TARGET_GPU" | tr -d ' ')
if [[ "$current" != "Enabled" ]]; then
  echo "[mig-a7]   still not Enabled; trying GPU reset..."
  nvidia-smi -i "$TARGET_GPU" -r || true
  sleep 2
fi
pending=$(nvidia-smi --query-gpu=mig.mode.pending --format=csv,noheader \
                   --id="$TARGET_GPU" | tr -d ' ')
current=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader \
                   --id="$TARGET_GPU" | tr -d ' ')
if [[ "$current" != "Enabled" ]]; then
  echo "FATAL: MIG mode on GPU $TARGET_GPU could not be enabled "
  echo "       (current=$current pending=$pending after -mig 0 + -r"
  echo "       + -mig 1 + -r retry chain)."
  echo ""
  echo "       Likely cause: a CUDA process (or torch.compile worker"
  echo "       pool, or vLLM child) is still holding /dev/nvidia$TARGET_GPU."
  echo ""
  # Surface non-root holders + ready-to-paste kill line.
  if command -v lsof >/dev/null 2>&1; then
    pids=$(lsof "/dev/nvidia$TARGET_GPU" 2>/dev/null \
            | awk 'NR>1 && $3!="root" {print $2}' \
            | sort -u | tr '\n' ' ' || true)
    if [[ -n "$pids" ]]; then
      echo "       The following $(echo $pids | wc -w) non-root PIDs are holding the device:"
      echo "         $pids"
      echo "       To unstick: "
      echo "         kill -9 $pids"
      echo "       Or, to nuke all torch.compile workers safely:"
      echo "         pkill -9 -f 'torch._inductor/compile_worker'"
      echo "       Or, if a parent vllm/seer.eval.runner is in the tree:"
      echo "         pkill -9 -f 'vllm\\|seer.eval.runner'"
      echo ""
      echo "       IMPORTANT: a torch.cuda probe in ANY python process"
      echo "       opens /dev/nvidia\${TARGET_GPU} even when the process"
      echo "       only computes on a DIFFERENT GPU (default torch"
      echo "       behaviour is to enumerate all visible GPUs).  If"
      echo "       you see non-root python PIDs holding the device"
      echo "       but nvidia-smi shows no compute-apps on GPU $TARGET_GPU,"
      echo "       a cross-GPU benchmark is the most common cause."
      echo "       Restart that benchmark with"
      echo "         CUDA_VISIBLE_DEVICES=<other-gpu> python ..."
      echo "       so it stops opening /dev/nvidia$TARGET_GPU."
    fi
  fi
  echo ""
  echo "       After killing, re-run: sudo -E bash $0"
  echo "       If nothing else works, a full host reboot is the"
  echo "       residual fix."
  exit 1
fi
echo "[mig-a7]   MIG mode = Enabled on GPU $TARGET_GPU"

echo "[mig-a7] step 3/6: create GI+CI from profile $PROFILE"
nvidia-smi mig -i "$TARGET_GPU" -cgi "$PROFILE" -C

echo "[mig-a7] step 4/6: read MIG UUID"
MIG_UUID=$(nvidia-smi -L | awk -F'UUID: ' '/MIG-/{print $2}' \
                       | tr -d ')' | head -1)
if [[ -z "$MIG_UUID" ]]; then
  echo "FATAL: no MIG UUID found. nvidia-smi -L output:"
  nvidia-smi -L
  exit 1
fi
echo "[mig-a7] MIG_UUID=$MIG_UUID"

echo "[mig-a7] step 5/6: run substrate_a7_mig probe under MIG"
CUDA_VISIBLE_DEVICES="$MIG_UUID" \
  python -m experiments.eC_bound_tightness.substrate_a7_mig \
    --cuda --device-index 0 \
    --contender-mb "$CONTENDER_MB" \
    --n-reps "$N_REPS" \
    --out-stem "$OUT_STEM"

echo "[mig-a7] step 6/6: cleanup (set KEEP_MIG=1 to skip)"
if [[ "$KEEP_MIG" != "1" ]]; then
  nvidia-smi mig -i "$TARGET_GPU" -dci || true
  nvidia-smi mig -i "$TARGET_GPU" -dgi || true
  nvidia-smi -i "$TARGET_GPU" -mig 0   || true
  echo "[mig-a7] MIG state on GPU $TARGET_GPU restored to Disabled."
else
  echo "[mig-a7] KEEP_MIG=1: MIG partition retained for cross-partition follow-up."
fi

echo "[mig-a7] done. Re-render paper:"
echo "  python scripts/gen_claim_evidence_table.py"
echo "  make paper"
echo "  git add -A paper/ experiments/eC_bound_tightness/results/${OUT_STEM}.{json,tex}"
echo "  git commit -m 'R36d MIG A7 probe: measured'"
