#!/usr/bin/env bash
# cloud/h100_l40_wcet/setup_env.sh
#
# Bootstraps a fresh CUDA-12.1+ cloud node so that
#   python -m seer.lap.wcet --backend tensorrt --plan ...
# runs successfully. Idempotent — safe to re-run.
#
# Tested on:
#   - RunPod official "PyTorch 2.4 CUDA 12.4" image
#   - Lambda Cloud "Ubuntu 22.04 + CUDA 12.1" base
#   - GCP a3-highgpu-1g (H100, Deep Learning VM CUDA 12.1)
#
# After this runs cleanly, run cloud/h100_l40_wcet/run.sh.

set -euo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_ROOT"

step() { printf "\033[1;36m[setup-env] %s\033[0m\n" "$*"; }
fail() { printf "\033[1;31m[setup-env] FAIL: %s\033[0m\n" "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. driver
step "1/6 verifying NVIDIA driver + nvidia-smi"
command -v nvidia-smi >/dev/null 2>&1 \
    || fail "nvidia-smi not on PATH; this kit assumes a CUDA cloud node"
nvidia-smi --query-gpu=name,driver_version,compute_cap \
    --format=csv,noheader || fail "nvidia-smi failed"

# ---------------------------------------------------------------- 2. python
step "2/6 verifying Python ≥ 3.11"
PYV=$(python3 --version 2>&1 | awk '{print $2}')
PY_MAJOR=$(echo "$PYV" | cut -d. -f1)
PY_MINOR=$(echo "$PYV" | cut -d. -f2)
if (( PY_MAJOR < 3 )) || (( PY_MAJOR == 3 && PY_MINOR < 11 )); then
    fail "Python $PYV; need 3.11+. On Ubuntu: sudo apt install python3.11 python3.11-venv"
fi
step "    using python $PYV"

# ---------------------------------------------------------------- 3. venv
VENV="${VENV:-$REPO_ROOT/.venv-cloud}"
step "3/6 creating venv at $VENV"
if [[ ! -d "$VENV" ]]; then
    python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --quiet --upgrade pip wheel setuptools

# ---------------------------------------------------------------- 4. torch
step "4/6 installing PyTorch 2.6 + CUDA 12.1 wheels"
pip install --quiet \
    "torch==2.6.0" "torchvision==0.21.0" \
    --index-url https://download.pytorch.org/whl/cu121 \
    || fail "torch install failed; check CUDA driver compatibility"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('  torch', torch.__version__, 'CUDA', torch.version.cuda, 'devices', torch.cuda.device_count())"

# ---------------------------------------------------------------- 5. onnx + tensorrt
step "5/6 installing ONNX-Runtime + TensorRT"
pip install --quiet \
    onnxruntime-gpu==1.20.* \
    onnx==1.17.* \
    pandas==2.2.* \
    numpy==1.26.* \
    pyarrow==17.* \
    || fail "onnxruntime / numpy install failed"

# TensorRT is on NVIDIA's index. The cuxx wheels match torch's CUDA major.
pip install --quiet \
    --extra-index-url https://pypi.nvidia.com \
    "tensorrt==10.7.*" "tensorrt-cu12==10.7.*" \
    || fail "tensorrt 10.7 install failed (try 10.6/10.5 if your driver is older)"
python -c "import tensorrt as trt; print('  tensorrt', trt.__version__)"

# ---------------------------------------------------------------- 6. seer
step "6/6 installing SEER package (editable)"
pip install --quiet -e "$REPO_ROOT" \
    || fail "pip install -e . failed"
python -c "from seer.lap import wcet; print('  seer.lap.wcet OK')"

step "DONE. Activate the venv with: source $VENV/bin/activate"
step "Then run: GPU_INDEX=0 bash cloud/h100_l40_wcet/run.sh"
