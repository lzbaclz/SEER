#!/usr/bin/env bash
# scripts/setup_env.sh — one-shot conda env setup for SEER reviewers.
#
# Creates a fresh `seer` conda environment with Python 3.11, installs
# every dependency (incl. TensorRT 10.6 from NVIDIA's pip index),
# installs the `seer` package in editable mode, and runs pytest.
#
# Usage:
#   bash scripts/setup_env.sh           # full install (~10 min, downloads ~5 GB)
#   bash scripts/setup_env.sh --skip-trt   # skip TensorRT (no GPU WCET claim)
#   bash scripts/setup_env.sh --env-name=foo   # custom env name (default seer)
#
# Idempotent: re-running upgrades pinned packages without recreating
# the env. Safe to interrupt; pip resumes downloads automatically.
set -euo pipefail

ENV_NAME="seer"
SKIP_TRT=0

for arg in "$@"; do
    case $arg in
        --skip-trt) SKIP_TRT=1 ;;
        --env-name=*) ENV_NAME="${arg#*=}" ;;
        --help|-h)
            sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "unknown arg: $arg"; exit 1 ;;
    esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[setup_env] root = $ROOT"
echo "[setup_env] target conda env = $ENV_NAME"

# 1. conda env (idempotent)
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup_env] env '$ENV_NAME' already exists; will pip-upgrade in place"
else
    echo "[setup_env] creating env '$ENV_NAME' (Python 3.11)"
    conda create -n "$ENV_NAME" python=3.11 -y >/dev/null
fi

# Resolve env-local python; we DON'T rely on `conda activate` inside scripts
PY="$(conda run -n "$ENV_NAME" --no-capture-output which python | tail -1)"
PIP="${PY%/*}/pip"
echo "[setup_env] using python = $PY"

# 2. main deps (CPU-resolvable from default PyPI)
echo "[setup_env] installing requirements.txt"
"$PIP" install --upgrade pip >/dev/null
"$PIP" install -r requirements.txt

# 3. TensorRT (optional, large)
if [ "$SKIP_TRT" -eq 1 ]; then
    echo "[setup_env] --skip-trt: NOT installing TensorRT. The WCET claim"
    echo "[setup_env] in §6.7 of the paper requires TensorRT; ONNX Runtime"
    echo "[setup_env] still works as an oracle (slower, but functional)."
else
    echo "[setup_env] installing TensorRT (3.4 GB; may take 5-15 min)"
    "$PIP" install --extra-index-url https://pypi.nvidia.com \
                   --resume-retries 50 --timeout 120 \
                   -r requirements-trt.txt
fi

# 4. seer package, editable
echo "[setup_env] installing seer package (editable)"
"$PIP" install -e .

# 5. quick verification
echo "[setup_env] running pytest"
"$PY" -m pytest tests/ -q

echo ""
echo "[setup_env] success. Activate with:"
echo "    conda activate $ENV_NAME"
echo ""
echo "[setup_env] reproduce paper headlines with:"
echo "    make e1               # GO/NO-GO predictability verdict"
echo "    bash experiments/eA_tail_latency/run_parallel.sh   # eA main table"
echo "    bash experiments/eE_lap_wcet/run.sh                # eE WCET sweep"
