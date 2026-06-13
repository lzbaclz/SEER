#!/usr/bin/env bash
# Finish the vllm 0.8.5 install in the seer conda env.
#
# Why this script exists: in the 2026-05-09 J.2 push, the seer
# conda env was bootstrapped from an offline cache for everything
# vllm needs *except* `numba` and its native dependency
# `llvmlite` (~56 MB). The reference machine's PyPI proxy
# (gost/2.11.5) caps file-transfer size around 13 MB and 503's
# above that, so neither pip download nor curl could pull the
# llvmlite wheel during the agent session. The agent stopped at
# "vllm.LLM() reaches the model loader, blocks on `from numba
# import jit`".
#
# Steps for the human operator (estimated 5 min on a stable
# network):
#
#   bash scripts/setup_env_vllm.sh
#   pytest tests/test_integration_vllm.py -q     # 7 passed
#   make eF-vllm                                 # if defined; else
#                                                # see work4.md §J.2
#
# If the proxy still blocks, options that worked for other team
# members on this host:
#   * connect from a tethered laptop and `pip install` from there,
#     then `pip wheel` the result back here as a local index;
#   * use the conda-forge channel: `conda install -n seer
#     -c conda-forge llvmlite numba`;
#   * `pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple
#      llvmlite numba`.
set -euo pipefail
cd "$(dirname "$0")/.."

PIP=${PIP:-/home/lzq/miniconda3/envs/seer/bin/pip}
PY=${PY:-/home/lzq/miniconda3/envs/seer/bin/python}

if ! $PY -c "import vllm" 2>/dev/null; then
    echo "[setup_env_vllm] vllm not importable — re-running offline-deps install"
    $PIP install --no-deps /tmp/vllm_85/vllm-0.8.5*.whl
    for d in cachetools blake3 cloudpickle msgspec lark partial-json-parser \
             openai prometheus-client gguf lm-format-enforcer compressed-tensors \
             py-cpuinfo pyzmq tiktoken mistral_common watchfiles depyf einops \
             fastapi uvicorn starlette pynvml outlines llguidance sniffio \
             distro annotated-doc importlib_metadata jsonschema outlines_core \
             xgrammar sentencepiece; do
        $PIP install --no-deps "$d" || echo "  [warn] $d not yet installed"
    done
fi

echo "[setup_env_vllm] installing numba + llvmlite ..."
$PIP install numba llvmlite

echo "[setup_env_vllm] verifying full vllm import + connector ..."
$PY -c "
import vllm
print('vllm', vllm.__version__)
from seer.integration.vllm_connector import _detect_vllm_api_version, make_seer_connector_class
print('SEER detected API:', _detect_vllm_api_version())
cls = make_seer_connector_class()
remaining = [m for m in dir(cls) if getattr(getattr(cls,m,None),'__isabstractmethod__',False)]
assert not remaining, f'unimplemented: {remaining}'
print('connector OK')
"
echo "[setup_env_vllm] done. Run pytest tests/test_integration_vllm.py -q to confirm."
