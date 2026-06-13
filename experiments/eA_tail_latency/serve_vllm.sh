#!/usr/bin/env bash
# experiments/e3_throughput/serve_vllm.sh
#
# Launch a vLLM OpenAI-compatible HTTP server for baseline throughput experiments.
# This script does NOT touch any Seer code — it serves the *vanilla* model so we
# can measure the baseline numbers Seer's policy will be compared against.
#
# Usage:
#   bash serve_vllm.sh                                  # default: Qwen2.5-7B on :8000
#   MODEL=meta-llama/Meta-Llama-3-8B-Instruct bash serve_vllm.sh
#   PORT=8001 TP=2 MAX_LEN=8192 bash serve_vllm.sh
#
# Stop the server with Ctrl-C in this terminal.

set -euo pipefail

# --- Configurable knobs (override via env) ---
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}     # any HF model id; small first
PORT=${PORT:-8000}                            # vLLM HTTP port
TP=${TP:-1}                                   # tensor-parallel size = #GPUs
MAX_LEN=${MAX_LEN:-4096}                      # max model context (prompt + output)
GPU_UTIL=${GPU_UTIL:-0.90}                    # vLLM HBM utilization fraction
DTYPE=${DTYPE:-auto}                          # bf16 / fp16 / auto
EXTRA_ARGS=${EXTRA_ARGS:-}                    # any extra vllm flags you want to append

# --- Sanity: make sure vllm is on PATH ---
if ! command -v vllm >/dev/null 2>&1; then
  echo "ERROR: 'vllm' command not found. Run: pip install vllm" >&2
  exit 1
fi

echo "[serve_vllm] model=$MODEL port=$PORT tp=$TP max_len=$MAX_LEN gpu_util=$GPU_UTIL"
echo "[serve_vllm] (Ctrl-C to stop. The first launch will download the model into ~/.cache/huggingface.)"
echo

# --- Launch ---
# --trust-remote-code is needed for DeepSeek/Qwen/etc. that ship custom modeling code.
# --disable-log-requests keeps the server log readable during benchmarking.
exec vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --dtype "$DTYPE" \
  --trust-remote-code \
  --disable-log-requests \
  $EXTRA_ARGS
