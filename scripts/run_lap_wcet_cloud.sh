#!/usr/bin/env bash
# scripts/run_lap_wcet_cloud.sh
#
# DEPRECATED — DO NOT USE.
#
# This early version invoked `python -m seer.lap.wcet --arch X --backend Y ...`
# but the wcet CLI does NOT accept --arch and uses backend names
# {torch, onnx, tensorrt} (not {TRT_fp16, PyTorch, ...}). Every
# invocation of the old grid silently failed.
#
# The replacement kit is self-contained at:
#
#     cloud/h100_l40_wcet/
#         setup_env.sh   - bootstrap CUDA-12.1+ cloud node
#         lock_clocks.sh - SM-clock lock (auto-detects SKU)
#         run.sh         - sweeps eE grid with the correct CLI flags
#         verify.py      - sanity-check JSONs against 200 µs budget
#         integrate.sh   - re-render eE figure + emit paper-update snippet
#         expected.json  - A100 reference numbers
#         README.md      - end-to-end runbook
#
# Use that instead.

set -euo pipefail
cat <<'EOF' >&2
This script is deprecated. See cloud/h100_l40_wcet/ for the working kit.
The CLI flags it used (--arch, --backend TRT_fp16) never matched
seer.lap.wcet, so this version of the sweep produced no output.
EOF
exit 1
