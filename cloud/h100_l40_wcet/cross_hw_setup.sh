#!/usr/bin/env bash
# cloud/h100_l40_wcet/cross_hw_setup.sh
#
# One-command cross-hardware WCET pipeline for any non-A100 cloud GPU
# (L40s, H100, future SKUs).  Handles the three things that bit us
# manually on L40s and H100:
#
#   1. setup_env.sh hardcodes the cu121 wheel index, but torch 2.6 has
#      no cu121 build on PyPI. We patch it to cu124 in place (idempotent).
#   2. The .plan files shipped via seer_checkpoints.tgz are TensorRT
#      engines compiled for A100 SM 8.0.  They will silently fail to
#      load on L40s (SM 8.9) or H100 (SM 9.0).  We detect the local
#      compute capability and rebuild from lap_prod_tiny.pt if needed.
#   3. lock_clocks.sh needs sudo; most rented pods don't have it.
#      We try it but don't fail if it's denied (matches the A100
#      reference config which also leaves SM clocks at default).
#
# Then we run the full WCET grid and print copy-pastable push
# instructions. The script does NOT auto-push, because pushing
# to main is a side-effect we want a human eyeball on.
#
# Prereqs (you do these once per pod, before this script):
#   - git clone the repo
#   - tar xzf /tmp/seer_checkpoints.tgz   (creates checkpoints/)
#   - git config user.email / user.name
#
# Usage:
#   bash cloud/h100_l40_wcet/cross_hw_setup.sh

set -euo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_ROOT"

step() { printf "\033[1;36m[cross-hw] %s\033[0m\n" "$*"; }
fail() { printf "\033[1;31m[cross-hw] FAIL: %s\033[0m\n" "$*" >&2; exit 1; }

# Sanity-check prereqs.
[[ -f checkpoints/lap_prod_tiny.pt ]] \
    || fail "checkpoints/lap_prod_tiny.pt missing; run 'tar xzf /tmp/seer_checkpoints.tgz' first"
command -v nvidia-smi >/dev/null \
    || fail "nvidia-smi not on PATH"

# ---- 1. patch setup_env.sh (cu121 → cu124) if not already done -------
step "1/6 patching setup_env.sh wheel index (cu121 → cu124) if needed"
if grep -q '/whl/cu121' cloud/h100_l40_wcet/setup_env.sh; then
    sed -i 's|/whl/cu121|/whl/cu124|' cloud/h100_l40_wcet/setup_env.sh
    step "  patched"
else
    step "  already patched (or upstream fixed)"
fi

# ---- 2. run setup_env.sh ----------------------------------------------
step "2/6 running setup_env.sh"
bash cloud/h100_l40_wcet/setup_env.sh
# shellcheck disable=SC1091
source .venv-cloud/bin/activate

# ---- 3. try lock_clocks.sh (skip on permission denied) ---------------
step "3/6 trying lock_clocks.sh (will skip if no sudo)"
if sudo -n bash cloud/h100_l40_wcet/lock_clocks.sh 2>/dev/null; then
    step "  SM clocks locked"
else
    step "  no sudo / permission denied; SM clocks left at default (matches A100 reference)"
fi

# ---- 4. SM-aware TRT plan rebuild ------------------------------------
step "4/6 inspecting GPU compute capability"
SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_TAG=$(echo "$GPU_NAME" | tr ' ' '_' | tr -cd '[:alnum:]_-')
step "  GPU = $GPU_NAME  (SM $SM)  → tag=$GPU_TAG"

if [[ "$SM" == "8.0" ]]; then
    step "  A100 SM 8.0 detected; shipped checkpoints/lap_prod_b*.plan are native — no rebuild"
else
    step "  non-A100 SM; rebuilding 5 TRT plans from lap_prod_tiny.pt"
    for B in 256 1024 4096 16384; do
        out="checkpoints/lap_prod_b${B}.plan"
        step "    fp32 plan b=$B → $out"
        python -m seer.lap.export \
            --ckpt checkpoints/lap_prod_tiny.pt \
            --out  "$out" \
            --backend tensorrt \
            --batch  "$B" \
            --static_shape \
            > /tmp/export_b${B}.log 2>&1 \
            || { cat /tmp/export_b${B}.log; fail "export failed for b=$B (see /tmp/export_b${B}.log)"; }
    done
    out="checkpoints/lap_prod_b4096_fp16.plan"
    step "    fp16 plan b=4096 → $out"
    python -m seer.lap.export \
        --ckpt checkpoints/lap_prod_tiny.pt \
        --out  "$out" \
        --backend tensorrt \
        --batch  4096 \
        --static_shape \
        --fp16 \
        > /tmp/export_fp16.log 2>&1 \
        || { cat /tmp/export_fp16.log; fail "export failed for fp16 (see /tmp/export_fp16.log)"; }
fi

# ---- 5. run WCET grid -------------------------------------------------
step "5/6 running WCET grid on GPU 0"
GPU_INDEX=0 bash cloud/h100_l40_wcet/run.sh

# ---- 6. verify --------------------------------------------------------
step "6/6 verifying"
python cloud/h100_l40_wcet/verify.py \
    --results_dir experiments/eE_lap_wcet/results \
    --sku "$GPU_TAG" \
    || step "  WARNING: verify exited non-zero — inspect output above before pushing"

# ---- print push instructions -----------------------------------------
N_JSON=$(ls experiments/eE_lap_wcet/results/lap_wcet_${GPU_TAG}_*.json 2>/dev/null | wc -l)
SUMMARY="experiments/eE_lap_wcet/results/_run_summary_${GPU_TAG}.txt"

cat <<EOF

============================================================
[cross-hw] DONE.  GPU_TAG = $GPU_TAG
            wrote $N_JSON JSONs + summary
============================================================
Next steps (manual; this script does NOT auto-push):

  # 1. Stash any local setup_env.sh edits, then sync with origin
  git stash push -m "cross-hw setup tweak" cloud/h100_l40_wcet/setup_env.sh 2>/dev/null || true
  git fetch origin main
  git pull --rebase origin main
  git stash pop 2>/dev/null || true

  # 2. Stage + commit + push
  git add -f experiments/eE_lap_wcet/results/lap_wcet_${GPU_TAG}_*.json \\
             $SUMMARY
  git status
  git commit -m "eE: ${GPU_TAG} cross-hardware WCET measurements"
  git push origin main

  # 3. Confirm it landed on origin, then destroy the pod
  git log origin/main -1 --oneline
============================================================
EOF
