#!/bin/bash
# One-shot script that pushes the SEER eviction-hook branch to the
# user's personal vllm-project/vllm fork and opens an upstream PR.
#
# PRECONDITIONS the user must satisfy first (Claude cannot do these):
#   1. gh auth login          (GitHub CLI logged in to the user's account)
#   2. A personal fork exists at https://github.com/<USER>/vllm
#      (run `gh repo fork vllm-project/vllm --clone=false --remote=false`
#      if you don't have one yet — gh will create it.)
#
# After those two preconditions are met, run:
#   bash seer/integration/vllm_patches/PUSH_PR.sh <your-github-handle>
# Example:
#   bash seer/integration/vllm_patches/PUSH_PR.sh lzbaclz
#
# What this script does:
#   (a) Add the user's fork as `userfork` remote (if not already set).
#   (b) Push the seer-eviction-hook branch to userfork.
#   (c) Open the upstream PR via gh pr create using UPSTREAM_PR.md
#       as the body.
#
set -e

USER_HANDLE=${1:-}
if [ -z "$USER_HANDLE" ]; then
  echo "USAGE: $0 <your-github-handle>"
  echo "(e.g.  bash $0 lzbaclz)"
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "ERROR: gh is not logged in. Run 'gh auth login' first."
  exit 1
fi

FORK_DIR=${FORK_DIR:-/home/lzq/codes/vllm-seer-fork}
BRANCH=${BRANCH:-seer-eviction-hook}
COVER=$(realpath "$(dirname "$0")/UPSTREAM_PR.md")

cd "$FORK_DIR"

echo "[1/3] Verifying we are on branch ${BRANCH}…"
CURR=$(git branch --show-current)
if [ "$CURR" != "$BRANCH" ]; then
  echo "Switching to ${BRANCH}…"
  git checkout "$BRANCH"
fi

echo "[2/3] Adding userfork remote https://github.com/${USER_HANDLE}/vllm.git"
if git remote get-url userfork >/dev/null 2>&1; then
  echo "  (userfork already configured: $(git remote get-url userfork))"
else
  git remote add userfork "https://github.com/${USER_HANDLE}/vllm.git"
fi

echo "[3a/3] Pushing ${BRANCH} → userfork (force-with-lease so we don't overwrite anything)"
git push --force-with-lease userfork "${BRANCH}:${BRANCH}"

echo "[3b/3] Opening upstream PR…"
gh pr create \
  --repo vllm-project/vllm \
  --base main \
  --head "${USER_HANDLE}:${BRANCH}" \
  --title "[Feature][KVConnector] Add optional get_eviction_order hook for policy-driven free-queue ordering" \
  --body-file "${COVER}"

echo "DONE."
