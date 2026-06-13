#!/usr/bin/env bash
# Downgrade nvidia-535 user-space to 535.288.01 to match the
# kernel module currently loaded (which is 535.288.01).
#
# Why this is non-trivial:
#   - apt has already purged 535.288.01 from noble-updates +
#     noble-security; only 535.309.01 (current) and 535.171.04
#     (noble release-pocket, way older) are in the mirror.
#   - The kernel module on disk for the running kernel
#     (6.17.0-14-generic) is 535.288.01.  The newer kernel
#     6.17.0-29-generic is installed but its
#     `linux-modules-nvidia-535-6.17.0-29-generic` is in
#     "half-installed" (hF) state -- nvidia.ko is missing from
#     /lib/modules/6.17.0-29-generic/kernel/nvidia-535/.
#
# Strategy this script takes:
#   1) Fetch 535.288.01-0ubuntu0.24.04.2 .debs from
#      launchpad.net (which keeps historical archives forever).
#   2) `dpkg -i --force-downgrade` the 14 user-space packages.
#   3) `apt-mark hold` so unattended-upgrades will not silently
#      re-bump them next week.
#
# Fallback (if launchpad fetch fails, e.g. network):
#   The script prints the "reinstall + reboot to 6.17.0-29" path,
#   which is the alternative way to converge (matches user-space
#   535.309 by getting the 535.309 kernel module).
#
# Usage:
#   sudo bash scripts/downgrade_nvidia_535.sh
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "FATAL: needs sudo."
  echo "       Re-run: sudo bash $0"
  exit 1
fi

TARGET="535.288.01-0ubuntu0.24.04.2"
ARCH="amd64"
WORK=$(mktemp -d -t seer-nvidia-downgrade-XXXX)
echo "[downgrade] work dir: $WORK"

# 14 user-space packages.  We deliberately do NOT touch
# linux-modules-nvidia-535-generic-hwe-24.04 because that tracks
# kernel version (not nvidia version) and would force a kernel
# downgrade.
PKGS=(
  libnvidia-compute-535
  libnvidia-common-535
  libnvidia-fbc1-535
  libnvidia-gl-535
  libnvidia-extra-535
  nvidia-compute-utils-535
  nvidia-driver-535
  libnvidia-encode-535
  nvidia-utils-535
  xserver-xorg-video-nvidia-535
  libnvidia-decode-535
  nvidia-kernel-common-535
  libnvidia-cfg1-535
  nvidia-kernel-source-535
)

# libnvidia-common-535 is arch=all; the rest are amd64.
declare -A PKG_ARCH
for p in "${PKGS[@]}"; do PKG_ARCH[$p]="amd64"; done
PKG_ARCH[libnvidia-common-535]="all"
PKG_ARCH[nvidia-kernel-common-535]="amd64"
PKG_ARCH[nvidia-kernel-source-535]="amd64"

BASE="https://launchpad.net/ubuntu/+archive/primary/+files"

echo "[downgrade] fetching .debs from launchpad..."
ok=()
fail=()
for p in "${PKGS[@]}"; do
  a=${PKG_ARCH[$p]}
  fn="${p}_${TARGET}_${a}.deb"
  url="$BASE/$fn"
  if curl -fsSL --connect-timeout 15 -o "$WORK/$fn" "$url"; then
    ok+=("$fn")
  else
    fail+=("$fn ($url)")
  fi
done

echo "[downgrade] fetched ${#ok[@]}/${#PKGS[@]}"
if (( ${#fail[@]} > 0 )); then
  echo "[downgrade] FAILED to fetch:"
  for f in "${fail[@]}"; do echo "  - $f"; done
  echo ""
  echo "Falling back to the alternative path: reinstall the new"
  echo "kernel-modules-535 for 6.17.0-29-generic, then reboot."
  echo "That makes the nvidia.ko match the (new) user-space libs."
  echo ""
  echo "  sudo apt install --reinstall linux-modules-nvidia-535-6.17.0-29-generic"
  echo "  sudo reboot"
  echo "  # ... after reboot:"
  echo "  uname -r                   # should be 6.17.0-29-generic"
  echo "  cat /proc/driver/nvidia/version | head -1   # should be 535.309.01"
  echo "  nvidia-smi                 # should work"
  exit 1
fi

echo "[downgrade] dpkg -i (this will downgrade)..."
dpkg -i --force-downgrade "$WORK"/*.deb

echo "[downgrade] holding so unattended-upgrades does not re-bump..."
apt-mark hold "${PKGS[@]}"

echo "[downgrade] verification:"
echo "  user-space lib: $(ls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.* 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
echo "  kernel module : $(cat /proc/driver/nvidia/version 2>/dev/null | head -1 | grep -oE '[0-9.]+' | head -1)"
echo "  nvidia-smi sanity:"
nvidia-smi --query-gpu=index,name --format=csv 2>&1 | head -5

cat <<NOTE

[downgrade] done.  user-space + kernel both 535.288.01.  No reboot.

To allow the new 535.309 (the version unattended-upgrades wanted):
  sudo apt-mark unhold $(echo "${PKGS[@]}")
  # then either reboot into a kernel that has 535.309 modules, or
  # stop all CUDA processes and rmmod+modprobe nvidia.

The 14 packages are now pinned at 535.288.01.  Inspect with:
  apt-mark showhold | grep nvidia-535
NOTE

rm -rf "$WORK"
