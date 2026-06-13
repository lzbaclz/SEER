#!/usr/bin/env bash
# Assemble nvme0n1 + nvme2n1 (RAID label gpu:0) and mount at /raid.
# Run from an interactive terminal after: sudo -v
set -euo pipefail

MOUNT=/raid
DEV0=/dev/nvme0n1
DEV1=/dev/nvme2n1

if [[ $EUID -ne 0 ]]; then
  echo "Run as root: sudo bash $0" >&2
  exit 1
fi

if ! command -v mdadm >/dev/null 2>&1; then
  echo "[raid] installing mdadm..."
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y mdadm
fi

echo "[raid] superblocks:"
mdadm --examine "$DEV0" "$DEV1" | grep -E '^(Device|Array|Raid|UUID|Events|Update Time|State|Level|Devices)'

if mountpoint -q "$MOUNT" 2>/dev/null; then
  echo "[raid] already mounted at $MOUNT"
  df -hT "$MOUNT"
  exit 0
fi

echo "[raid] assembling..."
if ! mdadm --assemble --scan --verbose; then
  mdadm --assemble --verbose /dev/md0 "$DEV0" "$DEV1" || \
    mdadm --assemble --verbose /dev/md127 "$DEV0" "$DEV1"
fi

MD=$(ls /dev/md* 2>/dev/null | head -1)
if [[ -z "${MD:-}" ]]; then
  echo "[raid] assembly failed; no /dev/md* device" >&2
  exit 1
fi
echo "[raid] array device: $MD"

FSTYPE=$(blkid -o value -s TYPE "$MD" || true)
UUID=$(blkid -o value -s UUID "$MD" || true)
if [[ -z "$FSTYPE" ]]; then
  echo "[raid] no filesystem on $MD — NOT formatting automatically." >&2
  echo "       Inspect with: sudo file -s $MD" >&2
  exit 1
fi
echo "[raid] filesystem: $FSTYPE uuid=$UUID"

mkdir -p "$MOUNT"
mount "$MD" "$MOUNT"
chmod 1777 "$MOUNT" 2>/dev/null || true

echo "[raid] mounted:"
df -hT "$MOUNT"
ls -la "$MOUNT" | head -15

# Persist assembly across reboots.
MDADM_CONF=/etc/mdadm/mdadm.conf
if [[ -f "$MDADM_CONF" ]] && ! grep -q "$UUID" "$MDADM_CONF" 2>/dev/null; then
  echo "[raid] updating $MDADM_CONF"
  mdadm --detail --scan >> "$MDADM_CONF"
fi

FSTAB=/etc/fstab
if ! grep -q "[[:space:]]$MOUNT[[:space:]]" "$FSTAB" 2>/dev/null; then
  echo "[raid] adding fstab entry"
  echo "UUID=$UUID  $MOUNT  $FSTYPE  defaults,nofail  0  2" >> "$FSTAB"
fi

echo "[raid] done. Use: mkdir -p $MOUNT/seer/e5_traces"
