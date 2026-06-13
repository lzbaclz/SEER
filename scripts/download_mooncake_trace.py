"""One-shot downloader for the real Mooncake production trace.

P1-6 (review round May 2026, reviewer #3): paper §A.2 setup says
"Mooncake-derived", which several reviewers read as "this paper does
not run against a real production trace". The trace **is** the real
upstream JSONL — Mooncake publishes anonymised hash_ids + lengths +
arrival timestamps, and we synthesise only the prompt text because no
raw text exists. This CLI explicitly downloads the JSONL into
``~/.cache/seer/mooncake/trace.jsonl`` so the eA / eB headline runs
do not depend on live network access.

Usage::

    python scripts/download_mooncake_trace.py
    python scripts/download_mooncake_trace.py --url <override-url>

Once cached, eA headline can use ``--workload mooncake-real`` to
forbid the RULER fallback that ``--workload mooncake`` silently
allows when the network is down.
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path


def _try_download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "seer/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as out:
            n = 0
            chunk_size = 1 << 20  # 1 MiB
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                n += len(chunk)
                print(f"\r  ... {n / (1 << 20):.1f} MiB", end="", flush=True)
            print()
        size = dest.stat().st_size
        if size == 0:
            dest.unlink(missing_ok=True)
            return False
        # Sanity: first line must look like JSON.
        with open(dest) as fp:
            head = fp.readline().strip()
        if not head.startswith("{"):
            print(f"  [warn] first line is not JSON: {head[:80]!r}")
            return False
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [fail] {exc!r}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=None,
                    help="Override URL to download (must be a JSONL).")
    ap.add_argument("--dest", default=None,
                    help="Override the cache destination "
                         "(default: ~/.cache/seer/mooncake/trace.jsonl).")
    args = ap.parse_args()

    from seer.trace.datasets import (
        _MOONCAKE_CACHE,
        _MOONCAKE_DEFAULT_URLS,
    )

    dest = Path(args.dest) if args.dest else _MOONCAKE_CACHE / "trace.jsonl"
    urls = [args.url] if args.url else list(_MOONCAKE_DEFAULT_URLS)

    if dest.exists() and dest.stat().st_size > 0:
        print(f"Trace already cached at {dest} "
              f"({dest.stat().st_size / (1 << 20):.1f} MiB). "
              "Delete to refresh.")
        return 0

    for url in urls:
        print(f"Trying {url} ...")
        if _try_download(url, dest):
            print(f"OK — cached at {dest}")
            print("Run paper-headline eA with:")
            print("  python -m seer.eval.runner --workload mooncake-real ...")
            return 0
    print("All upstream URLs failed; see paper/sections/A2_setup.tex for the "
          "schema if you want to vendor a local copy.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
