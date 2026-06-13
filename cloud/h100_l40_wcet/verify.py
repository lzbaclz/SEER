"""Sanity-check the JSONs produced by run.sh and print a one-line
P50/P99/P99.9 vs 200 µs budget report per (arch, backend, batch).

Run:
    python cloud/h100_l40_wcet/verify.py \\
        --results_dir experiments/eE_lap_wcet/results \\
        --sku NVIDIA_H100_80GB_HBM3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

WCET_BUDGET_US = 200.0   # Lemma 2 assumption
HEADLINE = ("tiny_mlp", "trt", 4096)
# Order matters: trt_fp16 must be matched before trt to avoid prefix-eat.
BACKEND_TAGS = ("trt_fp16", "trt", "onnx", "torch")
ARCH_TAGS = ("tiny_mlp", "block_rnn", "block_xfmr", "block_transformer")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--sku", default=None,
                    help="Restrict to a single GPU SKU (substring match on filename); "
                         "default = report all SKUs found.")
    ap.add_argument("--budget_us", type=float, default=WCET_BUDGET_US)
    args = ap.parse_args()

    pattern = os.path.join(args.results_dir, "lap_wcet_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[verify] no JSON files under {args.results_dir}", file=sys.stderr)
        return 2

    # Group by SKU.
    by_sku: dict[str, list[str]] = {}
    for f in files:
        name = os.path.basename(f)
        # Filename schema (both legacy A100 and new cloud-run):
        #   lap_wcet_<SKU>_<arch>_<backend>_b<batch>.json
        # Where backend ∈ {trt, trt_fp16, onnx, torch}.
        # The SKU is whatever comes between "lap_wcet_" and the arch.
        # Strip the .json + _b<digit> suffix first, then peel arch and backend.
        stem = name[:-5]  # drop ".json"
        bm = re.search(r"_b(\d+)$", stem)
        if not bm:
            continue
        stem = stem[: bm.start()]
        # Peel backend (longest match first).
        backend = next((b for b in BACKEND_TAGS if stem.endswith("_" + b)), None)
        if backend is None:
            continue
        stem = stem[: -(len(backend) + 1)]
        # Peel arch.
        arch = next((a for a in ARCH_TAGS if stem.endswith("_" + a)), None)
        if arch is None:
            continue
        stem = stem[: -(len(arch) + 1)]
        # Whatever remains after "lap_wcet_" is the SKU tag.
        sku = stem[len("lap_wcet_"):] if stem.startswith("lap_wcet_") else stem
        if args.sku and args.sku not in sku:
            continue
        by_sku.setdefault(sku, []).append(f)

    if not by_sku:
        print(f"[verify] no JSONs matched SKU filter '{args.sku}'", file=sys.stderr)
        return 2

    # Render per-SKU report.
    any_overbudget = False
    headline_ok_skus: list[str] = []
    headline_miss_skus: list[str] = []

    for sku, sku_files in sorted(by_sku.items()):
        print(f"\n[verify] {sku} — {len(sku_files)} measurements")
        rows = []
        for f in sorted(sku_files):
            try:
                d = json.load(open(f))
            except (OSError, json.JSONDecodeError) as e:
                print(f"  ! could not read {f}: {e}")
                continue
            name = os.path.basename(f)
            stem2 = name[:-5]
            bm2 = re.search(r"_b(\d+)$", stem2)
            batch = int(bm2.group(1)) if bm2 else 0
            stem2 = stem2[: bm2.start()] if bm2 else stem2
            backend = next((b for b in BACKEND_TAGS if stem2.endswith("_" + b)), "?")
            stem2 = stem2[: -(len(backend) + 1)] if backend != "?" else stem2
            arch = next((a for a in ARCH_TAGS if stem2.endswith("_" + a)), "?")
            rows.append((arch, backend, batch, d))

        rows.sort(key=lambda r: (r[0], r[1], r[2]))
        for arch, backend, batch, d in rows:
            p50 = d.get("p50_us", float("nan"))
            p99 = d.get("p99_us", float("nan"))
            p999 = d.get("p999_us", float("nan"))
            within = p999 <= args.budget_us
            tick = "✓" if within else "✗"
            if not within:
                any_overbudget = True
            print(f"  {arch:<10} {backend:<14} b={batch:<5}  "
                  f"P50={p50:6.1f}µs  P99={p99:6.1f}µs  P99.9={p999:6.1f}µs  {tick}")

            if (arch, backend, batch) == HEADLINE:
                if within:
                    headline_ok_skus.append(sku)
                else:
                    headline_miss_skus.append(sku)

    # Final headline summary.
    print()
    if headline_ok_skus:
        print(f"[verify] HEADLINE OK on: {', '.join(headline_ok_skus)}")
        print(f"[verify]   (TinyMLP / tensorrt-fp32 / b=4096 within {args.budget_us:.0f}µs)")
    if headline_miss_skus:
        print(f"[verify] ⚠ HEADLINE MISS on: {', '.join(headline_miss_skus)}")
        print( "[verify]   That SKU disqualifies TinyMLP at production batch — "
               "report this in §6.7 as an architecture/SKU exclusion.")
    if not headline_ok_skus and not headline_miss_skus:
        print(f"[verify] ⚠ headline ({HEADLINE}) not measured for any SKU")

    return 1 if any_overbudget else 0


if __name__ == "__main__":
    sys.exit(main())
