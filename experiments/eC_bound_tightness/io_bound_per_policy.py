"""eC IO-bound regime, per-policy projection (deployment what-if).

This extends ``io_bound_regime.py`` with per-policy projection: for each
of the eA b020 traces (full / h2o / quest / streaming / seer), we
inject substrate-class IO latency on top of the measured IO-free base
and report per-policy chat-miss under DRAM / NVMe / NVMe-GC. This is the
"what would per-policy chat-miss look like if KV were on NVMe rather
than HBM-resident" projection, *not* a deployment measurement.

Honesty boundary (must be in caption):
* Base distribution: measured eA mooncake_full per_step_us at b020,
  policy-specific (5 policies x ~620 steady-state samples).
* Per-step block-count: we use the analytical top-K = max(32, 0.10 * B_full),
  matching the wanted-set rule used in the substrate sizing tool.
* IO injection: per-step Bernoulli(eps_phi) per block, with eps_phi
  taken from the eA-calibrated eps_curve at phi=0.20.
* Substrate distributions: vendor datasheet means + lognormal jitter
  (DRAM 200us / NVMe 1500us / NVMe-GC 2000us).
* The output is therefore a *projection*. We do not run on an NVMe
  KV substrate; the table label says "what-if".

Output:
* per_policy_substrate.csv / .tex     -- policy x substrate -> chat-miss
* per_policy_substrate_summary.json   -- compact summary

CLI:
    python -m experiments.eC_bound_tightness.io_bound_per_policy \\
        --in_dir experiments/eA_tail_latency/results_mooncake_full \\
        --eps_curve experiments/eC_bound_tightness/results/eps_curve.csv \\
        --out_dir experiments/eC_bound_tightness/results
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


SUBSTRATES = [
    ("DRAM", 200.0, 0.30, 0.0, 4.0),
    ("NVMe", 1500.0, 0.50, 0.0, 4.0),
    ("NVMe-GC", 2000.0, 0.60, 0.02, 8.0),
]

_POLICY_FILES = {
    "full": "full_b020.json",
    "h2o": "h2o_b020.json",
    "quest": "quest_b020.json",
    "streaming": "recency_b020.json",  # in-tree streaming label
    "seer": "seer_b020.json",
}


def _load_eps_curve(path: Path) -> list[tuple[float, float]]:
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append((float(r["phi"]), float(r["epsilon"])))
    return sorted(rows)


def _eps_at(curve: list[tuple[float, float]], phi: float) -> float:
    if phi <= curve[0][0]:
        return curve[0][1]
    if phi >= curve[-1][0]:
        return curve[-1][1]
    for (a, ea), (b, eb) in zip(curve, curve[1:]):
        if a <= phi <= b:
            t = (phi - a) / max(1e-12, b - a)
            return ea + t * (eb - ea)
    return curve[-1][1]


def _load_per_step(path: Path) -> list[float]:
    pool: list[float] = []
    with open(path) as fh:
        d = json.load(fh)
    for r in d.get("results", []):
        ps = r.get("per_step_us", []) or []
        if len(ps) > 1:
            pool.extend(float(x) for x in ps[1:])
    return pool


def _draw_io_us(rng: random.Random, ell_bar: float, sigma_log: float,
                q_escape: float, esc_factor: float) -> float:
    if rng.random() < q_escape:
        return ell_bar * esc_factor * (1.0 + rng.expovariate(1.0))
    mu_log = math.log(ell_bar) - 0.5 * sigma_log ** 2
    return math.exp(rng.gauss(mu_log, sigma_log))


def _replay(base_pool: list[float], eps_phi: float, B_t: int,
            sub_ell: float, sub_sig: float, sub_q: float, sub_ef: float,
            seed: int, n_steps: int) -> list[float]:
    rng = random.Random(seed)
    out: list[float] = []
    n = len(base_pool)
    for _ in range(n_steps):
        b = base_pool[rng.randrange(n)]
        io = 0.0
        for _b in range(B_t):
            if rng.random() < eps_phi:
                io += _draw_io_us(rng, sub_ell, sub_sig, sub_q, sub_ef)
        out.append(b + io)
    return out


def _miss_at_d(samples: list[float], d_us: float) -> float:
    if not samples:
        return float("nan")
    return sum(1 for x in samples if x > d_us) / len(samples)


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] * (1 - (k - lo)) + s[hi] * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",
                    default="experiments/eA_tail_latency/results_mooncake_full")
    ap.add_argument("--eps_curve",
                    default="experiments/eC_bound_tightness/results/eps_curve.csv")
    ap.add_argument("--out_dir",
                    default="experiments/eC_bound_tightness/results")
    ap.add_argument("--phi", type=float, default=0.20)
    ap.add_argument("--B_full", type=float, default=64.0)
    ap.add_argument("--Ds_ms", type=float, nargs="+",
                    default=[50.0, 100.0, 200.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_steps_per_seed", type=int, default=2000)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eps_curve = _load_eps_curve(Path(args.eps_curve))
    eps_phi = _eps_at(eps_curve, args.phi)
    B_t = max(32, int(args.B_full * 0.10))

    per_pol_pools: dict[str, list[float]] = {}
    for pol, fname in _POLICY_FILES.items():
        p = in_dir / fname
        if not p.is_file():
            print(f"[per-pol] WARN missing {p}, skipping {pol}")
            continue
        pool = _load_per_step(p)
        if not pool:
            print(f"[per-pol] WARN empty per_step_us in {p}, skipping {pol}")
            continue
        per_pol_pools[pol] = pool
        print(f"[per-pol] {pol}: n={len(pool)} P50={_percentile(pool, 50)/1000:.2f}ms "
              f"P99={_percentile(pool, 99)/1000:.2f}ms")

    rows: list[dict] = []
    for pol, pool in per_pol_pools.items():
        for sub_label, sub_ell, sub_sig, sub_q, sub_ef in SUBSTRATES:
            for seed in args.seeds:
                trace = _replay(pool, eps_phi, B_t, sub_ell, sub_sig,
                                sub_q, sub_ef, seed, args.n_steps_per_seed)
                p50 = _percentile(trace, 50)
                p99 = _percentile(trace, 99)
                p999 = _percentile(trace, 99.9)
                for d_ms in args.Ds_ms:
                    miss = _miss_at_d(trace, d_ms * 1000.0)
                    rows.append({
                        "policy": pol, "substrate": sub_label,
                        "phi": args.phi, "eps_phi": eps_phi,
                        "ell_bar_us": sub_ell, "q_escape": sub_q,
                        "B_t": B_t, "seed": seed, "D_ms": d_ms,
                        "miss": miss,
                        "p50_us": p50, "p99_us": p99, "p999_us": p999,
                    })

    # CSV
    out_csv = out_dir / "per_policy_substrate.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[per-pol] wrote {out_csv} ({len(rows)} rows)")

    # Aggregate per (policy, substrate, D) over seeds
    keys = sorted({(r["policy"], r["substrate"], float(r["D_ms"])) for r in rows})
    by: dict[tuple[str, str, float], list[dict]] = {}
    for r in rows:
        by.setdefault((r["policy"], r["substrate"], float(r["D_ms"])), []).append(r)

    summary = {}
    for k, rs in by.items():
        miss = sum(float(r["miss"]) for r in rs) / len(rs)
        p99 = sum(float(r["p99_us"]) for r in rs) / len(rs)
        summary[f"{k[0]}/{k[1]}/{k[2]}ms"] = {
            "miss_mean": miss,
            "p99_us_mean": p99,
            "n_seeds": len(rs),
        }
    with open(out_dir / "per_policy_substrate_summary.json", "w") as fh:
        json.dump({
            "phi": args.phi, "eps_phi": eps_phi, "B_t": B_t,
            "n_steps_per_seed": args.n_steps_per_seed,
            "by_cell": summary,
        }, fh, indent=2)
    print("[per-pol] wrote per_policy_substrate_summary.json")

    # Per-policy LaTeX table: substrate x policy x D=50ms miss
    pol_order = ["full", "streaming", "h2o", "quest", "seer"]
    pol_order = [p for p in pol_order if p in per_pol_pools]
    sub_order = [s[0] for s in SUBSTRATES]
    D_show = 50.0

    lines = [
        "% Generated by experiments/eC_bound_tightness/io_bound_per_policy.py",
        "% Per-policy chat-miss projection at D=50ms under three substrates,",
        "% phi=0.20. Base = measured eA b020 per_step_us per policy. IO",
        "% injection = Bernoulli(eps(0.20)) per block, B_t=" + str(B_t) + ".",
        "% Substrate cost: lognormal centred at vendor mean. Three seeds.",
        "% This is a deployment WHAT-IF, not a measured run.",
        "\\begin{tabular}{l" + "c" * len(sub_order) + "}",
        "\\toprule",
        "Policy & " + " & ".join(sub_order) + " \\\\",
        "\\midrule",
    ]
    for pol in pol_order:
        cells = []
        for sub in sub_order:
            rs = by.get((pol, sub, D_show), [])
            if not rs:
                cells.append("--")
                continue
            m = sum(float(r["miss"]) for r in rs) / len(rs)
            if m == 0:
                cells.append("$0$")
            elif m >= 0.01:
                cells.append(f"${m:.3f}$")
            else:
                cells.append(f"${m:.4f}$")
        name = pol if pol != "streaming" else "Streaming"
        lines.append(f"{name.upper() if pol == 'seer' else name.title()} & "
                     + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    out_tex = out_dir / "per_policy_substrate.tex"
    out_tex.write_text("\n".join(lines) + "\n")
    print(f"[per-pol] wrote {out_tex}")


if __name__ == "__main__":
    main()
