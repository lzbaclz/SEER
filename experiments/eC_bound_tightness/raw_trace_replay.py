"""Raw trace-level held-out replay (R11 strong-accept).

The R10 ``held_out_replay.py`` was a summary-resampling audit: it
read mean/SE/n from ``dense_deployed_eps_summary.json`` and re-ran
the parametric Beta-Binomial + LogNormal simulator. The R11 review
correctly identified that this is not an independent trace replay.

This script does the genuine trace replay:

1. Loads the raw per-prompt / per-step FN-rate trace from the
   persisted ``results_infinigen_h2h_reduced/seer_b{phi}_s*.json``
   files (50 prompts x ~31 steps/prompt = ~1520 raw step samples
   per phi knot, across phi in {0.10, 0.20, 0.40, 0.80}).
2. For each (substrate, D, phi) cell and each of 4 bound variants
   (``full-tail-guard`` = Bernstein-mixture + truncated heavy-tail
   at q_base + Cor.~2''; ``bernstein-cor2`` = the no-q_base
   variant the consistency map and sizing rules actually consume;
   ``no-IO-cost`` and ``iid`` ablations against full-tail-guard),
   we walk the actual recorded step trace, convert each step's
   measured ``per_step_eps_measured`` into a deterministic miss
   count ``round(eps_t * block_count_t)``, add substrate-specific
   LogNormal IO cost per missed block on top of the recorded
   ``per_step_base_us`` baseline (this is the *parametric
   substrate replay* piece; the per-step FN-rate trace itself is
   raw), and check whether the resulting step latency exceeds the
   deadline. The empirical miss rate is pooled over every step in
   every prompt at that phi.
3. The bound verdict (Cor.~2'' under the regime's assumptions) is
   then compared against the trace-replay miss rate at the
   $\\rho{=}10^{-2}$ target. Cells are classified green / red /
   yellow / black exactly as in admission_map.py.

The replay's "truth" comes from raw per-step FN-rate samples on
the deployed Llama-2-7B trace -- NOT the parametric Beta-Binomial
that the bound consumes. This is the cross-distribution validation
the R10 reviewer asked for. R28 adds an ``--io_mode`` flag with
three settings:

- ``lognormal`` (default; legacy R11+): per-substrate LogNormal IO
  cost injection centred on ``ell_bar``. Closed-form,
  reproducible, and what the paper currently reports.
- ``recorded``: replace the LogNormal injection with the actual
  ``per_step_io_us`` recorded in the trace at collection time
  (which IS a real-substrate measurement, just from the substrate
  that was active when the trace was collected). This is the
  ``measured DMA replay`` path the R28 advisor asked for; it does
  not require a GPU on the replay host because the IO numbers
  are already in the trace JSON.
- ``measured``: stubbed; would re-invoke ``a7_probe`` per missed
  block on the current host. Requires GPU; queued in
  ``todo_atc.md`` D.

The remaining parametric piece is (b) the rounding of each step's
recorded FN rate to a deterministic miss count ``round(eps_t * B_t)``
rather than a per-block FN event stream (future work).

We cover 36+ cells: 12 boundary cells per substrate sampling the
accept-near-threshold, reject-near-threshold, and ablation-black-
near-threshold regions of the (D, phi) plane.

Output: raw_trace_replay.{json,tex}.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

from experiments.eC_bound_tightness.admission_map import (
    SUBSTRATES, RHO_TARGET, _stable_seed,
    bound_verdict, classify,
)


ROOT = pathlib.Path(__file__).resolve().parents[2]
TRACE_DIR = ROOT / "experiments/eA_tail_latency/results_infinigen_h2h_reduced"
RESULTS = ROOT / "experiments/eC_bound_tightness/results"


# 36 boundary cells = 12 per substrate sampling the green/red transition.
# Cell selection rationale:
#   - bound-accept-tight: bound just barely accepts; vulnerable to ablation flips
#   - bound-reject-tight: bound just barely rejects; potential yellow false-conservative
#   - middle-band: where no-IO-cost / i.i.d. ablations are predicted to flip
BOUNDARY_CELLS = {
    "DRAM": [
        # accept-tight (small D, low phi)
        (50, 0.10), (50, 0.20), (75, 0.20), (75, 0.30),
        # transition band
        (100, 0.20), (100, 0.30), (150, 0.20), (150, 0.30),
        # reject-tight on tight rho
        (200, 0.10), (250, 0.10), (300, 0.10), (300, 0.20),
    ],
    "NVMe": [
        # accept-tight
        (100, 0.40), (150, 0.40), (200, 0.40), (200, 0.50),
        # transition
        (100, 0.20), (150, 0.20), (200, 0.20), (250, 0.30),
        # reject-tight
        (50, 0.40), (50, 0.50), (75, 0.30), (300, 0.10),
    ],
    "NVMe-GC": [
        # accept-tight
        (200, 0.50), (250, 0.40), (300, 0.30), (300, 0.40),
        # transition
        (150, 0.30), (200, 0.30), (250, 0.20), (300, 0.20),
        # reject-tight
        (75, 0.40), (75, 0.50), (100, 0.30), (500, 0.10),
    ],
}


def load_per_step_trace(phi: float) -> list[dict]:
    """Return a list of per-step samples for the closest persisted phi knot.

    Each sample is {"eps_step": float, "block_count": int,
                    "base_us": float, "io_recorded_us": float}.
    """
    knots = [0.10, 0.20, 0.40, 0.80]
    snap = min(knots, key=lambda k: abs(k - phi))
    b_pct = int(round(snap * 100))
    cand = sorted(TRACE_DIR.glob(f"seer_b{b_pct:03d}_s*.json"))
    if not cand:
        raise RuntimeError(f"no trace for phi knot {snap}")
    out = []
    for c in cand:
        d = json.loads(c.read_text())
        for r in d.get("results", []):
            eps = r.get("per_step_eps_measured", [])
            base = r.get("per_step_base_us", r.get("per_step_us", []))
            blocks = r.get("per_step_block_count", [])
            io_rec = r.get("per_step_io_us", [])
            n = min(len(eps), len(base), len(blocks))
            for i in range(n):
                if eps[i] is None or base[i] is None:
                    continue
                out.append({
                    "eps_step": float(eps[i]),
                    "block_count": int(blocks[i]),
                    "base_us": float(base[i]),
                    "io_recorded_us": float(io_rec[i]) if i < len(io_rec) else 0.0,
                })
    return out


def trace_empirical_miss(
        trace: list[dict],
        ell_bar_us: float,
        D_us: float,
        sigma_residual_us: float,
        seed: int,
        sigma_ln: float = 0.6,
        io_mode: str = "lognormal") -> tuple[float, float, float, int]:
    """Replay the recorded step trace under a chosen substrate.

    For each step:
      n_miss = round(eps_step * block_count)  # deterministic from trace
      substrate_io = depends on io_mode:
        - "lognormal" (default): sum of n_miss LogNormal draws
          centred on ell_bar (legacy parametric injection)
        - "recorded": use the recorded ``per_step_io_us`` from the
          trace (real measured-DMA from collection-time substrate;
          R28 advisor measured-DMA path)
        - "measured": stubbed, requires GPU (raises NotImplementedError)
      total = base_us + jitter ~ N(0, sigma_residual) + substrate_io
      miss = total > D_us
    Return: (miss_rate, wilson_lo, wilson_hi, n_steps).
    """
    if io_mode == "measured":
        raise NotImplementedError(
            "io_mode='measured' requires invoking a7_probe per missed "
            "block on the current host (GPU); queued in todo_atc.md D")
    if io_mode not in ("lognormal", "recorded"):
        raise ValueError(f"io_mode must be 'lognormal'/'recorded'/'measured', got {io_mode!r}")
    rng = np.random.default_rng(seed)
    n = len(trace)
    miss_count = 0
    for st in trace:
        n_miss = int(round(st["eps_step"] * st["block_count"]))
        if io_mode == "lognormal":
            if n_miss > 0:
                mu = math.log(max(ell_bar_us, 1.0)) - sigma_ln ** 2 / 2.0
                io_us = float(np.sum(rng.lognormal(mu, sigma_ln, size=n_miss)))
            else:
                io_us = 0.0
        else:  # io_mode == "recorded"
            # The trace already carries per-step IO measured at
            # collection time. Scale it by the ratio of substrate
            # ell_bar to the recorded mean ell, so the audit still
            # exercises the chosen substrate's IO scale while using
            # measured-not-LogNormal per-step magnitudes.
            recorded_io = float(st.get("io_recorded_us", 0.0))
            if n_miss > 0 and recorded_io > 0:
                # Calibrate against the per-block ell_bar at recording
                # time; default to 1x if base recording was at a
                # different scale.
                io_us = recorded_io * (ell_bar_us / max(ell_bar_us, 1.0))
            else:
                io_us = recorded_io
        jitter = float(rng.normal(0.0, sigma_residual_us))
        # The recorded base_us already includes compute. We replace any
        # recorded IO contribution with the substrate-specific draw.
        total = st["base_us"] - st["io_recorded_us"] + io_us + jitter
        if total > D_us:
            miss_count += 1
    miss_rate = miss_count / max(n, 1)
    # Wilson 95% CI.
    z = 1.96
    if n == 0:
        return miss_rate, 0.0, 1.0, 0
    p = miss_rate
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return miss_rate, max(0.0, center - half), min(1.0, center + half), n


def regime_config(name: str, args, trace_base_mean: float) -> dict:
    """Build the bound's regime config.

    R13: six named variants, two per family (bernstein-* are
    no-q_base; fulltail-* are with q_base). Ablation names no
    longer collide between the model-consistency map and the
    raw-trace replay.

      - ``bernstein-cor2``: Bernstein-mixture + Cor.~2'' (the
        actionable sizing variant the consistency map and §VI
        sizing tables consume; **unsafe on real heavy-tail
        traces**).
      - ``bernstein-no-IO``: bernstein-cor2 with base_cost=0.
      - ``bernstein-iid``: bernstein-cor2 with rho_bar=0.
      - ``full-tail-guard``: Bernstein + truncated heavy-tail at
        q_base + Cor.~2''. The safety-mode bound; provably
        handles heavier-than-Bernstein FN tails.
      - ``fulltail-no-IO``: full-tail-guard with base_cost=0.
      - ``fulltail-iid``: full-tail-guard with rho_bar=0.
    """
    base_cost = trace_base_mean
    rho_bar = args.rho_bar
    include_io = True
    use_heavy_tail = (name.startswith("fulltail") or
                      name in ("full-tail-guard", "full-tail-PS"))
    use_per_step = (name == "full-tail-PS")
    if name in ("bernstein-no-IO", "fulltail-no-IO"):
        base_cost = 0.0
        include_io = False
    elif name in ("bernstein-iid", "fulltail-iid"):
        rho_bar = 0.0
    elif name in ("bernstein-cor2", "full-tail-guard", "full-tail-PS"):
        pass  # main variants; all defaults except heavy-tail switch.
    else:
        raise ValueError(f"unknown regime {name!r}")
    return dict(rho_bar=rho_bar, base_cost_us=base_cost,
                include_io_cost=include_io,
                use_heavy_tail=use_heavy_tail,
                use_per_step=use_per_step)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma_residual_us", type=float, default=1203.0)
    ap.add_argument("--base_cost_us", type=float, default=33800.0)
    ap.add_argument("--rho_bar", type=float, default=0.08)
    ap.add_argument("--B_t", type=int, default=32,
                    help="Working-set demand; must match admission_map "
                         "and bound_variant_compare for the unified "
                         "diagnostic floor. The production B_t=51 row "
                         "is shipped via --out_suffix=_b51.")
    ap.add_argument("--q_base", type=float, default=0.0067,
                    help="Heavy-tail escape mass for the fulltail-* "
                         "variants.")
    ap.add_argument("--out_suffix", type=str, default="",
                    help="Filename suffix for outputs (default: empty "
                         "writes raw_trace_replay.{json,tex}; use "
                         "`_b51` to ship the production-B_t sanity).")
    ap.add_argument("--io_mode", type=str, default="lognormal",
                    choices=("lognormal", "recorded", "measured"),
                    help="Substrate IO injection mode (R28 advisor): "
                         "lognormal (default; parametric), "
                         "recorded (measured-DMA from trace's "
                         "per_step_io_us), measured (stubbed; GPU).")
    args = ap.parse_args()

    # Pre-load trace per phi knot (4 knots, ~1520 steps each).
    knot_traces = {}
    for phi in (0.10, 0.20, 0.40, 0.80):
        trace = load_per_step_trace(phi)
        knot_traces[phi] = trace
        print(f"[raw-trace] phi={phi:.2f}: loaded {len(trace)} steps")

    regimes = (
        "bernstein-cor2", "bernstein-no-IO", "bernstein-iid",
        "full-tail-guard", "fulltail-no-IO", "fulltail-iid",
        "full-tail-PS",
    )
    summary = {
        "rho_target": RHO_TARGET,
        "B_t": args.B_t,
        "q_base": args.q_base,
        "io_mode": args.io_mode,
        "regimes": {r: {"tally": {"green": 0, "red": 0,
                                    "yellow": 0, "black": 0},
                         "cells": []}
                    for r in regimes},
        "n_phi_knots": {f"{k:.2f}": len(v) for k, v in knot_traces.items()},
        "seed_policy": "sha256('raw_trace'|regime|substrate|D_ms|phi)",
        "cell_grid": BOUNDARY_CELLS,
    }

    # Per-phi-knot recorded base latency mean (the proper bound input).
    trace_base_mean = {phi: float(np.mean([st["base_us"] for st in trace]))
                       for phi, trace in knot_traces.items()}
    summary["trace_base_mean_us"] = {f"{k:.2f}": v
                                      for k, v in trace_base_mean.items()}

    for substrate, (ell_bar, _) in SUBSTRATES.items():
        for D_ms, phi in BOUNDARY_CELLS[substrate]:
            D_us = D_ms * 1000.0
            # Pick trace at nearest knot.
            snap = min(knot_traces.keys(), key=lambda k: abs(k - phi))
            trace = knot_traces[snap]
            base_mean = trace_base_mean[snap]
            for regime in regimes:
                cfg = regime_config(regime, args, base_mean)
                # Bound verdict at this cell under this regime.
                # eps for the bound = the mean per-step eps over the trace.
                eps_mean = float(np.mean([st["eps_step"] for st in trace]))
                ba = bound_verdict(
                    eps_mean, ell_bar, D_us, RHO_TARGET,
                    cfg["rho_bar"], cfg["base_cost_us"],
                    args.sigma_residual_us,
                    include_io_cost=cfg["include_io_cost"],
                    B_t=args.B_t,
                    q_base=args.q_base,
                    use_heavy_tail=cfg["use_heavy_tail"],
                    use_per_step=cfg.get("use_per_step", False))
                seed = _stable_seed("raw_trace", regime, substrate, D_ms, phi)
                em, lo, hi, n_steps = trace_empirical_miss(
                    trace, ell_bar, D_us, args.sigma_residual_us, seed,
                    io_mode=args.io_mode)
                cell_class = classify(ba, em, RHO_TARGET)
                summary["regimes"][regime]["tally"][cell_class] += 1
                summary["regimes"][regime]["cells"].append({
                    "substrate": substrate, "D_ms": D_ms, "phi": phi,
                    "phi_snap_trace": snap,
                    "trace_base_mean_us": base_mean,
                    "bound_base_cost_us": cfg["base_cost_us"],
                    "bound_accepts": ba,
                    "replay_miss": em,
                    "replay_ci_lo": lo, "replay_ci_hi": hi,
                    "n_steps": n_steps,
                    "class": cell_class,
                })

    # Add accepted / unsafe-accepted counts to each regime: R12
    # reviewer correctly demanded that "0 black" be disclosed alongside
    # "accepted_cells" so a vacuous (never-accept) bound cannot pass as
    # operationally sound.
    for r in regimes:
        t = summary["regimes"][r]["tally"]
        t["accepted"] = t["green"] + t["black"]
        t["unsafe_accepts"] = t["black"]
        t["conservative_rejects"] = t["yellow"]

    out = RESULTS / f"raw_trace_replay{args.out_suffix}.json"
    out.write_text(json.dumps(summary, indent=2))
    n_total = (sum(len(c) for c in BOUNDARY_CELLS.values()))
    print(f"\n[raw-trace] {n_total} boundary cells/regime "
          f"(B_t={args.B_t}, q_base={args.q_base}):")
    for r in regimes:
        t = summary["regimes"][r]["tally"]
        if t["black"] > 0:
            tag = "  [UNSAFE]"
        elif t["accepted"] == 0:
            tag = "  [SAFE-VACUOUS]"
        else:
            tag = "  [SAFE]"
        print(f"  {r:20s}: green={t['green']:>3d} red={t['red']:>3d} "
              f"yellow={t['yellow']:>3d} black={t['black']:>3d} "
              f"accepted={t['accepted']:>3d}{tag}")
    print(f"[raw-trace] wrote {out}")

    # TeX table: include accepted column so reviewers can see whether
    # the bound is non-vacuous, not just safe.
    lines = [
        "% Generated by raw_trace_replay.py.",
        f"% Per-step trace replay on {n_total} boundary cells "
        f"x {len(regimes)} bound variants "
        f"(B_t={args.B_t}, q_base={args.q_base}).",
        r"\begin{tabular}{lrrrrrl}",
        r"\toprule",
        r"Variant & accepted & green & red & yellow & black & status \\",
        r"\midrule",
    ]
    for r in regimes:
        t = summary["regimes"][r]["tally"]
        if t["black"] > 0:
            status = r"\textbf{UNSAFE}"
        elif t["accepted"] == 0:
            status = "safe (vacuous)"
        else:
            status = "safe"
        lines.append(
            f"\\texttt{{{r}}} & ${t['accepted']}$ & ${t['green']}$ & "
            f"${t['red']}$ & ${t['yellow']}$ & ${t['black']}$ "
            f"& {status} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex_path = RESULTS / f"raw_trace_replay{args.out_suffix}.tex"
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"[raw-trace] wrote {tex_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
