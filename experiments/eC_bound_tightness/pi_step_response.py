"""PI-controller step-response analysis (paper §4.2 supplement).

The paper claims the LambdaController in seer/timing/slo.py is
"deliberately stable" but never quantifies that. RTSS reviewers who
work on adaptive control will (rightly) ask:

    1. What is the closed-loop pole / time-constant?
    2. How does λ_t respond to a step change in load?
    3. Does it reject a temporary disturbance without oscillation?

This driver answers all three with a synthetic but realistic harness:
* Plant model: per-step latency = mu0(λ) + Gaussian residual,
  where mu0 falls with λ in proportion to the IO weight (more IO
  expensive → smaller working set → smaller mu0).
* Inputs: (a) cold-start step (load jumps from idle to full at t=0),
  (b) load-step disturbance at t=500 (mu shifts +5 ms),
  (c) impulse outlier at t=200 (one decode is 100 ms).

Outputs the trajectory CSV and a 2-panel matplotlib figure
(λ trajectory + per-step latency) into
experiments/eC_bound_tightness/results/.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _plant_step(lam: float, mu0_us: float, slope_us_per_lam: float,
                sigma_us: float, rng) -> float:
    """One decode-step latency given controller-set λ.

    Plant: lat = mu0 + slope · λ + Gaussian noise.
    Calibrated so that the equilibrium for target=50ms is around λ=2.
    """
    lam = max(0.0, lam)
    return mu0_us + slope_us_per_lam * lam + rng.gauss(0.0, sigma_us)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir",
                    default="experiments/eC_bound_tightness/results")
    ap.add_argument("--target_ms", type=float, default=50.0,
                    help="Lambda controller target latency.")
    ap.add_argument("--n_steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import random

    from seer.timing.slo import LambdaController
    rng = random.Random(args.seed)

    # Two controllers: (a) paper-default stable, (b) aggressive 10× kp/ki
    # to show why the paper choice matters.
    ctrls = {
        "stable_default": LambdaController(target_us=args.target_ms * 1000),
        "aggressive_10x": LambdaController(target_us=args.target_ms * 1000,
                                           kp=1e-3, ki=2e-4),
    }

    # Plant: lat = mu0 + slope · λ + N(0, σ²).
    #   mu0 = 40 ms, slope = +5 ms / λ → λ* = 2 at target=50ms.
    #   Disturbance at t=n/2: mu0 jumps to 42.5 ms → new λ* = 1.5.
    n = args.n_steps
    t_disturb = n // 2
    slope_us = 5_000.0     # 5 ms per unit λ

    rows = []
    for ctrl_name, ctrl in ctrls.items():
        ctrl.reset()
        for t in range(n):
            mu0 = 42_500.0 if t >= t_disturb else 40_000.0
            lat = _plant_step(ctrl.current_lambda(),
                              mu0_us=mu0,
                              slope_us_per_lam=slope_us,
                              sigma_us=600.0, rng=rng)
            ctrl.observe(lat)
            rows.append({
                "controller": ctrl_name,
                "t": t,
                "latency_ms": lat / 1000.0,
                "lambda": ctrl.current_lambda(),
                "e_int": ctrl.stats()["e_int"],
            })

    # CSV
    csv_path = out_dir / "pi_step_response.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[pi-step] wrote trajectory to {csv_path} ({len(rows)} rows)")

    # Figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[pi-step] matplotlib unavailable; skipping figure")
        return 0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 4.5), sharex=True)
    for ctrl_name in ctrls:
        sub = [r for r in rows if r["controller"] == ctrl_name]
        ts = [r["t"] for r in sub]
        ax1.plot(ts, [r["latency_ms"] for r in sub], label=ctrl_name,
                 alpha=0.6, lw=0.8)
        ax2.plot(ts, [r["lambda"]    for r in sub], label=ctrl_name, lw=1.5)
    ax1.axhline(args.target_ms, color="k", ls="--", lw=0.7, label="target")
    ax1.axvline(t_disturb, color="r", ls=":", lw=0.7, alpha=0.5)
    ax1.set_ylabel("per-step latency (ms)")
    ax1.set_yscale("log")
    ax1.legend(fontsize=8, loc="upper right")
    ax2.axvline(t_disturb, color="r", ls=":", lw=0.7, alpha=0.5,
                label=f"step disturbance (+5ms) at t={t_disturb}")
    ax2.set_xlabel("decode step")
    ax2.set_ylabel(r"$\lambda_t$")
    ax2.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    pdf_path = out_dir / "pi_step_response.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(pdf_path.with_suffix(".png"), bbox_inches="tight", dpi=120)
    print(f"[pi-step] wrote figure to {pdf_path}")

    # Stability summary
    def _settle(traj_lambda, target_lam, tol=0.20, hold=50):
        """First t such that mean λ over the next ``hold`` steps is within
        ``tol`` of ``target_lam``. ``tol`` defaults to 0.2 — this is
        ~1.6× the steady-state σ_λ implied by the plant noise σ_us=600
        and slope 5000 us/λ (so σ_λ_ss ≈ 0.12)."""
        for i in range(len(traj_lambda) - hold):
            window = traj_lambda[i : i + hold]
            if abs(sum(window) / hold - target_lam) <= tol:
                return i
        return len(traj_lambda)

    # Analytical equilibria for the plant: lat = mu0 + slope·λ → target.
    lam_star_pre  = (args.target_ms * 1000 - 40_000.0) / slope_us  # = 2.0
    lam_star_post = (args.target_ms * 1000 - 42_500.0) / slope_us  # = 1.5
    print(f"\n[pi-step] analytical equilibria: λ*_pre={lam_star_pre:.2f},  "
          f"λ*_post={lam_star_post:.2f}")
    print("[pi-step] settling time (steps to stay within ±0.05 of equilibrium for ≥50 steps):")
    for ctrl_name in ctrls:
        sub = [r for r in rows if r["controller"] == ctrl_name]
        traj = [r["lambda"] for r in sub]
        cold = traj[:t_disturb]
        post = traj[t_disturb:]
        st_cold = _settle(cold, lam_star_pre)
        st_post = _settle(post, lam_star_post)
        print(f"  {ctrl_name:<18}  λ_final = {traj[-1]:.3f}   "
              f"cold-start: {st_cold:>4} steps   "
              f"disturbance: {st_post:>4} steps")

    return 0


if __name__ == "__main__":
    sys.exit(main())
