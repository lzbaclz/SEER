"""Production-trace measurement of attention-pattern shift magnitude
$\\sigma_\\text{shift}$ across consecutive decode-windows.

R4 W2 fix: the reactive-lag witness (\\cref{tab:reactive-lag})
activates only at $\\sigma \\ge 0.75$. Reviewers asked: what
fraction of production decode-windows have $\\sigma \\ge 0.75$? If
the answer is non-trivial (say $\\ge 1\\%$), Lemma 3 is
production-relevant for the workload tail.

Protocol:
  * Load every Llama-2 trace under ``data/traces/llama2_train_24/``
    (24 requests, 32 layers, 32 heads, 64 decode steps).
  * Aggregate attention over heads at a single middle-layer
    (layer 16 by default; the LAP was trained on per-(layer, head)
    features but we use the layer-aggregate for the shift-magnitude
    summary because $\\sigma$ is a property of the active block
    set, not of the predictor).
  * For each decode-step pair $(t, t+W)$ where $W = 8$ (one decode
    window of 8 token steps; matches block_size = 32 tokens for a
    Llama-2 generation rate of ~4 tokens/decode-block-shift):
      - $\\text{score}_b(t) = \\sum_{s \\in [t, t+W)} \\text{attn}(b, s)$
      - $\\text{top}_B(t) = \\text{top-B blocks by score}$
      - $\\sigma(t) = 1 - |\\text{top}_B(t) \\cap \\text{top}_B(t+W)| / B$
  * Aggregate $\\sigma$ over all $(request, t)$ pairs; report
    histogram, quantiles, fraction-above-thresholds.

Outputs:
  ``results/sigma_distribution_summary.json`` — quantiles + fractions.
  ``results/sigma_distribution.tex`` — small table for the paper.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def measure_request(df_req: pd.DataFrame, *, layer_id: int, window: int,
                    budget: int) -> list[float]:
    """Return one sigma value per decode-window pair within this request."""
    df_layer = df_req[df_req["layer_id"] == layer_id]
    if df_layer.empty:
        return []
    # Aggregate attention over heads.
    attn_per_block_step = (
        df_layer.groupby(["block_id", "step"])["attn_score"]
        .sum()
        .reset_index()
    )
    n_blocks = int(df_layer["block_id"].max()) + 1
    n_steps = int(df_layer["step"].max()) + 1
    grid = np.zeros((n_blocks, n_steps), dtype=np.float32)
    for _, row in attn_per_block_step.iterrows():
        grid[int(row["block_id"]), int(row["step"])] = float(row["attn_score"])
    # Slide a window of size W over the time axis; compare top-B
    # of the window starting at t vs the window starting at t + W.
    out: list[float] = []
    for t in range(0, n_steps - 2 * window + 1):
        w_old = grid[:, t : t + window].sum(axis=1)
        w_new = grid[:, t + window : t + 2 * window].sum(axis=1)
        # Drop blocks whose window-sum is zero in BOTH windows
        # (cold padding); they would dominate ties otherwise.
        live_mask = (w_old > 0) | (w_new > 0)
        live_idx = np.where(live_mask)[0]
        if len(live_idx) < budget:
            continue  # too few live blocks to measure top-B
        old_top = set(int(b) for b in
                      live_idx[np.argsort(-w_old[live_idx])[:budget]])
        new_top = set(int(b) for b in
                      live_idx[np.argsort(-w_new[live_idx])[:budget]])
        overlap = len(old_top & new_top) / budget
        sigma = 1.0 - overlap
        out.append(sigma)
    return out


def measure_cross_request(grids: list[np.ndarray], *,
                          budget: int) -> list[float]:
    """For each ordered pair (request_i, request_j) with i != j,
    compute sigma = 1 - |top_B(req_i, all-steps) cap top_B(req_j,
    all-steps)| / B. This captures the across-request shift that
    happens when a fresh conversation / prefill arrives. Top-B is
    computed over the entire decode-window of each request."""
    n = len(grids)
    if n < 2:
        return []
    block_set = max(g.shape[0] for g in grids)
    summed: list[np.ndarray] = []
    for g in grids:
        s = g.sum(axis=1)  # per-block total attention
        # Pad to common block_set so set ops are well defined.
        if s.shape[0] < block_set:
            s = np.concatenate([s, np.zeros(block_set - s.shape[0],
                                            dtype=s.dtype)])
        summed.append(s)
    out: list[float] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            live = (summed[i] > 0) | (summed[j] > 0)
            idx = np.where(live)[0]
            if len(idx) < budget:
                continue
            ti = set(int(b) for b in idx[np.argsort(-summed[i][idx])[:budget]])
            tj = set(int(b) for b in idx[np.argsort(-summed[j][idx])[:budget]])
            sigma = 1.0 - len(ti & tj) / budget
            out.append(sigma)
    return out


def collect_grid(df: pd.DataFrame, *, layer_id: int) -> np.ndarray:
    """Build the (block_id, step) per-block attention grid, summed
    over heads for the given layer."""
    df_layer = df[df["layer_id"] == layer_id]
    if df_layer.empty:
        return np.zeros((0, 0), dtype=np.float32)
    agg = (df_layer.groupby(["block_id", "step"])["attn_score"]
                   .sum().reset_index())
    n_blocks = int(df_layer["block_id"].max()) + 1
    n_steps = int(df_layer["step"].max()) + 1
    grid = np.zeros((n_blocks, n_steps), dtype=np.float32)
    for _, row in agg.iterrows():
        grid[int(row["block_id"]), int(row["step"])] = float(row["attn_score"])
    return grid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", default="data/traces/llama2_train_24")
    ap.add_argument("--layer_id", type=int, default=16,
                    help="middle layer; per-layer sigma is similar but "
                         "layer-16 attention is most policy-relevant")
    ap.add_argument("--window", type=int, default=8,
                    help="decode-window size in steps; W=8 matches "
                         "~1 generated KV block at block_size=32 tokens")
    ap.add_argument("--budget", type=int, default=16,
                    help="top-B; matches the budget used in "
                         "reactive_lag.py and the paper's bound sweep")
    ap.add_argument("--out_dir",
                    default="experiments/eD_adversarial/results")
    args = ap.parse_args()

    trace_dir = Path(args.trace_dir)
    files = sorted(trace_dir.glob("req_*.parquet"))
    if not files:
        raise SystemExit(f"no parquet traces under {trace_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_sigmas: list[float] = []
    grids: list[np.ndarray] = []
    print(f"[sigma-dist] loading {len(files)} traces from {trace_dir}")
    for f in files:
        df = pd.read_parquet(f, columns=[
            "layer_id", "block_id", "step", "attn_score",
        ])
        sigmas = measure_request(df, layer_id=args.layer_id,
                                 window=args.window, budget=args.budget)
        all_sigmas.extend(sigmas)
        grids.append(collect_grid(df, layer_id=args.layer_id))
    sigmas_arr = np.asarray(all_sigmas, dtype=np.float64)
    cross_sigmas = np.asarray(
        measure_cross_request(grids, budget=args.budget), dtype=np.float64)
    if sigmas_arr.size == 0 and cross_sigmas.size == 0:
        raise SystemExit("no sigma samples produced; check trace contents")

    # R5-residual W5-r: per-request grids retained for trace-level
    # bootstrap (resample requests with replacement, recompute
    # cross_request fraction).
    req_grids = grids
    req_in_sigmas: list[list[float]] = []
    # Build per-request in-request sigmas to enable trace-level
    # bootstrap of the in-request fraction too.
    for f in files:
        df = pd.read_parquet(f, columns=[
            "layer_id", "block_id", "step", "attn_score",
        ])
        s = measure_request(df, layer_id=args.layer_id,
                            window=args.window, budget=args.budget)
        req_in_sigmas.append(s)

    def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
        """Wilson 95% CI on a binomial fraction at sample size n."""
        if n <= 0:
            return (0.0, 0.0)
        denom = 1.0 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * (np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
        return (max(0.0, centre - half), min(1.0, centre + half))

    def _qfracs(arr: np.ndarray) -> tuple[dict, dict]:
        if arr.size == 0:
            return {}, {}
        q = {f"p{p}": float(np.percentile(arr, p))
             for p in (25, 50, 75, 90, 95, 99)}
        f: dict = {}
        # R5-W5: Wilson 95% CI on each fraction; small-N (n=552
        # cross-request pairs from n=24 requests) acknowledged.
        for thr_str, thr in (("0p25", 0.25), ("0p50", 0.50),
                              ("0p75", 0.75), ("0p90", 0.90),
                              ("1p00", 0.999)):
            mean_p = float((arr >= thr).mean())
            lo, hi = _wilson_ci(mean_p, int(arr.size))
            f[f"frac_ge_{thr_str}"] = mean_p
            f[f"frac_ge_{thr_str}_ci95_lo"] = lo
            f[f"frac_ge_{thr_str}_ci95_hi"] = hi
        return q, f

    quantiles_in, fractions_in = _qfracs(sigmas_arr)
    quantiles_cr, fractions_cr = _qfracs(cross_sigmas)

    # R5-residual W5-r: trace-level bootstrap. Resample requests
    # (not pairs) with replacement, recompute Pr[sigma>=0.75] on the
    # resampled trace. Captures the trace-level uncertainty the
    # Wilson CI on the 552 binomial pairs misses.
    def _bootstrap_trace_level(seed: int = 0, n_boot: int = 1000):
        rng = np.random.default_rng(seed)
        n_req = len(req_grids)
        frac_in_list = []
        frac_cr_list = []
        mean_cr_list = []
        for _ in range(n_boot):
            idx = rng.integers(0, n_req, n_req)
            # In-request: pool sigmas from resampled requests.
            in_pool: list[float] = []
            for i in idx:
                in_pool.extend(req_in_sigmas[i])
            in_pool_arr = np.asarray(in_pool, dtype=np.float64)
            frac_in = (float((in_pool_arr >= 0.75).mean())
                       if in_pool_arr.size else 0.0)
            frac_in_list.append(frac_in)
            # Cross-request: recompute from resampled grid list.
            resampled_grids = [req_grids[i] for i in idx]
            cr_sigmas = measure_cross_request(resampled_grids,
                                               budget=args.budget)
            cr_arr = np.asarray(cr_sigmas, dtype=np.float64)
            if cr_arr.size:
                frac_cr_list.append(float((cr_arr >= 0.75).mean()))
                mean_cr_list.append(float(cr_arr.mean()))
        return frac_in_list, frac_cr_list, mean_cr_list

    print(f"[sigma-dist] trace-level bootstrap (n_boot=1000, "
          f"resample {len(req_grids)} requests) ...")
    boot_in, boot_cr, boot_cr_mean = _bootstrap_trace_level()
    trace_level_ci = {
        "in_request_frac_ge_0p75_ci95": [
            float(np.quantile(boot_in, 0.025)) if boot_in else None,
            float(np.quantile(boot_in, 0.975)) if boot_in else None,
        ],
        "cross_request_frac_ge_0p75_ci95": [
            float(np.quantile(boot_cr, 0.025)) if boot_cr else None,
            float(np.quantile(boot_cr, 0.975)) if boot_cr else None,
        ],
        "cross_request_mean_sigma_ci95": [
            float(np.quantile(boot_cr_mean, 0.025)) if boot_cr_mean else None,
            float(np.quantile(boot_cr_mean, 0.975)) if boot_cr_mean else None,
        ],
        "n_boot": len(boot_cr),
    }
    print(f"  trace-level CI cross-request Pr[sigma>=0.75]: "
          f"[{trace_level_ci['cross_request_frac_ge_0p75_ci95'][0]:.3f}, "
          f"{trace_level_ci['cross_request_frac_ge_0p75_ci95'][1]:.3f}]")
    summary = {
        "trace_dir": str(trace_dir),
        "n_requests": len(files),
        "layer_id": args.layer_id,
        "window": args.window,
        "budget": args.budget,
        "in_request": {
            "n_window_pairs": int(sigmas_arr.size),
            "mean_sigma": float(sigmas_arr.mean()) if sigmas_arr.size else None,
            "quantiles": quantiles_in,
            "fraction_above_threshold": fractions_in,
        },
        "cross_request": {
            "n_pairs": int(cross_sigmas.size),
            "mean_sigma": float(cross_sigmas.mean()) if cross_sigmas.size else None,
            "quantiles": quantiles_cr,
            "fraction_above_threshold": fractions_cr,
        },
        "trace_level_bootstrap": trace_level_ci,
        "protocol": (
            "in_request: sliding decode-window pairs of size W on the "
            "per-(block,step) attention aggregated over heads at layer L; "
            "sigma = 1 - top-B set overlap; cold blocks excluded. "
            "cross_request: per-request all-step top-B sets pairwise; "
            "captures the shift between unrelated decoding trajectories "
            "(prefill -> decode boundary or fresh conversation)."
        ),
    }
    with open(out_dir / "sigma_distribution_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[sigma-dist] wrote {out_dir / 'sigma_distribution_summary.json'}")
    if sigmas_arr.size:
        print(f"[sigma-dist] in-request: n={sigmas_arr.size} "
              f"mean={sigmas_arr.mean():.3f} "
              f"p99={quantiles_in['p99']:.3f} "
              f"frac>=0.75={fractions_in['frac_ge_0p75']:.3f}")
    if cross_sigmas.size:
        print(f"[sigma-dist] cross-request: n={cross_sigmas.size} "
              f"mean={cross_sigmas.mean():.3f} "
              f"p99={quantiles_cr['p99']:.3f} "
              f"frac>=0.75={fractions_cr['frac_ge_0p75']:.3f}")

    # Tex table — two-regime summary with binomial Wilson CI +
    # trace-level bootstrap CI (R5-W5 / R5-residual W5-r).
    def _row(arr: np.ndarray, q: dict, f: dict, label: str,
             trace_lo: float | None, trace_hi: float | None) -> list[str]:
        if arr.size == 0:
            return [f"{label} & --- & --- & --- & --- & --- \\\\"]
        frac = f["frac_ge_0p75"]
        ci_lo = f.get("frac_ge_0p75_ci95_lo", float("nan"))
        ci_hi = f.get("frac_ge_0p75_ci95_hi", float("nan"))
        trace_str = ("---" if trace_lo is None else
                     f"$[{100*trace_lo:.1f}, {100*trace_hi:.1f}]$")
        return [
            f"{label} & ${arr.size}$ & "
            f"${arr.mean():.3f}$ & "
            f"${q['p99']:.3f}$ & "
            f"${100*frac:.1f}\\%$ "
            f"$[{100*ci_lo:.1f}, {100*ci_hi:.1f}]$ & "
            f"{trace_str} \\\\"
        ]

    lines = [
        "% Generated by experiments/eD_adversarial/sigma_distribution.py",
        "% Production-trace sigma-shift distribution (Llama-2 24-request",
        "% trace, layer-16, budget=16). Two regimes:",
        "%  - in-request: sliding (W,W) decode-window pairs within each",
        "%    request (captures within-decode attention drift).",
        "%  - cross-request: pairwise top-B-set overlap between distinct",
        "%    requests (captures the prefill -> decode boundary or fresh",
        "%    conversation shift, which is the regime Lemma 3 protects).",
        r"\begin{tabular}{lrrrll}",
        r"\toprule",
        r"Regime & $n$ & $\mathbb{E}[\sigma]$ & $p_{99}$ & "
        r"$\Pr[\sigma \ge 0.75]$ (Wilson $95\%$ CI) & trace-level $95\%$ CI \\",
        r"\midrule",
    ]
    trace_in_lo, trace_in_hi = trace_level_ci[
        "in_request_frac_ge_0p75_ci95"]
    trace_cr_lo, trace_cr_hi = trace_level_ci[
        "cross_request_frac_ge_0p75_ci95"]
    lines.extend(_row(sigmas_arr, quantiles_in, fractions_in,
                      "in-request (sliding $W{=}8$)",
                      trace_in_lo, trace_in_hi))
    lines.extend(_row(cross_sigmas, quantiles_cr, fractions_cr,
                      "cross-request",
                      trace_cr_lo, trace_cr_hi))
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    with open(out_dir / "sigma_distribution.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[sigma-dist] wrote {out_dir / 'sigma_distribution.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
