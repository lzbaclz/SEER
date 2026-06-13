"""Empirical check of Lemma 2's assumption A1 (per-block miss
independence).

R2-W4 follow-through (third review round): Lemma 2 assumes per-block
miss indicators $M_i$ are independent across $i$ within a decode step
(`paper/lemmas.tex:43-45`). LLM long-context attention is high-sparsity
and spatially correlated, so the reviewer asks whether the bound is
still an upper envelope when correlations are accounted for.

This script computes:

1. The empirical cross-block correlation matrix of `is_top_k` indicators
   on the Llama-2 training trace (per layer, per head_group), and
   reports its mean / median / max off-diagonal value.

2. The same correlation for the LAP false-negative indicator
   $M_i = is\\_top\\_k(t,i) \\wedge \\neg future\\_top\\_k\\_h1(t-K, i)$,
   using `future_top_k_h1` as a surrogate for the LAP prediction
   (it is the deterministic horizon-1 label the LAP was trained to
   match).

3. A correlation-aware Bernstein-mixture bound that replaces the
   i.i.d. variance term $\\nu = \\sum_i \\epsilon_i(1-\\epsilon_i)\\ell_i^2$
   with the correlation-adjusted form
   $\\nu_\\rho = \\nu + \\sum_{i \\ne j} \\rho_{ij}
   \\sqrt{\\epsilon_i(1-\\epsilon_i)} \\sqrt{\\epsilon_j(1-\\epsilon_j)}
   \\ell_i \\ell_j$
   and shows the resulting pessimism delta on the headline grid.

If the empirical $\\bar\\rho$ is small (say $\\le 0.05$), the bound
is robust to A1 in this regime; if not, the corollary 2'' in
`A1_proofs.tex` is the documented mitigation.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def avg_pairwise_corr(M: np.ndarray) -> tuple[float, float, float]:
    """Average / median / 95th-percentile of off-diagonal pairwise
    Pearson correlation across columns of M (rows = steps, cols =
    blocks). Skips zero-variance columns."""
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] < 2 or M.shape[1] < 2:
        return 0.0, 0.0, 0.0
    # Drop zero-variance columns.
    std = M.std(axis=0)
    keep = std > 1e-9
    M = M[:, keep]
    if M.shape[1] < 2:
        return 0.0, 0.0, 0.0
    R = np.corrcoef(M, rowvar=False)
    iu = np.triu_indices(R.shape[0], k=1)
    rhos = R[iu]
    rhos = rhos[~np.isnan(rhos)]
    if rhos.size == 0:
        return 0.0, 0.0, 0.0
    return (float(np.mean(rhos)),
            float(np.median(rhos)),
            float(np.percentile(rhos, 95)))


def lag_autocorr(M: np.ndarray, lags: tuple[int, ...] = (1, 2, 4, 8),
                 ) -> dict[int, float]:
    """R6-W4: per-block lag-k temporal autocorrelation of the FN
    indicator (mean across blocks). Each column of M is a block's
    FN time-series; we compute the Pearson lag-k autocorrelation per
    column, then average across columns. Zero-variance columns are
    skipped."""
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] < max(lags) + 2 or M.shape[1] < 1:
        return {lag: 0.0 for lag in lags}
    std = M.std(axis=0)
    keep = std > 1e-9
    M = M[:, keep]
    if M.shape[1] == 0:
        return {lag: 0.0 for lag in lags}
    out: dict[int, float] = {}
    for lag in lags:
        if M.shape[0] <= lag + 1:
            out[lag] = 0.0
            continue
        # Per-block: Pearson(x_t, x_{t-lag}).
        rhos = []
        for j in range(M.shape[1]):
            x_lag = M[:-lag, j]
            x_now = M[lag:, j]
            if x_lag.std() < 1e-9 or x_now.std() < 1e-9:
                continue
            r = np.corrcoef(x_lag, x_now)[0, 1]
            if not np.isnan(r):
                rhos.append(float(r))
        out[lag] = float(np.mean(rhos)) if rhos else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir",
                    default="data/traces/llama2_train_24")
    ap.add_argument("--max_req", type=int, default=24)
    ap.add_argument("--out_dir",
                    default="experiments/eC_bound_tightness/results")
    ap.add_argument("--layer_sample", nargs="+", type=int,
                    default=[0, 8, 16, 24, 31])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquets = sorted(glob.glob(f"{args.trace_dir}/req_*.parquet"))[
        :args.max_req]
    print(f"[A1-corr] loading {len(parquets)} requests from {args.trace_dir}")

    # Aggregate per (layer, head_group) miss-indicator matrices.
    is_top_k_rows = {lyr: [] for lyr in args.layer_sample}
    fn_indicator_rows = {lyr: [] for lyr in args.layer_sample}

    for p in parquets:
        df = pd.read_parquet(p, columns=["layer_id", "head_group", "block_id",
                                          "step", "is_top_k",
                                          "future_top_k_h1"])
        # Within a fixed (layer, head_group) at a step, the block index
        # ranges over blocks of the working set; this is the right
        # granularity for Lemma 2's per-block miss-indicator vector.
        # We always pin head_group=0 to avoid pooling correlated heads.
        for layer in args.layer_sample:
            sub = df[(df["layer_id"] == layer) & (df["head_group"] == 0)]
            if sub.empty:
                continue
            pivot_truth = sub.pivot_table(
                index="step", columns="block_id", values="is_top_k",
                aggfunc="max", fill_value=0).to_numpy()
            pivot_fhat = sub.pivot_table(
                index="step", columns="block_id", values="future_top_k_h1",
                aggfunc="max", fill_value=0).to_numpy()
            # FN indicator = (block is in truth's top-k) AND (LAP would
            # have missed it = future_top_k_h1 is 0).
            fn = pivot_truth * (1 - pivot_fhat)
            is_top_k_rows[layer].append(pivot_truth.astype(np.float64))
            fn_indicator_rows[layer].append(fn.astype(np.float64))

    per_layer = {}
    for layer in args.layer_sample:
        if not is_top_k_rows[layer]:
            continue
        # Stack with NaN-padding for differing block dims.
        max_b = max(m.shape[1] for m in is_top_k_rows[layer])
        truth_stack = []
        fn_stack = []
        for m, f in zip(is_top_k_rows[layer], fn_indicator_rows[layer]):
            pad = max_b - m.shape[1]
            if pad > 0:
                m = np.hstack([m, np.zeros((m.shape[0], pad))])
                f = np.hstack([f, np.zeros((f.shape[0], pad))])
            truth_stack.append(m)
            fn_stack.append(f)
        truth_M = np.vstack(truth_stack)
        fn_M = np.vstack(fn_stack)
        rho_t_mean, rho_t_med, rho_t_p95 = avg_pairwise_corr(truth_M)
        rho_f_mean, rho_f_med, rho_f_p95 = avg_pairwise_corr(fn_M)
        # R6-W4: lag-k temporal autocorrelation of the FN indicator
        # (per-block, averaged across blocks).
        fn_lag = lag_autocorr(fn_M, lags=(1, 2, 4, 8))
        # Empirical FN rate.
        fn_rate = float(fn_M.mean())
        truth_rate = float(truth_M.mean())
        per_layer[layer] = {
            "n_steps_pooled": int(truth_M.shape[0]),
            "n_blocks": int(truth_M.shape[1]),
            "truth_rate": truth_rate,
            "fn_rate": fn_rate,
            "rho_truth_mean": rho_t_mean,
            "rho_truth_median": rho_t_med,
            "rho_truth_p95": rho_t_p95,
            "rho_fn_mean": rho_f_mean,
            "rho_fn_median": rho_f_med,
            "rho_fn_p95": rho_f_p95,
            "rho_fn_lag1": fn_lag.get(1, 0.0),
            "rho_fn_lag2": fn_lag.get(2, 0.0),
            "rho_fn_lag4": fn_lag.get(4, 0.0),
            "rho_fn_lag8": fn_lag.get(8, 0.0),
        }
        print(f"[A1-corr] layer={layer}: "
              f"truth top-k rate={truth_rate:.3f}, FN rate={fn_rate:.3f}, "
              f"truth bar(rho)={rho_t_mean:+.3f} (median {rho_t_med:+.3f}, "
              f"p95 {rho_t_p95:+.3f}); "
              f"FN bar(rho)={rho_f_mean:+.3f} (median {rho_f_med:+.3f}, "
              f"p95 {rho_f_p95:+.3f}); "
              f"FN lag-(1,2,4,8) autocorr=("
              f"{fn_lag.get(1, 0.0):+.3f}, {fn_lag.get(2, 0.0):+.3f}, "
              f"{fn_lag.get(4, 0.0):+.3f}, {fn_lag.get(8, 0.0):+.3f})")

    # Corollary 2'': correlation-adjusted Bernstein-mixture variance
    # term. The original $\nu$ is $\sum \epsilon_i(1-\epsilon_i)\ell_i^2$;
    # with uniform $\bar\rho$, the adjusted term is approximately
    # $\nu(1 + (n_t-1)\bar\rho)$ at the worst-case-correlated regime.
    # We report the inflation factor relative to the headline grid
    # $B_t \in {32, 64, 128, 256}$ and $\bar\rho$ at the median value.
    if per_layer:
        rho_med = float(np.median(
            [v["rho_fn_median"] for v in per_layer.values()]))
        rho_mean = float(np.mean(
            [v["rho_fn_mean"] for v in per_layer.values()]))
    else:
        rho_med = 0.0
        rho_mean = 0.0

    inflation_grid: list[dict] = []
    for n_t in (32, 64, 128, 256, 512):
        inflation_grid.append({
            "n_t": n_t,
            "rho_used": rho_med,
            "variance_inflation_factor": 1.0 + (n_t - 1) * max(0.0, rho_med),
            "bound_inflation_factor": math.sqrt(
                1.0 + (n_t - 1) * max(0.0, rho_med)),  # since the bound
            # decays exponentially in variance, the geometric mid-point
            # is sqrt of the variance ratio.
        })
        print(f"[A1-corr] B_t={n_t}: variance inflation "
              f"{inflation_grid[-1]['variance_inflation_factor']:.2f}x, "
              f"bound inflation "
              f"{inflation_grid[-1]['bound_inflation_factor']:.2f}x")

    summary = {
        "n_requests": len(parquets),
        "layers_sampled": args.layer_sample,
        "per_layer": per_layer,
        "summary": {
            "rho_fn_median_across_layers": rho_med,
            "rho_fn_mean_across_layers": rho_mean,
            "interpretation": (
                "If rho is close to 0, Lemma 2's A1 i.i.d. assumption is "
                "approximately valid; the Bernstein-mixture variance term "
                "is unchanged. Positive correlation inflates the variance "
                "and the bound becomes optimistic by sqrt(1 + (n_t-1) rho); "
                "Corollary 2'' (A1_proofs.tex) reports the inflation."
            ),
        },
        "inflation_grid": inflation_grid,
    }
    out_json = out_dir / "a1_correlation_summary.json"
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[A1-corr] wrote {out_json}")

    # TeX table for the paper.
    lines = [
        "% Generated by experiments/eC_bound_tightness/a1_correlation_check.py",
        "% Empirical check of Lemma 2's A1 i.i.d. assumption.",
        "% Per-layer cross-block correlation of the FN indicator on the",
        "% Llama-2 training trace. R2-W4 follow-through.",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Layer & $n_\text{steps}$ & $n_\text{blocks}$ & "
        r"$\bar\rho_\text{FN}$ (mean) & median & p95 & FN rate \\",
        r"\midrule",
    ]
    for layer in args.layer_sample:
        v = per_layer.get(layer)
        if v is None:
            continue
        lines.append(
            f"{layer} & {v['n_steps_pooled']} & {v['n_blocks']} & "
            f"${v['rho_fn_mean']:+.3f}$ & "
            f"${v['rho_fn_median']:+.3f}$ & "
            f"${v['rho_fn_p95']:+.3f}$ & "
            f"${v['fn_rate']:.3f}$ \\\\"
        )
    lines.extend([
        r"\midrule",
        f"\\multicolumn{{7}}{{l}}{{\\textit{{Median $\\bar\\rho_\\text{{FN}}"
        f"$ across layers: ${rho_med:+.3f}$; bound inflation "
        f"$\\sqrt{{1 + (B_t-1)\\bar\\rho}}$ at $B_t{{=}}256$: "
        f"${inflation_grid[3]['bound_inflation_factor']:.2f}\\times$"
        f"}}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ])
    out_tex = out_dir / "a1_correlation.tex"
    with open(out_tex, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[A1-corr] wrote {out_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
