"""E1 analysis: AUC for LAP + heuristic baselines, emit GO/NO-GO verdict.

Writes ``predictability.json`` with per-horizon AUC for every method, plus

* a ``summary`` block containing the GO / NO-GO decision and the gap;
* an ``schedulability_view`` block (RTSS pivot, P6.1) that maps the LAP
  AUC to a horizon-K false-negative rate ``epsilon`` (we use the simple
  ``epsilon ≈ 1 − AUC`` linkage) and feeds it through Lemma 2 at the
  default ``chat-50ms`` operating point. The table appears verbatim in
  paper §6.2 — "AUC ↔ schedulability bound".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GO_ABS_THRESHOLD = 0.80
GO_GAP_THRESHOLD = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    # RTSS pivot: schedulability projection knobs.
    ap.add_argument("--ell_bar_us", type=float, default=200.0,
                    help="Average miss-tier IO latency (DRAM proxy)")
    ap.add_argument("--sigma_residual_us", type=float, default=100.0,
                    help="Sub-Gaussian residual proxy (FFN+attn jitter)")
    ap.add_argument("--B_t", type=float, default=64.0,
                    help="Per-step working-set size")
    ap.add_argument("--base_cost_us", type=float, default=1500.0,
                    help="HBM-resident baseline per-step cost (mean)")
    ap.add_argument("--deadline_ms", type=float, default=50.0,
                    help="Per-token deadline (TPOT) for the schedulability "
                         "projection. Default = chat-50ms.")
    args = ap.parse_args()

    import numpy as np
    import torch

    from seer.lap.features import build_features
    from seer.lap.model import build_model  # canonical (singular) path
    from seer.timing.schedulability import lemma2_miss_prob_bound
    from seer.trace.loader import load_traces, split_by_request
    from seer.trace.schema import HORIZONS

    df = load_traces(args.traces)
    _, _, test_df = split_by_request(df)
    print(f"[analyze] test rows: {len(test_df):,}")
    X, y, meta = build_features(test_df)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_model(
        ckpt["model_name"],
        input_dim=meta["input_dim"],
        n_horizons=meta["output_dim"],
        history_n=meta.get("history_n", 32),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float()).numpy()

    results = {
        "config": {
            "n_rows": int(len(X)),
            "horizons": list(HORIZONS),
            "model": ckpt["model_name"],
            "ell_bar_us": args.ell_bar_us,
            "sigma_residual_us": args.sigma_residual_us,
            "B_t": args.B_t,
            "base_cost_us": args.base_cost_us,
            "deadline_ms": args.deadline_ms,
        },
        "lap": {},
        "baselines": {},
    }

    for h_idx, h in enumerate(HORIZONS):
        results["lap"][f"h{h}"] = _safe_auc(y[:, h_idx], logits[:, h_idx])

    fn = meta["feature_names"]
    last_attn = X[:, fn.index("hist_0")]
    persistence = X[:, fn.index("persistence")]
    recency_neg = -X[:, fn.index("recency_log")]
    cum_hist = X[:, [fn.index(f"hist_{i}") for i in range(32)]].sum(axis=1)

    for name, score in [
        ("last_attn", last_attn),
        ("persistence_h2o", persistence),
        ("recency", recency_neg),
        ("cumulative_history", cum_hist),
    ]:
        results["baselines"][name] = {}
        for h_idx, h in enumerate(HORIZONS):
            results["baselines"][name][f"h{h}"] = _safe_auc(y[:, h_idx], score)

    lap_mean = float(np.nanmean(list(results["lap"].values())))
    baseline_means = {
        name: float(np.nanmean(list(vals.values())))
        for name, vals in results["baselines"].items()
    }
    best_baseline_name = max(baseline_means, key=baseline_means.get)
    best_baseline_mean = baseline_means[best_baseline_name]
    gap = lap_mean - best_baseline_mean
    verdict = ("GO" if (lap_mean >= GO_ABS_THRESHOLD and gap >= GO_GAP_THRESHOLD)
               else "NO-GO")
    results["summary"] = {
        "lap_mean_auc": lap_mean,
        "best_baseline": best_baseline_name,
        "best_baseline_mean_auc": best_baseline_mean,
        "baseline_means": baseline_means,
        "gap": gap,
        "verdict": verdict,
        "criteria": {
            "abs_threshold": GO_ABS_THRESHOLD,
            "gap_threshold": GO_GAP_THRESHOLD,
        },
    }

    # ---- RTSS P6.1 — AUC ↔ Lemma 2 bound projection -------------------
    deadline_us = args.deadline_ms * 1000.0
    sched: dict[str, object] = {
        "deadline_us": deadline_us,
        "B_t": args.B_t,
        "ell_bar_us": args.ell_bar_us,
        "sigma_residual_us": args.sigma_residual_us,
        "base_cost_us": args.base_cost_us,
        "lap": {},
        "baselines": {},
    }
    for h in HORIZONS:
        auc = results["lap"][f"h{h}"]
        if auc != auc:  # NaN
            sched["lap"][f"h{h}"] = None
            continue
        eps = max(0.0, min(1.0, 1.0 - float(auc)))
        bound = lemma2_miss_prob_bound(
            epsilon=eps,
            ell_bar_us=args.ell_bar_us,
            B_t=args.B_t,
            deadline_us=deadline_us,
            sigma_residual_us=args.sigma_residual_us,
            base_cost_us=args.base_cost_us,
        )
        sched["lap"][f"h{h}"] = {
            "auc": float(auc),
            "epsilon_proxy": float(eps),
            "lemma2_miss_bound": float(bound),
        }
    for name, vals in results["baselines"].items():
        sched["baselines"][name] = {}
        for h in HORIZONS:
            auc = vals[f"h{h}"]
            if auc != auc:
                sched["baselines"][name][f"h{h}"] = None
                continue
            eps = max(0.0, min(1.0, 1.0 - float(auc)))
            bound = lemma2_miss_prob_bound(
                epsilon=eps,
                ell_bar_us=args.ell_bar_us,
                B_t=args.B_t,
                deadline_us=deadline_us,
                sigma_residual_us=args.sigma_residual_us,
                base_cost_us=args.base_cost_us,
            )
            sched["baselines"][name][f"h{h}"] = {
                "auc": float(auc),
                "epsilon_proxy": float(eps),
                "lemma2_miss_bound": float(bound),
            }
    results["schedulability_view"] = sched

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[analyze] LAP         mean AUC = {lap_mean:.3f}")
    print(f"[analyze] best BL ({best_baseline_name}) mean AUC = {best_baseline_mean:.3f}")
    print(f"[analyze] gap              = {gap:+.3f}")
    print(f"[analyze] VERDICT          = {verdict}")
    print(f"[analyze] schedulability_view (chat-{args.deadline_ms:g}ms):")
    for h, payload in sched["lap"].items():
        if payload is None:
            continue
        print(f"  LAP {h}: AUC={payload['auc']:.3f}  ε≈{payload['epsilon_proxy']:.3f}  "
              f"Lemma2 miss bound={payload['lemma2_miss_bound']:.3e}")


def _safe_auc(y_true, scores) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return float("nan")


if __name__ == "__main__":
    main()
