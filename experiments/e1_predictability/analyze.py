"""E1 analysis: AUC for LAP + heuristic baselines, emit GO/NO-GO verdict.

Writes `predictability.json` with per-horizon AUC for every method plus a
`summary` block containing the go/no-go decision and the gap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


# ----------- criteria --------------------------------------------------------

GO_ABS_THRESHOLD = 0.80     # LAP mean AUC must exceed this
GO_GAP_THRESHOLD = 0.05     # and beat best baseline by at least this much


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    from seer.lap.features import build_features
    from seer.lap.models import build_model
    from seer.trace.loader import load_traces, split_by_request
    from seer.trace.schema import HORIZONS

    # ---------- Data
    df = load_traces(args.traces)
    _, _, test_df = split_by_request(df)
    print(f"[analyze] test rows: {len(test_df):,}")
    X, y, meta = build_features(test_df)

    # ---------- Load LAP
    ckpt = torch.load(args.ckpt, map_location="cpu")
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

    # ---------- LAP AUC
    results = {"config": {"n_rows": int(len(X)),
                          "horizons": list(HORIZONS),
                          "model": ckpt["model_name"]},
               "lap": {},
               "baselines": {}}

    for h_idx, h in enumerate(HORIZONS):
        auc = _safe_auc(y[:, h_idx], logits[:, h_idx])
        results["lap"][f"h{h}"] = auc

    # ---------- Baselines — use raw features as scores
    fn = meta["feature_names"]
    # hist_0 = t-1 attention (the most recent past score)
    last_attn = X[:, fn.index("hist_0")]
    # persistence is the H2O-style cumulative proxy
    persistence = X[:, fn.index("persistence")]
    # negated log-recency: more recent → higher score
    recency_neg = -X[:, fn.index("recency_log")]
    # cumulative history sum (broader H2O-like proxy)
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

    # ---------- GO / NO-GO
    import numpy as np
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

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[analyze] LAP         mean AUC = {lap_mean:.3f}")
    print(f"[analyze] best BL ({best_baseline_name}) mean AUC = {best_baseline_mean:.3f}")
    print(f"[analyze] gap              = {gap:+.3f}")
    print(f"[analyze] VERDICT          = {verdict}")


def _safe_auc(y_true, scores) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return float("nan")


if __name__ == "__main__":
    main()
