"""修改 8 — Fit H2O's effective discount γ_eff from cumulative-attention
scores on a single representative request, addressing reviewer m4
(Lemma 3' τ_π=156 cycles claim uses an unverified γ_eff=0.94).

Methodology
-----------
H2O accumulates per-block attention masses s_b(t) over decode steps.
At decision cadence K_dec, the cumulative score follows
    s_b(t) = γ · s_b(t-K_dec) + a_b(t)
where a_b(t) is the new attention mass in that K_dec-step window.
γ_eff is fit via OLS on the lagged regression:
    y = s_b(t) - a_b(t)   (left-shifted-by-a)
    x = s_b(t-K_dec)
    γ_eff = Σxy / Σx² .

We collect the (s, a) trajectory by running H2O through the masking
simulator with score logging enabled. The script writes
results/h2o_gamma_fit.json with median γ_eff + per-family distribution.

Requires running the simulator with a small instrumentation hook on
``seer.policy.baselines.H2OPolicy`` that exports the per-block score
deque at each decision (see _attach_score_logger below).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def fit_gamma(s_traj: list[float], a_traj: list[float]) -> float | None:
    """Lagged OLS on the cumulative-attention recurrence.

    Returns γ_eff = Σ s_{t-1}·(s_t - a_t) / Σ s_{t-1}², or None if
    the trajectory is too short or x has no variance.
    """
    s = np.asarray(s_traj, dtype=float)
    a = np.asarray(a_traj, dtype=float)
    n = min(len(s), len(a))
    if n < 5:
        return None
    s, a = s[:n], a[:n]
    x = s[:-1]
    y = s[1:] - a[1:]
    if x.var() < 1e-12:
        return None
    return float(np.dot(x, y) / np.dot(x, x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score_traces", default="results/h2o_score_traces.jsonl",
                    help="JSONL file: one line per (family, prompt_id, "
                         "layer, block) with {scores: [...], "
                         "attn_increments: [...]}")
    ap.add_argument("--out", default="results/h2o_gamma_fit.json")
    args = ap.parse_args()

    src = Path(__file__).parent / args.score_traces
    if not src.exists():
        print(f"[fit] {src} not found.")
        print("[fit] To produce it, instrument seer.policy.baselines.H2OPolicy")
        print("[fit] with a score_logger callback that appends one line per")
        print("[fit] (family, prompt, layer, block) to this jsonl, then re-run")
        print("[fit] experiments/eD_adversarial/run.sh.")
        return 1

    by_family: dict[str, list[float]] = {}
    with open(src) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            g = fit_gamma(r.get("scores", []), r.get("attn_increments", []))
            if g is None or not (0 < g < 1.5):
                continue
            by_family.setdefault(r.get("family", "?"), []).append(g)

    summary = {}
    all_g = []
    for fam, gs in by_family.items():
        arr = np.asarray(gs)
        summary[fam] = {
            "n": int(arr.size),
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
        }
        all_g.extend(gs)
    if all_g:
        a = np.asarray(all_g)
        summary["_overall"] = {
            "n": int(a.size),
            "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90)),
        }
        print(f"γ_eff overall: median={np.median(a):.3f}  "
              f"P10/P90 = [{np.percentile(a,10):.3f}, {np.percentile(a,90):.3f}]  "
              f"(n={a.size} fits)")

    out_path = Path(__file__).parent / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[fit] wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
