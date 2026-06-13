"""Per-step latency CDF — vLLM-async mode (J.2 / J.3 / stress).

Reads the per-task `stats` blocks from the chat results in
results_vllm_j3/ and results_vllm_stress/, treats each request's
P50/P90/P99/P999/max line as a 5-quantile sketch of its empirical
CDF, then renders a Patrick / ADSLab-style log-x CDF overlay
contrasting the light grid vs the stress grid.

Two figures land:
- `paper/figures/eF_vllm_cdf.pdf` — chat-stream CDFs at the J.3
  (light, mode-A) operating point.
- `paper/figures/eF_vllm_stress_cdf.pdf` — same at the stress
  (high-load) operating point. Shows the P999 tail jumping from
  ~24 ms to ~790 ms.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from seer import figstyle  # noqa: E402

# Anchor quantiles we recorded per-request in the JSON files.
QUANTILES = [
    ("p50_us", 0.50),
    ("p90_us", 0.90),
    ("p99_us", 0.99),
    ("p999_us", 0.999),
    ("max_us", 1.0),
]


def _load_chat_quantiles(results_dir: Path) -> dict[str, list[tuple[float, float]]]:
    """Map policy → list of (latency_ms, cumulative-prob) points
    pooled across all chat requests + all mix ratios."""
    pooled: dict[str, list[tuple[float, float]]] = {}
    for f in sorted(results_dir.glob("*.json")):
        if f.stem in ("summary",):
            continue
        d = json.load(open(f))
        if d.get("backend") != "vllm":
            continue
        policy = d.get("policy", "unknown")
        for r in d.get("results", []):
            if r.get("kind") != "chat":
                continue
            stats = r.get("stats", {})
            for key, q in QUANTILES:
                v = stats.get(key)
                if v is None:
                    continue
                pooled.setdefault(policy, []).append((v / 1000.0, q))
    return pooled


def _render(pooled: dict[str, list[tuple[float, float]]], out_path: Path,
            title_suffix: str, xlim: tuple[float, float] | None = None) -> None:
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(4.6, 2.6))

    policy_order = ["full", "streaming", "h2o", "seer"]
    for policy in policy_order:
        pts = pooled.get(policy)
        if not pts:
            continue
        pts.sort()  # sort by latency
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        # Empirical CDF of pooled quantiles: sort by x, take the
        # max y seen at each x to get a non-decreasing curve.
        order = np.argsort(xs)
        xs = xs[order]
        ys = np.maximum.accumulate(ys[order])
        s = figstyle.policy_style(policy)
        ax.plot(xs, ys,
                color=s["color"],
                linestyle=s["linestyle"],
                linewidth=s["lw"],
                label=figstyle.POLICY_PRETTY.get(policy, policy))

    ax.axvline(50, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
    # The "50 ms SLO" rider was floating in negative-x whitespace —
    # anchor it inside the axes at the top of the SLO knot.
    ax.text(50, 0.99, " 50 ms SLO", fontsize=6.5, color="0.35",
            ha="left", va="top",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.4))

    ax.set_xscale("log")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_ylim(0.4, 1.005)
    ax.set_xlabel("Per-step decode latency (ms, log scale)")
    ax.set_ylabel("Cumulative probability")
    ax.legend(loc="lower right", handlelength=2.4, labelspacing=0.2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"[per_step_cdf] wrote {out_path}")


def main() -> None:
    j3 = _load_chat_quantiles(_REPO / "experiments/eF_mixed_slo/results_vllm_j3")
    if j3:
        _render(j3, _REPO / "paper/figures/eF_vllm_cdf.pdf",
                title_suffix="J.3 light grid",
                xlim=(8, 200))

    stress = _load_chat_quantiles(_REPO / "experiments/eF_mixed_slo/results_vllm_stress")
    if stress:
        _render(stress, _REPO / "paper/figures/eF_vllm_stress_cdf.pdf",
                title_suffix="stress grid",
                xlim=(10, 2000))


if __name__ == "__main__":
    main()
