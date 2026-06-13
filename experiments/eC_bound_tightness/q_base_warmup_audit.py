"""R19 warm-up contamination audit on the legacy 1200-sample pool.

R18 reported a refit $q_\\mathrm{base}=0.0$ on a fresh 200-prompt
Mooncake pool and claimed the original $0.0067$ was
\\emph{warm-up contaminated}. The R18 reviewer correctly pointed
out that this is a post-hoc explanation: the 200-prompt pool
also drops the warm-up step, so the comparison does not isolate
the warm-up factor on the same pool.

This script does the controlled comparison on the \\emph{same}
legacy IO-free pool used in R12-R16
(\\texttt{experiments/eA\\_tail\\_latency/results\\_iofree/}):

* Variant A: include all per-step samples including step 0
  (no warm-up exclusion). This is the maximally-pessimistic
  reading of the pool.
* Variant B: drop step 0 per request (R12-R16 canonical, what
  \\texttt{io\\_bound\\_regime.\\_load\\_base\\_distribution} actually
  does).
* Variant C: drop steps 0--4 per request (more aggressive
  warm-up exclusion).
* Variant D: drop steps 0--9 (very aggressive).

For each variant we report $\\hat q_\\mathrm{base}$ at the same
$p_{50}+3\\sigma$ threshold the headline calibrator uses, plus
the Clopper-Pearson 95\\% upper bound. If the warm-up hypothesis
holds, $q_\\mathrm{base}(A) \\gg q_\\mathrm{base}(B,C,D)$. If not,
$q_\\mathrm{base}$ stays roughly invariant across variants and
the R18 prose must be retracted.

Output: \\texttt{q\\_base\\_warmup\\_audit.\\{json,tex\\}}.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/eC_bound_tightness/results"
IOFREE_DIR = ROOT / "experiments/eA_tail_latency/results_iofree"


def _load_per_request(iofree_dir: pathlib.Path) -> list[list[float]]:
    """Return list of per-request per-step latency lists from
    the legacy IO-free pool (matches io_bound_regime._load_base_distribution
    selection but keeps the per-request grouping)."""
    grouped: list[list[float]] = []
    for fname in ("seer_b100_iofree.json", "seer_b020_decomp.json"):
        p = iofree_dir / fname
        if not p.is_file():
            continue
        d = json.loads(p.read_text())
        for r in d.get("results", []):
            ps = r.get("per_step_base_us") or r.get("per_step_us") or []
            ps = [float(x) for x in ps if x == x]
            if len(ps) >= 2:
                grouped.append(ps)
    return grouped


def _q_base_with_ci(xs: list[float],
                     heavy_tail_factor: float = 3.0) -> dict:
    """Use the canonical io_bound_regime estimator so the result
    matches R12-R16's calibrated 0.0067 numerically (R19 reviewer
    point: my earlier reimplementation used sample-stdev while the
    paper uses (P99.5 - P50) / sqrt(2 ln 200) as the sigma proxy)."""
    if len(xs) < 50:
        return {"q_base": 0.0, "n": len(xs), "n_over": 0,
                "ci_hi": 1.0, "note": "n<50"}
    from experiments.eC_bound_tightness import io_bound_regime as iobr
    p50 = iobr._percentile(xs, 50)
    sigma = iobr._empirical_sigma_us(xs)
    thresh = p50 + heavy_tail_factor * sigma
    n_over = sum(1 for x in xs if x > thresh)
    n = len(xs)
    q = n_over / n
    if n_over == 0:
        ci_hi = 1.0 - 0.05 ** (1.0 / n)
        ci_method = "rule-of-three"
    else:
        try:
            from scipy.stats import beta
            ci_hi = float(beta.ppf(0.975, n_over + 1, n - n_over))
        except Exception:
            # Wilson upper
            z = 1.96
            phat = n_over / n
            denom = 1 + z * z / n
            centre = (phat + z * z / (2 * n)) / denom
            half = (z / denom) * math.sqrt(
                phat * (1 - phat) / n + z * z / (4 * n * n))
            ci_hi = min(1.0, centre + half)
        ci_method = "Clopper-Pearson"
    return {
        "q_base": q, "n": n, "n_over": n_over,
        "p50_us": p50, "sigma_us": sigma, "threshold_us": thresh,
        "ci_hi_95": ci_hi, "ci_method": ci_method,
    }


def _drop_first_k(grouped: list[list[float]], k: int) -> list[float]:
    pool: list[float] = []
    for ps in grouped:
        if len(ps) > k:
            pool.extend(ps[k:])
    return pool


def main() -> int:
    grouped = _load_per_request(IOFREE_DIR)
    if not grouped:
        print(f"[q-base-warmup-audit] no IO-free pool at {IOFREE_DIR}")
        return 1
    variants = []
    for k, label in [(0, "include-all"), (1, "drop-1 (R12-R16)"),
                     (5, "drop-5"), (10, "drop-10")]:
        pool = _drop_first_k(grouped, k)
        res = _q_base_with_ci(pool)
        res["drop_first"] = k
        res["label"] = label
        variants.append(res)
        print(f"[warmup-audit] {label:>16s}: n={res['n']:>5d}, "
              f"n_over={res['n_over']:>3d}, "
              f"q_base={res['q_base']:.4f}, "
              f"95\\% upper={res['ci_hi_95']:.2e}")

    out = {
        "iofree_dir": str(IOFREE_DIR.relative_to(ROOT)),
        "n_requests": len(grouped),
        "calibrated_q_base_2024": 0.0067,
        "variants": variants,
        "interpretation": (
            "The 'drop-1' variant is what io_bound_regime._load_"
            "base_distribution does (and matches the R12-R16 0.0067 "
            "calibration). 'include-all' tests the warm-up "
            "contamination hypothesis: if q_base(include-all) >> "
            "q_base(drop-1), the original 0.0067 is dominated by "
            "step-0 outliers."),
    }
    out_json = RESULTS / "q_base_warmup_audit.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"[warmup-audit] wrote {out_json}")

    # TeX rendering.
    lines = [
        "% Auto-generated by q_base_warmup_audit.py (R19).",
        "% Tests whether the R12-R16 q_base=0.0067 is dominated by",
        "% warm-up step contamination, by comparing q_base under",
        "% different drop-warm-up policies on the SAME legacy pool.",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Variant & $n$ & $n_\mathrm{over}$ & "
        r"$\hat q_\mathrm{base}$ & $95\%$ upper \\",
        r"\midrule",
    ]
    for v in variants:
        ci_str = f"${v['ci_hi_95']:.2e}$"
        lines.append(
            rf"\texttt{{{v['label']}}} & {v['n']} & {v['n_over']} & "
            rf"${v['q_base']:.4f}$ & {ci_str} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_tex = RESULTS / "q_base_warmup_audit.tex"
    out_tex.write_text("\n".join(lines) + "\n")
    print(f"[warmup-audit] wrote {out_tex}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
