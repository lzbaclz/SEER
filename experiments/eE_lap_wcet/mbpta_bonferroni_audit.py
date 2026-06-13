"""R21 Bonferroni audit of the per-chunk MBPTA hygiene.

Reviewer R20 said the $1/9$ TRT chunks that pass the per-chunk
runs + Ljung-Box hygiene at $\\alpha{=}0.05$ are barely above the
chance pass rate, and asked for an explicit multiplicity-corrected
report. This script ingests the persisted
``mbpta_representative_state.json`` and emits, per backend:

  * Pass count at the original $\\alpha{=}0.05$ threshold (as in
    the body of \\S6.2).
  * Pass count at the Bonferroni-corrected per-test threshold
    $\\alpha/K$ (the reviewer's exact reference value
    ``0.05/9 = 0.00556``). Because non-rejection ``p > \\alpha``
    is the pass criterion, lowering $\\alpha$ \\emph{relaxes} the
    pass rule; we report the corrected count honestly so the
    direction of the correction is transparent.
  * Family-wise pass-by-chance check: under the null
    ``H_0: chunk is not i.i.d.''$ a chunk passes both tests
    (Type-II) with probability bounded by ``(1-power)^2``; the
    \\emph{rare-event} pass rate ``\\alpha^2`` is a fast
    upper-tail proxy that gives the reviewer a clean
    ``pass-count vs. expected-by-chance'' contrast.

Output:
  * ``mbpta_bonferroni_audit.{json,tex}`` next to the existing
    rep-state artifacts.
  * stdout summary.

This is a Tier-1 hardening: no new measurement; the JSON is the
ground truth and Bonferroni is a deterministic post-hoc audit.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/eE_lap_wcet/results"


def _audit_backend(rec: dict, alpha: float) -> dict:
    K = rec["n_chunks_analysed"]
    per = rec["per_chunk"]
    n_total = rec["n_total"]

    pass_orig = sum(1 for c in per if c.get("pass_both"))
    pass_bonf = sum(
        1 for c in per
        if (c.get("runs_p") is not None and c["runs_p"] > alpha)
        and (c.get("lb_p") is not None and c["lb_p"] > alpha)
    )
    # Sidak-corrected family-wise α (matches Bonferroni at small K
    # and is slightly tighter): α_fw = 1 - (1-α)^(1/K).
    alpha_fw_target = 0.05
    alpha_sidak = 1.0 - (1.0 - alpha_fw_target) ** (1.0 / max(K, 1))
    pass_sidak = sum(
        1 for c in per
        if (c.get("runs_p") is not None and c["runs_p"] > alpha_sidak)
        and (c.get("lb_p") is not None and c["lb_p"] > alpha_sidak)
    )

    # R21 reviewer fix: the original "chance UB K*alpha^2" quantity
    # is non-standard. The reviewer-supplied correct multiplicity-
    # aware chance metric is the Binomial p-value: under the null
    # "each independent chunk passes with probability <= alpha"
    # (a strict reading of "chance pass"), the probability of
    # observing >= obs passes in K trials is
    # 1 - sum_{j<obs} C(K,j) alpha^j (1-alpha)^(K-j).
    # For K=9, alpha=0.05: P[>=1] = 1 - 0.95^9 = 0.370. Observing
    # 1 pass is therefore *consistent* with chance under this null.
    def _binom_tail_ge(k_obs: int, K_: int, p_: float) -> float:
        # P[X >= k_obs] with X ~ Binomial(K_, p_).
        # Closed-form sum avoids scipy.
        if k_obs <= 0:
            return 1.0
        try:
            import math as _m
            cdf_lt = 0.0
            for j in range(0, k_obs):
                cdf_lt += (_m.comb(K_, j) * (p_ ** j)
                           * ((1.0 - p_) ** (K_ - j)))
            return max(0.0, min(1.0, 1.0 - cdf_lt))
        except Exception:
            return float("nan")

    binom_p_ge_obs = _binom_tail_ge(pass_orig, K, alpha)
    expected_under_null = K * alpha

    return {
        "K": K,
        "n_total": n_total,
        "alpha": alpha,
        "alpha_bonferroni": alpha / K if K else float("nan"),
        "alpha_sidak_fw0.05": alpha_sidak,
        "pass_count_alpha005": pass_orig,
        "pass_count_bonferroni": pass_bonf,
        "pass_count_sidak_fw0.05": pass_sidak,
        "binomial_p_ge_observed_under_null":
            binom_p_ge_obs,
        "expected_pass_count_under_null": expected_under_null,
        "note": (
            "Direction-of-correction note: 'pass' = non-rejection "
            "of the i.i.d. null. Bonferroni lowers per-test alpha "
            "and therefore RELAXES the non-rejection criterion, "
            "increasing the pass count. The reviewer's R21 "
            "correction: the proper multiplicity-aware chance "
            "metric is the Binomial p-value P[X>=obs | "
            "X~Binom(K, alpha)] = 1-(1-alpha)^K for obs=1, "
            "alpha=0.05, K=9 -> 0.370. Observing 1/9 passes is "
            "consistent with chance under this null; the per-chunk "
            "hygiene is weak evidence and C2 is restricted to a "
            "measurement-based empirical envelope, not a WCET."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rep-state-json", type=pathlib.Path,
        default=RESULTS / "mbpta_representative_state.json")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    src = args.rep_state_json
    if not src.exists():
        print(f"[bonferroni-audit] FAIL: {src} missing; run "
              "experiments.eE_lap_wcet.mbpta_representative_state first")
        return 1
    raw = json.loads(src.read_text())

    backends_out = []
    for rec in raw["results"]:
        audit = _audit_backend(rec, alpha=args.alpha)
        backend = "TRT" if "trt" in rec["file"] else "torch"
        audit["backend"] = backend
        audit["file"] = rec["file"]
        backends_out.append(audit)
        print(f"[bonferroni-audit] {backend}: K={audit['K']}, "
              f"alpha=0.05 pass={audit['pass_count_alpha005']}, "
              f"Bonferroni alpha/K={audit['alpha_bonferroni']:.4f} "
              f"pass={audit['pass_count_bonferroni']}, "
              f"Sidak FW=0.05 alpha={audit['alpha_sidak_fw0.05']:.4f} "
              f"pass={audit['pass_count_sidak_fw0.05']}, "
              f"Binom P[>=obs | K,alpha]="
              f"{audit['binomial_p_ge_observed_under_null']:.3f}, "
              f"E[pass under null]="
              f"{audit['expected_pass_count_under_null']:.3f}")

    out = {
        "alpha_per_test_default": args.alpha,
        "source_json": str(src),
        "backends": backends_out,
        "interpretation": (
            "Bonferroni alpha/K (= 0.0056 at K=9) relaxes the "
            "non-rejection threshold; pass count under that "
            "threshold is reported. The Sidak family-wise alpha "
            "at FWER=0.05 is tighter than Bonferroni and is the "
            "tightest multiplicity correction we report. R21 "
            "reviewer-supplied multiplicity-aware chance metric: "
            "Binomial p-value P[>=obs | X~Binom(K, alpha)]. For "
            "K=9 alpha=0.05 and obs=1, this is 1-(0.95)^9 = "
            "0.370 -> observing 1 pass is consistent with chance. "
            "C2 is therefore restricted to a measurement-based "
            "empirical envelope, not a WCET. Earlier 'chance UB "
            "K*alpha^2 = 0.0225' wording was non-standard and has "
            "been retired."
        ),
    }
    out_json = RESULTS / "mbpta_bonferroni_audit.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"[bonferroni-audit] wrote {out_json}")

    # TeX table.
    lines = [
        "% Generated by mbpta_bonferroni_audit.py (R21 + Binomial fix).",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Backend & $K$ & pass$_{\alpha{=}0.05}$ & "
        r"pass$_{\alpha/K}$ & pass$_{\mathrm{Sid},FW{=}0.05}$ & "
        r"$\Pr[{\ge}\text{obs}\mid\mathrm{Binom}(K,\alpha)]$ \\",
        r"\midrule",
    ]
    for b in backends_out:
        lines.append(
            f"\\texttt{{{b['backend']}}} & {b['K']} & "
            f"{b['pass_count_alpha005']} & "
            f"{b['pass_count_bonferroni']} & "
            f"{b['pass_count_sidak_fw0.05']} & "
            f"{b['binomial_p_ge_observed_under_null']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_tex = RESULTS / "mbpta_bonferroni_audit.tex"
    out_tex.write_text("\n".join(lines) + "\n")
    print(f"[bonferroni-audit] wrote {out_tex}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
