"""R21 q_base sanity sweep on the in-tree 40-request llama2_prod trace.

The R12--R16 calibration used a 24-request Mooncake-style synthetic
trace; R18--R20 attempted a 200-prompt ShareGPT/LMSys refit but the
HuggingFace mirror was unreachable in our submission environment so
the headline 0.0067 value remained the 24-request calibration.
R20 reviewer asked for at least a $>24$-request real-trace sanity
check before claiming representativeness.

This script consumes the in-tree, IO-free,
LLM-checkpoint-derived ``data/traces/llama2_prod`` parquet pool
(40~requests, $2{,}560$ unique (request, step) tuples; all rows
have ``c_io_us=0`` so the IO-free regime is exact, no filter needed)
and re-runs the canonical ``q_base`` calibrator from
``experiments.eC_bound_tightness.io_bound_regime``
(``\\sigma = (P_{99.5}{-}P_{50})/\\sqrt{2\\ln n}``, heavy-tail
factor $3.0$, Clopper-Pearson 95\\% CI). This is a $40$-request
real-prod sanity proxy for the queued ShareGPT-200 follow-up
(\\texttt{todo\\_atc.md}~B); we do NOT claim it replaces the 200-conv
production sweep.

Output: ``q_base_refit_llama2prod40.{json,tex}``.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/eC_bound_tightness/results"
TRACE_DIR = ROOT / "data/traces/llama2_prod"


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (p / 100.0) * (len(s) - 1)
    lo = math.floor(k)
    hi = min(len(s) - 1, lo + 1)
    return s[lo] + (k - lo) * (s[hi] - s[lo])


def _cp_two_sided_95(k: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    try:
        from scipy.stats import beta as _beta  # type: ignore
        if k == 0:
            lo = 0.0
        else:
            lo = float(_beta.ppf(0.025, k, n - k + 1))
        if k >= n:
            hi = 1.0
        else:
            hi = float(_beta.ppf(0.975, k + 1, n - k))
        return lo, hi
    except Exception:
        # Approximate fallback (rule of three for k=0).
        if k == 0:
            return 0.0, 1.0 - 0.05 ** (1.0 / n)
        phat = k / n
        z = 1.96
        half = z * math.sqrt(phat * (1 - phat) / n)
        return max(0.0, phat - half), min(1.0, phat + half)


def load_io_free_pool() -> tuple[list[float], int, int]:
    """Return (per_step_us, n_requests, n_kept_after_dropping_warmup)."""
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        print(f"[refit-prod40] FAIL: pandas missing: {exc!r}")
        return [], 0, 0
    files = sorted(TRACE_DIR.glob("req_*.parquet"))
    if not files:
        print(f"[refit-prod40] FAIL: no parquets in {TRACE_DIR}")
        return [], 0, 0
    n_req = len(files)
    pool: list[float] = []
    for fp in files:
        df = pd.read_parquet(
            fp, columns=["request_id", "step", "c_io_us",
                          "step_latency_us"])
        # Each (request, step) row appears once per (layer, block); the
        # latency is the same so we dedupe.
        g = df.groupby("step", as_index=False).first()
        # Drop the first 2 warm-up steps per request (matches the
        # io_bound_regime convention of dropping the prefill / first
        # token from the calibration pool).
        g = g[g["step"] >= 2]
        # Sanity: pool is already IO-free (c_io_us == 0) for this trace.
        g = g[g["c_io_us"].fillna(0) == 0]
        pool.extend(float(x) for x in g["step_latency_us"].dropna().tolist())
    return pool, n_req, len(pool)


def main() -> int:
    pool, n_req, n_steps = load_io_free_pool()
    if not pool:
        out = {"status": "TRACE_UNAVAILABLE",
               "trace_dir": str(TRACE_DIR)}
        (RESULTS / "q_base_refit_llama2prod40.json").write_text(
            json.dumps(out, indent=2))
        return 1

    from experiments.eC_bound_tightness import io_bound_regime as iobr
    p50 = iobr._percentile(pool, 50)
    sigma = iobr._empirical_sigma_us(pool)
    threshold = p50 + 3.0 * sigma
    n_over = sum(1 for x in pool if x > threshold)
    n = len(pool)
    point = n_over / n
    ci_lo, ci_hi = _cp_two_sided_95(n_over, n)

    chat_nonvac_threshold = 1e-2 / 32.0  # 3.125e-4 (B_t=32, rho=1e-2)
    qbase_calib = 0.0067
    ci_calib = (0.0058, 0.0125)

    in_calib_ci = ci_calib[0] <= point <= ci_calib[1]
    upper_below_chat_nonvac = ci_hi < chat_nonvac_threshold
    point_below_chat_nonvac = point < chat_nonvac_threshold

    out = {
        "status": "REFIT_COMPLETE",
        "workload": "llama2_prod_40req",
        "trace_dir": str(TRACE_DIR),
        "n_requests": n_req,
        "n_steps_kept": n_steps,
        "refit": {
            "q_base": point,
            "n": n,
            "n_over": n_over,
            "p50_us": p50,
            "sigma_us": sigma,
            "threshold_us": threshold,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "ci_method": "Clopper-Pearson (two-sided 95%)",
        },
        "calibrated_qbase_2024_mooncake_24req": qbase_calib,
        "ci95_calibrated_2024": list(ci_calib),
        "chat_tier_nonvacuous_threshold_qstar": chat_nonvac_threshold,
        "point_within_calibrated_ci": in_calib_ci,
        "ci_upper_below_chat_nonvacuous_threshold":
            upper_below_chat_nonvac,
        "point_below_chat_nonvacuous_threshold":
            point_below_chat_nonvac,
        "verdict":
            ("R21 sanity: 40-request real-prod q_base point falls "
             f"{'WITHIN' if in_calib_ci else 'OUTSIDE'} the calibrated "
             "24-req Mooncake CI [0.0058, 0.0125]; the CP 95% upper "
             f"{'IS' if upper_below_chat_nonvac else 'is NOT'} below "
             "the union-cover chat-tier non-vacuous threshold "
             "q*<3.125e-4. ShareGPT-200 GPU run remains queued "
             "(todo_atc.md~B)."),
    }
    out_json = RESULTS / "q_base_refit_llama2prod40.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"[refit-prod40] n_req={n_req} n_steps={n_steps} "
          f"q_base={point:.4f} CP95=[{ci_lo:.4f}, {ci_hi:.4f}] "
          f"calib=0.0067 CI[0.0058,0.0125]")
    print(f"[refit-prod40] point in calib CI: {in_calib_ci}; "
          f"CP upper < q*=3.125e-4: {upper_below_chat_nonvac}")
    print(f"[refit-prod40] wrote {out_json}")

    # Concise TeX.
    yes_no_in_ci = r"\textsc{yes}" if in_calib_ci else r"\textsc{no}"
    yes_no_below = (r"\textsc{yes}" if upper_below_chat_nonvac
                    else r"\textsc{no}")
    tex_lines = [
        "% Generated by q_base_refit_llama2prod40.py (R21).",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Field & Value \\",
        r"\midrule",
        rf"Workload & \texttt{{llama2\_prod}} ({n_req} reqs, {n_steps} steps) \\",
        rf"$P_{{50}}$ ($\mu$s) & {p50:.1f} \\",
        rf"$\sigma$ ($\mu$s, canonical) & {sigma:.1f} \\",
        rf"$\#\{{\text{{step}}>P_{{50}}{{+}}3\sigma\}}$ & {n_over}/{n} \\",
        rf"$\hat q_\mathrm{{base}}$ & {point:.4f} \\",
        rf"Clopper-Pearson 95\% CI & [${ci_lo:.4f}$, ${ci_hi:.4f}$] \\",
        r"Calibrated 24-req $q_\mathrm{base}$ & "
        r"0.0067 (CI [0.0058, 0.0125]) \\",
        rf"Point in calib CI & {yes_no_in_ci} \\",
        rf"CP upper $< q^*{{=}}3.125{{\cdot}}10^{{-4}}$ & {yes_no_below} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (RESULTS / "q_base_refit_llama2prod40.tex").write_text(
        "\n".join(tex_lines) + "\n")
    print(f"[refit-prod40] wrote "
          f"{RESULTS / 'q_base_refit_llama2prod40.tex'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
