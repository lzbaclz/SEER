"""R23 W2: side-by-side substrate-grade vs vLLM-connector-integrated
A7 probe on the same GPU window.

R22 reviewer (W2) flagged that the operator-runnable probe
``seer/timing/a7_probe.py`` only measures the bare CUDA DMA path
and may under-measure the deployment path through
``KVConnector_V1.start_load_kv``. R23 W2 ships an
``--mode integrated`` extension to the probe that mimics the
dominant per-block overheads of
``seer/integration/vllm_connector.py`` (per-block pinned-host
pool lookup, per-block ``cudaEvent`` record+sync, periodic
stream drain every 32 blocks). This driver runs both modes
back-to-back on the same GPU window and writes a comparison
artifact with the gap.

Output: ``a7_probe_substrate_vs_integrated.{json,tex}``.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from seer.timing.a7_probe import run_probe
from seer.timing.substrate_measure import safe_write_stub

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/eC_bound_tightness/results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-sizes-kb", nargs="+", type=int,
                    default=[4, 16, 32, 64])
    ap.add_argument("--n-reps", type=int, default=5000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--rho", type=float, default=1e-2)
    ap.add_argument("--qbase-calibrated", type=float, default=0.0067)
    ap.add_argument("--pool-size-blocks", type=int, default=1024)
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--force-stub-overwrite", action="store_true",
                    help="R28: clobber existing measured artifact when "
                         "emitting the harness-only stub. Default refuses.")
    args = ap.parse_args()

    def _has_cuda() -> bool:
        try:
            import torch  # noqa: F401
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    if not _has_cuda():
        stub = {"status": "HARNESS_READY_AWAITING_GPU_RUN",
                "config": vars(args)}
        tex_content = (
            "% R23 W2 harness-only\n"
            r"\begin{tabular}{ll}\toprule Field & Value \\ \midrule "
            r"Status & \textsc{harness-only} \\ \bottomrule \end{tabular}"
            "\n")
        wrote = safe_write_stub(
            RESULTS / "a7_probe_substrate_vs_integrated.json",
            RESULTS / "a7_probe_substrate_vs_integrated.tex",
            stub, tex_content,
            force=args.force_stub_overwrite,
        )
        if wrote:
            print("[a7-cmp] STATUS: harness-only")
        return 0

    print("[a7-cmp] R23 W2 substrate-vs-integrated A7 probe comparison")
    print(f"[a7-cmp] Running substrate-mode on cuda:{args.device_index}...")
    sub = run_probe(args.block_sizes_kb, n_reps=args.n_reps,
                    warmup=args.warmup, rho=args.rho,
                    q_base_calibrated=args.qbase_calibrated,
                    mode="substrate",
                    device_index=args.device_index)
    print(f"[a7-cmp] Running integrated-mode on cuda:{args.device_index}...")
    integ = run_probe(args.block_sizes_kb, n_reps=args.n_reps,
                      warmup=args.warmup, rho=args.rho,
                      q_base_calibrated=args.qbase_calibrated,
                      mode="integrated",
                      pool_size_blocks=args.pool_size_blocks,
                      device_index=args.device_index)

    if sub.get("status") != "MEASUREMENT_COMPLETE" or \
       integ.get("status") != "MEASUREMENT_COMPLETE":
        print("[a7-cmp] FAIL: probe run did not complete")
        print(json.dumps({"sub": sub, "integ": integ}, indent=2))
        return 1

    pair_rows = []
    for s, i in zip(sub["rows"], integ["rows"]):
        pair_rows.append({
            "block_size_kb": s["block_size_kb"],
            "substrate_mean_us": s["mean_us"],
            "integrated_mean_us": i["mean_us"],
            "mean_inflation_int_over_sub":
                i["mean_us"] / max(s["mean_us"], 1e-6),
            "substrate_p999_us": s["p999_us"],
            "integrated_p999_us": i["p999_us"],
            "substrate_qUB95": s["q_step_a2_upper95"],
            "integrated_qUB95": i["q_step_a2_upper95"],
            "substrate_a7": s.get("a7_holds_at_rho"),
            "integrated_a7": i.get("a7_holds_at_rho"),
        })
        print(f"[a7-cmp] {s['block_size_kb']}KB: "
              f"sub mean={s['mean_us']:.2f}us "
              f"int mean={i['mean_us']:.2f}us "
              f"inflation={i['mean_us']/max(s['mean_us'],1e-6):.2f}x  "
              f"sub qUB={s['q_step_a2_upper95']:.3e} "
              f"int qUB={i['q_step_a2_upper95']:.3e}  "
              f"sub A7={s.get('a7_holds_at_rho')} "
              f"int A7={i.get('a7_holds_at_rho')}")

    overall_agree = all(
        r["substrate_a7"] == r["integrated_a7"] for r in pair_rows)
    out = {
        "status": "MEASUREMENT_COMPLETE",
        "rho": args.rho,
        "qbase_calibrated": args.qbase_calibrated,
        "pool_size_blocks": args.pool_size_blocks,
        "n_reps": args.n_reps,
        "block_sizes_kb": args.block_sizes_kb,
        "substrate_run": sub,
        "integrated_run": integ,
        "pairs": pair_rows,
        "substrate_verdict": sub.get("verdict"),
        "integrated_verdict": integ.get("verdict"),
        "verdicts_agree": overall_agree,
        "interpretation": (
            "R23 W2 reviewer concern: the substrate-grade probe "
            "under-measures the vLLM-integrated path. This driver "
            "runs both modes on the same GPU window. If the "
            "integrated verdict diverges from the substrate "
            "verdict (or if integrated mean ≫ substrate mean), "
            "the substrate probe is under-measuring; if they "
            "agree closely, the integration overhead is dominated "
            "by substrate load and the substrate probe is a "
            "defensible operator-side proxy."
        ),
    }
    out_json = RESULTS / "a7_probe_substrate_vs_integrated.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"[a7-cmp] wrote {out_json}")
    print(f"[a7-cmp] substrate verdict={sub.get('verdict')}, "
          f"integrated verdict={integ.get('verdict')}, "
          f"agree={overall_agree}")

    tex_lines = [
        "% Generated by a7_probe_substrate_vs_integrated.py (R23 W2).",
        r"\begin{tabular}{rrrrrrll}",
        r"\toprule",
        r"Block & $\bar\ell_\text{sub}$ & $\bar\ell_\text{int}$ & "
        r"inflation & $q_\mathrm{step}^\text{A2,sub,UB}$ & "
        r"$q_\mathrm{step}^\text{A2,int,UB}$ & A7$_\text{sub}$ & "
        r"A7$_\text{int}$ \\",
        r"(KB) & ($\mu$s) & ($\mu$s) & ($\times$) & & & & \\",
        r"\midrule",
    ]
    for r in pair_rows:
        sub_a7 = ("\\textsc{holds}" if r["substrate_a7"]
                  else "\\textsc{fails}")
        int_a7 = ("\\textsc{holds}" if r["integrated_a7"]
                  else "\\textsc{fails}")
        tex_lines.append(
            f"{r['block_size_kb']} & "
            f"{r['substrate_mean_us']:.2f} & "
            f"{r['integrated_mean_us']:.2f} & "
            f"{r['mean_inflation_int_over_sub']:.2f} & "
            f"${r['substrate_qUB95']:.2e}$ & "
            f"${r['integrated_qUB95']:.2e}$ & "
            f"{sub_a7} & {int_a7} \\\\"
        )
    tex_lines += [r"\bottomrule", r"\end{tabular}"]
    out_tex = RESULTS / "a7_probe_substrate_vs_integrated.tex"
    out_tex.write_text("\n".join(tex_lines) + "\n")
    print(f"[a7-cmp] wrote {out_tex}")
    return 0 if overall_agree else 2


if __name__ == "__main__":
    sys.exit(main())
