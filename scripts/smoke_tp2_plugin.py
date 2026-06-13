"""Smoke test: vLLM 0.8.5 TP=2 + SeerKVConnector plugin.

Verifies the J.4 fix that LAP is loaded only on the SCHEDULER role.
Pre-fix behavior: 2nd worker rank hangs at LAP CUDA init.
Post-fix:        TP=2 starts, decodes 1 short prompt, exits clean.

Run:
    CUDA_VISIBLE_DEVICES=0,1 python scripts/smoke_tp2_plugin.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    from seer.integration.vllm_connector import register_with_vllm_factory
    register_with_vllm_factory("SeerKVConnector")
    import vllm
    from vllm.config import KVTransferConfig

    print(f"[tp2-smoke] vllm {vllm.__version__}")
    kv_cfg = KVTransferConfig(
        kv_connector="SeerKVConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "lap_path": "checkpoints/lap_prod_b1024.onnx",
            "budget_blocks": 32,
            "horizon_steps": 4,
            "p_threshold": 0.4,
            "policy": "seer",
        },
    )
    t0 = time.perf_counter()
    llm = vllm.LLM(
        model="meta-llama/Llama-2-7b-hf",
        dtype="float16",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.45,
        max_model_len=1024,
        enforce_eager=True,
        disable_log_stats=True,
        kv_transfer_config=kv_cfg,
    )
    print(f"[tp2-smoke] LLM loaded in {time.perf_counter()-t0:.1f}s")

    sp = vllm.SamplingParams(temperature=0.0, max_tokens=16)
    out = llm.generate(["Hello, world."], sp)
    txt = out[0].outputs[0].text.strip()
    print(f"[tp2-smoke] generated: {txt!r}")
    print("[tp2-smoke] OK — TP=2 + SeerKVConnector boots and decodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
