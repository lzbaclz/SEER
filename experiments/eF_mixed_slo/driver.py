"""eF: concurrent driver for mixed-SLO workloads.

Two thread pools share a single model: chat tasks come in at a
configurable arrival rate with TPOT-class SLO, doc tasks arrive
sparsely with TTFT-class SLO. Each task is run through
``simulate_attention_mask`` from :mod:`seer.eval.sim`, the per-step
latencies and TTFT are recorded, and an aggregate JSON is emitted.

This is the runtime side of §6.8. Admission control (the inverse
direction of Lemma 2) is delegated to a small helper that decides,
for each new task, whether the current SEER ``λ_t`` plus the
empirical $\bar{\\ell}$ leave room for the new SLO without violating
the existing ones.

This is the canonical multi-policy driver used by every β2/β3/β4
patched-fork sweep (~750 LoC). It supports both the in-process
``simulate_attention_mask`` path (for offline ablation) and the
``--use_engine_step`` patched-fork path that produces the
``tpot_stats`` distributions consumed by §6.8 tables. The
``vllm_plugin_seer`` whitelist accepts six policies
(seer / h2o / streaming / full / snapkv / quest); the canonical
sweep (``scripts/run_path_beta4_scaled_sweep.sh``) exercises all
six. Multi-tenant single-host scheduling beyond a single EDF queue
is queued as the ATC follow-up in ``todo_atc.md``.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time
from pathlib import Path


def _build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lap", default=None,
                    help="LAP checkpoint for SEER policy (.onnx/.pt/.plan)")
    ap.add_argument("--policy", default="seer",
                    choices=["full", "seer", "h2o", "snapkv", "streaming",
                             "quest", "recency"])
    ap.add_argument("--n_chat", type=int, default=10)
    ap.add_argument("--n_doc", type=int, default=2)
    ap.add_argument("--chat_slo", default="P99=50ms")
    ap.add_argument("--doc_slo",  default="TTFT-P99=1s")
    ap.add_argument("--hbm_budget", type=float, default=0.2)
    ap.add_argument("--max_workers", type=int, default=2,
                    help="Concurrency. With 2 GPUs = 2.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--backend", default="simulator",
                    choices=["simulator", "vllm"],
                    help="simulator | vllm. simulator is the published "
                         "§6.8 numbers; vllm routes through real vllm.")
    ap.add_argument("--vllm_plugin_seer", action="store_true",
                    help="J.3: register SeerKVConnector as a vllm plugin "
                         "and pass kv_transfer_config so vllm calls SEER "
                         "for prefetch decisions. Active for --policy seer "
                         "only; the other policies fall back to vllm "
                         "default prefix-cache + FIFO eviction.")
    ap.add_argument("--max_tokens", type=int, default=64,
                    help="Per-task generation length. Default 64 (chat) is "
                         "what the published numbers used; bump to 256 "
                         "for J.3 stress tests where we want to expose the "
                         "P99 tail.")
    ap.add_argument("--use_engine_step", action="store_true",
                    help="vllm backend: use LLMEngine.step() loop instead of "
                         "high-level LLM.generate(), which exposes per-step "
                         "decode latency for honest TPOT P99 / P999 numbers.")
    ap.add_argument("--long_context", type=int, default=0,
                    help="If > 0, generate prompts of approximately this many "
                         "input tokens (Lorem-ipsum padded). 0 = the short "
                         "prompts used by published §6.8.")
    ap.add_argument("--workload", default="default",
                    choices=["default", "multi_turn_chat"],
                    help="Workload generator. 'default' uses the templated "
                         "chat/doc prompts in _make_chat_prompt/_make_doc_prompt. "
                         "'multi_turn_chat' (Path beta) uses "
                         "workloads.multi_turn_chat.generate to produce "
                         "n_chat threads x n_turns turns with a shared "
                         "system prompt -- the regime where vLLM's "
                         "prefix-cache absorber triggers cross-prompt KV "
                         "reuse and the scheduler/worker namespace gap "
                         "closes (n_load > 0 expected).")
    ap.add_argument("--n_turns", type=int, default=5,
                    help="Turns per chat thread when --workload "
                         "multi_turn_chat. Default 5.")
    ap.add_argument("--max_model_len", type=int, default=2048,
                    help="vllm max_model_len. Bump to 4096/8192 for "
                         "long-context stress (capped by model's "
                         "max_position_embeddings).")
    ap.add_argument("--gpu_mem_util", type=float, default=0.85,
                    help="vllm gpu_memory_utilization. Lower to 0.20-0.30 "
                         "to force KV pressure for the stress run.")
    ap.add_argument("--tensor_parallel", type=int, default=1,
                    help="vllm tensor_parallel_size. >1 shards the model "
                         "across multiple GPUs (validates the multi-GPU "
                         "code path; on Llama-2-7B + 2x A100 mostly tests "
                         "that the connector resolves under TP).")
    ap.add_argument("--disable_prefix_cache", action="store_true",
                    help="Pass enable_prefix_caching=False to vllm. "
                         "Without this, vLLM's prefix cache absorbs the "
                         "policy gap on benign chat workloads, masking "
                         "any differentiation between SEER / H2O / "
                         "Streaming / Full. Use for the J.5 high-pressure "
                         "separation regime.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for prompt index offset + Python/torch RNG. "
                         "Two seeds with the same workload spec produce "
                         "structurally similar prompt sets (same template, "
                         "different (request {i}) ids and different submission "
                         "order), so the across-seed variance in chat-miss "
                         "captures real-time scheduling jitter on the GPU + "
                         "vLLM scheduler, not prompt content variance.")
    return ap.parse_args()


def _load_model(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"float16": torch.float16,
             "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager",
    ).to(args.device).eval()
    return tok, model


def _build_policy(args):
    from seer.policy import build_policy
    from seer.timing import parse_slo
    kw: dict = {}
    if args.policy == "seer":
        if not args.lap:
            raise SystemExit("--lap required for --policy seer")
        from seer.lap.infer import LAPPredictor
        kw["lap_predictor"] = LAPPredictor.from_path(args.lap, device=args.device)
        kw["slo"] = parse_slo(args.chat_slo)  # tighter SLO drives the controller
    return build_policy(args.policy, **kw)


def _make_chat_prompt(i: int, long_context_tokens: int = 0) -> str:
    if long_context_tokens > 0:
        # Stress mode: ~long_context_tokens-token prompt that
        # exercises real KV-cache pressure. Each "Lorem ipsum"
        # word is ~1.5 tokens; pad to the requested length.
        n_repeats = max(1, long_context_tokens // 5)
        body = "Lorem ipsum dolor sit amet consectetur adipiscing elit. " * n_repeats
        return (
            f"You are a helpful assistant. Background context for "
            f"request {i}:\n\n{body}\n\n"
            f"Now please summarise the difference between solar PV "
            f"and solar thermal in two sentences.\nAnswer:"
        )
    return (
        f"You are a helpful assistant. The user asks: please summarise "
        f"the difference between solar PV and solar thermal in two "
        f"sentences (request {i}).\nAnswer:"
    )


def _make_doc_prompt(i: int, long_context_tokens: int = 0) -> str:
    n = max(60, long_context_tokens // 5) if long_context_tokens else 60
    body = "Lorem ipsum dolor sit amet, " * n
    return (
        f"Please summarise the following document in three short "
        f"bullet points.\n\n{body}\n\nSummary (request {i}):"
    )


def _run_vllm_backend(args) -> dict:
    """vllm-async-prefetch backend for J.2.

    Routes the eF chat/doc workload through real vllm 0.8.x or
    0.20.x, with SEER's connector registered via vllm's
    ``--kv-transfer-config`` plumbing. Same SLO accounting as the
    simulator path, but per-request latency comes from vllm
    directly rather than our hand-rolled mask-based simulator.

    Pre-conditions (validated up front):
      1. ``vllm`` importable end-to-end (full deps, not just the
         wheel + our two pure-Python stubs).
      2. The LAP plan path resolves (TRT or ONNX).
      3. A free GPU is available (>= 16 GB free for Llama-2-7B).

    If any pre-condition fails, raises a SystemExit with a precise
    message — we never silently fall back to the simulator path,
    because that would produce numbers labelled "vllm-async" that
    were actually measured with the in-process driver.
    """
    from seer.integration import is_vllm_available
    if not is_vllm_available():
        raise SystemExit(
            "[eF] --backend vllm requires `pip install vllm` (0.8.x or "
            "0.20.x). Currently not installed in this env. Run with "
            "--backend simulator to use the published §6.8 path."
        )
    if not args.lap:
        raise SystemExit("[eF] --backend vllm requires --lap (the LAP plan path).")

    import json
    import time
    from pathlib import Path

    import vllm

    from seer.eval.metrics import LatencyStats
    from seer.timing import derive_deadline_us, parse_slo

    print(f"[eF/vllm] vllm version: {vllm.__version__}")
    chat_slo = parse_slo(args.chat_slo)
    doc_slo = parse_slo(args.doc_slo)
    chat_D = derive_deadline_us(chat_slo)
    doc_D = derive_deadline_us(doc_slo)

    # Sanity-check the connector class WITHOUT instantiating it
    # (instantiating loads LAP onto CUDA, which conflicts with
    # vllm's later cuda init in worker process). The class
    # construction itself is verified by the unit tests.
    from seer.integration.vllm_connector import (
        _detect_vllm_api_version,
        make_seer_connector_class,
    )
    print(f"[eF/vllm] vllm API: {_detect_vllm_api_version()}")
    cls = make_seer_connector_class()
    remaining = [m for m in dir(cls) if getattr(getattr(cls, m, None),
                                                "__isabstractmethod__", False)]
    if remaining:
        raise SystemExit(f"[eF/vllm] connector has unimplemented abstracts: {remaining}")
    print(f"[eF/vllm] connector class {cls.__name__} (no unimplemented abstracts)")
    print("[eF/vllm] loading vllm.LLM ...")

    # J.3 / Task 5: register SeerKVConnector and route the chosen
    # policy through it. The connector's planner branches on
    # extra_config["policy"]:
    #   seer       → LAP-driven prefetch
    #   h2o        → cumulative-attention prefetch
    #   streaming  → sliding-window prefetch
    #   full       → never prefetch (= vLLM default)
    # When --vllm_plugin_seer is off, the path falls back to vLLM's
    # built-in prefix cache for every policy.
    # T2-F: reset the process-local connector registry before vllm
    # constructs the new connector instance(s). Otherwise stats from
    # any prior driver invocation in the same Python session bleed
    # into this run's summary.
    try:
        from seer.integration.vllm_connector import _CONNECTOR_INSTANCES
        _CONNECTOR_INSTANCES.clear()
    except Exception:  # noqa: BLE001
        pass
    # T2-F-bis (May 2026 reviewer round B1 follow-up): set
    # SEER_CONNECTOR_STATS_PATH so the connector running in vLLM's
    # worker subprocess can dump its stats here via atexit; the
    # driver reads this file post-generation via
    # :func:`get_connector_stats`. Each run gets its own path so
    # concurrent driver invocations don't collide.
    import os as _os
    import tempfile
    _stats_dump = _os.path.join(
        tempfile.gettempdir(),
        f"seer_connector_stats_{_os.getpid()}.json",
    )
    _os.environ["SEER_CONNECTOR_STATS_PATH"] = _stats_dump
    # Remove any stale file from a previous invocation.
    try:
        _os.remove(_stats_dump)
    except FileNotFoundError:
        pass

    kv_transfer_config = None
    if args.vllm_plugin_seer and args.policy in (
        "seer", "h2o", "streaming", "full", "snapkv", "quest"
    ):
        from seer.integration.vllm_connector import register_with_vllm_factory
        register_with_vllm_factory("SeerKVConnector")
        from vllm.config import KVTransferConfig
        kv_transfer_config = KVTransferConfig(
            kv_connector="SeerKVConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "lap_path": args.lap or "",
                "budget_blocks": int(args.hbm_budget * 1024),
                "horizon_steps": 4,
                "p_threshold": 0.4,
                "policy": args.policy,
            },
        )
        print(f"[eF/vllm] plugin path: SeerKVConnector(policy={args.policy})")

    llm_kwargs = {
        "model": args.model,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_mem_util,
        "max_model_len": args.max_model_len,
        "enforce_eager": True,
        "disable_log_stats": True,
        "tensor_parallel_size": args.tensor_parallel,
    }
    if args.disable_prefix_cache:
        llm_kwargs["enable_prefix_caching"] = False
        print("[eF/vllm] J.5 mode: enable_prefix_caching=False "
              "(forces policy differentiation by removing vLLM's hidden "
              "absorber)")
    if kv_transfer_config is not None:
        llm_kwargs["kv_transfer_config"] = kv_transfer_config
    llm = None
    t_load_0 = time.time()
    try:
        llm = vllm.LLM(**llm_kwargs)
    except Exception as e:
        raise SystemExit(f"[eF/vllm] vllm.LLM() failed: {e!r}\n"
                         f"likely cause: GPU OOM or missing model weights. "
                         f"Verify nvidia-smi and ~/.cache/huggingface/hub/.")
    t_load = time.time() - t_load_0
    print(f"[eF/vllm] LLM loaded in {t_load:.1f}s")

    # T2-L (sixth-round R1-Cut6): install the forward-hook patch that
    # surfaces an InfiniGen-style per-block attention proxy
    # (Q·K_centroid), so SeerKVConnector receives a signal correlated
    # with actual attention rather than the recency-decay fallback.
    # Cost is one (n_blocks, n_heads) matmul per layer per decode
    # step, ~10us on A100, well inside Lemma 2's 200us LAP budget.
    if args.vllm_plugin_seer:
        try:
            import os as _os_inst

            from seer.integration.vllm_forward_hook import (
                get_hook_counters_remote,
                install_via_engine,
            )
            _use_monkey = _os_inst.environ.get(
                "SEER_USE_MONKEY_PATCH", "").strip() in (
                "1", "true", "True")
            _use_batched = _os_inst.environ.get(
                "SEER_USE_BATCHED_HOOK", "").strip() in (
                "1", "true", "True")
            _use_batched_compute = _os_inst.environ.get(
                "SEER_USE_BATCHED_COMPUTE", "").strip() in (
                "1", "true", "True")
            if _os_inst.environ.get("SEER_NO_HOOK_INSTALL", "").strip() \
                    in ("1", "true", "True"):
                print("[eF/vllm] forward-hook install SKIPPED via "
                      "SEER_NO_HOOK_INSTALL=1")
                n_h = 0
                _seer_attn_hook_diag = lambda: {"fire": 0, "skipped_install": 1}
            elif _use_monkey:
                from seer.integration.vllm_attention_monkey import (
                    get_patch_counters_remote,
                )
                from seer.integration.vllm_attention_monkey import (
                    install_via_engine as _install_monkey,
                )
                n_h = _install_monkey(llm)
                if n_h > 0:
                    print(f"[eF/vllm] Attention.forward MONKEY-PATCHED "
                          f"on {n_h} class(es) (Step 0 dispatch-bypass)")
                    _seer_attn_hook_diag = (
                        lambda: get_patch_counters_remote(llm))
                else:
                    print("[eF/vllm] monkey-patch install: 0 classes")
                    _seer_attn_hook_diag = (
                        lambda: get_patch_counters_remote(llm))
            elif _use_batched:
                from seer.integration.vllm_attention_batched import (
                    get_counters_remote as _bcr,
                )
                from seer.integration.vllm_attention_batched import (
                    install_via_engine as _install_batched,
                )
                n_h = _install_batched(llm)
                if n_h > 0:
                    print(f"[eF/vllm] BATCHED hook installed: 1 callback "
                          f"on layer-0 will process {n_h} Attention "
                          f"layers/fire (Step 0b)")
                    _seer_attn_hook_diag = lambda: _bcr(llm)
                else:
                    print("[eF/vllm] batched install: 0 layers")
                    _seer_attn_hook_diag = lambda: _bcr(llm)
            elif _use_batched_compute:
                from seer.integration.vllm_attention_batched_compute import (
                    get_counters_remote as _bccr,
                )
                from seer.integration.vllm_attention_batched_compute import (
                    install_via_engine as _install_bc,
                )
                n_h = _install_bc(llm)
                if n_h > 0:
                    print(f"[eF/vllm] BATCHED-COMPUTE hook installed: "
                          f"1 callback + stacked tensor compute over "
                          f"{n_h} layers (Step 0c)")
                    _seer_attn_hook_diag = lambda: _bccr(llm)
                else:
                    _seer_attn_hook_diag = lambda: _bccr(llm)
            else:
                # Legacy path: standard per-layer forward_pre_hook via
                # vllm_forward_hook.install_via_engine. Only this branch
                # uses get_hook_counters_remote; the other branches set
                # their own ``_seer_attn_hook_diag`` above.
                n_h = install_via_engine(llm)
                if n_h > 0:
                    print(f"[eF/vllm] forward-hook installed on {n_h} "
                          f"Attention layers (T2-L; apply_model path)")
                    _seer_attn_hook_diag = (
                        lambda: get_hook_counters_remote(llm))
                else:
                    print("[eF/vllm] forward-hook install: 0 layers "
                          "(no Attention modules found via apply_model; "
                          "recency fallback)")
                    _seer_attn_hook_diag = (
                        lambda: get_hook_counters_remote(llm))
        except Exception as _e:  # noqa: BLE001
            print(f"[eF/vllm] forward-hook install skipped: {_e!r}")
            _seer_attn_hook_diag = lambda: {"fire": 0, "skip_error": 1}
    else:
        _seer_attn_hook_diag = lambda: {"fire": 0}

    sampling = vllm.SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    sampling_doc = vllm.SamplingParams(temperature=0.0,
                                        max_tokens=max(32, args.max_tokens // 2))

    # Seed plumbing for 3-seed methodology (R1/R2 W3 follow-through).
    # The seed offsets prompt request-ids (so KV cache keys differ across
    # seeds) and seeds Python/torch RNGs so the (deterministic-by-default)
    # vLLM scheduler still sees a different prompt-submission order.
    import random as _random
    _random.seed(args.seed)
    try:
        import torch as _torch
        _torch.manual_seed(args.seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(args.seed)
    except Exception:  # noqa: BLE001
        pass
    _seed_offset = args.seed * max(args.n_chat, 1) * 1000
    if args.workload == "multi_turn_chat":
        # Path beta: cross-prompt KV-reuse workload. n_chat is the
        # thread count; each thread runs n_turns turns interleaved
        # across threads (turn 0 of A, turn 0 of B, ..., turn 1 of A,
        # ...). The shared system prefix lets vLLM's prefix-cache
        # absorber promote real cache slot reuse, closing the
        # scheduler/worker namespace gap.
        from experiments.eF_mixed_slo.workloads.multi_turn_chat import (
            generate as _mt_generate,
        )
        chat_turns = _mt_generate(
            n_threads=args.n_chat,
            n_turns=args.n_turns,
            seed=args.seed,
        )
        chat_prompts = [t.prompt for t in chat_turns]
        # Do not shuffle: the interleave order is the controlled
        # signal (turn t of all threads, then turn t+1, ...). The
        # multi_turn_chat generator emits exactly that order.
    else:
        chat_prompts = [_make_chat_prompt(i + _seed_offset, args.long_context)
                        for i in range(args.n_chat)]
        _random.shuffle(chat_prompts)
    doc_prompts = [_make_doc_prompt(i + _seed_offset, args.long_context)
                   for i in range(args.n_doc)]
    _random.shuffle(doc_prompts)

    def _generate(prompts: list[str], sp: vllm.SamplingParams, kind: str):
        """High-level path: per-task wall-clock divided by number of
        generated tokens gives mean TPOT (no per-step distribution).
        For honest P99 / P999, prefer --use_engine_step."""
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        wall = time.perf_counter() - t0
        results = []
        for out in outs:
            n_gen = len(out.outputs[0].token_ids) if out.outputs else 0
            tpot_us = (wall / max(1, n_gen)) * 1e6
            stats = LatencyStats.from_vector(
                [tpot_us] * max(1, n_gen),
                deadline_us=chat_D if kind == "chat" else doc_D,
            )
            results.append({"kind": kind, "stats": stats.to_dict(),
                            "ttft_us": wall * 1e6, "n_gen": n_gen})
        return results

    def _generate_with_engine_step(prompts: list[str], sp: vllm.SamplingParams,
                                   kind: str):
        """Low-level LLMEngine.step() loop. Each step() handles one
        decode iteration across all in-flight requests; we time it
        with perf_counter and credit the elapsed wall to every
        request that decoded a token in that step. This gives the
        per-step latency distribution that paper §6.8 actually wants
        (honest P50 / P99 / P999 / max under SEER's plugin)."""
        engine = llm.llm_engine
        # Submit all prompts up-front so vllm batches them.
        for i, p in enumerate(prompts):
            req_id = f"{kind}_{i}"
            try:
                engine.add_request(req_id, p, sp)
            except TypeError:
                # vllm 0.8.x signature: add_request(request_id, prompt, params, ...)
                engine.add_request(request_id=req_id, prompt=p, params=sp)
        # Per-request bookkeeping: list of per-step latencies (us)
        per_step: dict[str, list[float]] = {f"{kind}_{i}": [] for i in range(len(prompts))}
        ttft_us: dict[str, float] = {}
        first_step_t = time.perf_counter()
        prev_unfinished = None
        while engine.has_unfinished_requests():
            t0 = time.perf_counter()
            step_outs = engine.step()
            t1 = time.perf_counter()
            step_us = (t1 - t0) * 1e6
            for o in step_outs:
                rid = o.request_id
                if rid not in per_step:
                    continue
                # First decode → record TTFT
                if not per_step[rid]:
                    ttft_us[rid] = (t1 - first_step_t) * 1e6
                per_step[rid].append(step_us)
        results = []
        for i in range(len(prompts)):
            rid = f"{kind}_{i}"
            steps = per_step[rid]
            # T2-M (R1-Cut1/Cut3 follow-up, 2026-05-14): chat-miss is a
            # TPOT (time-per-output-token) SLO, defined over *decode*
            # tokens — the first sample in ``per_step[rid]`` is the
            # batched-prefill step that vLLM credits to the request
            # when its first decode-token is emitted, and is naturally
            # an outlier (50 prompts $\times$ ~32 tokens of prefill in
            # one engine.step). Including it in the per-prompt
            # miss-ratio meant every prompt produced exactly one
            # mis-step, giving the structurally-uniform chat\_miss =
            # 1/max_tokens artefact that masked the policy
            # differential at the SLO threshold. We now split:
            #   * ``tpot_stats`` -- decode-only TPOT samples (skip the
            #     first per-request step). This is what the 50ms SLO
            #     gate operates on; chat\_miss reads from here.
            #   * ``stats`` -- legacy full-trace samples (kept for
            #     back-compat with results_vllm_w3_p25L_fix archival).
            #   * ``ttft_us`` -- unchanged; TTFT is its own SLO line.
            decode_steps = steps[1:] if len(steps) > 1 else []
            tpot_stats = LatencyStats.from_vector(
                decode_steps if decode_steps else [0.0],
                deadline_us=chat_D if kind == "chat" else doc_D,
            )
            stats = LatencyStats.from_vector(
                steps if steps else [0.0],
                deadline_us=chat_D if kind == "chat" else doc_D,
            )
            results.append({"kind": kind,
                            "stats": stats.to_dict(),
                            "tpot_stats": tpot_stats.to_dict(),
                            "ttft_us": ttft_us.get(rid, 0.0),
                            "n_gen": len(steps),
                            "n_decode": len(decode_steps)})
        return results

    gen_fn = _generate_with_engine_step if args.use_engine_step else _generate
    mode = "engine.step" if args.use_engine_step else "LLM.generate"
    print(f"[eF/vllm] generating {len(chat_prompts)} chat + {len(doc_prompts)} "
          f"doc, max_tokens={args.max_tokens}, mode={mode} ...")
    t0 = time.time()
    results = []
    if chat_prompts:
        results.extend(gen_fn(chat_prompts, sampling, "chat"))
    if doc_prompts:
        results.extend(gen_fn(doc_prompts, sampling_doc, "doc"))
    dt = time.time() - t0
    print(f"[eF/vllm] generation wall: {dt:.1f}s")

    chat_results = [r for r in results if r["kind"] == "chat"]
    doc_results = [r for r in results if r["kind"] == "doc"]
    # T2-M (R1-Cut1 follow-up, second iteration): two distinct
    # aggregations of per-prompt decode-only TPOT samples:
    #
    #   chat_miss_step_level  = total misses across all decode steps
    #                            / total decode steps. Operationally
    #                            equivalent to a step-level CDF
    #                            crossing the SLO threshold; this is
    #                            the metric reviewers ask for because
    #                            it does not suffer the per-prompt
    #                            mean-of-bimodal artefact below.
    #
    #   chat_miss             = mean of per-prompt miss_ratio. With
    #                            n_chat=50 and chunked prefill, the
    #                            distribution is bimodal: most prompts
    #                            never miss, a few hit the slow step
    #                            and accrue miss_ratio = 1/n_decode.
    #                            Mean is dominated by the count of
    #                            unlucky prompts (small) divided by 50,
    #                            so it tracks the *fraction of unlucky
    #                            requests* rather than the *fraction
    #                            of slow steps*. Reviewer R1
    #                            (sixth-round) flags this as the
    #                            "12 cells identical chat-miss to 7
    #                            decimals" artefact.
    #
    # We expose both in the JSON; the aggregator + paper report
    # ``chat_miss_step_level`` as the headline.
    tpot_n_total = 0
    tpot_miss_total = 0
    for r in chat_results:
        ts = r.get("tpot_stats") or r.get("stats", {}) or {}
        tpot_n_total += int(ts.get("n", 0))
        tpot_miss_total += int(ts.get("miss_count",
                                       ts.get("misses", 0)))
    chat_miss_step_level = (tpot_miss_total / tpot_n_total
                            if tpot_n_total > 0 else 0.0)
    chat_miss = (sum(r.get("tpot_stats", r["stats"])["miss_ratio"]
                     for r in chat_results)
                 / max(1, len(chat_results)))
    chat_miss_legacy = (sum(r["stats"]["miss_ratio"] for r in chat_results)
                        / max(1, len(chat_results)))
    doc_miss = sum(1 for r in doc_results if r["ttft_us"] > doc_D) / max(1, len(doc_results))

    # T2-F (May 2026 reviewer round R1-N1, R2-#5): collect the
    # mechanistic counters from every SeerKVConnector instance
    # constructed in this process so the paper's J.5 chat-miss
    # reduction can be traced back to real planner.plan() / prefetch
    # hits rather than asking the reader to trust the signal.
    try:
        from seer.integration.vllm_connector import get_connector_stats
        connector_stats_snapshot = get_connector_stats()
    except Exception as e:  # noqa: BLE001
        connector_stats_snapshot = {"error": repr(e)}

    # Pull hook counters one more time so the JSON has the post-run
    # tally (these accumulate across the whole run).
    try:
        _hook_counter_snapshot = _seer_attn_hook_diag() \
            if callable(_seer_attn_hook_diag) else {}
    except Exception:  # noqa: BLE001
        _hook_counter_snapshot = {}
    summary = {
        "backend": "vllm",
        "vllm_version": vllm.__version__,
        "model": args.model,
        "policy": args.policy,
        "seed": int(args.seed),
        "forward_hook_counters": _hook_counter_snapshot,
        "vllm_plugin_seer": bool(args.vllm_plugin_seer and args.policy == "seer"),
        "use_engine_step": bool(args.use_engine_step),
        "max_tokens": int(args.max_tokens),
        "n_chat": args.n_chat,
        "n_doc": args.n_doc,
        "chat_slo": args.chat_slo,
        "doc_slo": args.doc_slo,
        "hbm_budget": args.hbm_budget,
        # T2-F: mechanistic explanation of where chat-miss reduction
        # comes from. ``aggregate.planner_decisions`` > 0 confirms the
        # LAP actually fired; ``aggregate.hit_rate`` quantifies
        # prefetch effectiveness; ``aggregate.mean_blocks_per_decision``
        # bounds the fan-out per planner call.
        "seer_connector_stats": connector_stats_snapshot,
        "chat_miss": chat_miss,
        "chat_miss_step_level": chat_miss_step_level,
        "chat_miss_legacy_prefill_inclusive": chat_miss_legacy,
        "doc_miss": doc_miss,
        "wall_time_s": dt,
        "load_time_s": t_load,
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"[eF/vllm] chat_miss={chat_miss:.4f}  doc_miss={doc_miss:.4f}  "
          f"wall={dt:.1f}s  → {args.out}")
    llm = None
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return summary


def main():
    import os

    if os.environ.get("SEER_E3_HOT_PATH") == "1":
        try:
            from seer.integration.vllm_hot_path_hook import (
                install_hot_path_hook, register_router,
            )
            from seer.integration.hot_path_router import HotPathRouter
            from seer.lap.infer import load_lap_scorer

            scorer = load_lap_scorer()
            router = HotPathRouter(scorer=scorer, ell_bar_us=200.0)
            install_hot_path_hook()
            register_router(router)
            print("[eF/vllm] SEER_E3_HOT_PATH hook installed", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[eF/vllm] SEER_E3_HOT_PATH hook skipped: {exc}", flush=True)

    args = _build_args()

    if args.backend == "vllm":
        # All vllm-specific setup is encapsulated in the helper so
        # the simulator path stays untouched.
        _run_vllm_backend(args)
        return 0

    tok, model = _load_model(args)
    policy = _build_policy(args)

    from seer.eval.metrics import LatencyStats
    from seer.eval.sim import simulate_attention_mask
    from seer.timing import derive_deadline_us, parse_slo

    chat_slo = parse_slo(args.chat_slo)
    doc_slo  = parse_slo(args.doc_slo)
    chat_D = derive_deadline_us(chat_slo)
    doc_D  = derive_deadline_us(doc_slo)

    def _run_task(prompt: str, kind: str, max_new: int) -> dict:
        # Each task gets a fresh policy state so they don't cross-contaminate;
        # in production the substrate isolates tenants but here we share one
        # process and need to reset.
        policy.reset()
        r = simulate_attention_mask(
            model, tok, prompt, policy,
            budget_frac=args.hbm_budget,
            max_new_tokens=max_new,
            decision_period=8,
        )
        per_step = r.get("per_step_us", [])
        D = chat_D if kind == "chat" else doc_D
        stats = LatencyStats.from_vector(per_step, deadline_us=D)
        return {"kind": kind, "stats": stats.to_dict(),
                "ttft_us": r.get("prefill_us", 0.0)}

    tasks: list[tuple[str, str, int]] = []
    for i in range(args.n_chat):
        tasks.append((_make_chat_prompt(i, args.long_context), "chat", args.max_tokens))
    for i in range(args.n_doc):
        tasks.append((_make_doc_prompt(i, args.long_context), "doc", max(32, args.max_tokens // 2)))

    t0 = time.time()
    results = []
    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = [pool.submit(_run_task, *t) for t in tasks]
        for fut in cf.as_completed(futs):
            results.append(fut.result())
    dt = time.time() - t0

    chat_results = [r for r in results if r["kind"] == "chat"]
    doc_results  = [r for r in results if r["kind"] == "doc"]

    chat_miss = (sum(r["stats"]["miss_ratio"] for r in chat_results)
                 / max(1, len(chat_results)))
    doc_ttft_violations = sum(1 for r in doc_results if r["ttft_us"] > doc_D)
    doc_miss = doc_ttft_violations / max(1, len(doc_results))

    summary = {
        "model": args.model,
        "policy": args.policy,
        "wall_time_s": dt,
        "chat_slo": {"name": chat_slo.name, "deadline_us": chat_D},
        "doc_slo":  {"name": doc_slo.name,  "deadline_us": doc_D},
        "n_chat": len(chat_results),
        "n_doc":  len(doc_results),
        "chat_miss_ratio": chat_miss,
        "doc_miss_ratio":  doc_miss,
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[eF] {args.policy}: chat_miss={chat_miss:.4f} "
          f"doc_miss={doc_miss:.4f}  t={dt:.1f}s  → {args.out}")


def run_one_cell(
    policy_name: str,
    workload: str,
    slo_p99_us: float,
    out_dir: str | Path,
    fork_path: str | None = None,
    *,
    model: str | None = None,
    lap: str | None = None,
    seed: int = 0,
    n_chat: int | None = None,
    install_hot_path: bool = False,
) -> dict:
    """Programmatic single-cell runner for eR/e3 hot-path binding sweeps."""
    import os
    import subprocess
    import sys

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "cell.json"

    if "-" in workload:
        base, tail = workload.rsplit("-", 1)
        if tail.isdigit():
            n_chat = n_chat or int(tail)

    model = model or os.environ.get("SEER_MODEL", "meta-llama/Llama-2-7b-hf")
    lap = lap or os.environ.get("SEER_LAP", "checkpoints/lap_prod_tiny.pt")
    slo_ms = slo_p99_us / 1000.0
    n_chat = n_chat or 16

    env = os.environ.copy()
    if fork_path:
        env["VLLM_FORK_PATH"] = fork_path
    elif install_hot_path:
        env.pop("VLLM_FORK_PATH", None)
    if install_hot_path:
        env["SEER_E3_HOT_PATH"] = "1"

    cmd = [
        sys.executable, "-m", "experiments.eF_mixed_slo.driver",
        "--backend", "vllm",
        "--model", model,
        "--lap", lap,
        "--policy", policy_name,
        "--n_chat", str(n_chat),
        "--n_doc", "0",
        "--chat_slo", f"P99={slo_ms:g}ms",
        "--hbm_budget", "0.20",
        "--max_tokens", "64",
        "--workload", "multi_turn_chat",
        "--n_turns", "4",
        "--max_model_len", "2048",
        "--gpu_mem_util", "0.85",
        "--vllm_plugin_seer",
        "--seed", str(seed),
        "--out", str(out_json),
    ]
    subprocess.run(cmd, check=True, env=env)

    with open(out_json) as fh:
        summary = json.load(fh)
    summary.setdefault("chat_miss_ratio", summary.get("chat_miss", float("nan")))
    return summary


if __name__ == "__main__":
    main()
