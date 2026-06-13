"""LAP WCET measurement harness.

The RTSS schedulability argument requires a defensible upper bound on
``C_LAP`` — the time the predictor itself takes per decision point.
This module provides the empirical leg of that argument: it runs the
predictor many thousand times under controlled conditions and reports
the latency distribution.

Methodology
-----------
1. **Warmup**: 100 inferences at full batch size before measurement.
   Pulls weights into SM caches and lets cuDNN finish autotuning.
2. **Measurement loop**: ``n_reps`` inferences using ``torch.cuda.Event``
   pairs (start / end) to get GPU-side timing without host clock noise.
   Each rep uses the same input tensor (we want to characterize the
   *forward pass*, not the data path).
3. **Statistics**: P50 / P90 / P99 / P99.9 / max / mean / std on the
   latency vector. Output is JSON.

The reference machine has 2× A100 SXM4 NV12 NVLink. RTSS 2027 follow-up
extends to H100 + L40 (P3.6).

The ONNX-Runtime path is included as a sanity oracle — when TensorRT
isn't available we still want to report *some* WCET number rather than
none. The output JSON tags which backend produced each run.
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class WCETResult:
    backend: str            # "torch" / "onnx" / "tensorrt"
    device: str             # "cuda:0" / "cpu" / "A100" / ...
    batch_size: int
    feat_dim: int
    n_reps: int
    n_warmup: int
    p50_us: float
    p90_us: float
    p99_us: float
    p999_us: float
    max_us: float
    mean_us: float
    std_us: float


# ---------------------------------------------------------------------------
#  Core measurement
# ---------------------------------------------------------------------------

def measure_torch_wcet(
    model,
    batch_size: int,
    feat_dim: int,
    device: str = "cuda",
    n_reps: int = 10000,
    n_warmup: int = 100,
) -> WCETResult:
    """Time a PyTorch ``model`` forward pass on ``device``.

    Uses CUDA events when available (GPU-side timer); falls back to
    ``time.perf_counter`` on CPU.
    """
    import torch

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    dev = torch.device(device if use_cuda else "cpu")
    model = model.to(dev).eval()
    x = torch.randn(batch_size, feat_dim, device=dev)

    with torch.no_grad():
        # warmup
        for _ in range(n_warmup):
            _ = model(x)
        if use_cuda:
            torch.cuda.synchronize()

        latencies_us: list[float] = []
        if use_cuda:
            for _ in range(n_reps):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(x)
                end.record()
                end.synchronize()
                latencies_us.append(float(start.elapsed_time(end)) * 1000.0)
        else:
            import time
            for _ in range(n_reps):
                t0 = time.perf_counter()
                _ = model(x)
                t1 = time.perf_counter()
                latencies_us.append((t1 - t0) * 1e6)

    return _summarize(
        backend="torch",
        device=device,
        batch_size=batch_size,
        feat_dim=feat_dim,
        n_reps=n_reps,
        n_warmup=n_warmup,
        latencies_us=latencies_us,
    )


def measure_onnx_wcet(
    onnx_path: str,
    batch_size: int,
    feat_dim: int,
    device: str = "cuda",
    n_reps: int = 10000,
    n_warmup: int = 100,
) -> WCETResult:
    """Time an ONNX Runtime InferenceSession."""
    import time

    import numpy as np
    import onnxruntime as ort

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device.startswith("cuda")
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = sess.get_inputs()[0].name
    x = np.random.randn(batch_size, feat_dim).astype(np.float32)

    for _ in range(n_warmup):
        _ = sess.run(None, {input_name: x})
    latencies_us: list[float] = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        _ = sess.run(None, {input_name: x})
        t1 = time.perf_counter()
        latencies_us.append((t1 - t0) * 1e6)

    return _summarize(
        backend="onnx",
        device=device,
        batch_size=batch_size,
        feat_dim=feat_dim,
        n_reps=n_reps,
        n_warmup=n_warmup,
        latencies_us=latencies_us,
    )


def measure_tensorrt_wcet(
    plan_path: str,
    batch_size: int,
    feat_dim: int,
    device: str = "cuda",
    n_reps: int = 10000,
    n_warmup: int = 100,
    capture_cuda_graph: bool = True,
) -> WCETResult:
    """TensorRT plan replay WCET timer.

    The canonical "production-grade C_LAP" measurement path. Pipeline:

    1. Deserialize the plan into a TensorRT runtime engine.
    2. Allocate static input/output buffers on the GPU.
    3. Optionally capture an entire ``execute_async_v3`` call into a
       CUDA Graph. Once captured, every replay is a single
       ``cudaGraphLaunch`` with no host-side decision making — exactly
       what the schedulability bound assumes.
    4. Time each replay with a fresh ``torch.cuda.Event`` pair, since
       PyTorch ships the simplest GPU-side timer in our env.

    The result is meant to be the lowest-variance ``C_LAP`` we can
    produce on the deployment hardware without writing C++. A separate
    sanity run with ``capture_cuda_graph=False`` lets us quantify how
    much of the WCET comes from kernel launch overhead vs the actual
    compute (typically 30-50% on small predictors).
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise RuntimeError(
            "TensorRT not installed. Install with:\n"
            "    pip install --extra-index-url https://pypi.nvidia.com "
            "'tensorrt-cu12'\n"
            "or use --backend torch / --backend onnx."
        ) from e
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT WCET measurement requires CUDA")

    # --- Deserialize engine
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(plan_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"failed to deserialize engine from {plan_path}")
    context = engine.create_execution_context()

    # Look up tensor names (TensorRT 8.5+ tensor API).
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    inp_names = [n for n in tensor_names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    out_names = [n for n in tensor_names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
    if len(inp_names) != 1 or len(out_names) != 1:
        raise RuntimeError(
            f"expected 1 input + 1 output, got {inp_names} / {out_names}"
        )
    inp_name, out_name = inp_names[0], out_names[0]

    # Static shape: confirm it matches our caller-requested batch/feat.
    expected_shape = tuple(engine.get_tensor_shape(inp_name))
    if expected_shape != (batch_size, feat_dim):
        # Try to coerce the dynamic context to our shape; this only works
        # when the engine was built with an optimization profile that
        # admits this shape. For static-shape engines (the common case)
        # the shape must already match.
        try:
            context.set_input_shape(inp_name, (batch_size, feat_dim))
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"engine input shape {expected_shape} does not match "
                f"requested {(batch_size, feat_dim)}; rebuild with "
                f"--batch {batch_size} or pass matching --feat_dim"
            ) from e

    # --- Allocate device buffers
    out_shape = tuple(engine.get_tensor_shape(out_name))
    if any(d < 0 for d in out_shape):
        # Output may have an inferred batch dim; stamp it from input.
        out_shape = (batch_size,) + tuple(d for d in out_shape[1:])
    dev = torch.device("cuda:0")
    inp_buf = torch.randn(batch_size, feat_dim, dtype=torch.float32, device=dev)
    out_buf = torch.empty(out_shape, dtype=torch.float32, device=dev)
    context.set_tensor_address(inp_name, inp_buf.data_ptr())
    context.set_tensor_address(out_name, out_buf.data_ptr())

    stream = torch.cuda.Stream()

    def _execute_once() -> None:
        with torch.cuda.stream(stream):
            context.execute_async_v3(stream.cuda_stream)

    # --- Warmup
    for _ in range(n_warmup):
        _execute_once()
    stream.synchronize()

    # --- Optionally capture CUDA Graph
    graph = None
    if capture_cuda_graph:
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                _execute_once()
            stream.synchronize()
        except Exception as e:  # noqa: BLE001
            print(f"[wcet] CUDA Graph capture failed ({e}); falling back to "
                  "ungraphed replay")
            graph = None

    # --- Measure
    latencies_us: list[float] = []
    for _ in range(n_reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if graph is not None:
            start.record(stream)
            graph.replay()
            end.record(stream)
        else:
            start.record(stream)
            _execute_once()
            end.record(stream)
        end.synchronize()
        latencies_us.append(float(start.elapsed_time(end)) * 1000.0)

    backend_tag = "tensorrt+cudagraph" if graph is not None else "tensorrt"
    return _summarize(
        backend=backend_tag,
        device=device,
        batch_size=batch_size,
        feat_dim=feat_dim,
        n_reps=n_reps,
        n_warmup=n_warmup,
        latencies_us=latencies_us,
    )


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs_sorted) - 1)
    frac = k - lo
    return float(xs_sorted[lo] * (1 - frac) + xs_sorted[hi] * frac)


_LAST_SAMPLES_US: list[float] = []  # populated by _summarize for --samples_out


def _summarize(
    *,
    backend: str,
    device: str,
    batch_size: int,
    feat_dim: int,
    n_reps: int,
    n_warmup: int,
    latencies_us: list[float],
) -> WCETResult:
    if not latencies_us:
        raise RuntimeError("no latencies recorded")
    global _LAST_SAMPLES_US
    _LAST_SAMPLES_US = list(latencies_us)
    return WCETResult(
        backend=backend,
        device=device,
        batch_size=batch_size,
        feat_dim=feat_dim,
        n_reps=n_reps,
        n_warmup=n_warmup,
        p50_us=_percentile(latencies_us, 50.0),
        p90_us=_percentile(latencies_us, 90.0),
        p99_us=_percentile(latencies_us, 99.0),
        p999_us=_percentile(latencies_us, 99.9),
        max_us=max(latencies_us),
        mean_us=float(statistics.fmean(latencies_us)),
        std_us=float(statistics.pstdev(latencies_us)) if len(latencies_us) > 1 else 0.0,
    )


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m seer.lap.wcet",
        description="Measure LAP inference WCET on a target device.",
    )
    ap.add_argument("--ckpt", default=None,
                    help="PyTorch checkpoint (.pt) — used with --backend torch")
    ap.add_argument("--onnx", default=None,
                    help="ONNX model path — used with --backend onnx")
    ap.add_argument("--plan", default=None,
                    help="TensorRT plan path — used with --backend tensorrt (stub)")
    ap.add_argument("--backend", default="torch",
                    choices=["torch", "onnx", "tensorrt"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=4096,
                    help="Batched blocks per inference (default 4096 — 1 layer × 1 head_group "
                         "of a long-context request).")
    ap.add_argument("--feat_dim", type=int, default=37,
                    help="Input feature dim. Default 37 = 32 hist + 5 aux from features.py")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--out", default=None,
                    help="Optional JSON output path")
    ap.add_argument("--samples_out", default=None,
                    help="Optional sidecar JSON with raw per-rep latencies (µs). "
                         "Used for true-histogram figures (e.g. cross-card "
                         "asymmetry); skip if disk / token budget is tight.")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    if args.backend == "torch":
        if not args.ckpt:
            raise SystemExit("--ckpt required for --backend torch")
        import torch

        from seer.lap.model import build_model
        ckpt = torch.load(args.ckpt, map_location="cpu")
        meta = ckpt["meta"]
        model = build_model(
            ckpt["model_name"],
            input_dim=meta["input_dim"],
            n_horizons=meta["output_dim"],
            history_n=meta.get("history_n", 32),
        )
        model.load_state_dict(ckpt["state_dict"])
        result = measure_torch_wcet(
            model=model,
            batch_size=args.batch,
            feat_dim=meta["input_dim"],
            device=args.device,
            n_reps=args.reps,
            n_warmup=args.warmup,
        )
    elif args.backend == "onnx":
        if not args.onnx:
            raise SystemExit("--onnx required for --backend onnx")
        result = measure_onnx_wcet(
            onnx_path=args.onnx,
            batch_size=args.batch,
            feat_dim=args.feat_dim,
            device=args.device,
            n_reps=args.reps,
            n_warmup=args.warmup,
        )
    else:
        if not args.plan:
            raise SystemExit("--plan required for --backend tensorrt")
        result = measure_tensorrt_wcet(
            plan_path=args.plan,
            batch_size=args.batch,
            feat_dim=args.feat_dim,
            device=args.device,
            n_reps=args.reps,
            n_warmup=args.warmup,
        )

    payload = asdict(result)
    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    if args.samples_out:
        Path(args.samples_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.samples_out).write_text(json.dumps({
            "backend": result.backend, "device": result.device,
            "batch_size": result.batch_size, "n_reps": result.n_reps,
            "samples_us": _LAST_SAMPLES_US,
        }))
    print(text)
    print(
        f"\n[wcet] {result.backend} / {result.device}: "
        f"P50={result.p50_us:.1f}µs  P99={result.p99_us:.1f}µs  "
        f"P99.9={result.p999_us:.1f}µs  max={result.max_us:.1f}µs",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
