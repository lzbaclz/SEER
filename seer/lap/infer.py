"""Inference wrapper for a LAP — supports ONNX, PyTorch, or (stub) TensorRT.

Typical use from the SEER policy::

    predictor = LAPPredictor.from_onnx("checkpoints/lap.onnx")
    probs = predictor(X)   # [N, input_dim]  ->  [N, n_horizons]

The wrapper hides backend differences: caller always sees a callable
``X (np.float32) -> probs (np.float32)`` returning sigmoid probabilities.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class LAPPredictor:
    """Callable predictor over feature matrices. Thread-safe at inference."""

    def __init__(self, backend: str, runner):
        self.backend = backend
        self._run = runner  # callable: np.ndarray -> np.ndarray (logits)

    # --- Factory constructors -------------------------------------------

    @classmethod
    def from_onnx(cls, path: str | Path, device: str = "cuda") -> LAPPredictor:
        import onnxruntime as ort
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        sess = ort.InferenceSession(str(path), providers=providers)
        input_name = sess.get_inputs()[0].name

        def runner(X: np.ndarray) -> np.ndarray:
            return sess.run(None, {input_name: X.astype(np.float32)})[0]

        return cls(backend="onnx", runner=runner)

    @classmethod
    def from_torch_ckpt(cls, path: str | Path, device: str = "cpu") -> LAPPredictor:
        import torch

        from seer.lap.model import build_model

        ckpt = torch.load(path, map_location=device)
        meta = ckpt["meta"]
        model = build_model(
            ckpt["model_name"],
            input_dim=meta["input_dim"],
            n_horizons=meta["output_dim"],
            history_n=meta.get("history_n", 32),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        model.to(device)

        def runner(X: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                x = torch.from_numpy(X).float().to(device)
                return model(x).cpu().numpy()

        return cls(backend="torch", runner=runner)

    @classmethod
    def from_tensorrt(
        cls,
        path: str | Path,
        device: str = "cuda",
        feat_dim: int | None = None,
        capture_cuda_graph: bool = True,
    ) -> LAPPredictor:
        """Load a TensorRT plan file with optional CUDA-Graph capture.

        The plan is built by :func:`seer.lap.export.export_tensorrt` with
        a fully static input shape ``[B, feat_dim]``. At first call we
        allocate device buffers, set tensor addresses, and (by default)
        capture the entire ``execute_async_v3`` invocation into a CUDA
        Graph so subsequent inferences are a single graph replay.

        For policy use the runner expects a callable
        ``predictor(X: np.ndarray) -> np.ndarray``. We honour the
        contract by copying X → device buffer, replaying, copying out.
        The host-side copy dominates the latency for tiny X (~32 blocks)
        and amortizes for large X (~4096 blocks). When the policy passes
        a batch that doesn't match the engine's locked shape, we pad
        with zeros and slice the output back — the WCET claim only
        applies at the locked shape, so padding is honest.
        """
        try:
            import tensorrt as trt
        except ImportError as e:
            raise RuntimeError(
                "TensorRT not installed — from_tensorrt unavailable. "
                "Install with: pip install --extra-index-url "
                "https://pypi.nvidia.com 'tensorrt-cu12'. Or use from_onnx."
            ) from e
        import torch

        if device != "cuda" and not device.startswith("cuda"):
            raise ValueError("from_tensorrt requires device='cuda'")
        if not torch.cuda.is_available():
            raise RuntimeError("from_tensorrt requires CUDA GPU")

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"failed to deserialize {path}")
        context = engine.create_execution_context()

        tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        inp_names = [n for n in tensor_names
                     if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        out_names = [n for n in tensor_names
                     if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        inp_name, out_name = inp_names[0], out_names[0]

        # Locked engine shape — this is what defines the WCET.
        engine_shape = tuple(engine.get_tensor_shape(inp_name))
        if len(engine_shape) != 2:
            raise RuntimeError(f"unexpected engine input rank: {engine_shape}")
        engine_batch, engine_feat = int(engine_shape[0]), int(engine_shape[1])
        if feat_dim is not None and feat_dim != engine_feat:
            raise RuntimeError(
                f"feat_dim mismatch: engine wants {engine_feat}, caller {feat_dim}"
            )

        out_shape = tuple(engine.get_tensor_shape(out_name))
        if any(d < 0 for d in out_shape):
            out_shape = (engine_batch,) + tuple(int(d) for d in out_shape[1:])
        else:
            out_shape = tuple(int(d) for d in out_shape)

        dev = torch.device("cuda:0")
        inp_buf = torch.zeros(engine_batch, engine_feat,
                              dtype=torch.float32, device=dev)
        out_buf = torch.empty(out_shape, dtype=torch.float32, device=dev)
        context.set_tensor_address(inp_name, inp_buf.data_ptr())
        context.set_tensor_address(out_name, out_buf.data_ptr())

        stream = torch.cuda.Stream()

        def _execute_once():
            with torch.cuda.stream(stream):
                context.execute_async_v3(stream.cuda_stream)

        # Warmup so kernel selection settles before graph capture.
        for _ in range(8):
            _execute_once()
        stream.synchronize()

        graph = None
        if capture_cuda_graph:
            try:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    _execute_once()
                stream.synchronize()
            except Exception as e:  # noqa: BLE001
                print(f"[lap-trt] CUDA Graph capture failed ({e}); using "
                      "ungraphed replay")
                graph = None

        def runner(X: np.ndarray) -> np.ndarray:
            X = np.ascontiguousarray(X.astype(np.float32))
            n, d = X.shape
            if d != engine_feat:
                raise ValueError(f"X.shape[1]={d} but engine wants {engine_feat}")
            if n > engine_batch:
                # Chunk through engine_batch slices when caller hands us
                # more rows than the engine's locked batch admits.
                outs = []
                for s in range(0, n, engine_batch):
                    slice_n = min(engine_batch, n - s)
                    outs.append(runner(X[s:s + slice_n]))
                return np.concatenate(outs, axis=0)

            # Pad up to engine_batch if needed (the WCET claim is *at*
            # engine_batch; padding rows ride for free).
            inp_buf.zero_()
            x_src = torch.from_numpy(X).to(dev, non_blocking=True)
            inp_buf[:n].copy_(x_src)
            if graph is not None:
                graph.replay()
            else:
                _execute_once()
            stream.synchronize()
            out = out_buf[:n].clone().cpu().numpy()
            return out

        wrapper = cls(backend="tensorrt+cudagraph" if graph is not None
                                else "tensorrt", runner=runner)
        # Stash references so they don't get GC'd while wrapper is alive.
        wrapper._engine = engine          # type: ignore[attr-defined]
        wrapper._context = context        # type: ignore[attr-defined]
        wrapper._buffers = (inp_buf, out_buf)  # type: ignore[attr-defined]
        wrapper._graph = graph            # type: ignore[attr-defined]
        wrapper._stream = stream          # type: ignore[attr-defined]
        return wrapper

    # --- Auto-dispatch helper -------------------------------------------

    @classmethod
    def from_path(cls, path: str | Path, device: str = "cuda") -> LAPPredictor:
        """Pick the right backend by file extension."""
        s = str(path).lower()
        if s.endswith(".onnx"):
            return cls.from_onnx(path, device=device)
        if s.endswith(".plan") or s.endswith(".engine"):
            return cls.from_tensorrt(path, device=device)
        return cls.from_torch_ckpt(path, device=device)

    # --- Call / predict -------------------------------------------------

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Return sigmoid probabilities of shape [N, n_horizons]."""
        logits = self._run(X)
        return _sigmoid(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.__call__(X)


def load_lap_scorer(device: str = "cuda") -> LAPPredictor:
    """Load the production LAP checkpoint for hot-path / vLLM hooks."""
    import os

    path = os.environ.get("SEER_LAP", "checkpoints/lap_prod_tiny.pt")
    return LAPPredictor.from_path(path, device=device)


def compute_eps_at_phi(
    lap_checkpoint: str,
    trace_path: str,
    phi: float,
    horizon: int = 4,
    n_samples: int = 2000,
    sink: int = 4,
    window: int = 4,
) -> float:
    """Estimate deployed false-negative rate ε(φ) under SEER policy.

    Wraps the schedulability estimator used by Lemma~2 inversion.
    ``trace_path`` may be a parquet directory or single parquet file.
    ``n_samples`` is reserved for future subsampling; the current
    implementation uses the full held-out test split.
    """
    del n_samples  # reserved
    from seer.timing.schedulability import _estimate_epsilon_from_lap
    from seer.trace.schema import HORIZONS

    try:
        horizon_idx = list(HORIZONS).index(horizon)
    except ValueError:
        horizon_idx = 1
    return _estimate_epsilon_from_lap(
        lap_path=lap_checkpoint,
        workload=trace_path,
        hbm_budget=float(phi),
        sink=sink,
        window=window,
        horizon_idx=horizon_idx,
    )
