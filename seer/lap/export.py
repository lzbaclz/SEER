"""CLI: export a trained LAP checkpoint to ONNX or (stub) TensorRT.

Examples
--------

ONNX (works everywhere):

    python -m seer.lap.export \\
        --ckpt checkpoints/lap_llama3_8b.pt \\
        --backend onnx \\
        --out checkpoints/lap_llama3_8b.onnx

TensorRT static-shape (RTSS-pivot path; requires tensorrt installed):

    python -m seer.lap.export \\
        --ckpt checkpoints/lap_llama3_8b.pt \\
        --backend tensorrt --static_shape \\
        --batch 4096 \\
        --out checkpoints/lap_llama3_8b.plan

The TensorRT path goes via an intermediate ONNX file: ``ckpt → onnx →
trt.plan``. Static-shape build means the input batch dimension is
locked, which is what makes CUDA Graph capture viable downstream.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="onnx", choices=["onnx", "tensorrt"])
    ap.add_argument("--batch", type=int, default=1024,
                    help="Dummy batch size for export. With --static_shape this "
                         "is the LOCKED batch dim used at inference.")
    ap.add_argument("--static_shape", action="store_true",
                    help="Lock the batch dim (TensorRT WCET-friendly).")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--fp16", action="store_true",
                    help="Build with fp16 weights (TensorRT only).")
    return ap


def export_onnx(ckpt_path: str, out_path: str, batch: int, opset: int,
                static_shape: bool) -> str:
    import torch

    from seer.lap.model import build_model

    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt["meta"]
    history_n = meta.get("history_n", 32)
    model = build_model(
        ckpt["model_name"],
        input_dim=meta["input_dim"],
        n_horizons=meta["output_dim"],
        history_n=history_n,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dummy = torch.randn(batch, meta["input_dim"])
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = None if static_shape else {
        "features": {0: "batch"},
        "logits": {0: "batch"},
    }
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
    )
    print(f"[export] wrote ONNX → {out_path} "
          f"(opset={opset}, static_shape={static_shape}, batch={batch})")
    return out_path


def export_tensorrt(
    onnx_path: str,
    plan_path: str,
    batch: int,
    fp16: bool,
    feat_dim: int = 37,
    workspace_mb: int = 1024,
) -> str:
    """Convert ONNX → TensorRT plan with statically locked batch shape.

    Implementation uses the TensorRT Python API directly (no ``trtexec``
    dependency). The build pipeline:

    1. Create a Builder + NetworkDefinition with EXPLICIT_BATCH flag.
    2. Parse the ONNX model.
    3. Set the input shape to ``[batch, feat_dim]`` exactly (no
       optimization profile needed for fully static shape).
    4. Optionally enable fp16.
    5. Serialize the engine and dump to ``plan_path``.

    Static shape is the prerequisite for CUDA-Graph capture in
    :class:`seer.lap.infer.LAPPredictor.from_tensorrt`, which is what
    keeps ``C_LAP`` repeatable at the µs scale required by Lemma 2.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise RuntimeError(
            "TensorRT not installed. Install with:\n"
            "    pip install --extra-index-url https://pypi.nvidia.com "
            "'tensorrt-cu12'\n"
            "or use --backend onnx."
        ) from e

    Path(plan_path).parent.mkdir(parents=True, exist_ok=True)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"ONNX parse failed:\n{errs}")

    # Force input to a fully static shape — required for CUDA Graph replay.
    inp = network.get_input(0)
    inp.shape = (batch, feat_dim)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024
    )
    if fp16:
        if not builder.platform_has_fast_fp16:
            print("[export] platform reports no fast fp16; building anyway")
        config.set_flag(trt.BuilderFlag.FP16)
    # CUDA-Graph friendly: prefer kernels with deterministic shape behaviour
    # so replay latency is stable.
    if hasattr(trt.BuilderFlag, "PREFER_PRECISION_CONSTRAINTS"):
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    print(f"[export] building TensorRT plan (batch={batch}, fp16={fp16}, "
          f"workspace={workspace_mb}MB)")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None — "
                           "check the logger output above for layer errors")
    # IHostMemory doesn't support len(); cast to bytes via memoryview.
    payload = bytes(serialized)
    with open(plan_path, "wb") as f:
        f.write(payload)
    print(f"[export] wrote TensorRT plan → {plan_path} "
          f"({len(payload) / 1024:.1f} KB)")
    return plan_path


def main():
    args = _build_argparser().parse_args()

    out = Path(args.out)
    if args.backend == "onnx":
        export_onnx(args.ckpt, str(out), args.batch, args.opset, args.static_shape)
        return

    # tensorrt: go via an ONNX intermediate. Read the ckpt feat_dim so the
    # builder can lock the network input to the correct shape regardless of
    # whether the user passed --feat_dim.
    import torch
    ckpt = torch.load(args.ckpt, map_location="cpu")
    feat_dim = int(ckpt["meta"]["input_dim"])

    onnx_intermediate = out.with_suffix(".onnx")
    export_onnx(args.ckpt, str(onnx_intermediate), args.batch, args.opset,
                static_shape=True)  # tensorrt always wants static shape
    export_tensorrt(str(onnx_intermediate), str(out), args.batch, args.fp16,
                    feat_dim=feat_dim)


if __name__ == "__main__":
    main()
