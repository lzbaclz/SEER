"""CLI: export a trained LAP checkpoint to ONNX.

Example
-------
    python -m seer.lap.export \\
        --ckpt checkpoints/lap_llama3_8b.pt \\
        --out checkpoints/lap_llama3_8b.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=1024,
                    help="dummy batch size for export (axis is dynamic)")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    import torch

    from seer.lap.models import build_model

    ckpt = torch.load(args.ckpt, map_location="cpu")
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

    dummy = torch.randn(args.batch, meta["input_dim"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={
            "features": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=args.opset,
    )
    print(f"[export] wrote {out_path}  (opset={args.opset})")


if __name__ == "__main__":
    main()
