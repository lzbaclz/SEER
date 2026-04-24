"""Inference wrapper for a LAP — supports ONNX or a loaded PyTorch checkpoint.

Typical use from the SEER policy:

    predictor = LAPPredictor.from_onnx("checkpoints/lap.onnx")
    probs = predictor(X)   # [N, input_dim]  ->  [N, n_horizons]
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
    def from_onnx(cls, path: str | Path, device: str = "cuda") -> "LAPPredictor":
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
    def from_torch_ckpt(cls, path: str | Path, device: str = "cpu") -> "LAPPredictor":
        import torch

        from seer.lap.models import build_model

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

    # --- Call / predict -------------------------------------------------

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Return sigmoid probabilities of shape [N, n_horizons]."""
        logits = self._run(X)
        return _sigmoid(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.__call__(X)
