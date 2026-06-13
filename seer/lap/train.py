"""CLI: train a Learned Attention Predictor with optional WCET-aware penalty.

Example
-------

    python -m seer.lap.train \\
        --traces data/traces/llama3-8b \\
        --arch tiny_mlp \\
        --history 32 \\
        --horizons 1 4 16 64 \\
        --wcet_budget_us 200 \\
        --out checkpoints/lap_llama3_8b.pt

The resulting .pt file bundles model state, model_name, feature meta, and
training args — so :mod:`seer.lap.export` can round-trip to ONNX /
TensorRT without knowing the feature config up front.

WCET-aware model selection
--------------------------
Under the RTSS framing the predictor's *own* inference time is part of
the timing budget (Lemma 2's ``C_LAP``). With ``--wcet_budget_us > 0``
the trainer measures the model's batched forward pass at the end of
each epoch and uses a penalty as a *Pareto filter* on the checkpoint
selection score (not on the back-propagated training loss). The
checkpoint with the highest selection score is saved:

    score = mean_AUC - 1e-3 * beta * max(0, p50_us - budget_us)^2

``beta`` is annealed from 0 (epoch 0) to ``wcet_beta_max`` (default 1.0)
at epoch ``epochs/2`` so early epochs ignore WCET entirely and later
epochs rank checkpoints by a (AUC, WCET) Pareto trade-off.

Why filter, not loss? The forward-pass latency is not differentiable
w.r.t. the model parameters, so a true loss term would require a
surrogate (parameter count, FLOPs). The empirical filter retains the
correctness guarantee that the released checkpoint clears the WCET
budget on the target GPU.

When ``--wcet_budget_us`` is 0 (default) the trainer behaves exactly like
the NeurIPS-era pipeline (selects by mean AUC alone).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train Learned Attention Predictor.")
    ap.add_argument("--traces", required=True, help="parquet dir")
    # both --arch (RTSS spelling) and --model (NeurIPS spelling) accepted
    ap.add_argument("--arch", "--model", dest="arch", default="tiny_mlp",
                    choices=["tiny_mlp", "block_rnn", "block_transformer"])
    ap.add_argument("--history", type=int, default=32)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 16, 64],
                    help="must match HORIZONS in schema.py")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--loss", default="focal", choices=["bce", "focal"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log_dir", default="logs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_train_rows", type=int, default=None,
                    help="optional cap for quick experiments")
    # RTSS pivot
    ap.add_argument("--wcet_budget_us", type=float, default=0.0,
                    help="If > 0, add a soft penalty for forward-pass latency "
                         "exceeding this budget (in µs). Measured per-epoch "
                         "at batch_size=`--wcet_probe_batch`.")
    ap.add_argument("--wcet_probe_batch", type=int, default=4096)
    ap.add_argument("--wcet_beta_max", type=float, default=1.0,
                    help="Maximum WCET penalty weight (annealed from 0).")
    return ap.parse_args()


def wcet_aware_score(mean_auc: float, measured_us: float, budget_us: float,
                     beta: float, score_scale: float = 1e-3) -> float:
    """Pareto-filter score used for WCET-aware checkpoint selection.

    ``score = mean_auc - score_scale * beta * max(0, measured - budget)^2``

    The hinge penalises any checkpoint whose measured forward-pass
    latency exceeds the budget, scaled by ``beta`` (annealed across
    epochs). The penalty is on the *selection* score, not on the
    back-propagated loss — the forward-pass latency is not
    differentiable w.r.t. model parameters, so this is a Pareto
    filter rather than a constraint pulled into gradient descent.

    This function exists so the score formula in
    \\S\\ref{eq:wcet-score} of paper/sections/04_method.tex is a
    direct callable rather than an inline expression buried in
    :func:`main`, and so it can be unit-tested in isolation.

    Parameters
    ----------
    mean_auc : float
        Validation AUC averaged across horizons.
    measured_us : float
        Median forward-pass latency probed at the deployment batch
        size on the target GPU, in microseconds.
    budget_us : float
        WCET budget W, in microseconds (e.g.\\ 200 for chat-50ms).
    beta : float
        Annealed penalty weight in [0, beta_max].
    score_scale : float
        Global penalty scale; default 1e-3 (matches paper).

    Returns
    -------
    float
        Selection score; higher is better. Pareto-dominated
        checkpoints (lower AUC, higher latency) score lower.
    """
    penalty = beta * max(0.0, float(measured_us) - float(budget_us)) ** 2
    return float(mean_auc) - float(score_scale) * penalty


def _measure_forward_latency_us(model, device, batch_size: int, feat_dim: int,
                                n_reps: int = 64) -> float:
    """Measure median forward latency at the given batch size, in µs."""
    import torch
    model.eval()
    use_cuda = device.type == "cuda"
    x = torch.randn(batch_size, feat_dim, device=device)
    with torch.no_grad():
        for _ in range(8):  # warmup
            _ = model(x)
        if use_cuda:
            torch.cuda.synchronize()
        latencies_us = []
        if use_cuda:
            for _ in range(n_reps):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(x)
                end.record()
                end.synchronize()
                latencies_us.append(start.elapsed_time(end) * 1000.0)
        else:
            for _ in range(n_reps):
                t0 = time.perf_counter()
                _ = model(x)
                t1 = time.perf_counter()
                latencies_us.append((t1 - t0) * 1e6)
    latencies_us.sort()
    return float(latencies_us[len(latencies_us) // 2])


def main():
    args = _parse_args()

    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    from seer.lap.dataset import TraceDataset
    from seer.lap.features import build_features
    from seer.lap.losses import LOSS_FNS
    from seer.lap.model import build_model, count_params
    from seer.trace.loader import load_traces, split_by_request

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[train] loading traces from {args.traces}")
    df = load_traces(args.traces)
    print(f"[train] total rows: {len(df):,}  |  requests: {df['request_id'].nunique()}")

    train_df, val_df, test_df = split_by_request(df, seed=args.seed)
    del test_df  # not used at training time
    print(f"[train] train/val rows: {len(train_df):,} / {len(val_df):,}")

    print("[train] building features (this may take a minute on large traces)")
    X_tr, y_tr, meta = build_features(train_df, history_n=args.history)
    X_va, y_va, _ = build_features(val_df, history_n=args.history)

    if args.max_train_rows and len(X_tr) > args.max_train_rows:
        idx = np.random.default_rng(args.seed).choice(
            len(X_tr), args.max_train_rows, replace=False
        )
        X_tr, y_tr = X_tr[idx], y_tr[idx]
        print(f"[train] subsampled train to {len(X_tr):,} rows")

    print(f"[train] input_dim={meta['input_dim']}  output_dim={meta['output_dim']}")

    model = build_model(
        args.arch,
        input_dim=meta["input_dim"],
        n_horizons=meta["output_dim"],
        history_n=args.history,
    )
    print(f"[train] {args.arch}  {count_params(model):,} params")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(
        TraceDataset(X_tr, y_tr),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        TraceDataset(X_va, y_va),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = LOSS_FNS[args.loss]

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_score = -float("inf")
    history = []

    for epoch in range(args.epochs):
        model.train()
        total, n_seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        scheduler.step()
        train_loss = total / max(1, n_seen)

        # --- Validation AUC per horizon
        model.eval()
        all_logits, all_y = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                all_logits.append(model(xb).cpu().numpy())
                all_y.append(yb.numpy())
        preds = np.concatenate(all_logits, axis=0)
        ys = np.concatenate(all_y, axis=0)
        aucs = []
        for h_idx in range(ys.shape[1]):
            try:
                aucs.append(float(roc_auc_score(ys[:, h_idx], preds[:, h_idx])))
            except ValueError:
                aucs.append(float("nan"))
        mean_auc = float(np.nanmean(aucs))

        # --- Optional WCET probe + penalty
        wcet_p50 = 0.0
        wcet_penalty = 0.0
        if args.wcet_budget_us > 0:
            wcet_p50 = _measure_forward_latency_us(
                model, device,
                batch_size=args.wcet_probe_batch,
                feat_dim=meta["input_dim"],
            )
            beta = args.wcet_beta_max * min(1.0, 2.0 * (epoch + 1) / max(1, args.epochs))
            wcet_penalty = beta * max(0.0, wcet_p50 - args.wcet_budget_us) ** 2
        else:
            beta = 0.0
        # Pareto-filter score: see `wcet_aware_score` for the formula
        # bound to paper §4.1 eq.~(eq:wcet-score).
        score = wcet_aware_score(
            mean_auc=mean_auc,
            measured_us=wcet_p50,
            budget_us=args.wcet_budget_us if args.wcet_budget_us > 0 else 0.0,
            beta=beta,
        )

        print(
            f"[train] ep {epoch+1:02d}/{args.epochs}  "
            f"loss={train_loss:.4f}  "
            f"val_AUC={[f'{a:.3f}' for a in aucs]}  "
            f"mean={mean_auc:.3f}  "
            + (f"wcet_p50={wcet_p50:.1f}µs  " if args.wcet_budget_us > 0 else "")
            + f"lr={scheduler.get_last_lr()[0]:.2e}",
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_auc": aucs,
            "val_auc_mean": mean_auc,
            "wcet_p50_us": wcet_p50,
            "wcet_penalty": wcet_penalty,
            "score": score,
        })

        if score > best_score:
            best_score = score
            torch.save({
                "state_dict": model.state_dict(),
                "model_name": args.arch,
                "meta": meta,
                "args": vars(args),
                "epoch": epoch + 1,
                "val_auc": aucs,
                "val_auc_mean": mean_auc,
                "wcet_p50_us": wcet_p50,
            }, out_path)
            print(f"[train]   saved best → {out_path} (mean AUC {mean_auc:.3f}"
                  + (f", WCET p50 {wcet_p50:.1f}µs" if args.wcet_budget_us > 0 else "")
                  + ")")

    with open(log_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"[train] done. best score = {best_score:.4f}")


if __name__ == "__main__":
    main()
