"""CLI: train a LAP on collected attention traces.

Example
-------
    python -m seer.lap.train \\
        --traces data/traces/llama3-8b \\
        --model tiny_mlp \\
        --epochs 10 \\
        --out checkpoints/lap_llama3_8b.pt

The resulting .pt file bundles model state, model_name, feature meta, and
training args — so `seer.lap.export` can round-trip to ONNX without knowing
the feature config up front.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train Learned Attention Predictor.")
    ap.add_argument("--traces", required=True, help="parquet dir")
    ap.add_argument("--model", default="tiny_mlp",
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
    return ap.parse_args()


def main():
    args = _parse_args()

    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    from seer.lap.dataset import TraceDataset
    from seer.lap.features import build_features
    from seer.lap.losses import LOSS_FNS
    from seer.lap.models import build_model, count_params
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
        idx = np.random.default_rng(args.seed).choice(len(X_tr), args.max_train_rows, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]
        print(f"[train] subsampled train to {len(X_tr):,} rows")

    print(f"[train] input_dim={meta['input_dim']}  output_dim={meta['output_dim']}")

    model = build_model(
        args.model,
        input_dim=meta["input_dim"],
        n_horizons=meta["output_dim"],
        history_n=args.history,
    )
    print(f"[train] {args.model}  {count_params(model):,} params")

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

    best_mean_auc = -1.0
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

        print(f"[train] ep {epoch+1:02d}/{args.epochs}  "
              f"loss={train_loss:.4f}  "
              f"val_AUC={['%.3f' % a for a in aucs]}  "
              f"mean={mean_auc:.3f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_auc": aucs,
            "val_auc_mean": mean_auc,
        })

        if mean_auc > best_mean_auc:
            best_mean_auc = mean_auc
            torch.save({
                "state_dict": model.state_dict(),
                "model_name": args.model,
                "meta": meta,
                "args": vars(args),
                "epoch": epoch + 1,
                "val_auc": aucs,
                "val_auc_mean": mean_auc,
            }, out_path)
            print(f"[train]   saved best → {out_path} (mean AUC {best_mean_auc:.3f})")

    with open(log_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"[train] done. best val mean AUC = {best_mean_auc:.3f}")


if __name__ == "__main__":
    main()
