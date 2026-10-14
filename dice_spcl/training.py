"""Train DICE-SPCL and extract the fused shared representations."""
from __future__ import annotations
import argparse
import csv
import random
from pathlib import Path
from typing import Any
import numpy as np

def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

def _load_torch():
    try:
        import torch
        from .model import DICESPCL, dice_loss
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "DICE-SPCL requires PyTorch. Activate a Python environment with torch installed."
        ) from exc
    return torch, DICESPCL, dice_loss

def _train_seed(
    x: np.ndarray,
    agreement: np.ndarray,
    fusion: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    out_dir: Path,
) -> np.ndarray:
    torch, model_class, loss_function = _load_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model_class(
        n_views=x.shape[1],
        in_dim=x.shape[2],
        hidden_dim=args.hidden_dim,
        shared_dim=args.shared_dim,
        private_dim=args.private_dim,
        proj_dim=args.proj_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x_tensor = torch.from_numpy(x)
    agreement_tensor = torch.from_numpy(agreement)
    fusion_tensor = torch.from_numpy(fusion)
    history: list[dict[str, Any]] = []
    generator = np.random.default_rng(seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = generator.permutation(len(x))
        epoch_rows = []
        for start in range(0, len(order), args.batch_size):
            index = order[start : start + args.batch_size]
            if len(index) < 4:
                continue
            batch_x = x_tensor[index].to(device)
            batch_q = agreement_tensor[index].to(device)
            batch_alpha = fusion_tensor[index].to(device)
            outputs = model(batch_x, batch_alpha)
            loss, metrics = loss_function(
                outputs,
                batch_x,
                batch_q,
                {
                    "reconstruction": args.loss_reconstruction,
                    "contrast": args.loss_contrast,
                    "orthogonality": args.loss_orthogonality,
                    "redundancy": args.loss_redundancy,
                },
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_rows.append(metrics)
        if not epoch_rows:
            raise RuntimeError("No valid training batches; decrease --batch-size.")
        history.append(
            {
                "epoch": epoch,
                **{
                    key: float(np.mean([row[key] for row in epoch_rows]))
                    for key in epoch_rows[0]
                },
            }
        )

    model.eval()
    fused = []
    with torch.no_grad():
        for start in range(0, len(x), 512):
            batch = torch.from_numpy(x[start : start + 512]).to(device)
            alpha = torch.from_numpy(fusion[start : start + 512]).to(device)
            fused.append(model(batch, alpha)["fused"].cpu().numpy())
    latent = np.concatenate(fused, axis=0).astype(np.float32)
    _write_csv(out_dir / "training_loss.csv", history)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seed": seed,
            "dimensions": {
                "n_views": x.shape[1],
                "in_dim": x.shape[2],
                "hidden_dim": args.hidden_dim,
                "shared_dim": args.shared_dim,
                "private_dim": args.private_dim,
                "proj_dim": args.proj_dim,
            },
        },
        out_dir / "model.pt",
    )
    return latent
