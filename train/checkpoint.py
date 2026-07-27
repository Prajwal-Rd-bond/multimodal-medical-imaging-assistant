"""Checkpoint save/resume -- non-negotiable given Colab's session disconnects."""
from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: Path, model, optimizer, scaler, epoch: int, global_step: int, best_metric: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "global_step": global_step,
            "best_metric": best_metric,
        },
        tmp_path,
    )
    tmp_path.replace(path)  # atomic on same filesystem -- avoids truncated ckpt if interrupted mid-write


def load_checkpoint(path: Path, model, optimizer=None, scaler=None, map_location="cpu") -> dict:
    # weights_only=False: this repo's own checkpoints, not third-party files -- torch 2.6+
    # defaults to True, which rejects the optimizer state saved alongside the model weights.
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    return {
        "epoch": ckpt.get("epoch", 0),
        "global_step": ckpt.get("global_step", 0),
        "best_metric": ckpt.get("best_metric", float("-inf")),
    }


def find_latest_checkpoint(checkpoint_dir: Path, name: str) -> Path | None:
    latest = checkpoint_dir / f"{name}_latest.pt"
    return latest if latest.exists() else None
