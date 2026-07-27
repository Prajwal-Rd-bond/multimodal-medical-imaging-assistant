"""Fit + save a temperature scaler for one modality's trained classifier,
using its held-out validation set.

    python -m calibrate.fit_temperature --modality cxr
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from configs.config import load_config
from calibrate.temperature_scaling import compute_ece, fit_temperature
from train.device import get_device
from train.train import build_datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=["cxr", "derm"], required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    task = "multilabel" if args.modality == "cxr" else "multiclass"
    device = get_device()

    from models.backbone import build_classifier

    _, val_ds, _, label_names = build_datasets(cfg, args.modality)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    model = build_classifier(
        modality=args.modality,
        num_classes=len(label_names),
        backbone_name=cfg.model.backbone,
        pretrained=False,
        freeze_stages=0,
        dropout=cfg.model.dropout,
    ).to(device)
    ckpt_path = Path(cfg.train.checkpoint_dir) / f"{args.modality}_best.pt"
    # weights_only=False: this repo's own checkpoint, not a third-party file.
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()

    all_logits, all_labels, all_masks = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch["image"].to(device)).cpu()
            all_logits.append(logits)
            if task == "multilabel":
                all_labels.append(batch["labels"])
                all_masks.append(batch["loss_mask"])
            else:
                all_labels.append(batch["label"])

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    loss_mask = torch.cat(all_masks) if task == "multilabel" else None

    scaler = fit_temperature(
        logits, labels, task=task, per_class=(task == "multilabel"),
        max_iter=cfg.calibrate.max_iter, lr=cfg.calibrate.lr, loss_mask=loss_mask,
    )

    # Report ECE before/after so the improvement is visible immediately.
    if task == "multilabel":
        import numpy as np
        raw_probs = torch.sigmoid(logits).numpy()
        cal_probs = torch.sigmoid(scaler(logits)).detach().numpy()
        labels_np = labels.numpy()
        mask_np = loss_mask.numpy()
        for i, name in enumerate(label_names):
            valid = mask_np[:, i] > 0
            before = compute_ece(raw_probs[valid, i], labels_np[valid, i], n_bins=cfg.calibrate.n_bins)
            after = compute_ece(cal_probs[valid, i], labels_np[valid, i], n_bins=cfg.calibrate.n_bins)
            print(f"{name}: ECE before={before.ece:.4f} after={after.ece:.4f}")
    else:
        raw_probs = torch.softmax(logits, dim=1).numpy()
        cal_probs = torch.softmax(scaler(logits), dim=1).detach().numpy()
        preds = raw_probs.argmax(axis=1)
        labels_np = labels.numpy()
        correct = (preds == labels_np).astype(float)
        conf_before = raw_probs.max(axis=1)
        conf_after = cal_probs.max(axis=1)
        before = compute_ece(conf_before, correct, n_bins=cfg.calibrate.n_bins)
        after = compute_ece(conf_after, correct, n_bins=cfg.calibrate.n_bins)
        print(f"overall: ECE before={before.ece:.4f} after={after.ece:.4f}")

    out_path = Path(cfg.train.checkpoint_dir) / f"{args.modality}_temperature.pt"
    torch.save(scaler.state_dict(), out_path)
    print(f"Saved fitted temperature scaler to {out_path} (T={scaler.temperature.detach().numpy()})")


if __name__ == "__main__":
    main()
