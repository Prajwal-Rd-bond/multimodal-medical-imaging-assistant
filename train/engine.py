"""Train/eval loops shared by the CXR (multi-label) and dermoscopy (multi-class)
classifiers. The two differ only in loss + metric computation, both handled
by the `task` argument.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from train.checkpoint import save_checkpoint

Task = Literal["multilabel", "multiclass"]


def cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float) -> float:
    """Linear warmup for `warmup_steps`, then cosine decay to `min_lr` over the
    remaining steps. Pure function of `step` -- no scheduler object/state to
    checkpoint, since `global_step` is already saved/restored on resume."""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    remaining = max(total_steps - warmup_steps, 1)
    progress = min((step - warmup_steps) / remaining, 1.0)
    return float(min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress)))


def compute_pos_weight(dataset, num_classes: int) -> torch.Tensor:
    """Per-class BCE pos_weight = n_negative / n_positive, for class imbalance
    (per the plan: "these datasets are heavily imbalanced")."""
    pos = np.zeros(num_classes)
    total = 0
    for i in range(len(dataset)):
        item = dataset[i]
        pos += item["labels"].numpy() * item["loss_mask"].numpy()
        total += 1
    pos = np.clip(pos, 1, None)  # avoid div-by-zero for never-positive classes
    neg = total - pos
    return torch.tensor(neg / pos, dtype=torch.float32)


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _multilabel_loss(logits, batch, pos_weight):
    loss_unreduced = F.binary_cross_entropy_with_logits(
        logits, batch["labels"], pos_weight=pos_weight, reduction="none"
    )
    mask = batch["loss_mask"]
    return (loss_unreduced * mask).sum() / mask.sum().clamp(min=1)


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer,
    device,
    task: Task,
    scaler=None,
    pos_weight=None,
    class_weight=None,
    grad_clip_norm: float = 5.0,
    global_step: int = 0,
    log_fn=None,
    lr_schedule_fn=None,
):
    model.train()
    running_loss = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        if lr_schedule_fn is not None:
            lr = lr_schedule_fn(global_step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=scaler is not None):
            logits = model(images)
            if task == "multilabel":
                labels = batch["labels"].to(device)
                mask = batch["loss_mask"].to(device)
                loss = _multilabel_loss(logits, {"labels": labels, "loss_mask": mask}, pos_weight)
            else:
                labels = batch["label"].to(device)
                loss = F.cross_entropy(logits, labels, weight=class_weight)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        running_loss += loss.item()
        n_batches += 1
        global_step += 1
        if log_fn is not None:
            log_fn({"train/loss": loss.item(), "train/step": global_step, "train/lr": optimizer.param_groups[0]["lr"]})

    return running_loss / max(n_batches, 1), global_step


@torch.no_grad()
def evaluate(model, loader: DataLoader, device, task: Task, label_names: list[str]):
    model.eval()
    all_logits, all_labels, all_masks = [], [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        all_logits.append(logits.cpu())
        if task == "multilabel":
            all_labels.append(batch["labels"])
            all_masks.append(batch["loss_mask"])
        else:
            all_labels.append(batch["label"])

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    metrics = {}
    if task == "multilabel":
        masks = torch.cat(all_masks)
        probs = torch.sigmoid(logits).numpy()
        labels_np = labels.numpy()
        masks_np = masks.numpy()
        aucs = []
        for i, name in enumerate(label_names):
            valid = masks_np[:, i] > 0
            y = labels_np[valid, i]
            if len(np.unique(y)) < 2:
                continue  # AUC undefined with a single class present
            auc = roc_auc_score(y, probs[valid, i])
            metrics[f"auc/{name}"] = auc
            aucs.append(auc)
        metrics["auc/mean"] = float(np.mean(aucs)) if aucs else float("nan")
    else:
        probs = torch.softmax(logits, dim=1).numpy()
        labels_np = labels.numpy()
        try:
            metrics["auc/macro_ovr"] = roc_auc_score(labels_np, probs, multi_class="ovr", average="macro")
        except ValueError:
            metrics["auc/macro_ovr"] = float("nan")
        preds = probs.argmax(axis=1)
        metrics["accuracy"] = float((preds == labels_np).mean())

    return metrics, logits, labels
