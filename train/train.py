"""CLI training entrypoint for either modality.

    python -m train.train --modality cxr --config configs/default.yaml
    python -m train.train --modality derm --set train.epochs=5

Designed to survive Colab disconnects: checkpoints every `checkpoint_every_steps`
steps to `train.checkpoint_dir`, and resumes automatically if a checkpoint for
this modality already exists there (`--resume`/`train.resume: true`).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from configs.config import load_config
from data.datasets import DermoscopyDataset, MultiLabelImageDataset, split_dataframe
from data.transforms import eval_transforms, train_transforms
from models.backbone import build_classifier
from train.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from train.device import get_device
from train.engine import compute_class_weights, compute_pos_weight, cosine_lr, evaluate, train_one_epoch

try:
    import wandb
except ImportError:
    wandb = None


class NullLogger:
    def log(self, *_args, **_kwargs):
        pass

    def finish(self):
        pass


def build_logger(cfg, run_name: str):
    mode = cfg.train.wandb_mode
    if mode == "disabled" or wandb is None:
        return NullLogger()
    return wandb.init(project=cfg.train.wandb_project, name=run_name, mode=mode, config=vars(cfg))


def build_datasets(cfg, modality: str):
    import pandas as pd

    if modality == "cxr":
        dcfg = cfg.data.cxr
        df = pd.read_csv(dcfg.labels_csv)
        train_df, val_df, test_df = split_dataframe(df, cfg.data.val_fraction, cfg.data.test_fraction, cfg.seed)

        def make(df_split, transform):
            tmp_csv = Path(dcfg.labels_csv).with_name(".tmp_split.csv")
            df_split.to_csv(tmp_csv, index=False)
            ds = MultiLabelImageDataset(tmp_csv, dcfg.root, dcfg.image_col, list(dcfg.label_cols), transform)
            tmp_csv.unlink()
            return ds

        train_ds = make(train_df, train_transforms(dcfg.image_size))
        val_ds = make(val_df, eval_transforms(dcfg.image_size))
        test_ds = make(test_df, eval_transforms(dcfg.image_size))
        return train_ds, val_ds, test_ds, list(dcfg.label_cols)

    dcfg = cfg.data.derm
    df = pd.read_csv(dcfg.labels_csv)
    train_df, val_df, test_df = split_dataframe(df, cfg.data.val_fraction, cfg.data.test_fraction, cfg.seed)

    def make(df_split, transform):
        tmp_csv = Path(dcfg.labels_csv).with_name(".tmp_split.csv")
        df_split.to_csv(tmp_csv, index=False)
        ds = DermoscopyDataset(tmp_csv, dcfg.root, dcfg.image_col, list(dcfg.label_cols), transform)
        tmp_csv.unlink()
        return ds

    train_ds = make(train_df, train_transforms(dcfg.image_size))
    val_ds = make(val_df, eval_transforms(dcfg.image_size))
    test_ds = make(test_df, eval_transforms(dcfg.image_size))
    return train_ds, val_ds, test_ds, list(dcfg.label_cols)


def run(cfg, modality: str):
    torch.manual_seed(cfg.seed)
    device = get_device()
    task = "multilabel" if modality == "cxr" else "multiclass"

    train_ds, val_ds, test_ds, label_names = build_datasets(cfg, modality)
    num_classes = len(label_names)

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.data.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    model = build_classifier(
        modality=modality,
        num_classes=num_classes,
        backbone_name=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        freeze_stages=cfg.model.freeze_backbone_stages,
        dropout=cfg.model.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp and device.type == "cuda")

    pos_weight, class_weight = None, None
    if task == "multilabel":
        pos_weight = compute_pos_weight(train_ds, num_classes).to(device)
    else:
        labels = [int(train_ds[i]["label"]) for i in range(len(train_ds))]
        class_weight = compute_class_weights(labels, num_classes).to(device)

    checkpoint_dir = Path(cfg.train.checkpoint_dir)
    ckpt_path = checkpoint_dir / f"{modality}_latest.pt"
    best_ckpt_path = checkpoint_dir / f"{modality}_best.pt"

    start_epoch, global_step, best_metric = 0, 0, float("-inf")
    if cfg.train.resume:
        existing = find_latest_checkpoint(checkpoint_dir, modality)
        if existing is not None:
            state = load_checkpoint(existing, model, optimizer, scaler, map_location=device)
            start_epoch = state["epoch"]
            global_step = state["global_step"]
            best_metric = state["best_metric"]
            print(f"Resumed {modality} from {existing} at epoch {start_epoch}, step {global_step}")

    logger = build_logger(cfg, run_name=f"{modality}-{cfg.model.backbone}")
    steps_since_ckpt = 0

    lr_schedule_fn = None
    if getattr(cfg.train, "scheduler", "none") == "cosine":
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * cfg.train.epochs
        warmup_steps = steps_per_epoch * cfg.train.warmup_epochs
        lr_schedule_fn = lambda step: cosine_lr(  # noqa: E731
            step, total_steps, warmup_steps, cfg.train.lr, cfg.train.min_lr
        )

    for epoch in range(start_epoch, cfg.train.epochs):
        def log_fn(d):
            nonlocal steps_since_ckpt
            logger.log(d)
            steps_since_ckpt += 1
            if steps_since_ckpt >= cfg.train.checkpoint_every_steps:
                save_checkpoint(ckpt_path, model, optimizer, scaler, epoch, d["train/step"], best_metric)
                steps_since_ckpt = 0

        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, device, task,
            scaler=scaler if scaler.is_enabled() else None,
            pos_weight=pos_weight, class_weight=class_weight,
            grad_clip_norm=cfg.train.grad_clip_norm, global_step=global_step, log_fn=log_fn,
            lr_schedule_fn=lr_schedule_fn,
            focal_gamma=getattr(cfg.train, "focal_gamma", 0.0) if task == "multilabel" else 0.0,
        )

        val_metrics, _, _ = evaluate(model, val_loader, device, task, label_names)
        logger.log({"epoch": epoch, "train/epoch_loss": train_loss, **{f"val/{k}": v for k, v in val_metrics.items()}})

        primary_metric = val_metrics.get("auc/mean", val_metrics.get("auc/macro_ovr", float("nan")))
        print(f"[{modality}] epoch {epoch}: train_loss={train_loss:.4f} val={val_metrics}")

        save_checkpoint(ckpt_path, model, optimizer, scaler, epoch + 1, global_step, best_metric)
        if primary_metric == primary_metric and primary_metric > best_metric:  # NaN-safe comparison
            best_metric = primary_metric
            save_checkpoint(best_ckpt_path, model, optimizer, scaler, epoch + 1, global_step, best_metric)
            print(f"[{modality}] new best {primary_metric:.4f} -> saved {best_ckpt_path}")

    test_metrics, test_logits, test_labels = evaluate(model, test_loader, device, task, label_names)
    print(f"[{modality}] final test metrics: {test_metrics}")
    logger.log({f"test/{k}": v for k, v in test_metrics.items()})
    logger.finish()

    return {
        "model": model,
        "test_metrics": test_metrics,
        "test_logits": test_logits,
        "test_labels": test_labels,
        "label_names": label_names,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modality", choices=["cxr", "derm"], required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="key.subkey=value overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    run(cfg, args.modality)


if __name__ == "__main__":
    main()
