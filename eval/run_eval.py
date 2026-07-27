"""Produces the eval artifacts the plan's deliverables checklist asks for:
AUC table, ECE plots (before/after calibration), OOD AUROC/FPR95, and a
failure-case gallery of the highest-confidence wrong predictions.

    python -m eval.run_eval --modality cxr --ood-dataset path/to/ood_images/

Writes everything under `eval_out/<modality>/`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from calibrate.temperature_scaling import compute_ece, multilabel_ece_per_class, plot_reliability_diagram
from configs.config import load_config
from data.transforms import eval_transforms
from models.backbone import build_classifier
from ood.detectors import MahalanobisDetector, entropy_score, evaluate_ood
from train.device import get_device
from train.train import build_datasets


def load_model(cfg, modality: str, device):
    from data.datasets import DermoscopyDataset, MultiLabelImageDataset  # noqa: F401  (kept for reference)

    dcfg = cfg.data.cxr if modality == "cxr" else cfg.data.derm
    label_names = list(dcfg.label_cols)
    model = build_classifier(
        modality=modality, num_classes=len(label_names), backbone_name=cfg.model.backbone,
        pretrained=False, freeze_stages=0, dropout=cfg.model.dropout,
    ).to(device)
    ckpt_path = Path(cfg.train.checkpoint_dir) / f"{modality}_best.pt"
    # weights_only=False: this repo's own checkpoint, not a third-party file.
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, label_names


def collect_test_outputs(model, loader, task, device):
    all_logits, all_labels, all_masks, all_paths = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device)).cpu()
            all_logits.append(logits)
            all_paths.extend(batch["path"])
            if task == "multilabel":
                all_labels.append(batch["labels"])
                all_masks.append(batch["loss_mask"])
            else:
                all_labels.append(batch["label"])
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    masks = torch.cat(all_masks) if task == "multilabel" else None
    return logits, labels, masks, all_paths


def run(cfg, modality: str, ood_dir: str | None, out_dir: Path):
    device = get_device()
    task = "multilabel" if modality == "cxr" else "multiclass"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, label_names = load_model(cfg, modality, device)
    train_ds, val_ds, test_ds, _ = build_datasets(cfg, modality)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    logits, labels, masks, paths = collect_test_outputs(model, test_loader, task, device)

    # --- AUC table ---
    from sklearn.metrics import roc_auc_score

    lines = ["| Label | AUC |", "|---|---|"]
    if task == "multilabel":
        probs = torch.sigmoid(logits).numpy()
        labels_np, masks_np = labels.numpy(), masks.numpy()
        for i, name in enumerate(label_names):
            valid = masks_np[:, i] > 0
            y = labels_np[valid, i]
            if len(np.unique(y)) < 2:
                continue
            auc = roc_auc_score(y, probs[valid, i])
            lines.append(f"| {name} | {auc:.3f} |")
    else:
        probs = torch.softmax(logits, dim=1).numpy()
        labels_np = labels.numpy()
        for i, name in enumerate(label_names):
            y = (labels_np == i).astype(int)
            if len(np.unique(y)) < 2:
                continue
            auc = roc_auc_score(y, probs[:, i])
            lines.append(f"| {name} | {auc:.3f} |")
    (out_dir / "auc_table.md").write_text("\n".join(lines))
    print(f"Wrote {out_dir / 'auc_table.md'}")

    # --- ECE + reliability diagrams ---
    if task == "multilabel":
        ece_results = multilabel_ece_per_class(probs, labels_np, masks_np, n_bins=cfg.calibrate.n_bins)
        for i, name in enumerate(label_names):
            plot_reliability_diagram(ece_results[i], name, str(out_dir / f"reliability_{name.replace(' ', '_')}.png"))
    else:
        preds = probs.argmax(axis=1)
        correct = (preds == labels_np).astype(float)
        ece_result = compute_ece(probs.max(axis=1), correct, n_bins=cfg.calibrate.n_bins)
        plot_reliability_diagram(ece_result, "overall", str(out_dir / "reliability_overall.png"))
    print(f"Wrote reliability diagrams to {out_dir}")

    # --- OOD: entropy baseline vs. Mahalanobis-on-features upgrade ---
    # AUROC/FPR95 are reported against the held-out test set (as they should
    # be -- it's the honest final number). The *deployed threshold*, though,
    # is tuned on the validation set instead: fitting it against the same
    # test set used for the headline metric would leak test data into an
    # operational decision, even though it doesn't affect the (threshold-free)
    # AUROC number itself.
    import json

    id_scores_entropy = entropy_score(probs, multilabel=(task == "multilabel"))

    val_logits, _, val_masks, _ = collect_test_outputs(model, val_loader, task, device)
    val_probs = torch.sigmoid(val_logits).numpy() if task == "multilabel" else torch.softmax(val_logits, dim=1).numpy()
    val_id_scores_entropy = entropy_score(val_probs, multilabel=(task == "multilabel"))

    # Mahalanobis needs penultimate features (not logits) for train/val/test,
    # fit against a per-class Gaussian estimated from the training set. For
    # multi-label CXR there's no single ground-truth class per image to group
    # by, so MahalanobisDetector.extract_features falls back to each sample's
    # argmax label as a pseudo-class -- an approximation, not a limitation of
    # the method itself.
    train_features, train_pseudo_labels = MahalanobisDetector.extract_features(model, train_loader, device)
    mahalanobis = MahalanobisDetector.fit(train_features, train_pseudo_labels, num_classes=len(label_names))
    np.savez(
        Path(cfg.train.checkpoint_dir) / f"{modality}_mahalanobis.npz",
        means=mahalanobis.means, precision=mahalanobis.precision,
    )

    val_features, _ = MahalanobisDetector.extract_features(model, val_loader, device)
    test_features, _ = MahalanobisDetector.extract_features(model, test_loader, device)
    id_scores_maha = mahalanobis.score(test_features)
    val_id_scores_maha = mahalanobis.score(val_features)

    thresholds = {
        "entropy": float(np.percentile(val_id_scores_entropy, (1 - cfg.ood.fpr_target_tpr) * 100)),
        "mahalanobis": float(np.percentile(val_id_scores_maha, (1 - cfg.ood.fpr_target_tpr) * 100)),
    }
    threshold_path = Path(cfg.train.checkpoint_dir) / f"{modality}_ood_threshold.json"
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_path.write_text(json.dumps({
        "thresholds": thresholds, "active_method": cfg.ood.method, "fpr_target_tpr": cfg.ood.fpr_target_tpr,
    }))
    print(f"Wrote OOD thresholds (fit on val) to {threshold_path}: {thresholds}")

    if ood_dir:
        from PIL import Image

        dcfg = cfg.data.cxr if modality == "cxr" else cfg.data.derm
        transform = eval_transforms(dcfg.image_size)
        ood_images = list(Path(ood_dir).glob("*"))
        ood_logits, ood_feats = [], []
        with torch.no_grad():
            for p in ood_images:
                try:
                    img = Image.open(p)
                except Exception:
                    continue
                t = transform(img).unsqueeze(0).to(device)
                ood_feats.append(model.forward_features(t).cpu())
                ood_logits.append(model(t).cpu())
        if ood_logits:
            ood_logits = torch.cat(ood_logits)
            ood_feats_np = torch.cat(ood_feats).numpy()
            ood_probs = torch.sigmoid(ood_logits).numpy() if task == "multilabel" else torch.softmax(ood_logits, dim=1).numpy()
            ood_scores_entropy = entropy_score(ood_probs, multilabel=(task == "multilabel"))
            ood_scores_maha = mahalanobis.score(ood_feats_np)

            metrics_entropy = evaluate_ood(id_scores_entropy, ood_scores_entropy, target_tpr=cfg.ood.fpr_target_tpr)
            metrics_maha = evaluate_ood(id_scores_maha, ood_scores_maha, target_tpr=cfg.ood.fpr_target_tpr)
            fpr_key = f"fpr_at_{int(cfg.ood.fpr_target_tpr*100)}tpr"
            (out_dir / "ood_metrics.md").write_text(
                "| Method | AUROC | FPR@{0}TPR |\n|---|---|---|\n"
                "| entropy | {1:.3f} | {2:.3f} |\n"
                "| mahalanobis | {3:.3f} | {4:.3f} |\n".format(
                    int(cfg.ood.fpr_target_tpr * 100),
                    metrics_entropy["auroc"], metrics_entropy[fpr_key],
                    metrics_maha["auroc"], metrics_maha[fpr_key],
                )
            )
            print(f"Wrote {out_dir / 'ood_metrics.md'}: entropy={metrics_entropy}, mahalanobis={metrics_maha}")
    else:
        print("No --ood-dataset given; skipping OOD AUROC/FPR95 (per plan: feed CIFAR/random web images "
              "or the other modality's images through this model to compute it).")

    # --- Failure gallery: highest-confidence wrong predictions ---
    gallery = []
    if task == "multilabel":
        preds_bin = (probs >= 0.5).astype(int)
        wrong_mask = (preds_bin != labels_np) & (masks_np > 0)
        confidence = np.abs(probs - 0.5)
        for i in range(len(paths)):
            wrong_labels = [label_names[c] for c in range(len(label_names)) if wrong_mask[i, c]]
            if wrong_labels:
                gallery.append((paths[i], wrong_labels, float(confidence[i].max())))
    else:
        preds = probs.argmax(axis=1)
        confidence = probs.max(axis=1)
        for i in range(len(paths)):
            if preds[i] != labels_np[i]:
                gallery.append((paths[i], [f"predicted {label_names[preds[i]]}, true {label_names[labels_np[i]]}"], float(confidence[i])))

    gallery.sort(key=lambda x: -x[2])
    top_gallery = gallery[:10]
    gallery_lines = ["# Failure case gallery (top 10 highest-confidence misses)", ""]
    for path, wrong_labels, conf in top_gallery:
        gallery_lines.append(f"- `{path}` (confidence {conf:.2f}): {', '.join(wrong_labels)}")
    (out_dir / "failure_gallery.md").write_text("\n".join(gallery_lines))
    print(f"Wrote {out_dir / 'failure_gallery.md'} ({len(top_gallery)} cases)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=["cxr", "derm"], required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--ood-dataset", type=str, default=None, help="Directory of OOD images (CIFAR samples, random web images, or the other modality)")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    out_dir = Path(args.out or f"eval_out/{args.modality}")
    run(cfg, args.modality, args.ood_dataset, out_dir)


if __name__ == "__main__":
    main()
