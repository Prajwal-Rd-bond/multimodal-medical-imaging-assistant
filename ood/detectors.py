"""Out-of-distribution detection.

Two tiers, per the plan: start with max-softmax-probability / entropy
thresholding (needs nothing but the classifier's own outputs), upgrade to
Mahalanobis distance on penultimate features if time allows (needs a fitted
per-class Gaussian over training features).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def max_softmax_score(probs: np.ndarray) -> np.ndarray:
    """Higher score = more in-distribution. For multi-label heads, pass
    per-class sigmoid outputs collapsed via max (highest-confidence finding);
    for multi-class, pass the softmax vector directly."""
    return probs.max(axis=1)


def entropy_score(probs: np.ndarray, multilabel: bool = False) -> np.ndarray:
    """Negative entropy as an in-distribution score (higher = more confident
    = more in-distribution)."""
    # 1e-12 is below float32 precision near 1.0 -- `1 - 1e-12` rounds right
    # back to exactly 1.0 there, so clipping silently does nothing and an
    # overconfident (saturated) prediction produces log(0). A saturated
    # model on OOD inputs is exactly the case this function needs to handle,
    # so this isn't a corner case -- it's the common one.
    eps = 1e-6
    if multilabel:
        # Independent Bernoulli entropy per class, summed.
        p = np.clip(probs, eps, 1 - eps)
        ent = -(p * np.log(p) + (1 - p) * np.log(1 - p)).sum(axis=1)
    else:
        p = np.clip(probs, eps, 1)
        ent = -(p * np.log(p)).sum(axis=1)
    return -ent


@dataclass
class MahalanobisDetector:
    """Fits one Gaussian (mean + shared covariance) per class over penultimate
    features from the training set, following Lee et al. 2018. Score is the
    negative minimum Mahalanobis distance to any class centroid."""

    means: np.ndarray            # (num_classes, feat_dim)
    precision: np.ndarray        # (feat_dim, feat_dim), inverse of shared covariance

    @classmethod
    def fit(cls, features: np.ndarray, labels: np.ndarray, num_classes: int, eps: float = 1e-6) -> "MahalanobisDetector":
        feat_dim = features.shape[1]
        means = np.zeros((num_classes, feat_dim))
        centered_all = []
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() == 0:
                continue
            means[c] = features[mask].mean(axis=0)
            centered_all.append(features[mask] - means[c])
        centered_all = np.concatenate(centered_all, axis=0)
        cov = np.cov(centered_all, rowvar=False) + eps * np.eye(feat_dim)
        precision = np.linalg.inv(cov)
        return cls(means=means, precision=precision)

    def score(self, features: np.ndarray) -> np.ndarray:
        """Higher = more in-distribution (closer to nearest class centroid)."""
        dists = np.stack(
            [np.einsum("ij,jk,ik->i", features - m, self.precision, features - m) for m in self.means],
            axis=1,
        )
        min_dist = dists.min(axis=1)
        return -min_dist

    @staticmethod
    @torch.no_grad()
    def extract_features(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        feats, labels = [], []
        for batch in loader:
            images = batch["image"].to(device)
            f = model.forward_features(images).cpu().numpy()
            feats.append(f)
            if "label" in batch:
                labels.append(batch["label"].numpy())
            else:
                labels.append(batch["labels"].argmax(dim=1).numpy())  # best-effort for multilabel
        return np.concatenate(feats), np.concatenate(labels)


def evaluate_ood(id_scores: np.ndarray, ood_scores: np.ndarray, target_tpr: float = 0.95) -> dict:
    """AUROC + FPR@target_tpr for separating in-distribution (label 1) from
    out-of-distribution (label 0) inputs, using the score as the ID-ness signal.
    """
    y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    y_score = np.concatenate([id_scores, ood_scores])
    auroc = roc_auc_score(y_true, y_score)

    # Threshold at the target_tpr-th percentile of ID scores, then measure
    # what fraction of OOD scores fall above it (false positives at that operating point).
    threshold = float(np.percentile(id_scores, (1 - target_tpr) * 100))
    fpr = float((ood_scores >= threshold).mean())

    return {"auroc": float(auroc), f"fpr_at_{int(target_tpr*100)}tpr": fpr, "threshold": float(threshold)}


def flag_uncertain(score: float, threshold: float) -> bool:
    return score < threshold
