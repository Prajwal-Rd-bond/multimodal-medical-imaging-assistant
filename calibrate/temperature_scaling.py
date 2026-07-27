"""Temperature scaling calibration + ECE, for both multi-label (CXR, one
temperature per class via a vector) and multi-class (dermoscopy, single
scalar temperature) heads.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    """Learns one temperature per output logit (a vector of size num_classes
    for multi-label; effectively one shared scalar works fine for
    multi-class too since softmax is scale-invariant per-sample only in
    relative logit differences -- a single scalar is standard for that case,
    so `per_class=False` collapses to one parameter)."""

    def __init__(self, num_classes: int, per_class: bool = True):
        super().__init__()
        n = num_classes if per_class else 1
        self.log_temperature = nn.Parameter(torch.zeros(n))
        self.per_class = per_class

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    task: str,
    per_class: bool = True,
    max_iter: int = 50,
    lr: float = 0.01,
    loss_mask: torch.Tensor | None = None,
) -> TemperatureScaler:
    """Fit temperature on held-out validation logits by minimizing NLL, per
    Guo et al. 2017. `task` is "multilabel" or "multiclass"."""
    num_classes = logits.shape[1]
    scaler = TemperatureScaler(num_classes, per_class=per_class)
    optimizer = torch.optim.LBFGS(scaler.parameters(), lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        scaled = scaler(logits)
        if task == "multilabel":
            loss_unreduced = F.binary_cross_entropy_with_logits(scaled, labels, reduction="none")
            if loss_mask is not None:
                loss = (loss_unreduced * loss_mask).sum() / loss_mask.sum().clamp(min=1)
            else:
                loss = loss_unreduced.mean()
        else:
            loss = F.cross_entropy(scaled, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler


@dataclass
class ECEResult:
    ece: float
    bin_confidences: np.ndarray
    bin_accuracies: np.ndarray
    bin_counts: np.ndarray


def compute_ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> ECEResult:
    """Standard binned ECE: |mean_confidence - mean_accuracy| weighted by bin size.

    `confidences`: predicted probability of the predicted (or positive) class, shape (N,)
    `correct`: 1 if the prediction was correct (or label==1 for a binary head), shape (N,)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_confidences = np.zeros(n_bins)
    bin_accuracies = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        count = in_bin.sum()
        bin_counts[i] = count
        if count > 0:
            bin_confidences[i] = confidences[in_bin].mean()
            bin_accuracies[i] = correct[in_bin].mean()

    total = bin_counts.sum()
    ece = float(np.sum(bin_counts / total * np.abs(bin_confidences - bin_accuracies))) if total > 0 else float("nan")
    return ECEResult(ece=ece, bin_confidences=bin_confidences, bin_accuracies=bin_accuracies, bin_counts=bin_counts)


def multilabel_ece_per_class(probs: np.ndarray, labels: np.ndarray, loss_mask: np.ndarray, n_bins: int = 15) -> dict[int, ECEResult]:
    results = {}
    for c in range(probs.shape[1]):
        valid = loss_mask[:, c] > 0
        results[c] = compute_ece(probs[valid, c], labels[valid, c], n_bins=n_bins)
    return results


def plot_reliability_diagram(ece_result: ECEResult, title: str, out_path: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    bin_edges = np.linspace(0, 1, len(ece_result.bin_accuracies) + 1)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.bar(centers, ece_result.bin_accuracies, width=1 / len(centers), edgecolor="black", alpha=0.7, label="accuracy")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title} (ECE={ece_result.ece:.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
