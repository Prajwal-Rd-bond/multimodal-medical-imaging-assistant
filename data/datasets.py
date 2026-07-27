"""Dataset classes for the two modalities.

Both NIH ChestX-ray14 and CheXpert distribute a CSV of image paths + one
column per finding, so a single `MultiLabelImageDataset` covers both --
point it at whichever CSV is currently available. ISIC ships one label per
image (single-label), so it gets its own thin subclass.

None of this downloads data for you -- see `data/download.py` for that.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class MultiLabelImageDataset(Dataset):
    """Generic CSV-backed multi-label image dataset (NIH14 / CheXpert).

    Expects a CSV with an image-path column and one 0/1 column per label.
    CheXpert's uncertainty labels (u/-1) are mapped via `uncertainty_policy`:
      - "zeros": treat uncertain as negative (0)
      - "ones": treat uncertain as positive (1)
      - "ignore": drop rows containing any uncertain label for that column mask
                  (implemented as a per-sample loss mask, not row drop, so
                  batches stay a fixed size)
    """

    def __init__(
        self,
        csv_path: str | Path,
        image_root: str | Path,
        image_col: str,
        label_cols: list[str],
        transform=None,
        uncertainty_policy: str = "zeros",
    ):
        self.df = pd.read_csv(csv_path)
        self.image_root = Path(image_root)
        self.image_col = image_col
        self.label_cols = label_cols
        self.transform = transform
        self.uncertainty_policy = uncertainty_policy

        missing = [c for c in label_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Label columns not found in {csv_path}: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_label(self, raw) -> tuple[float, float]:
        """Return (value, loss_mask) for one label cell."""
        if pd.isna(raw):
            return 0.0, 0.0  # missing label entirely -> excluded from loss
        val = float(raw)
        if val == -1:  # CheXpert uncertain marker
            if self.uncertainty_policy == "zeros":
                return 0.0, 1.0
            if self.uncertainty_policy == "ones":
                return 1.0, 1.0
            if self.uncertainty_policy == "ignore":
                return 0.0, 0.0
            raise ValueError(f"Unknown uncertainty_policy {self.uncertainty_policy}")
        return max(val, 0.0), 1.0

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.image_root / row[self.image_col]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        labels = np.zeros(len(self.label_cols), dtype=np.float32)
        mask = np.zeros(len(self.label_cols), dtype=np.float32)
        for i, col in enumerate(self.label_cols):
            labels[i], mask[i] = self._resolve_label(row[col])

        return {
            "image": image,
            "labels": torch.from_numpy(labels),
            "loss_mask": torch.from_numpy(mask),
            "path": str(img_path),
        }


class DermoscopyDataset(Dataset):
    """Single-label dermoscopy dataset (ISIC-style CSV: image_path,label)."""

    def __init__(
        self,
        csv_path: str | Path,
        image_root: str | Path,
        image_col: str,
        label_cols: list[str],
        transform=None,
        label_col: str = "label",
    ):
        self.df = pd.read_csv(csv_path)
        self.image_root = Path(image_root)
        self.image_col = image_col
        self.label_cols = label_cols
        self.label_col = label_col
        self.transform = transform

        if label_col not in self.df.columns:
            # Fall back to one-hot style columns like the CXR dataset.
            self._onehot = True
            missing = [c for c in label_cols if c not in self.df.columns]
            if missing:
                raise ValueError(f"Neither '{label_col}' nor {missing} found in {csv_path}")
        else:
            self._onehot = False
            self.class_to_idx = {c: i for i, c in enumerate(label_cols)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.image_root / row[self.image_col]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        if self._onehot:
            label_idx = int(np.argmax([row[c] for c in self.label_cols]))
        else:
            label_idx = self.class_to_idx[row[self.label_col]]

        return {
            "image": image,
            "label": torch.tensor(label_idx, dtype=torch.long),
            "path": str(img_path),
        }


def split_dataframe(df: pd.DataFrame, val_fraction: float, test_fraction: float, seed: int = 42):
    """Deterministic random split; for real runs prefer a patient-level split
    (NIH14/CheXpert provide patient IDs) to avoid leakage across sets."""
    n = len(df)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
