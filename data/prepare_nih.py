"""Turn the raw NIH ChestX-ray14 Kaggle download into the labels.csv format
MultiLabelImageDataset expects: one image_path column + one 0/1 column per
finding.

Handles both Kaggle mirrors, since they lay out identically-schemaed CSVs
differently:
  - `nih-chest-xrays/data` (full, ~42GB): `Data_Entry_2017.csv` at the root,
    images spread across `images_001` .. `images_012` subfolders (each with
    its own `images/` dir).
  - `nih-chest-xrays/sample` (5,606-image official sample, ~4GB -- the
    practical choice on a laptop/free-tier budget): `sample_labels.csv` at
    the root, images under a single `sample/images/` dir.

Both ship the same `Image Index` / pipe-separated `Finding Labels` columns,
so this script just needs to locate the CSV and the image dir(s) and
flattens either into one flat labels.csv.

Usage:
    python -m data.prepare_nih --root data_raw/nih_cxr
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ALL_NIH_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural_Thickening", "Hernia", "No Finding",
]

# Map to the naming used elsewhere in this repo (CheXpert calls it "Pleural Effusion").
RENAME = {"Effusion": "Pleural Effusion"}


def _find_labels_csv(root: Path) -> Path:
    for name in ("Data_Entry_2017.csv", "sample_labels.csv"):
        candidates = list(root.glob(f"{name}")) or list(root.rglob(name))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(
        f"Could not find Data_Entry_2017.csv (full dataset) or sample_labels.csv "
        f"(sample dataset) under {root}"
    )


def build_labels_csv(root: Path) -> Path:
    entry_csv = _find_labels_csv(root)
    df = pd.read_csv(entry_csv)

    # Build an image_index -> relative_path lookup across every images/ dir
    # this layout might use: images_NNN/images/ (full dataset), sample/images/
    # (sample dataset), or a flat images/ (fallback).
    image_dirs = sorted(root.glob("images_*/images")) + sorted(root.glob("sample/images"))
    if not image_dirs:
        image_dirs = [root / "images"] if (root / "images").exists() else []
    index_to_path: dict[str, str] = {}
    for d in image_dirs:
        for f in d.glob("*.png"):
            index_to_path[f.name] = str(f.relative_to(root))

    rows = []
    for _, row in df.iterrows():
        image_name = row["Image Index"]
        rel_path = index_to_path.get(image_name)
        if rel_path is None:
            continue  # image not actually downloaded (partial download)
        labels = set(row["Finding Labels"].split("|"))
        out_row = {"image_path": rel_path}
        for label in ALL_NIH_LABELS:
            out_row[RENAME.get(label, label)] = int(label in labels)
        rows.append(out_row)

    out_df = pd.DataFrame(rows)
    out_path = root / "labels.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    build_labels_csv(args.root)


if __name__ == "__main__":
    main()
