"""Turn a raw ISIC metadata export into the labels.csv format the dermoscopy
dataloader expects: image_path, melanoma, nevus, other (one-hot).

ISIC metadata's diagnosis field has many fine-grained values across
collection years; this collapses them into the 3-way scheme the project plan
uses (melanoma / benign nevus / other lesion). Adjust DIAGNOSIS_MAP if your
export uses different category names -- check with:
    python -c "import pandas as pd; print(pd.read_csv('metadata.csv')['diagnosis'].value_counts())"

Usage:
    python -m data.prepare_isic --root data_raw/isic
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DIAGNOSIS_MAP = {
    "melanoma": "melanoma",
    "nevus": "nevus",
    "melanocytic nevus": "nevus",
}
DEFAULT_CLASS = "other"


def classify(diagnosis: str) -> str:
    if not isinstance(diagnosis, str):
        return DEFAULT_CLASS
    return DIAGNOSIS_MAP.get(diagnosis.strip().lower(), DEFAULT_CLASS)


def build_labels_csv(root: Path) -> Path:
    meta_path = root / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Expected {meta_path} (from `isic metadata download`)")

    meta = pd.read_csv(meta_path)
    id_col = "isic_id" if "isic_id" in meta.columns else meta.columns[0]
    diag_col = "diagnosis" if "diagnosis" in meta.columns else None
    if diag_col is None:
        raise ValueError(f"No 'diagnosis' column in {meta_path}; inspect the export and update this script.")

    images_dir = root / "images"
    rows = []
    for _, row in meta.iterrows():
        image_id = row[id_col]
        # isic-cli typically saves as <isic_id>.jpg
        candidates = [images_dir / f"{image_id}.jpg", images_dir / f"{image_id}.png"]
        image_path = next((c for c in candidates if c.exists()), None)
        if image_path is None:
            continue

        label = classify(row[diag_col])
        rows.append(
            {
                "image_path": str(image_path.relative_to(root)),
                "melanoma": int(label == "melanoma"),
                "nevus": int(label == "nevus"),
                "other": int(label == "other"),
                "label": label,
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = root / "labels.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")
    print(out_df["label"].value_counts())
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    build_labels_csv(args.root)


if __name__ == "__main__":
    main()
