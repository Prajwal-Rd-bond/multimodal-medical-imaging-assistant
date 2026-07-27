"""Merge the per-class ISIC download folders (data_raw/isic_raw/<bucket>/*.jpg)
produced by targeted `isic image download -s '<query>'` pulls into the flat
layout `data/datasets.py` expects: data_raw/isic/images/*.jpg + labels.csv.

Bypasses prepare_isic.py's diagnosis-column guessing entirely -- since each
source folder was already downloaded via a diagnosis-specific search query,
the class is known up front by construction, which is more reliable than
re-parsing ISIC's multi-level diagnosis_1..5 taxonomy after the fact.

Usage:
    python -m data.merge_isic_raw --raw-root data_raw/isic_raw --out-root data_raw/isic
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

BUCKET_TO_LABEL = {
    "melanoma": "melanoma",
    "nevus": "nevus",
    "other_bcc": "other",
    "other_sk": "other",
    "other_df": "other",
}


def merge(raw_root: Path, out_root: Path) -> Path:
    images_dir = out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for bucket, label in BUCKET_TO_LABEL.items():
        bucket_dir = raw_root / bucket
        if not bucket_dir.exists():
            print(f"skip {bucket}: {bucket_dir} not found")
            continue
        image_files = sorted(list(bucket_dir.glob("*.jpg")) + list(bucket_dir.glob("*.png")))
        for f in image_files:
            dest = images_dir / f.name
            if not dest.exists():
                shutil.copy2(f, dest)
            rows.append(
                {
                    "image_path": f"images/{f.name}",
                    "melanoma": int(label == "melanoma"),
                    "nevus": int(label == "nevus"),
                    "other": int(label == "other"),
                    "label": label,
                }
            )
        print(f"{bucket}: {len(image_files)} images -> label={label}")

    df = pd.DataFrame(rows).drop_duplicates(subset="image_path")
    out_csv = out_root / "labels.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {len(df)} rows to {out_csv}")
    print(df["label"].value_counts())
    return out_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data_raw/isic_raw"))
    parser.add_argument("--out-root", type=Path, default=Path("data_raw/isic"))
    args = parser.parse_args()
    merge(args.raw_root, args.out_root)


if __name__ == "__main__":
    main()
