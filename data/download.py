"""Dataset acquisition helpers.

These are thin wrappers, not magic -- each dataset has different access
terms (see project plan section 2), so this module automates the mechanical
parts and prints the manual steps you still have to do yourself.

Usage:
    python -m data.download nih --dest data_raw/nih_cxr
    python -m data.download isic --dest data_raw/isic
    python -m data.download chexpert-instructions
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def download_nih(dest: Path) -> None:
    """NIH ChestX-ray14 via Kaggle (nih-chest-xrays/data), no access request needed.

    Requires `pip install kaggle` and a Kaggle API token at ~/.kaggle/kaggle.json.
    """
    dest.mkdir(parents=True, exist_ok=True)
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("Install the Kaggle CLI first: pip install kaggle", file=sys.stderr)
        print("Then place your API token at ~/.kaggle/kaggle.json (from kaggle.com/settings).", file=sys.stderr)
        sys.exit(1)

    import subprocess

    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "nih-chest-xrays/data", "-p", str(dest), "--unzip"],
        check=True,
    )
    print(f"Downloaded NIH ChestX-ray14 to {dest}.")
    print("Next: run `python -m data.prepare_nih --root {dest}` to build labels.csv "
          "in the {image_path, label columns...} format the dataloader expects.")


ISIC_SEARCH_QUERIES = {
    "melanoma": ('diagnosis_2:"Malignant melanocytic proliferations (Melanoma)"', 1200),
    "nevus": ('diagnosis_3:"Nevus"', 1200),
    "other_bcc": ('diagnosis_3:"Basal cell carcinoma"', 500),
    "other_sk": ('diagnosis_3:"Seborrheic keratosis"', 500),
    "other_df": ('diagnosis_3:"Dermatofibroma"', 200),
}


def download_isic(dest: Path, limit: int | None = None) -> None:
    """ISIC Archive via the public isic-cli / API v2, no login required.

    A plain unfiltered `isic image download --limit N` pulls images in
    dataset order, which front-loads a huge nevus-heavy historical
    collection -- at N=8000 that's >97% nevus and essentially zero
    non-melanoma/non-nevus lesions. So instead this runs one search-filtered
    download per target class (`ISIC_SEARCH_QUERIES`) and merges them via
    `data.merge_isic_raw`, which gives a usable (if still imbalanced --
    melanoma is a minority class in reality too) 3-way split.

    `limit` overrides every per-class limit above uniformly if passed;
    otherwise each class uses its own tuned default.
    """
    raw_root = dest.parent / f"{dest.name}_raw" if dest.name != "isic_raw" else dest
    raw_root.mkdir(parents=True, exist_ok=True)
    try:
        import subprocess

        for bucket, (query, default_limit) in ISIC_SEARCH_QUERIES.items():
            bucket_dir = raw_root / bucket
            n = limit or default_limit
            print(f"Downloading {bucket} (limit={n}, query={query})...")
            subprocess.run(
                ["isic", "image", "download", str(bucket_dir), "-s", query, "--limit", str(n)],
                check=True,
            )
    except FileNotFoundError:
        print("Install the ISIC CLI first: pip install isic-cli", file=sys.stderr)
        sys.exit(1)

    from data.merge_isic_raw import merge

    merge(raw_root, dest)
    print(f"Downloaded and merged ISIC images to {dest} (labels.csv already built).")


def chexpert_instructions() -> None:
    print(
        """
CheXpert is gated behind a free request form (Stanford AIMI), not an API:

  1. Go to https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2
  2. Submit the access request form (usually approved within hours-to-days).
  3. Download the "small" (~11GB downsampled) version once approved.
  4. Unzip into data_raw/chexpert/ -- it ships train.csv/valid.csv already in
     the {Path, Sex, Age, Frontal/Lateral, ..., pathology columns with u/0/1/-1}
     format that MultiLabelImageDataset expects directly (uncertainty_policy
     handles the -1 values).

Do this on day 1: it's the only lead-time risk in the whole plan. Build and
validate the pipeline against NIH ChestX-ray14 in the meantime.
""".strip()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_nih = sub.add_parser("nih", help="Download NIH ChestX-ray14 via Kaggle")
    p_nih.add_argument("--dest", type=Path, default=Path("data_raw/nih_cxr"))

    p_isic = sub.add_parser("isic", help="Download a class-stratified ISIC sample (melanoma/nevus/other) + build labels.csv")
    p_isic.add_argument("--dest", type=Path, default=Path("data_raw/isic"))
    p_isic.add_argument("--limit", type=int, default=None, help="Override every per-class limit uniformly (default: tuned per class, see ISIC_SEARCH_QUERIES)")

    sub.add_parser("chexpert-instructions", help="Print CheXpert access steps")

    args = parser.parse_args()
    if args.cmd == "nih":
        download_nih(args.dest)
    elif args.cmd == "isic":
        download_isic(args.dest, args.limit)
    elif args.cmd == "chexpert-instructions":
        chexpert_instructions()


if __name__ == "__main__":
    main()
