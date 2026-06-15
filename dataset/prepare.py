"""
Dataset preparation utility.

Expected input structure:
    raw/
    ├── sfw/    (safe-for-work images)
    └── nsfw/   (explicit content images)

Output structure:
    processed/
    ├── train/
    │   ├── sfw/
    │   └── nsfw/
    └── val/
        ├── sfw/
        └── nsfw/
"""

import argparse
import os
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split


def collect_images(directory: str) -> list[str]:
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return [
        str(p) for p in Path(directory).rglob("*")
        if p.suffix.lower() in extensions
    ]


def split_and_copy(files: list[str], label: str, output_dir: str, val_ratio: float):
    train_files, val_files = train_test_split(files, test_size=val_ratio, random_state=42)

    for split, split_files in [("train", train_files), ("val", val_files)]:
        dest = os.path.join(output_dir, split, label)
        os.makedirs(dest, exist_ok=True)
        for f in split_files:
            shutil.copy2(f, dest)

    return len(train_files), len(val_files)


def main():
    parser = argparse.ArgumentParser(description="Prepare SIRAN dataset")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to raw/ directory")
    parser.add_argument("--output_dir", type=str, default="./data/processed")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    args = parser.parse_args()

    for label in ["sfw", "nsfw"]:
        source = os.path.join(args.input_dir, label)
        if not os.path.isdir(source):
            raise FileNotFoundError(f"Expected directory: {source}")

        files = collect_images(source)
        train_count, val_count = split_and_copy(files, label, args.output_dir, args.val_ratio)
        print(f"[{label.upper()}] train: {train_count}, val: {val_count}")

    print(f"\nDataset prepared at: {args.output_dir}")


if __name__ == "__main__":
    main()
