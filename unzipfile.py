#!/usr/bin/env python3
"""Extract a local dataset zip into the ignored data directory."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a zip archive.")
    parser.add_argument("--zip", type=Path, default=Path("data.zip"), help="Zip file path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data"),
        help="Extraction directory. This repo ignores data/ by default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zip_path = args.zip.expanduser()
    output_dir = args.output.expanduser()

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    print(f"Extracted {zip_path} to {output_dir}")


if __name__ == "__main__":
    main()
