"""Download Zensus grid data (or any large census ZIP) from a provided URL.

Hard truth
----------
Zensus grid datasets can be VERY large. If you download/extract them blindly, you will burn disk.
Therefore:
- You must provide the direct URL yourself (the official portals change URLs).
- Extraction is OFF by default.

This script is a *safe downloader* with guardrails.

Usage
-----
# Download only
python raw_data/code_external_data/download_09_zensus_grid_download_stub.py --url "<DIRECT_DOWNLOAD_URL>"

# Download + extract (DANGEROUS: disk usage)
python raw_data/code_external_data/download_09_zensus_grid_download_stub.py --url "<DIRECT_DOWNLOAD_URL>" --extract --max-extract-gb 10

Dependencies: requests
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path
from typing import Optional

import requests

from abs_path_utils import get_base_dir, get_output_dir, require_absolute_path




def _download(url: str, out_path: Path, max_gb: Optional[float] = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        length = r.headers.get("Content-Length")
        if length and max_gb is not None:
            gb = int(length) / (1024**3)
            if gb > max_gb:
                raise SystemExit(f"Refusing to download because Content-Length={gb:.2f}GB > --max-gb={max_gb}")

        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    tmp.replace(out_path)


def _dir_size_gb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024**3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=None, help="ABSOLUTE base directory for scripts + outputs")
    ap.add_argument("--url", required=True, help="Direct download URL")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: <base-dir>/_external_data/zensus)")
    ap.add_argument("--max-gb", type=float, default=None, help="Abort download if Content-Length exceeds this size")
    ap.add_argument("--extract", action="store_true", help="Extract ZIP after download (OFF by default)")
    ap.add_argument("--max-extract-gb", type=float, default=10.0, help="Abort if extracted size exceeds this (default: 10GB)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing file")
    args = ap.parse_args()

    base_dir = get_base_dir(args.base_dir)

    out_dir = require_absolute_path(args.out_dir, "--out-dir") if args.out_dir else get_output_dir(base_dir, "zensus")
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = args.url.split("/")[-1].split("?", 1)[0]
    if not filename:
        filename = "zensus_download.bin"

    out_path = out_dir / filename

    if out_path.exists() and not args.force:
        print(f"File exists, skipping: {out_path}")
    else:
        print(f"Downloading {args.url} -> {out_path}")
        _download(args.url, out_path, max_gb=args.max_gb)
        print("Download completed.")

    if args.extract:
        if out_path.suffix.lower() != ".zip":
            print("--extract set, but file is not a .zip. Skipping extraction.")
            return 0

        extract_dir = out_dir / out_path.stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting -> {extract_dir}")

        with zipfile.ZipFile(out_path, "r") as z:
            z.extractall(extract_dir)

        size_gb = _dir_size_gb(extract_dir)
        print(f"Extracted size: {size_gb:.2f}GB")
        if size_gb > args.max_extract_gb:
            raise SystemExit(
                f"Extraction exceeded max size ({size_gb:.2f}GB > {args.max_extract_gb}GB). "
                "Delete the extracted folder if you don't have the disk."
            )

    print("Done.")
    print("Storage warning:")
    print("- Zensus grid data can be huge. Prefer clipping/aggregating to store catchments later instead of keeping full grids.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
