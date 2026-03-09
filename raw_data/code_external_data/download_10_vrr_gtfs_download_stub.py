"""Download a GTFS ZIP (e.g., VRR schedule data) from a provided URL.

Hard truth
----------
- GTFS feeds are updated frequently and are not necessarily archived publicly.
- For model training on historical sales, a current GTFS snapshot may be mismatched.
  If you can't get historical GTFS, treat GTFS as static accessibility proxy, not a daily signal.

This is intentionally a *dumb downloader*:
- You provide the direct feed URL.
- It downloads the ZIP and optionally extracts it.

Usage
-----
python raw_data/code_external_data/download_10_vrr_gtfs_download_stub.py --url "<DIRECT_GTFS_ZIP_URL>" --extract

Dependencies: requests
"""

from __future__ import annotations

import argparse
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=None, help="ABSOLUTE base directory for scripts + outputs")
    ap.add_argument("--url", required=True, help="Direct GTFS ZIP download URL")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: <base-dir>/_external_data/vrr_gtfs)")
    ap.add_argument("--max-gb", type=float, default=None, help="Abort download if Content-Length exceeds this size")
    ap.add_argument("--extract", action="store_true", help="Extract ZIP after download")
    ap.add_argument("--force", action="store_true", help="Overwrite existing file")
    args = ap.parse_args()

    base_dir = get_base_dir(args.base_dir)

    out_dir = require_absolute_path(args.out_dir, "--out-dir") if args.out_dir else get_output_dir(base_dir, "vrr_gtfs")
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = args.url.split("/")[-1].split("?", 1)[0] or "gtfs.zip"
    if not filename.lower().endswith(".zip"):
        filename += ".zip"

    out_path = out_dir / filename

    if out_path.exists() and not args.force:
        print(f"File exists, skipping: {out_path}")
    else:
        print(f"Downloading {args.url} -> {out_path}")
        _download(args.url, out_path, max_gb=args.max_gb)
        print("Download completed.")

    if args.extract:
        extract_dir = out_dir / out_path.stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting -> {extract_dir}")
        with zipfile.ZipFile(out_path, "r") as z:
            z.extractall(extract_dir)
        print("Extraction completed.")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
