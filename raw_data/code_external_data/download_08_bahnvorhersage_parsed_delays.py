"""Download Bahn-Vorhersage open data (parsed train delays) archives.

Hard truth (read this)
---------------------
- Parsed train delays are BIG. Yearly archives can be multiple GB to tens of GB.
- If you download everything blindly, you will waste disk and time.

What this script does
---------------------
- Scrapes the Bahn-Vorhersage open-data pages for direct file links (tar/parquet).
- Downloads only the years you request.
- Optionally downloads station metadata (stations.parquet / stops.parquet) from the /open-data/stops page.

This script is intentionally only a DOWNLOADER.
Feature extraction (e.g., disruption indices near stores) should be done later, because
it depends heavily on your modeling decisions.

Usage (from repo root)
---------------------
# Download 2023 and 2024 archives (dry run first)
python raw_data/code_external_data/download_08_bahnvorhersage_parsed_delays.py --years 2023,2024 --dry-run

# Actually download
python raw_data/code_external_data/download_08_bahnvorhersage_parsed_delays.py --years 2023,2024 --download-stops

Dependencies: requests
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import requests

from abs_path_utils import get_base_dir, get_output_dir, require_absolute_path


DEFAULT_PAGE_DELAYS = "https://bahnvorhersage.de/open-data/parsed-train-delays"
DEFAULT_PAGE_STOPS = "https://bahnvorhersage.de/open-data/stops"



def _http_get_text(url: str, timeout_s: int = 60) -> str:
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    return r.text


def _extract_urls(text: str) -> List[str]:
    # Extract absolute URLs and relative ones.
    urls = set()

    # Absolute links
    for m in re.finditer(r"https?://[^\s\"\'>]+", text):
        u = m.group(0)
        if any(u.lower().endswith(ext) for ext in (".tar", ".tar.gz", ".tgz", ".parquet", ".zip")):
            urls.add(u)

    # Relative links
    for m in re.finditer(r"href=\"([^\"]+)\"", text):
        u = m.group(1)
        if any(u.lower().endswith(ext) for ext in (".tar", ".tar.gz", ".tgz", ".parquet", ".zip")):
            urls.add(u)

    return sorted(urls)


def _normalize(url: str, base: str) -> str:
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if url.startswith("//"):
        return "https:" + url
    # Relative
    return base.rstrip("/") + "/" + url.lstrip("/")


def _download(url: str, out_path: Path, max_gb: Optional[float] = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        length = r.headers.get("Content-Length")
        if length and max_gb is not None:
            gb = int(length) / (1024**3)
            if gb > max_gb:
                raise SystemExit(f"Refusing to download {url} because Content-Length={gb:.2f}GB > --max-gb={max_gb}")

        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    tmp.replace(out_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=None, help="ABSOLUTE base directory for scripts + outputs")
    ap.add_argument("--years", required=True, help="Comma-separated years to download, e.g. 2023,2024")
    ap.add_argument("--page-delays", default=DEFAULT_PAGE_DELAYS, help="Page URL for parsed delays")
    ap.add_argument("--page-stops", default=DEFAULT_PAGE_STOPS, help="Page URL for station/stop metadata")
    ap.add_argument("--download-stops", action="store_true", help="Also download stations.parquet/stops.parquet (if linked)")
    ap.add_argument("--out-dir", default=None, help="ABSOLUTE output directory. If omitted, a default path under <base-dir> is used.")
    ap.add_argument("--dry-run", action="store_true", help="Only list URLs, do not download")
    ap.add_argument("--max-gb", type=float, default=None, help="Abort downloads if Content-Length exceeds this size")
    ap.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = ap.parse_args()

    years = [y.strip() for y in args.years.split(",") if y.strip()]
    if not years:
        raise SystemExit("No years parsed from --years")

    out_dir = require_absolute_path(args.out_dir, "--out-dir") if args.out_dir else get_output_dir(base_dir, "bahnvorhersage")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Parse delay archive links
    html = _http_get_text(args.page_delays)
    urls = _extract_urls(html)

    # Make a best-effort base for relative URLs
    base = args.page_delays
    base = base.split("#", 1)[0]

    urls_norm = [_normalize(u, base=base) for u in urls]

    # Filter by years
    year_urls: List[str] = []
    for u in urls_norm:
        if any(re.search(rf"\b{re.escape(y)}\b", u) for y in years):
            year_urls.append(u)

    if not year_urls:
        print("No direct file URLs found on the parsed-train-delays page.")
        print("Possible reasons:")
        print("- The page renders links via JavaScript (scraping won't see them).")
        print("- The URL patterns changed.")
        print("What to do:")
        print("- Open the page in a browser, copy the direct download link(s), and pass them via --page-delays to a simple text file (or adjust this script).")
        return 1

    print("Found these candidate URLs:")
    for u in year_urls:
        print("-", u)

    # 2) Optionally parse stops metadata links
    stops_urls: List[str] = []
    if args.download_stops:
        html2 = _http_get_text(args.page_stops)
        u2 = _extract_urls(html2)
        base2 = args.page_stops.split("#", 1)[0]
        u2n = [_normalize(u, base=base2) for u in u2]
        # Keep only parquet
        stops_urls = [u for u in u2n if u.lower().endswith(".parquet")]
        if stops_urls:
            print("Found stop/station metadata URLs:")
            for u in stops_urls:
                print("-", u)

    if args.dry_run:
        print("Dry-run enabled; not downloading.")
        return 0

    # 3) Download
    for u in year_urls + stops_urls:
        filename = u.split("/")[-1]
        out_path = out_dir / filename
        if out_path.exists() and not args.force:
            print(f"Skip existing: {out_path}")
            continue
        print(f"Downloading: {u} -> {out_path}")
        _download(u, out_path, max_gb=args.max_gb)

    print("Done.")
    print("Storage warning:")
    print("- If you downloaded yearly archives, expect multiple GB per year.")
    print("- Do NOT extract the full dataset unless you really need it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
