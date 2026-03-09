"""Download OSM POIs around each store using Overpass API and compute static POI counts.

This script intentionally outputs *aggregated* features (counts) instead of raw geometries.
Reason: raw OSM geometries can explode disk/RAM usage with zero benefit for a daily sales forecast.

Outputs (static, store-level):
  - store_id
  - lat, lon
  - radius_m
  - n_bakery, n_supermarket
  - n_school, n_university
  - n_bus_stop, n_rail_station, n_transit_platform

Usage (from repo root)
---------------------
python raw_data/code_external_data/download_03_osm_pois_overpass.py --radius-m 1000

If your stores file is not auto-detected:
python raw_data/code_external_data/download_03_osm_pois_overpass.py --stores-file raw_data/20260218_144523_stores.parquet

Dependencies: requests, pandas
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from abs_path_utils import (
    get_base_dir,
    get_default_plz_centroids_path,
    get_output_dir,
    get_repo_root,
    require_absolute_path,
)
from store_location_utils import load_store_locations

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"



def _find_latest_file(repo_root: Path, pattern: str) -> Optional[Path]:
    """Find newest file matching pattern in common project dirs.

    This repo stores the generated parquet files in different places (e.g.
    raw_data/, visualized_raw_data_analysis/). We search both to avoid
    fragile path assumptions.
    """
    root = repo_root
    search_dirs = [root / 'raw_data', root / 'visualized_raw_data_analysis']
    candidates: list[Path] = []
    for d in search_dirs:
        if not d.exists():
            continue
        candidates.extend(d.glob(pattern))
        # Also look one level deeper (cheap)
        for sub in d.iterdir():
            if sub.is_dir():
                candidates.extend(sub.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _autodetect_column(cols: List[str], preferred: List[str], contains_any: List[str]) -> str:
    lower = {c.lower(): c for c in cols}
    for p in preferred:
        if p.lower() in lower:
            return lower[p.lower()]
    for c in cols:
        cl = c.lower()
        if any(tok in cl for tok in contains_any):
            return c
    raise SystemExit(f"Could not autodetect column. Have: {cols}")


def _read_stores(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            raise SystemExit(
                f"Failed to read parquet stores file: {path}\n"
                "Install pyarrow: pip install pyarrow\n"
                f"Original error: {e}"
            )
    return df


def _overpass_query(lat: float, lon: float, radius_m: int) -> str:
    # Request tags only (no geometry) to keep payload small.
    # We ask for node/way/relation for each feature group.
    return f"""[out:json][timeout:180];
(
  node(around:{radius_m},{lat},{lon})[shop=bakery];
  way(around:{radius_m},{lat},{lon})[shop=bakery];
  relation(around:{radius_m},{lat},{lon})[shop=bakery];

  node(around:{radius_m},{lat},{lon})[shop=supermarket];
  way(around:{radius_m},{lat},{lon})[shop=supermarket];
  relation(around:{radius_m},{lat},{lon})[shop=supermarket];

  node(around:{radius_m},{lat},{lon})[amenity=school];
  way(around:{radius_m},{lat},{lon})[amenity=school];
  relation(around:{radius_m},{lat},{lon})[amenity=school];

  node(around:{radius_m},{lat},{lon})[amenity=university];
  way(around:{radius_m},{lat},{lon})[amenity=university];
  relation(around:{radius_m},{lat},{lon})[amenity=university];

  node(around:{radius_m},{lat},{lon})[highway=bus_stop];

  node(around:{radius_m},{lat},{lon})[railway=station];

  node(around:{radius_m},{lat},{lon})[public_transport=platform];
);
out tags;"""


def _http_post_json(url: str, payload: str, timeout_s: int = 180, max_retries: int = 6) -> Dict[str, Any]:
    backoff = 2
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(url, data=payload.encode("utf-8"), timeout=timeout_s)
            if r.status_code in (429, 504, 502, 503):
                time.sleep(min(120, backoff))
                backoff *= 2
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == max_retries:
                raise
            time.sleep(min(120, backoff))
            backoff *= 2
    raise RuntimeError("Unreachable")


def _count_elements(resp: Dict[str, Any]) -> Dict[str, int]:
    counts = {
        "n_bakery": 0,
        "n_supermarket": 0,
        "n_school": 0,
        "n_university": 0,
        "n_bus_stop": 0,
        "n_rail_station": 0,
        "n_transit_platform": 0,
    }
    elements = resp.get("elements", [])
    for el in elements:
        tags = el.get("tags", {}) or {}
        if tags.get("shop") == "bakery":
            counts["n_bakery"] += 1
        if tags.get("shop") == "supermarket":
            counts["n_supermarket"] += 1
        if tags.get("amenity") == "school":
            counts["n_school"] += 1
        if tags.get("amenity") == "university":
            counts["n_university"] += 1
        if tags.get("highway") == "bus_stop":
            counts["n_bus_stop"] += 1
        if tags.get("railway") == "station":
            counts["n_rail_station"] += 1
        if tags.get("public_transport") == "platform":
            counts["n_transit_platform"] += 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=None, help="ABSOLUTE base directory for scripts + outputs")
    ap.add_argument(
        "--stores-file",
        default=None,
        help="ABSOLUTE path to stores file (parquet/csv). If omitted, the newest *stores*.parquet is auto-detected under the repo.",
    )
    ap.add_argument(
        "--plz-centroids-file",
        default=None,
        help=(
            "ABSOLUTE path to plz_centroids_nrw.csv (ZIP -> lat/lon). "
            "Required if your stores file has no lat/lon columns."
        ),
    )
    ap.add_argument("--radius-m", type=int, default=1000, help="Catchment radius in meters (default: 1000)")
    ap.add_argument(
        "--out",
        default=None,
        help="ABSOLUTE output FILE path (parquet/csv). If omitted, a default path under <base-dir> is used.",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="ABSOLUTE output DIRECTORY. If provided, the script writes its default file name into this directory.",
    )
    ap.add_argument("--sleep-s", type=float, default=1.0, help="Sleep between store queries (default: 1.0)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing output")
    ap.add_argument("--max-stores", type=int, default=None, help="Limit number of stores (debug)")
    args = ap.parse_args()

    base_dir = get_base_dir(args.base_dir)
    repo_root = get_repo_root(base_dir)

    stores_path = require_absolute_path(args.stores_file, "--stores-file") if args.stores_file else _find_latest_file(repo_root, "*stores*.parquet")
    if stores_path is None or not stores_path.exists():
        raise SystemExit("Could not auto-detect stores file. Pass --stores-file <path>.")

    # Stores must have coordinates for any location-based enrichment.
    # If the stores table has no lat/lon columns, we derive them from ZIP centroids.
    plz_centroids_path = (
        require_absolute_path(args.plz_centroids_file, "--plz-centroids-file")
        if args.plz_centroids_file
        else None
    )
    if plz_centroids_path is None:
        default_plz = get_default_plz_centroids_path(base_dir)
        if default_plz.exists():
            plz_centroids_path = default_plz

    df_stores = load_store_locations(stores_path, plz_centroids_path, require_complete=True)

    out_path: Path
    if args.out:
        out_path = require_absolute_path(args.out, "--out")
    elif args.out_dir:
        out_dir = require_absolute_path(args.out_dir, "--out-dir")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = (out_dir / f"osm_pois_{args.radius_m}m.parquet").resolve()
    else:
        out_dir = get_output_dir(base_dir, "osm_pois")
        out_path = (out_dir / f"osm_pois_{args.radius_m}m.parquet").resolve()

    if out_path.exists() and not args.force:
        print(f"Output exists, skipping: {out_path}")
        return 0

    if args.max_stores is not None:
        df_stores = df_stores.head(args.max_stores)

    results: List[Dict[str, Any]] = []

    total = len(df_stores)
    for i, row in df_stores.iterrows():
        store_id = row["store_id"]
        lat = float(row["lat"])
        lon = float(row["lon"])
        q = _overpass_query(lat, lon, args.radius_m)
        try:
            resp = _http_post_json(OVERPASS_URL, q)
            counts = _count_elements(resp)
        except Exception as e:
            print(f"WARN: Overpass failed for store_id={store_id}: {e}", file=sys.stderr)
            counts = {k: None for k in [
                "n_bakery",
                "n_supermarket",
                "n_school",
                "n_university",
                "n_bus_stop",
                "n_rail_station",
                "n_transit_platform",
            ]}

        results.append(
            {
                "store_id": store_id,
                "lat": lat,
                "lon": lon,
                "radius_m": args.radius_m,
                **counts,
                "source": "OpenStreetMap/Overpass",
                "fetched_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
            }
        )

        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{total}")
        time.sleep(max(0.0, args.sleep_s))

    df_out = pd.DataFrame(results)

    try:
        if out_path.suffix.lower() == ".csv":
            df_out.to_csv(out_path, index=False)
        else:
            df_out.to_parquet(out_path, index=False)
    except Exception as e:
        csv_fallback = out_path.with_suffix(".csv")
        df_out.to_csv(csv_fallback, index=False)
        print(
            "Parquet write failed (missing pyarrow/fastparquet?). "
            f"Wrote CSV instead: {csv_fallback}\nError: {e}",
            file=sys.stderr,
        )
        return 2

    print(f"Wrote: {out_path} ({len(df_out):,} stores)")
    print("NOTE: This script may be slow for many stores because Overpass is queried per store.")
    print("      If you have >500 stores, consider running per-city batches or caching results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

