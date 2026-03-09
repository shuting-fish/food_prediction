"""Download historical weather from Bright Sky API (DWD-based) and produce daily features per store.

Why Bright Sky?
- Simple coordinate-based API.
- Uses DWD data under the hood.

Hard truth
----------
- If you have many stores and a long date range, naive per-store API calls will hit rate limits.
  This script deduplicates locations by rounding lat/lon, so multiple stores can share one fetch.

Outputs
-------
1) daily weather per unique rounded location
2) daily weather per store_id (store inherits its location's daily features)

Usage (from repo root)
---------------------
python raw_data/code_external_data/download_07_dwd_weather_brightsky.py --start 2025-04-01 --end 2025-06-30

Dependencies: requests, pandas
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from abs_path_utils import get_base_dir, get_output_dir, get_repo_root, require_absolute_path

import requests

API_BASE = "https://api.brightsky.dev"



def _parse_date(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError as e:
        raise SystemExit(f"Invalid date '{s}'. Expected YYYY-MM-DD.") from e


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
        return pd.read_csv(path)
    try:
        return pd.read_parquet(path)
    except Exception as e:
        raise SystemExit(
            f"Failed to read stores parquet: {path}\n"
            "Install pyarrow: pip install pyarrow\n"
            f"Original error: {e}"
        )


def _http_get_json(url: str, timeout_s: int = 60, max_retries: int = 6) -> Any:
    backoff = 2
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, timeout=timeout_s)
            if r.status_code in (429, 502, 503, 504):
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


def _fetch_weather(lat: float, lon: float, start: date, end: date, tz: str = "Europe/Berlin") -> pd.DataFrame:
    url = (
        f"{API_BASE}/weather?lat={lat}&lon={lon}"
        f"&date={start.isoformat()}&last_date={end.isoformat()}&tz={tz}"
    )
    data = _http_get_json(url)
    weather = data.get("weather") if isinstance(data, dict) else None
    if not isinstance(weather, list):
        return pd.DataFrame()
    df = pd.DataFrame(weather)
    if df.empty:
        return df
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).copy()
    return df


def _daily_aggregate(df_hourly: pd.DataFrame) -> pd.DataFrame:
    if df_hourly.empty:
        return pd.DataFrame()

    df = df_hourly.copy()
    df["date"] = df["timestamp"].dt.date.astype(str)

    numeric_cols = [c for c in df.columns if c not in ("timestamp", "date") and pd.api.types.is_numeric_dtype(df[c])]

    # Heuristic: precipitation/sunshine are additive; most others use mean/max.
    sum_cols = [c for c in numeric_cols if any(tok in c.lower() for tok in ["precip", "rain", "snow", "sunshine", "sun", "solar", "radiation"]) ]
    mean_cols = [c for c in numeric_cols if c not in sum_cols]

    agg_parts = []
    if mean_cols:
        agg_parts.append(df.groupby("date")[mean_cols].mean().add_suffix("_mean"))
        agg_parts.append(df.groupby("date")[mean_cols].max().add_suffix("_max"))
        agg_parts.append(df.groupby("date")[mean_cols].min().add_suffix("_min"))
    if sum_cols:
        agg_parts.append(df.groupby("date")[sum_cols].sum().add_suffix("_sum"))

    out = pd.concat(agg_parts, axis=1).reset_index()
    out["n_hours"] = df.groupby("date").size().values
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=None, help="ABSOLUTE base directory for scripts + outputs")
    ap.add_argument("--start", default="2025-04-01", help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", default="2025-06-30", help="End date (YYYY-MM-DD)")
    ap.add_argument("--stores-file", default=None, help="ABSOLUTE stores file path (parquet/csv). If omitted, the newest *stores*.parquet is auto-detected under the repo.")
    ap.add_argument("--round", type=int, default=3, help="Round lat/lon decimals to deduplicate locations (default: 3 ~ 100m)")
    ap.add_argument("--tz", default="Europe/Berlin", help="Timezone for daily bucketing (default: Europe/Berlin)")
    ap.add_argument("--sleep-s", type=float, default=0.3, help="Sleep between API calls (default: 0.3)")
    ap.add_argument("--out-dir", default=None, help="ABSOLUTE output directory. If omitted, a default path under <base-dir> is used.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    base_dir = get_base_dir(args.base_dir)
    repo_root = get_repo_root(base_dir)

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if end < start:
        raise SystemExit("--end must be >= --start")

    stores_path = require_absolute_path(args.stores_file, "--stores-file") if args.stores_file else _find_latest_file(repo_root, "*stores*.parquet")
    if stores_path is None or not stores_path.exists():
        raise SystemExit("Could not auto-detect stores file. Pass --stores-file <path>.")

    out_dir = require_absolute_path(args.out_dir, "--out-dir") if args.out_dir else get_output_dir(base_dir, "dwd_weather_brightsky")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_loc_daily = out_dir / f"weather_daily_locations_{start.isoformat()}_{end.isoformat()}_r{args.round}.parquet"
    out_store_daily = out_dir / f"weather_daily_store_{start.isoformat()}_{end.isoformat()}_r{args.round}.parquet"

    if out_store_daily.exists() and not args.force:
        print(f"Output exists, skipping: {out_store_daily}")
        return 0

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

    df_stores["lat_r"] = df_stores["lat"].round(args.round)
    df_stores["lon_r"] = df_stores["lon"].round(args.round)

    locations = df_stores[["lat_r", "lon_r"]].drop_duplicates().reset_index(drop=True)

    daily_loc_parts: List[pd.DataFrame] = []
    for i, loc in locations.iterrows():
        lat = float(loc["lat_r"])
        lon = float(loc["lon_r"])
        df_hourly = _fetch_weather(lat, lon, start, end, tz=args.tz)
        if df_hourly.empty:
            print(f"WARN: empty weather for lat={lat} lon={lon}", file=sys.stderr)
            continue
        df_daily = _daily_aggregate(df_hourly)
        if df_daily.empty:
            continue
        df_daily["lat_r"] = lat
        df_daily["lon_r"] = lon
        daily_loc_parts.append(df_daily)

        if (i + 1) % 10 == 0:
            print(f"Progress locations: {i+1}/{len(locations)}")
        time.sleep(max(0.0, args.sleep_s))

    if not daily_loc_parts:
        print("No weather data returned. Check API availability or date range.", file=sys.stderr)
        return 1

    df_loc_daily = pd.concat(daily_loc_parts, axis=0, ignore_index=True)

    # Map to stores
    df_store_daily = df_stores.merge(df_loc_daily, on=["lat_r", "lon_r"], how="left")

    try:
        df_loc_daily.to_parquet(out_loc_daily, index=False)
        df_store_daily.to_parquet(out_store_daily, index=False)
    except Exception as e:
        df_loc_daily.to_csv(out_loc_daily.with_suffix(".csv"), index=False)
        df_store_daily.to_csv(out_store_daily.with_suffix(".csv"), index=False)
        print(
            "Parquet write failed (missing pyarrow/fastparquet?). Wrote CSV instead. "
            f"Error: {e}",
            file=sys.stderr,
        )
        return 2

    print("Wrote:")
    print(f"- {out_loc_daily} ({len(df_loc_daily):,} rows)")
    print(f"- {out_store_daily} ({len(df_store_daily):,} rows)")

    print("RAM/storage note:")
    print("- You store daily aggregates, not hourly raw. That's deliberate to avoid 24x row blow-up.")
    print("- If you need hourly later, change the script, but expect disk/RAM to grow drastically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
