"""Download air quality measurements from Umweltbundesamt (UBA) Air Data API (v4 via api-proxy).

This script is built for *training/feature enrichment*:
- It downloads measurements per station for a date range.
- It maps each store to the nearest UBA station (within a configurable max distance).
- It outputs DAILY aggregates (mean/max) per pollutant and per store.

Important (hard truth)
---------------------
- Hourly data over multiple years + many stations becomes large quickly.
- For forecasting, you typically want daily aggregates. That is what this script produces.

API endpoints used (documented by UBA)
-------------------------------------
- Station/component metadata: /api-proxy/meta/json
- Measurements: /api-proxy/measures/json

Usage (from repo root)
---------------------
python raw_data/code_external_data/download_04_uba_air_quality.py --start 2025-04-01 --end 2025-06-30

Dependencies: requests, pandas, numpy
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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

API_BASE = "https://luftdaten.umweltbundesamt.de/api-proxy"

# UBA component IDs (per UBA documentation)
COMPONENTS_DEFAULT = {
    1: "PM10",
    3: "O3",
    5: "NO2",
    9: "PM2_5",
}



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


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


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


def _fetch_station_metadata(start: date, end: date, lang: str = "en") -> pd.DataFrame:
    # UBA docs: meta endpoint returns stations + what is available for given period.
    url = (
        f"{API_BASE}/meta/json?use=airquality"
        f"&date_from={start.isoformat()}&date_to={end.isoformat()}&time_from=1&time_to=24&lang={lang}"
    )
    data = _http_get_json(url)

    # We try multiple structures. Keep it defensive.
    stations = None
    if isinstance(data, dict):
        for key in ("stations", "station", "data"):
            if key in data and isinstance(data[key], list):
                stations = data[key]
                break

    if stations is None:
        # Some versions may return a list directly.
        if isinstance(data, list):
            stations = data
        else:
            raise SystemExit(f"Unexpected meta response shape: {type(data)}")

    # Normalize columns
    rows: List[Dict[str, Any]] = []
    for s in stations:
        if not isinstance(s, dict):
            continue
        code = s.get("station") or s.get("station_code") or s.get("code") or s.get("id")
        name = s.get("name") or s.get("station_name")
        lat = s.get("lat") or s.get("latitude")
        lon = s.get("lon") or s.get("longitude")
        if code is None or lat is None or lon is None:
            continue
        rows.append(
            {
                "station_code": str(code),
                "station_name": name,
                "lat": float(lat),
                "lon": float(lon),
            }
        )

    df = pd.DataFrame(rows).drop_duplicates(subset=["station_code"]).reset_index(drop=True)
    if df.empty:
        raise SystemExit("No stations parsed from UBA meta endpoint. The API response structure may have changed.")
    return df


def _fetch_measures(
    station_code: str,
    component_id: int,
    start: date,
    end: date,
    lang: str = "en",
    scope: Optional[int] = None,
) -> pd.DataFrame:
    # UBA docs example:
    # /measures/json?date_from=2022-08-01&date_to=2022-08-01&time_from=1&time_to=24&station=DENW207&component=5
    url = (
        f"{API_BASE}/measures/json?date_from={start.isoformat()}&date_to={end.isoformat()}"
        f"&time_from=1&time_to=24&station={station_code}&component={component_id}&lang={lang}"
    )
    if scope is not None:
        url += f"&scope={scope}"

    data = _http_get_json(url)
    if not isinstance(data, list):
        # Sometimes API returns dict with 'data'
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            data = data["data"]
        else:
            # empty result might be dict
            return pd.DataFrame(columns=["station_code", "component_id", "datetime_end", "value"])

    rows: List[Dict[str, Any]] = []
    for rec in data:
        if not isinstance(rec, dict):
            continue
        # The API uses 'date start' / 'date end' (space in key) according to docs.
        dt_end = rec.get("date end") or rec.get("date_end") or rec.get("datetime") or rec.get("timestamp")
        val = rec.get("value")
        if dt_end is None or val is None:
            continue
        # Some responses may include hour as integer fields; keep tolerant.
        try:
            dt_parsed = pd.to_datetime(dt_end)
        except Exception:
            continue
        rows.append(
            {
                "station_code": station_code,
                "component_id": component_id,
                "datetime_end": dt_parsed,
                "value": float(val),
            }
        )

    return pd.DataFrame(rows)


def _daily_aggregate(measures: pd.DataFrame, component_name: str) -> pd.DataFrame:
    if measures.empty:
        return pd.DataFrame(columns=["station_code", "date", f"{component_name}_mean", f"{component_name}_max"])

    measures = measures.copy()
    measures["date"] = measures["datetime_end"].dt.date.astype(str)
    agg = (
        measures.groupby(["station_code", "date"], as_index=False)["value"]
        .agg([("mean", "mean"), ("max", "max"), ("n", "count")])
        .reset_index()
    )
    agg.rename(
        columns={
            "mean": f"{component_name}_mean",
            "max": f"{component_name}_max",
            "n": f"{component_name}_n",
        },
        inplace=True,
    )
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=None, help="ABSOLUTE base directory for scripts + outputs")
    ap.add_argument("--start", default="2025-04-01", help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", default="2025-06-30", help="End date (YYYY-MM-DD)")
    ap.add_argument("--lang", default="en", choices=["en", "de"], help="API language (default: en)")
    ap.add_argument(
        "--components",
        default=",".join(str(k) for k in COMPONENTS_DEFAULT.keys()),
        help="Comma-separated component IDs (default: 1,3,5,9)",
    )
    ap.add_argument(
        "--stores-file",
        default=None,
        help="ABSOLUTE stores file path (parquet/csv). If omitted, the newest *stores*.parquet is auto-detected under the repo.",
    )
    ap.add_argument("--max-distance-km", type=float, default=20.0, help="Max store-to-station distance (default: 20km)")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="ABSOLUTE output directory. If omitted, a default path under <base-dir> is used.",
    )
    ap.add_argument("--sleep-s", type=float, default=0.2, help="Sleep between API calls (default: 0.2)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    base_dir = get_base_dir(args.base_dir)
    repo_root = get_repo_root(base_dir)

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if end < start:
        raise SystemExit("--end must be >= --start")

    out_dir = require_absolute_path(args.out_dir, "--out-dir") if args.out_dir else get_output_dir(base_dir, "uba_air_quality")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_stations = out_dir / f"uba_stations_meta_{start.isoformat()}_{end.isoformat()}.parquet"
    out_mapping = out_dir / f"store_to_uba_station_{start.isoformat()}_{end.isoformat()}.parquet"
    out_station_daily = out_dir / f"uba_station_daily_{start.isoformat()}_{end.isoformat()}.parquet"
    out_store_daily = out_dir / f"uba_store_daily_{start.isoformat()}_{end.isoformat()}.parquet"

    if out_store_daily.exists() and not args.force:
        print(f"Output exists, skipping: {out_store_daily}")
        return 0

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

    # 1) Station metadata
    df_stations = _fetch_station_metadata(start, end, lang=args.lang)

    # 2) Map store -> nearest station
    mapping_rows: List[Dict[str, Any]] = []
    for _, s in df_stores.iterrows():
        sid = s["store_id"]
        slat = float(s["lat"])
        slon = float(s["lon"])
        # brute force; station list is not huge
        dists = (
            (df_stations["lat"].to_numpy(), df_stations["lon"].to_numpy())
        )
        best_code = None
        best_km = None
        for i in range(len(df_stations)):
            km = _haversine_km(slat, slon, float(df_stations.loc[i, "lat"]), float(df_stations.loc[i, "lon"]))
            if best_km is None or km < best_km:
                best_km = km
                best_code = str(df_stations.loc[i, "station_code"])
        if best_km is not None and best_km <= args.max_distance_km:
            mapping_rows.append(
                {
                    "store_id": sid,
                    "station_code": best_code,
                    "distance_km": float(best_km),
                    "store_lat": slat,
                    "store_lon": slon,
                }
            )
        else:
            mapping_rows.append(
                {
                    "store_id": sid,
                    "station_code": None,
                    "distance_km": None,
                    "store_lat": slat,
                    "store_lon": slon,
                }
            )

    df_map = pd.DataFrame(mapping_rows)

    used_stations = sorted({c for c in df_map["station_code"].dropna().astype(str).tolist()})
    if not used_stations:
        print(
            "No store could be mapped to an UBA station within max distance. "
            "Increase --max-distance-km or verify store coordinates.",
            file=sys.stderr,
        )
        return 1

    # 3) Download + aggregate station daily
    comp_ids = [int(x.strip()) for x in args.components.split(",") if x.strip()]

    station_daily_parts: List[pd.DataFrame] = []
    for station in used_stations:
        for comp_id in comp_ids:
            comp_name = COMPONENTS_DEFAULT.get(comp_id, f"C{comp_id}")
            df_meas = _fetch_measures(station, comp_id, start, end, lang=args.lang)
            df_daily = _daily_aggregate(df_meas, comp_name)
            if not df_daily.empty:
                station_daily_parts.append(df_daily)
            time.sleep(max(0.0, args.sleep_s))

    if not station_daily_parts:
        print("No measurement data returned for requested period/components.", file=sys.stderr)
        return 1

    df_station_daily = station_daily_parts[0]
    for part in station_daily_parts[1:]:
        df_station_daily = df_station_daily.merge(part, on=["station_code", "date"], how="outer")

    df_station_daily.sort_values(["station_code", "date"], inplace=True)

    # 4) Map to store daily
    df_store_daily = df_map.merge(df_station_daily, on="station_code", how="left")

    # 5) Write outputs
    try:
        df_stations.to_parquet(out_stations, index=False)
        df_map.to_parquet(out_mapping, index=False)
        df_station_daily.to_parquet(out_station_daily, index=False)
        df_store_daily.to_parquet(out_store_daily, index=False)
    except Exception as e:
        # Fallback to CSV
        df_stations.to_csv(out_stations.with_suffix(".csv"), index=False)
        df_map.to_csv(out_mapping.with_suffix(".csv"), index=False)
        df_station_daily.to_csv(out_station_daily.with_suffix(".csv"), index=False)
        df_store_daily.to_csv(out_store_daily.with_suffix(".csv"), index=False)
        print(
            "Parquet write failed (missing pyarrow/fastparquet?). Wrote CSV instead. "
            f"Error: {e}",
            file=sys.stderr,
        )
        return 2

    print("Wrote:")
    print(f"- {out_stations}")
    print(f"- {out_mapping}")
    print(f"- {out_station_daily} ({len(df_station_daily):,} rows)")
    print(f"- {out_store_daily} ({len(df_store_daily):,} rows)")

    print("RAM/storage note:")
    print("- This script writes daily aggregates. If you later decide to store hourly data, expect 24x more rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
