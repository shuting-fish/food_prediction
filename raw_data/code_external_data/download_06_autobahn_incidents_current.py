"""Download current Autobahn incidents (roadworks, warnings, closures) from Autobahn API.

Hard truth
----------
This API is *current state*, not historical. You cannot reliably backfill past incidents for model
training. If you want a historical dataset, you must snapshot it regularly going forward.

Outputs
-------
- Raw snapshot table with a fetch timestamp.
- OPTIONAL: store-proximity aggregates if you provide a stores file.

Usage (from repo root)
---------------------
# Just snapshot Autobahn incidents
python raw_data/code_external_data/download_06_autobahn_incidents_current.py

# Snapshot + aggregate incidents near stores (within 10km)
python raw_data/code_external_data/download_06_autobahn_incidents_current.py --stores-file visualized_raw_data_analysis/20260218_144523_stores.parquet --max-distance-km 10

Dependencies: requests, pandas
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from abs_path_utils import get_base_dir, get_output_dir, get_repo_root, require_absolute_path

import requests

# Public endpoint used in the official OpenAPI examples.
DEFAULT_BASE = "https://verkehr.autobahn.de/o/autobahn"

SERVICES = {
    "roadworks": "services/roadworks",
    "warnings": "services/warning",
    "closures": "services/closure",
}



def _find_latest_file(repo_root: Path, pattern: str) -> Optional[Path]:
    root = repo_root
    search_dirs = [root / "raw_data", root / "visualized_raw_data_analysis"]
    candidates: List[Path] = []
    for d in search_dirs:
        if not d.exists():
            continue
        candidates.extend(d.glob(pattern))
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


def _http_get_json(url: str, timeout_s: int = 60) -> Any:
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def _list_roads(base_url: str) -> List[str]:
    # Try a couple of patterns, because proxies/redirects exist.
    for url in (base_url, base_url + "/"):
        try:
            data = _http_get_json(url)
            if isinstance(data, dict) and isinstance(data.get("roads"), list):
                return [str(x) for x in data["roads"]]
            if isinstance(data, list):
                return [str(x) for x in data]
        except Exception:
            continue
    raise SystemExit("Failed to list roads from Autobahn API. Try --base-url with a different endpoint.")


def _flatten_records(obj: Any, road: str, kind: str, fetched_at: str) -> List[Dict[str, Any]]:
    # Keep it simple: store raw JSON blob + try to pull coordinates.
    records: List[Dict[str, Any]] = []

    if isinstance(obj, dict):
        payload = None
        for k in (kind, "items", "data"):
            if isinstance(obj.get(k), list):
                payload = obj.get(k)
                break
        if payload is None and isinstance(obj.get("features"), list):
            payload = obj.get("features")
        if payload is None:
            payload = []

        for it in payload:
            rec: Dict[str, Any] = {
                "road": road,
                "kind": kind,
                "fetched_at_utc": fetched_at,
                "raw": it,
            }
            try:
                if isinstance(it, dict):
                    geom = it.get("geometry")
                    if isinstance(geom, dict) and geom.get("type") == "Point":
                        coords = geom.get("coordinates")
                        if isinstance(coords, list) and len(coords) >= 2:
                            rec["lon"] = float(coords[0])
                            rec["lat"] = float(coords[1])
                    if "lat" in it and "lon" in it:
                        rec["lat"] = float(it["lat"])
                        rec["lon"] = float(it["lon"])
            except Exception:
                pass

            records.append(rec)

    elif isinstance(obj, list):
        for it in obj:
            records.append({"road": road, "kind": kind, "fetched_at_utc": fetched_at, "raw": it})

    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=None, help="ABSOLUTE base directory for scripts + outputs")
    ap.add_argument("--base-url", default=DEFAULT_BASE, help="Base URL (default: https://verkehr.autobahn.de/o/autobahn)")
    ap.add_argument("--stores-file", default=None, help="Optional stores file (parquet/csv) to compute proximity counts")
    ap.add_argument("--max-distance-km", type=float, default=10.0, help="Max distance for proximity features (default: 10km)")
    ap.add_argument("--out-dir", default=None, help="ABSOLUTE output directory. If omitted, a default path under <base-dir> is used.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    base_url = args.base_url.rstrip("/")

    out_dir = require_absolute_path(args.out_dir, "--out-dir") if args.out_dir else get_output_dir(base_dir, "autobahn_api")
    out_dir.mkdir(parents=True, exist_ok=True)

    fetched_at = datetime.utcnow().isoformat(timespec="seconds")
    stamp = fetched_at.replace(":", "").replace("-", "")

    out_raw = out_dir / f"autobahn_snapshot_{stamp}.parquet"
    out_store = out_dir / f"autobahn_store_counts_{stamp}.parquet"

    if out_raw.exists() and not args.force:
        print(f"Output exists, skipping: {out_raw}")
        return 0

    roads = _list_roads(base_url)

    all_records: List[Dict[str, Any]] = []
    for road in roads:
        for kind, rel_path in SERVICES.items():
            url = f"{base_url}/{road}/{rel_path}"
            try:
                data = _http_get_json(url)
                all_records.extend(_flatten_records(data, road=road, kind=kind, fetched_at=fetched_at))
            except Exception as e:
                print(f"WARN: failed {url}: {e}", file=sys.stderr)

    df_raw = pd.DataFrame(all_records)

    try:
        df_raw.to_parquet(out_raw, index=False)
    except Exception:
        df_raw.to_csv(out_raw.with_suffix(".csv"), index=False)

    print(f"Wrote raw snapshot: {out_raw} ({len(df_raw):,} records)")

    # Optional store proximity aggregates
    stores_path = require_absolute_path(args.stores_file, "--stores-file") if args.stores_file else _find_latest_file(repo_root, "*stores*.parquet")

    if stores_path and stores_path.exists():
        df_stores = _read_stores(stores_path)
        cols = list(df_stores.columns)
        store_id_col = _autodetect_column(cols, ["store_id", "storeid", "id"], ["store", "filiale", "shop"])
        lat_col = _autodetect_column(cols, ["lat", "latitude"], ["lat"])
        lon_col = _autodetect_column(cols, ["lon", "lng", "longitude"], ["lon", "lng", "long"])

        df_geo = df_raw.dropna(subset=["lat", "lon"], how="any").copy()
        if df_geo.empty:
            print("No geocoded incidents in this snapshot; skipping store proximity features.")
            return 0

        store_rows: List[Dict[str, Any]] = []
        for _, s in df_stores.iterrows():
            sid = s[store_id_col]
            slat = float(s[lat_col])
            slon = float(s[lon_col])

            counts = {"roadworks": 0, "warnings": 0, "closures": 0}
            for _, r in df_geo.iterrows():
                km = _haversine_km(slat, slon, float(r["lat"]), float(r["lon"]))
                if km <= args.max_distance_km:
                    k = str(r["kind"])
                    if k in counts:
                        counts[k] += 1

            store_rows.append(
                {
                    "store_id": sid,
                    "max_distance_km": args.max_distance_km,
                    "n_roadworks_near": counts["roadworks"],
                    "n_warnings_near": counts["warnings"],
                    "n_closures_near": counts["closures"],
                    "fetched_at_utc": fetched_at,
                }
            )

        df_store = pd.DataFrame(store_rows)
        try:
            df_store.to_parquet(out_store, index=False)
        except Exception:
            df_store.to_csv(out_store.with_suffix(".csv"), index=False)
        print(f"Wrote store proximity counts: {out_store}")
    else:
        print("No stores file found; skipping store proximity features.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
