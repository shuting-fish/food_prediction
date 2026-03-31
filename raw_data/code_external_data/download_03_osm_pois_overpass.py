from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import pandas as pd
import requests

DEFAULT_OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_SLEEP_SECONDS = 1.25
DEFAULT_RETRY_ATTEMPTS = 4
DEFAULT_RETRY_BACKOFF_SECONDS = 3.0
DEFAULT_RADII_METERS = [300, 500, 1000]
DEFAULT_SELF_EXCLUSION_METERS = 30.0

CATEGORY_DEFINITIONS = {
    "bakery": {"shop": "bakery"},
    "supermarket": {"shop": "supermarket"},
    "cafe": {"amenity": "cafe"},
    "school": {"amenity": "school"},
    "bus_stop": {"highway": "bus_stop"},
    "station": [
        {"railway": "station"},
        {"public_transport": "station"},
    ],
}


def get_repo_root() -> Path:
    """
    Return the repository root based on this file location.
    """
    return Path(__file__).resolve().parents[2]


def get_base_dir() -> Path:
    """
    Return the external data code directory.
    """
    return get_repo_root() / "raw_data" / "code_external_data"


def get_default_stores_path() -> Path:
    """
    Return the default canonical stores parquet path.
    """
    return get_repo_root() / "raw_data" / "20260218_144523_stores.parquet"


def get_default_centroids_path() -> Path:
    """
    Return the default NRW centroid CSV path.
    """
    return get_base_dir() / "_reference_geo" / "plz_centroids_nrw.csv"


def get_default_output_path() -> Path:
    """
    Return the default curated OSM context output path.
    """
    return (
        get_base_dir()
        / "_external_data"
        / "osm_pois_overpass"
        / "store_static_context_osm.parquet"
    )


def get_default_cache_dir() -> Path:
    """
    Return the default disk cache directory for Overpass responses.
    """
    return get_base_dir() / "_external_data" / "osm_pois_overpass" / "cache"


def ensure_parent_dir(path: Path) -> None:
    """
    Ensure the parent directory exists.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> None:
    """
    Ensure the directory exists.
    """
    path.mkdir(parents=True, exist_ok=True)


def normalize_zipcode(value: object) -> Optional[str]:
    """
    Convert a zipcode-like value to a 5-digit string if possible.
    """
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None

    return digits.zfill(5)


def _normalize_column_name(name: object) -> str:
    """
    Normalize a column name for flexible matching.
    """
    text = str(name).strip().lower()
    text = text.replace(" ", "_").replace("-", "_")
    return text


def _resolve_column_name(columns: Iterable[object], candidates: List[str]) -> Optional[str]:
    """
    Resolve the first matching column name from a candidate list.
    """
    normalized_to_original = {
        _normalize_column_name(column): str(column)
        for column in columns
    }

    for candidate in candidates:
        normalized_candidate = _normalize_column_name(candidate)
        if normalized_candidate in normalized_to_original:
            return normalized_to_original[normalized_candidate]

    return None


def load_plz_centroids(centroids_path: str | Path) -> pd.DataFrame:
    """
    Load NRW zipcode centroids with robust column auto-detection.
    """
    df = pd.read_csv(centroids_path)

    zipcode_candidates = [
        "zipcode",
        "zip",
        "plz",
        "postal_code",
        "postalcode",
        "zip_code",
        "postleitzahl",
    ]
    lat_candidates = [
        "lat",
        "latitude",
        "y",
    ]
    lon_candidates = [
        "lon",
        "lng",
        "long",
        "longitude",
        "x",
    ]

    zipcode_col = _resolve_column_name(df.columns, zipcode_candidates)
    lat_col = _resolve_column_name(df.columns, lat_candidates)
    lon_col = _resolve_column_name(df.columns, lon_candidates)

    missing_logical_cols: List[str] = []
    if zipcode_col is None:
        missing_logical_cols.append("zipcode")
    if lat_col is None:
        missing_logical_cols.append("lat")
    if lon_col is None:
        missing_logical_cols.append("lon")

    if missing_logical_cols:
        raise KeyError(
            "Could not auto-detect required centroid columns. "
            f"Missing logical columns: {missing_logical_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    out = df[[zipcode_col, lat_col, lon_col]].copy()
    out = out.rename(
        columns={
            zipcode_col: "zipcode",
            lat_col: "lat",
            lon_col: "lon",
        }
    )

    out["zipcode"] = out["zipcode"].map(normalize_zipcode)
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")

    out = out.dropna(subset=["zipcode", "lat", "lon"]).drop_duplicates(
        subset=["zipcode"]
    )

    if out.empty:
        raise ValueError(
            f"No usable centroid rows found after normalization in: {centroids_path}"
        )

    return out.reset_index(drop=True)


def load_store_locations(
    stores_path: str | Path,
    centroids_path: str | Path,
) -> pd.DataFrame:
    """
    Load store locations from the canonical stores parquet.

    Preferred:
    - direct lat/lon in stores parquet

    Fallback:
    - zipcode + centroid CSV
    """
    stores_df = pd.read_parquet(stores_path)

    store_id_candidates = [
        "store",
        "store_id",
        "store_nr",
        "branch_id",
        "branch",
        "id",
    ]
    store_name_candidates = [
        "store_name",
        "branch_name",
        "name",
    ]
    zipcode_candidates = [
        "zipcode",
        "zip",
        "plz",
        "postal_code",
        "postleitzahl",
    ]
    lat_candidates = [
        "lat",
        "latitude",
        "y",
    ]
    lon_candidates = [
        "lon",
        "lng",
        "long",
        "longitude",
        "x",
    ]

    store_id_col = _resolve_column_name(stores_df.columns, store_id_candidates)
    store_name_col = _resolve_column_name(stores_df.columns, store_name_candidates)
    zipcode_col = _resolve_column_name(stores_df.columns, zipcode_candidates)
    lat_col = _resolve_column_name(stores_df.columns, lat_candidates)
    lon_col = _resolve_column_name(stores_df.columns, lon_candidates)

    if store_id_col is None:
        raise KeyError(
            "Could not auto-detect a store identifier column. "
            f"Available columns: {list(stores_df.columns)}"
        )

    out = pd.DataFrame()
    out["store_id"] = stores_df[store_id_col].astype(str).str.strip()

    if store_name_col is not None:
        out["store_name"] = stores_df[store_name_col].astype(str).str.strip()
    else:
        out["store_name"] = out["store_id"]

    if zipcode_col is not None:
        out["zipcode"] = stores_df[zipcode_col].map(normalize_zipcode)
    else:
        out["zipcode"] = None

    if lat_col is not None and lon_col is not None:
        out["lat"] = pd.to_numeric(stores_df[lat_col], errors="coerce")
        out["lon"] = pd.to_numeric(stores_df[lon_col], errors="coerce")
    else:
        out["lat"] = None
        out["lon"] = None

    need_centroids_mask = out["lat"].isna() | out["lon"].isna()

    if need_centroids_mask.any():
        if out["zipcode"].isna().all():
            raise ValueError(
                "Some stores are missing lat/lon and no usable zipcode fallback is available."
            )

        centroids_df = load_plz_centroids(centroids_path=centroids_path)
        out = out.merge(
            centroids_df.rename(columns={"lat": "centroid_lat", "lon": "centroid_lon"}),
            on="zipcode",
            how="left",
        )
        out["lat"] = out["lat"].fillna(out["centroid_lat"])
        out["lon"] = out["lon"].fillna(out["centroid_lon"])
        out = out.drop(columns=["centroid_lat", "centroid_lon"])

    out = out.dropna(subset=["store_id", "lat", "lon"]).drop_duplicates(
        subset=["store_id"]
    )

    if out.empty:
        raise ValueError("No usable store locations available after normalization.")

    return out.reset_index(drop=True)


def haversine_distance_m(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Compute great-circle distance in meters.
    """
    radius_m = 6371000.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return radius_m * c


def build_overpass_query(
    lat: float,
    lon: float,
    radius_m: int,
) -> str:
    """
    Build one compact Overpass query around one store location.
    """
    lines: List[str] = [
        "[out:json][timeout:60];",
        "(",
    ]

    for selector in CATEGORY_DEFINITIONS.values():
        selectors = selector if isinstance(selector, list) else [selector]
        for selector_item in selectors:
            for key, value in selector_item.items():
                lines.append(
                    f'  nwr(around:{radius_m},{lat},{lon})["{key}"="{value}"];'
                )

    lines.extend(
        [
            ");",
            "out center tags;",
        ]
    )

    return "\n".join(lines)


def build_cache_key(
    store_id: str,
    lat: float,
    lon: float,
    radius_m: int,
    query: str,
) -> str:
    """
    Build a stable disk cache key for one store query.
    """
    raw = f"{store_id}|{lat:.6f}|{lon:.6f}|{radius_m}|{query}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_cached_elements(cache_dir: Path, cache_key: str) -> Optional[List[dict]]:
    """
    Load cached Overpass elements if available.
    """
    cache_path = cache_dir / f"{cache_key}.json"
    if not cache_path.exists():
        return None

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    elements = payload.get("elements", [])
    if not isinstance(elements, list):
        return []
    return elements


def save_cached_elements(cache_dir: Path, cache_key: str, elements: List[dict]) -> None:
    """
    Save Overpass elements to disk cache.
    """
    ensure_dir(cache_dir)
    cache_path = cache_dir / f"{cache_key}.json"
    cache_path.write_text(
        json.dumps({"elements": elements}, ensure_ascii=False),
        encoding="utf-8",
    )


def fetch_overpass_elements_with_retry(
    session: requests.Session,
    overpass_urls: List[str],
    query: str,
    timeout_seconds: int,
    retry_attempts: int,
    retry_backoff_seconds: float,
) -> List[dict]:
    """
    Execute one Overpass query with retry, exponential backoff, and endpoint fallback.
    """
    last_error: Optional[Exception] = None

    for overpass_url in overpass_urls:
        for attempt in range(1, retry_attempts + 1):
            try:
                response = session.post(
                    overpass_url,
                    data=query.encode("utf-8"),
                    timeout=timeout_seconds,
                    headers={"Content-Type": "text/plain; charset=utf-8"},
                )

                if response.status_code == 429:
                    raise requests.HTTPError(
                        f"429 Too Many Requests for {overpass_url}",
                        response=response,
                    )

                if 500 <= response.status_code <= 599:
                    raise requests.HTTPError(
                        f"{response.status_code} server error for {overpass_url}",
                        response=response,
                    )

                response.raise_for_status()

                payload = response.json()
                elements = payload.get("elements", [])
                if not isinstance(elements, list):
                    return []
                return elements

            except (requests.RequestException, ValueError) as exc:
                last_error = exc

                is_last_attempt_for_endpoint = attempt == retry_attempts
                if not is_last_attempt_for_endpoint:
                    sleep_seconds = retry_backoff_seconds * (2 ** (attempt - 1))
                    sleep_seconds += random.uniform(0.0, 0.75)
                    print(
                        f"[WARN] Overpass request failed on {overpass_url} "
                        f"(attempt {attempt}/{retry_attempts}): {exc}"
                    )
                    print(f"[INFO] Sleeping {sleep_seconds:.2f}s before retry.")
                    time.sleep(sleep_seconds)
                else:
                    print(
                        f"[WARN] Overpass endpoint exhausted: {overpass_url}. "
                        f"Switching to next endpoint if available."
                    )

    if last_error is not None:
        raise last_error

    return []


def classify_osm_element(tags: dict) -> Optional[str]:
    """
    Map OSM tags to one of the target feature categories.
    """
    if not isinstance(tags, dict):
        return None

    if tags.get("shop") == "bakery":
        return "bakery"
    if tags.get("shop") == "supermarket":
        return "supermarket"
    if tags.get("amenity") == "cafe":
        return "cafe"
    if tags.get("amenity") == "school":
        return "school"
    if tags.get("highway") == "bus_stop":
        return "bus_stop"
    if tags.get("railway") == "station" or tags.get("public_transport") == "station":
        return "station"

    return None


def extract_element_coordinates(element: dict) -> tuple[Optional[float], Optional[float]]:
    """
    Extract coordinates from a node, way, or relation result.
    """
    if "lat" in element and "lon" in element:
        lat = pd.to_numeric(element.get("lat"), errors="coerce")
        lon = pd.to_numeric(element.get("lon"), errors="coerce")
        if pd.notna(lat) and pd.notna(lon):
            return float(lat), float(lon)

    center = element.get("center")
    if isinstance(center, dict):
        lat = pd.to_numeric(center.get("lat"), errors="coerce")
        lon = pd.to_numeric(center.get("lon"), errors="coerce")
        if pd.notna(lat) and pd.notna(lon):
            return float(lat), float(lon)

    return None, None


def build_store_context_row(
    store_id: str,
    store_name: str,
    zipcode: Optional[str],
    store_lat: float,
    store_lon: float,
    elements: List[dict],
    radii_m: List[int],
    self_exclusion_m: float,
) -> Dict[str, object]:
    """
    Convert Overpass elements around one store into one feature row.
    """
    row: Dict[str, object] = {
        "store_id": store_id,
        "store_name": store_name,
        "zipcode": zipcode,
        "store_lat": store_lat,
        "store_lon": store_lon,
    }

    categories = list(CATEGORY_DEFINITIONS.keys())
    for category in categories:
        row[f"dist_nearest_{category}_m"] = None
        for radius in radii_m:
            row[f"count_{category}_{radius}m"] = 0

    seen_features = set()

    for element in elements:
        tags = element.get("tags", {})
        category = classify_osm_element(tags=tags)
        if category is None:
            continue

        poi_lat, poi_lon = extract_element_coordinates(element)
        if poi_lat is None or poi_lon is None:
            continue

        element_type = element.get("type")
        element_id = element.get("id")
        unique_key = (element_type, element_id, category)
        if unique_key in seen_features:
            continue
        seen_features.add(unique_key)

        distance_m = haversine_distance_m(
            lat1=store_lat,
            lon1=store_lon,
            lat2=poi_lat,
            lon2=poi_lon,
        )

        if category == "bakery" and distance_m < self_exclusion_m:
            continue

        nearest_key = f"dist_nearest_{category}_m"
        nearest_value = row[nearest_key]
        if nearest_value is None or distance_m < nearest_value:
            row[nearest_key] = float(distance_m)

        for radius in radii_m:
            if distance_m <= radius:
                row[f"count_{category}_{radius}m"] += 1

    return row


def build_store_static_context_osm(
    stores_df: pd.DataFrame,
    overpass_urls: List[str],
    cache_dir: Path,
    radii_m: List[int],
    timeout_seconds: int,
    sleep_seconds: float,
    self_exclusion_m: float,
    retry_attempts: int,
    retry_backoff_seconds: float,
) -> pd.DataFrame:
    """
    Build a curated static OSM context table with one row per store.
    """
    if not radii_m:
        raise ValueError("At least one radius must be provided.")

    max_radius = max(radii_m)
    rows: List[Dict[str, object]] = []

    with requests.Session() as session:
        session.headers.update(
            {
                "User-Agent": "food_prediction/1.0",
                "Accept": "application/json",
            }
        )

        total_stores = len(stores_df)

        for idx, store_row in stores_df.iterrows():
            store_id = str(store_row["store_id"])
            store_name = str(store_row["store_name"])
            zipcode = store_row["zipcode"]
            store_lat = float(store_row["lat"])
            store_lon = float(store_row["lon"])

            print(f"[INFO] Processing store {idx + 1}/{total_stores}: {store_id}")

            query = build_overpass_query(
                lat=store_lat,
                lon=store_lon,
                radius_m=max_radius,
            )
            cache_key = build_cache_key(
                store_id=store_id,
                lat=store_lat,
                lon=store_lon,
                radius_m=max_radius,
                query=query,
            )

            elements = load_cached_elements(cache_dir=cache_dir, cache_key=cache_key)
            if elements is None:
                elements = fetch_overpass_elements_with_retry(
                    session=session,
                    overpass_urls=overpass_urls,
                    query=query,
                    timeout_seconds=timeout_seconds,
                    retry_attempts=retry_attempts,
                    retry_backoff_seconds=retry_backoff_seconds,
                )
                save_cached_elements(
                    cache_dir=cache_dir,
                    cache_key=cache_key,
                    elements=elements,
                )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
            else:
                print(f"[INFO] Cache hit for store {store_id}")

            row = build_store_context_row(
                store_id=store_id,
                store_name=store_name,
                zipcode=zipcode,
                store_lat=store_lat,
                store_lon=store_lon,
                elements=elements,
                radii_m=radii_m,
                self_exclusion_m=self_exclusion_m,
            )
            rows.append(row)

    return pd.DataFrame(rows).sort_values("store_id").reset_index(drop=True)


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the command line parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build curated static store-context OSM features "
            "for bakery competition, POIs, and transit proximity."
        )
    )
    parser.add_argument(
        "--stores-path",
        type=Path,
        default=get_default_stores_path(),
        help=f"Path to canonical stores parquet. Default: {get_default_stores_path()}",
    )
    parser.add_argument(
        "--plz-centroids-path",
        type=Path,
        default=get_default_centroids_path(),
        help=f"Path to NRW centroid CSV. Default: {get_default_centroids_path()}",
    )
    parser.add_argument(
        "--overpass-urls",
        nargs="*",
        default=DEFAULT_OVERPASS_URLS,
        help=(
            "Ordered list of Overpass interpreter URLs. "
            "Default includes two public instances."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=get_default_cache_dir(),
        help=f"Disk cache directory. Default: {get_default_cache_dir()}",
    )
    parser.add_argument(
        "--radii-m",
        nargs="*",
        type=int,
        default=DEFAULT_RADII_METERS,
        help="Radii in meters for count features. Default: 300 500 1000",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Sleep between uncached store queries.",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=DEFAULT_RETRY_ATTEMPTS,
        help="Retry attempts per endpoint.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=DEFAULT_RETRY_BACKOFF_SECONDS,
        help="Base exponential backoff seconds.",
    )
    parser.add_argument(
        "--self-exclusion-m",
        type=float,
        default=DEFAULT_SELF_EXCLUSION_METERS,
        help="Distance threshold to exclude the store itself as bakery competitor.",
    )
    parser.add_argument(
        "--limit-stores",
        type=int,
        default=None,
        help="Optional limit for debugging.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=get_default_output_path(),
        help=f"Output parquet path. Default: {get_default_output_path()}",
    )
    return parser


def main() -> int:
    """
    Run the store-static-context OSM feature builder.
    """
    parser = build_argument_parser()
    args = parser.parse_args()

    stores_df = load_store_locations(
        stores_path=Path(args.stores_path).resolve(),
        centroids_path=Path(args.plz_centroids_path).resolve(),
    )

    if args.limit_stores is not None:
        stores_df = stores_df.head(args.limit_stores).copy()

    output_path = Path(args.output_path).resolve()
    cache_dir = Path(args.cache_dir).resolve()

    ensure_parent_dir(output_path)
    ensure_dir(cache_dir)

    context_df = build_store_static_context_osm(
        stores_df=stores_df,
        overpass_urls=list(args.overpass_urls),
        cache_dir=cache_dir,
        radii_m=sorted(set(args.radii_m)),
        timeout_seconds=args.timeout_seconds,
        sleep_seconds=args.sleep_seconds,
        self_exclusion_m=args.self_exclusion_m,
        retry_attempts=args.retry_attempts,
        retry_backoff_seconds=args.retry_backoff_seconds,
    )
    context_df.to_parquet(output_path, index=False)

    print(f"[OK] Saved {len(context_df)} store rows to {output_path}")
    print("[INFO] Columns:")
    print(context_df.columns.tolist())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
