from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests

from date_range_utils import (
    format_date_range_for_logs,
    get_default_sales_path,
    resolve_date_range,
)


META_API_URL = "https://luftdaten.umweltbundesamt.de/api-proxy/meta/json"
MEASURES_API_URL = "https://luftdaten.umweltbundesamt.de/api-proxy/measures/json"

DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_SLEEP_SECONDS = 0.10

# UBA component ids and scope ids used for hourly measurements where possible.
# Official UBA documentation lists:
# PM10=1, O3=3, NO2=5, PM2.5=9
# scope 2 = hourly mean
POLLUTANTS = {
    "pm10": {"component_id": 1, "scope_id": 2},
    "o3": {"component_id": 3, "scope_id": 2},
    "no2": {"component_id": 5, "scope_id": 2},
    "pm25": {"component_id": 9, "scope_id": 2},
}

DEFAULT_STATE_CODES = ["NW"]
STATION_CODE_PATTERN = re.compile(r"^DE[A-Z]{2}\d{3}$")


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


def get_default_output_path(start_date: pd.Timestamp, end_date: pd.Timestamp) -> Path:
    """
    Return the default parquet output path.
    """
    file_name = f"air_quality_hourly_{start_date.date()}_{end_date.date()}.parquet"
    return get_base_dir() / "_external_data" / "air_quality_uba" / file_name


def ensure_parent_dir(path: Path) -> None:
    """
    Ensure the parent directory of a file exists.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


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


def fetch_json(
    session: requests.Session,
    url: str,
    timeout_seconds: int,
) -> object:
    """
    Fetch JSON content and raise on HTTP errors.
    """
    response = session.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    return response.json()


def load_store_zipcodes(
    stores_path: str | Path,
    zipcode_col: str = "zipcode",
) -> List[str]:
    """
    Load distinct store zipcodes from the canonical stores parquet.
    """
    stores_df = pd.read_parquet(stores_path)

    if zipcode_col not in stores_df.columns:
        raise KeyError(f"Missing required stores column: '{zipcode_col}'")

    zipcodes = (
        stores_df[zipcode_col]
        .map(normalize_zipcode)
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if not zipcodes:
        raise ValueError("No usable zipcodes found in stores parquet.")

    return zipcodes


def haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Compute great-circle distance in kilometers.
    """
    radius_km = 6371.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return radius_km * c


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


def meta_url(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> str:
    """
    Build the UBA metadata endpoint URL.
    """
    return (
        f"{META_API_URL}"
        f"?use=airquality"
        f"&date_from={start_date.date()}"
        f"&date_to={end_date.date()}"
        f"&time_from=1"
        f"&time_to=24"
        f"&lang=de"
    )


def measures_url(
    station_code: str,
    component_id: int,
    scope_id: int,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> str:
    """
    Build the UBA measures endpoint URL.
    """
    return (
        f"{MEASURES_API_URL}"
        f"?date_from={start_date.date()}"
        f"&date_to={end_date.date()}"
        f"&time_from=1"
        f"&time_to=24"
        f"&station={station_code}"
        f"&component={component_id}"
        f"&scope={scope_id}"
        f"&lang=de"
    )


def _flatten_objects(payload: object) -> List[dict]:
    """
    Recursively extract all dict objects from nested JSON content.
    """
    objects: List[dict] = []

    if isinstance(payload, dict):
        objects.append(payload)
        for value in payload.values():
            objects.extend(_flatten_objects(value))
    elif isinstance(payload, list):
        for item in payload:
            objects.extend(_flatten_objects(item))

    return objects


def _parse_station_code(record: dict) -> Optional[str]:
    """
    Extract a valid station code from a record if possible.
    """
    candidate_keys = [
        "station",
        "station_code",
        "code",
        "stationId",
        "station_id",
        "id",
    ]

    for key in candidate_keys:
        if key not in record:
            continue
        value = str(record.get(key)).strip()
        if STATION_CODE_PATTERN.match(value):
            return value

    for value in record.values():
        text = str(value).strip()
        if STATION_CODE_PATTERN.match(text):
            return text

    return None


def _parse_station_name(record: dict) -> Optional[str]:
    """
    Extract a station name if available.
    """
    for key in ["name", "station_name", "stationsname", "title"]:
        if key in record and pd.notna(record.get(key)):
            text = str(record.get(key)).strip()
            if text:
                return text
    return None


def _parse_lat_lon(record: dict) -> tuple[Optional[float], Optional[float]]:
    """
    Extract latitude and longitude from a station-like record.
    """
    lat_keys = ["lat", "latitude", "geo_lat", "y"]
    lon_keys = ["lon", "lng", "long", "longitude", "geo_lon", "x"]

    lat = None
    lon = None

    for key in lat_keys:
        if key in record:
            lat_value = pd.to_numeric(record.get(key), errors="coerce")
            if pd.notna(lat_value):
                lat = float(lat_value)
                break

    for key in lon_keys:
        if key in record:
            lon_value = pd.to_numeric(record.get(key), errors="coerce")
            if pd.notna(lon_value):
                lon = float(lon_value)
                break

    return lat, lon


def parse_meta_payload(payload: object, allowed_state_codes: Iterable[str]) -> pd.DataFrame:
    """
    Parse station metadata from the UBA meta endpoint.

    The endpoint is documented as providing available stations and metadata
    for a given period. The response shape can vary, so parsing is defensive.
    """
    allowed_state_codes_upper = {str(code).strip().upper() for code in allowed_state_codes}
    rows: List[Dict[str, object]] = []

    for record in _flatten_objects(payload):
        station_code = _parse_station_code(record)
        lat, lon = _parse_lat_lon(record)

        if station_code is None or lat is None or lon is None:
            continue

        state_code = station_code[2:4].upper()
        if allowed_state_codes_upper and state_code not in allowed_state_codes_upper:
            continue

        rows.append(
            {
                "station_code": station_code,
                "station_name": _parse_station_name(record),
                "state_code": state_code,
                "lat": lat,
                "lon": lon,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(
            columns=[
                "station_code",
                "station_name",
                "state_code",
                "lat",
                "lon",
            ]
        )

    return (
        df.drop_duplicates(subset=["station_code"])
        .dropna(subset=["lat", "lon"])
        .reset_index(drop=True)
    )


def assign_nearest_station_per_zipcode(
    store_zipcodes: List[str],
    centroids_df: pd.DataFrame,
    stations_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Map each store zipcode to the nearest air quality station.
    """
    zipcode_df = centroids_df.loc[centroids_df["zipcode"].isin(store_zipcodes)].copy()

    if zipcode_df.empty:
        raise ValueError("No overlapping zipcodes between stores and centroid file.")

    if stations_df.empty:
        raise ValueError(
            "No usable UBA stations found for the requested date range. "
            "Check the debug meta payload JSON."
        )

    rows: List[Dict[str, object]] = []

    for _, zipcode_row in zipcode_df.iterrows():
        zipcode = zipcode_row["zipcode"]
        zip_lat = float(zipcode_row["lat"])
        zip_lon = float(zipcode_row["lon"])

        stations_work = stations_df.copy()
        stations_work["distance_km"] = stations_work.apply(
            lambda station_row: haversine_distance_km(
                zip_lat,
                zip_lon,
                float(station_row["lat"]),
                float(station_row["lon"]),
            ),
            axis=1,
        )

        nearest = stations_work.sort_values("distance_km").iloc[0]

        rows.append(
            {
                "zipcode": zipcode,
                "station_code": nearest["station_code"],
                "station_name": nearest["station_name"],
                "station_distance_km": float(nearest["distance_km"]),
            }
        )

    return pd.DataFrame(rows).drop_duplicates(subset=["zipcode"]).reset_index(drop=True)


def _extract_timestamp_from_record(record: dict) -> Optional[pd.Timestamp]:
    """
    Extract a timestamp from a measures record if possible.
    """
    timestamp_keys = [
        "date",
        "timestamp",
        "date_end",
        "dateEnd",
        "datum",
    ]

    for key in timestamp_keys:
        if key in record:
            ts = pd.to_datetime(record.get(key), errors="coerce")
            if pd.notna(ts):
                return pd.Timestamp(ts)

    return None


def _extract_value_from_record(record: dict) -> Optional[float]:
    """
    Extract a numeric value from a measures record if possible.
    """
    value_keys = [
        "value",
        "measurement",
        "val",
    ]

    for key in value_keys:
        if key in record:
            value = pd.to_numeric(record.get(key), errors="coerce")
            if pd.notna(value):
                return float(value)

    return None


def parse_measures_payload(
    payload: object,
    station_code: str,
    pollutant_name: str,
) -> pd.DataFrame:
    """
    Parse UBA measures payload into a flat hourly table.
    """
    rows: List[Dict[str, object]] = []

    for record in _flatten_objects(payload):
        timestamp = _extract_timestamp_from_record(record)
        value = _extract_value_from_record(record)

        if timestamp is None or value is None:
            continue

        rows.append(
            {
                "station_code": station_code,
                "pollutant": pollutant_name,
                "timestamp": timestamp,
                "value": value,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(
            columns=[
                "station_code",
                "pollutant",
                "timestamp",
                "value",
            ]
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df.dropna(subset=["timestamp", "value"]).reset_index(drop=True)


def build_air_quality_dataset(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    stores_path: Path,
    centroids_path: Path,
    state_codes: Iterable[str],
    timeout_seconds: int,
    sleep_seconds: float,
    debug_save_payloads: bool = False,
) -> pd.DataFrame:
    """
    Download and assemble hourly air quality measurements for store zipcodes.
    """
    store_zipcodes = load_store_zipcodes(stores_path=stores_path)
    centroids_df = load_plz_centroids(centroids_path=centroids_path)

    with requests.Session() as session:
        session.headers.update(
            {
                "User-Agent": "food_prediction/1.0",
                "Accept": "application/json",
            }
        )

        meta_payload = fetch_json(
            session=session,
            url=meta_url(
                start_date=start_date,
                end_date=end_date,
            ),
            timeout_seconds=timeout_seconds,
        )

        if debug_save_payloads:
            debug_path = (
                get_base_dir()
                / "_external_data"
                / "air_quality_uba"
                / "debug_meta_payload.json"
            )
            ensure_parent_dir(debug_path)
            debug_path.write_text(
                json.dumps(meta_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        stations_df = parse_meta_payload(
            payload=meta_payload,
            allowed_state_codes=state_codes,
        )

        zipcode_station_df = assign_nearest_station_per_zipcode(
            store_zipcodes=store_zipcodes,
            centroids_df=centroids_df,
            stations_df=stations_df,
        )

        all_frames: List[pd.DataFrame] = []

        for _, station_row in zipcode_station_df.iterrows():
            station_code = str(station_row["station_code"])
            zipcode = station_row["zipcode"]

            for pollutant_name, pollutant_cfg in POLLUTANTS.items():
                payload = fetch_json(
                    session=session,
                    url=measures_url(
                        station_code=station_code,
                        component_id=pollutant_cfg["component_id"],
                        scope_id=pollutant_cfg["scope_id"],
                        start_date=start_date,
                        end_date=end_date,
                    ),
                    timeout_seconds=timeout_seconds,
                )

                pollutant_df = parse_measures_payload(
                    payload=payload,
                    station_code=station_code,
                    pollutant_name=pollutant_name,
                )

                if pollutant_df.empty:
                    continue

                pollutant_df["zipcode"] = zipcode
                pollutant_df["station_name"] = station_row["station_name"]
                pollutant_df["station_distance_km"] = station_row["station_distance_km"]

                all_frames.append(pollutant_df)

                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

    if not all_frames:
        return pd.DataFrame(
            columns=[
                "zipcode",
                "station_code",
                "station_name",
                "station_distance_km",
                "pollutant",
                "timestamp",
                "value",
            ]
        )

    df = pd.concat(all_frames, ignore_index=True)
    df = df.drop_duplicates(
        subset=["zipcode", "station_code", "pollutant", "timestamp"]
    ).sort_values(["zipcode", "pollutant", "timestamp"]).reset_index(drop=True)

    return df


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the command line parser.
    """
    parser = argparse.ArgumentParser(
        description="Download historical UBA air quality data using an automatic sales-based date range."
    )
    parser.add_argument(
        "--sales-path",
        type=Path,
        default=get_default_sales_path(),
        help=f"Path to canonical sales parquet. Default: {get_default_sales_path()}",
    )
    parser.add_argument(
        "--stores-path",
        type=Path,
        default=get_repo_root() / "raw_data" / "20260218_144523_stores.parquet",
        help="Path to canonical stores parquet.",
    )
    parser.add_argument(
        "--plz-centroids-path",
        type=Path,
        default=get_base_dir() / "_reference_geo" / "plz_centroids_nrw.csv",
        help="Path to NRW zipcode centroid CSV.",
    )
    parser.add_argument(
        "--date-col",
        default="date",
        help="Date column in the sales parquet.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional manual override for start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional manual override for end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--state-codes",
        nargs="*",
        default=DEFAULT_STATE_CODES,
        help="UBA state codes to query. Default: NW",
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
        help="Sleep between measurement requests.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional explicit output path.",
    )
    parser.add_argument(
        "--debug-save-payloads",
        action="store_true",
        help="Save raw meta payload JSON for debugging.",
    )
    return parser


def main() -> int:
    """
    Run the historical UBA air quality download workflow.
    """
    parser = build_argument_parser()
    args = parser.parse_args()

    start_date, end_date = resolve_date_range(
        start_date=args.start_date,
        end_date=args.end_date,
        sales_path=args.sales_path,
        date_col=args.date_col,
    )

    output_path = (
        Path(args.output_path).resolve()
        if args.output_path is not None
        else get_default_output_path(start_date=start_date, end_date=end_date).resolve()
    )
    ensure_parent_dir(output_path)

    print(
        "[INFO] Resolved air quality date range:",
        format_date_range_for_logs(start_date, end_date),
    )

    df = build_air_quality_dataset(
        start_date=start_date,
        end_date=end_date,
        stores_path=Path(args.stores_path).resolve(),
        centroids_path=Path(args.plz_centroids_path).resolve(),
        state_codes=args.state_codes,
        timeout_seconds=args.timeout_seconds,
        sleep_seconds=args.sleep_seconds,
        debug_save_payloads=args.debug_save_payloads,
    )
    df.to_parquet(output_path, index=False)

    print(f"[OK] Saved {len(df)} rows to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())