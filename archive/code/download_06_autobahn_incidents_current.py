from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import pandas as pd
import requests

API_BASE_URL = "https://verkehr.autobahn.de/o/autobahn"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_SLEEP_SECONDS = 0.10

# Focus is NRW/West Germany relevant motorways.
DEFAULT_ROADS = [
    "A1",
    "A2",
    "A3",
    "A4",
    "A30",
    "A31",
    "A33",
    "A40",
    "A42",
    "A43",
    "A44",
    "A45",
    "A46",
    "A52",
    "A57",
    "A59",
    "A61",
    "A555",
    "A562",
    "A565",
]

SERVICE_TYPES = [
    "warning",
    "closure",
    "roadworks",
]


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


def get_default_output_path() -> Path:
    """
    Return the default parquet output path.
    """
    return (
        get_base_dir()
        / "_external_data"
        / "autobahn_incidents"
        / "autobahn_incidents_current.parquet"
    )


def require_parent_dir(path: Path) -> None:
    """
    Ensure the parent directory exists.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def fetch_json(
    session: requests.Session,
    url: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Any:
    """
    Fetch JSON content from a URL and raise on HTTP errors.
    """
    response = session.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    return response.json()


def normalize_roads(roads: Iterable[str]) -> List[str]:
    """
    Normalize motorway identifiers.
    """
    normalized: List[str] = []

    for road in roads:
        value = str(road).strip().upper()
        if not value:
            continue
        if not value.startswith("A"):
            value = f"A{value}"
        normalized.append(value)

    unique_sorted = sorted(set(normalized), key=lambda x: (len(x), x))
    return unique_sorted


def road_service_url(road: str, service_type: str) -> str:
    """
    Build the service endpoint URL for a motorway and service type.
    """
    return f"{API_BASE_URL}/{road}/services/{service_type}"


def detail_url_from_id(service_type: str, item_id: str) -> str:
    """
    Build the detail endpoint URL from a service type and item id.
    """
    return f"{API_BASE_URL}/details/{service_type}/{item_id}"


def _extract_strings_recursive(payload: Any) -> List[str]:
    """
    Recursively extract all string values from nested JSON content.
    """
    values: List[str] = []

    if isinstance(payload, str):
        values.append(payload)
        return values

    if isinstance(payload, list):
        for item in payload:
            values.extend(_extract_strings_recursive(item))
        return values

    if isinstance(payload, dict):
        for value in payload.values():
            values.extend(_extract_strings_recursive(value))
        return values

    return values


def extract_detail_urls(payload: Any, service_type: str) -> List[str]:
    """
    Extract detail URLs from a service response.

    The Autobahn service endpoints may return nested structures with detail URLs.
    This function is intentionally defensive to avoid fragile assumptions.
    """
    all_strings = _extract_strings_recursive(payload)
    urls = [
        value
        for value in all_strings
        if isinstance(value, str)
        and value.startswith("http")
        and f"/details/{service_type}/" in value
    ]

    return sorted(set(urls))


def flatten_location_fields(location_payload: Any) -> Dict[str, Any]:
    """
    Flatten common location fields from nested location content.
    """
    result: Dict[str, Any] = {
        "latitude": None,
        "longitude": None,
        "direction": None,
        "roadway": None,
        "description": None,
        "extent": None,
    }

    if not isinstance(location_payload, dict):
        return result

    for key in ["direction", "roadway", "description", "extent"]:
        result[key] = location_payload.get(key)

    if "coordinate" in location_payload and isinstance(location_payload["coordinate"], dict):
        result["latitude"] = location_payload["coordinate"].get("lat")
        result["longitude"] = location_payload["coordinate"].get("long")

    if result["latitude"] is None:
        result["latitude"] = location_payload.get("lat")

    if result["longitude"] is None:
        result["longitude"] = location_payload.get("long")

    return result


def parse_detail_payload(
    road: str,
    service_type: str,
    detail_url: str,
    detail_payload: Any,
) -> Dict[str, Any]:
    """
    Convert a raw detail payload into a flat row structure.
    """
    row: Dict[str, Any] = {
        "road": road,
        "service_type": service_type,
        "detail_url": detail_url,
        "id": None,
        "title": None,
        "subtitle": None,
        "description": None,
        "start_timestamp": None,
        "end_timestamp": None,
        "is_blocking": None,
        "source_updated_at": None,
        "latitude": None,
        "longitude": None,
        "direction": None,
        "roadway": None,
        "location_description": None,
        "extent": None,
        "raw_json": json.dumps(detail_payload, ensure_ascii=False),
        "snapshot_timestamp_utc": pd.Timestamp.utcnow(),
    }

    if not isinstance(detail_payload, dict):
        return row

    row["id"] = detail_payload.get("identifier") or detail_payload.get("id")
    row["title"] = detail_payload.get("title")
    row["subtitle"] = detail_payload.get("subtitle")
    row["description"] = detail_payload.get("description")
    row["start_timestamp"] = detail_payload.get("startTimestamp") or detail_payload.get("start")
    row["end_timestamp"] = detail_payload.get("endTimestamp") or detail_payload.get("end")
    row["is_blocking"] = detail_payload.get("isBlocked") or detail_payload.get("blocked")
    row["source_updated_at"] = (
        detail_payload.get("updateTimestamp")
        or detail_payload.get("updated")
        or detail_payload.get("lastModified")
    )

    location_fields = flatten_location_fields(detail_payload.get("location"))
    row["latitude"] = location_fields["latitude"]
    row["longitude"] = location_fields["longitude"]
    row["direction"] = location_fields["direction"]
    row["roadway"] = location_fields["roadway"]
    row["location_description"] = location_fields["description"]
    row["extent"] = location_fields["extent"]

    return row


def fetch_service_rows(
    session: requests.Session,
    road: str,
    service_type: str,
    timeout_seconds: int,
    sleep_seconds: float,
) -> List[Dict[str, Any]]:
    """
    Fetch all detail rows for one road and one service type.
    """
    service_endpoint = road_service_url(road=road, service_type=service_type)
    service_payload = fetch_json(
        session=session,
        url=service_endpoint,
        timeout_seconds=timeout_seconds,
    )

    detail_urls = extract_detail_urls(service_payload, service_type=service_type)
    rows: List[Dict[str, Any]] = []

    for detail_url in detail_urls:
        detail_payload = fetch_json(
            session=session,
            url=detail_url,
            timeout_seconds=timeout_seconds,
        )
        rows.append(
            parse_detail_payload(
                road=road,
                service_type=service_type,
                detail_url=detail_url,
                detail_payload=detail_payload,
            )
        )
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return rows


def build_incidents_dataframe(
    roads: Iterable[str],
    timeout_seconds: int,
    sleep_seconds: float,
) -> pd.DataFrame:
    """
    Download current motorway incident rows for the requested roads.
    """
    normalized_roads = normalize_roads(roads)
    rows: List[Dict[str, Any]] = []

    with requests.Session() as session:
        session.headers.update(
            {
                "User-Agent": "food_prediction/1.0",
                "Accept": "application/json",
            }
        )

        for road in normalized_roads:
            for service_type in SERVICE_TYPES:
                try:
                    service_rows = fetch_service_rows(
                        session=session,
                        road=road,
                        service_type=service_type,
                        timeout_seconds=timeout_seconds,
                        sleep_seconds=sleep_seconds,
                    )
                    rows.extend(service_rows)
                except requests.HTTPError as exc:
                    print(
                        f"[WARN] HTTP error for road={road}, service_type={service_type}: {exc}"
                    )
                except requests.RequestException as exc:
                    print(
                        f"[WARN] Request error for road={road}, service_type={service_type}: {exc}"
                    )
                except Exception as exc:
                    print(
                        f"[WARN] Unexpected error for road={road}, service_type={service_type}: {exc}"
                    )

    if not rows:
        return pd.DataFrame(
            columns=[
                "road",
                "service_type",
                "detail_url",
                "id",
                "title",
                "subtitle",
                "description",
                "start_timestamp",
                "end_timestamp",
                "is_blocking",
                "source_updated_at",
                "latitude",
                "longitude",
                "direction",
                "roadway",
                "location_description",
                "extent",
                "raw_json",
                "snapshot_timestamp_utc",
            ]
        )

    df = pd.DataFrame(rows).drop_duplicates(
        subset=["road", "service_type", "detail_url"]
    ).reset_index(drop=True)

    for col in ["snapshot_timestamp_utc", "start_timestamp", "end_timestamp", "source_updated_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the command line parser.
    """
    repo_root = get_repo_root()
    base_dir = get_base_dir()

    parser = argparse.ArgumentParser(
        description="Download current Autobahn incidents for NRW-relevant roads."
    )
    parser.add_argument(
        "--roads",
        nargs="*",
        default=DEFAULT_ROADS,
        help="Motorways to query, e.g. A1 A3 A40 A57.",
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
        help="Sleep between detail requests to avoid aggressive polling.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=get_default_output_path(),
        help=(
            "Output parquet path. "
            f"Default: {get_default_output_path()}"
        ),
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print a compact summary after saving.",
    )
    parser.epilog = (
        f"repo_root={repo_root}\n"
        f"base_dir={base_dir}"
    )
    return parser


def main() -> int:
    """
    Run the Autobahn incident download workflow.
    """
    parser = build_argument_parser()
    args = parser.parse_args()

    output_path = Path(args.output_path).resolve()
    require_parent_dir(output_path)

    incidents_df = build_incidents_dataframe(
        roads=args.roads,
        timeout_seconds=args.timeout_seconds,
        sleep_seconds=args.sleep_seconds,
    )
    incidents_df.to_parquet(output_path, index=False)

    print(f"[OK] Saved {len(incidents_df)} rows to {output_path}")

    if args.print_summary and not incidents_df.empty:
        summary_df = (
            incidents_df.groupby(["road", "service_type"], dropna=False)
            .size()
            .reset_index(name="n_rows")
            .sort_values(["road", "service_type"])
        )
        print(summary_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())