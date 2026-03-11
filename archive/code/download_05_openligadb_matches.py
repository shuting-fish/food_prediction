from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import requests

from date_range_utils import (
    format_date_range_for_logs,
    get_default_sales_path,
    resolve_date_range,
)


API_BASE_URL = "https://api.openligadb.de"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_LEAGUES = ["bl1"]


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
    file_name = f"openligadb_matches_{start_date.date()}_{end_date.date()}.parquet"
    return get_base_dir() / "_external_data" / "openligadb_matches" / file_name


def ensure_parent_dir(path: Path) -> None:
    """
    Ensure the parent directory exists.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def fetch_json(
    session: requests.Session,
    url: str,
    timeout_seconds: int,
) -> object:
    """
    Fetch JSON payload and raise on HTTP errors.
    """
    response = session.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    return response.json()


def infer_candidate_seasons(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> List[int]:
    """
    Infer candidate league seasons that may overlap the requested date range.

    Example:
    A date range in spring 2025 can still belong to the 2024/2025 season.
    """
    years = set(range(start_date.year - 1, end_date.year + 2))
    return sorted(years)


def league_season_url(league_shortcut: str, season: int) -> str:
    """
    Build the OpenLigaDB season endpoint URL.
    """
    return f"{API_BASE_URL}/getmatchdata/{league_shortcut}/{season}"


def parse_match_rows(
    payload: object,
    league_shortcut: str,
    season: int,
) -> pd.DataFrame:
    """
    Parse OpenLigaDB match payload for one league season.
    """
    if not isinstance(payload, list):
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []

    for item in payload:
        if not isinstance(item, dict):
            continue

        group = item.get("Group") or {}
        team1 = item.get("Team1") or {}
        team2 = item.get("Team2") or {}

        match_results = item.get("MatchResults") or []
        final_result = None

        if isinstance(match_results, list):
            for result in match_results:
                if not isinstance(result, dict):
                    continue
                result_name = str(result.get("ResultName") or "").strip().lower()
                result_type_id = result.get("ResultTypeID")

                if result_type_id == 2 or "end" in result_name or "final" in result_name:
                    final_result = result
                    break

            if final_result is None and match_results:
                first_result = match_results[0]
                if isinstance(first_result, dict):
                    final_result = first_result

        rows.append(
            {
                "league_shortcut": league_shortcut,
                "league_season": season,
                "league_name": item.get("LeagueName"),
                "match_id": item.get("MatchID"),
                "match_date_time_utc": item.get("MatchDateTimeUTC"),
                "match_date_time_local": item.get("MatchDateTime"),
                "match_is_finished": item.get("MatchIsFinished"),
                "group_id": group.get("GroupID"),
                "group_name": group.get("GroupName"),
                "group_order_id": group.get("GroupOrderID"),
                "team1_id": team1.get("TeamId"),
                "team1_name": team1.get("TeamName"),
                "team1_short_name": team1.get("ShortName"),
                "team2_id": team2.get("TeamId"),
                "team2_name": team2.get("TeamName"),
                "team2_short_name": team2.get("ShortName"),
                "location_name": (item.get("Location") or {}).get("LocationStadium"),
                "team1_score_final": (
                    final_result.get("PointsTeam1")
                    if isinstance(final_result, dict)
                    else None
                ),
                "team2_score_final": (
                    final_result.get("PointsTeam2")
                    if isinstance(final_result, dict)
                    else None
                ),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["match_date_time_utc"] = pd.to_datetime(
        df["match_date_time_utc"], errors="coerce", utc=True
    )
    df["match_date_time_local"] = pd.to_datetime(
        df["match_date_time_local"], errors="coerce"
    )
    df["match_date"] = df["match_date_time_local"].dt.normalize()

    return df


def download_openligadb_matches(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    league_shortcuts: Iterable[str],
    timeout_seconds: int,
) -> pd.DataFrame:
    """
    Download season-level match data and filter it to the resolved date range.
    """
    seasons = infer_candidate_seasons(start_date=start_date, end_date=end_date)
    frames: List[pd.DataFrame] = []

    with requests.Session() as session:
        session.headers.update(
            {
                "User-Agent": "food_prediction/1.0",
                "Accept": "application/json",
            }
        )

        for league_shortcut in league_shortcuts:
            for season in seasons:
                payload = fetch_json(
                    session=session,
                    url=league_season_url(league_shortcut=league_shortcut, season=season),
                    timeout_seconds=timeout_seconds,
                )
                season_df = parse_match_rows(
                    payload=payload,
                    league_shortcut=league_shortcut,
                    season=season,
                )
                if not season_df.empty:
                    frames.append(season_df)

    if not frames:
        return pd.DataFrame(
            columns=[
                "league_shortcut",
                "league_season",
                "league_name",
                "match_id",
                "match_date_time_utc",
                "match_date_time_local",
                "match_is_finished",
                "group_id",
                "group_name",
                "group_order_id",
                "team1_id",
                "team1_name",
                "team1_short_name",
                "team2_id",
                "team2_name",
                "team2_short_name",
                "location_name",
                "team1_score_final",
                "team2_score_final",
                "match_date",
            ]
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["league_shortcut", "match_id"]).reset_index(drop=True)

    mask = (df["match_date"] >= start_date) & (df["match_date"] <= end_date)
    df = df.loc[mask].sort_values(["match_date", "league_shortcut", "match_id"]).reset_index(drop=True)

    return df


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the command line parser.
    """
    parser = argparse.ArgumentParser(
        description="Download historical OpenLigaDB matches using an automatic sales-based date range."
    )
    parser.add_argument(
        "--sales-path",
        type=Path,
        default=get_default_sales_path(),
        help=f"Path to canonical sales parquet. Default: {get_default_sales_path()}",
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
        "--league-shortcuts",
        nargs="*",
        default=DEFAULT_LEAGUES,
        help="League shortcuts, e.g. bl1 bl2 bl3 dfb.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional explicit output path.",
    )
    return parser


def main() -> int:
    """
    Run the historical OpenLigaDB download workflow.
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
        "[INFO] Resolved OpenLigaDB date range:",
        format_date_range_for_logs(start_date, end_date),
    )

    df = download_openligadb_matches(
        start_date=start_date,
        end_date=end_date,
        league_shortcuts=args.league_shortcuts,
        timeout_seconds=args.timeout_seconds,
    )
    df.to_parquet(output_path, index=False)

    print(f"[OK] Saved {len(df)} rows to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())