"""Download football match schedule/results from OpenLigaDB and produce DAILY aggregates.

Why daily?
- Your sales/production targets are daily.
- Event signals must be aligned to the same temporal resolution to be mergeable later.

Outputs (parquet/csv)
---------------------
1) Daily overall table:
   - date
   - n_matches_total
   - n_matches_finished
   - n_matches_nrw_involved

2) Daily-by-city table:
   - date
   - location_city
   - n_matches_total
   - n_matches_finished
   - n_matches_nrw_involved

Optional (debug): match-level table.

Notes
-----
OpenLigaDB's "season" semantics can be confusing depending on league/provider (start year vs end year).
To reduce manual trial-and-error, the script automatically tries adjacent seasons if the requested
season yields no matches in the date window.

Dependencies: requests, pandas
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from abs_path_utils import get_base_dir, get_output_dir, require_absolute_path

API_BASE = "https://api.openligadb.de"

# Heuristic list: team names change over time. We keep this simple.
NRW_TEAM_SUBSTRINGS = [
    "Köln",
    "Koeln",
    "Dortmund",
    "Leverkusen",
    "Mönchengladbach",
    "Moenchengladbach",
    "Bochum",
    "Gelsenkirchen",  # Schalke naming sometimes
    "Düsseldorf",
    "Duesseldorf",
    "Paderborn",
    "Bielefeld",
]

DEFAULT_START = "2025-04-01"
DEFAULT_END = "2025-06-30"


def _parse_date(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError as e:
        raise SystemExit(f"Invalid date '{s}'. Expected YYYY-MM-DD.") from e


def _http_get_json(url: str, timeout_s: int = 60, max_retries: int = 4) -> Any:
    """HTTP GET with small retry/backoff.

    This is defensive: public APIs occasionally throttle (429) or have transient 5xx.
    """
    backoff = 2
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, timeout=timeout_s)
            if r.status_code in (429, 502, 503, 504):
                time.sleep(min(60, backoff))
                backoff *= 2
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == max_retries:
                raise
            time.sleep(min(60, backoff))
            backoff *= 2
    raise RuntimeError("Unreachable")


def _is_nrw_team(name: str) -> bool:
    n = (name or "").lower()
    return any(sub.lower() in n for sub in NRW_TEAM_SUBSTRINGS)


def _is_nrw_match(home: str, away: str) -> bool:
    return _is_nrw_team(home) or _is_nrw_team(away)


def _to_iso(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    try:
        return pd.to_datetime(s).isoformat()
    except Exception:
        return s


def _extract_rows(
    data: Any,
    *,
    league: str,
    season: int,
    start: date,
    end: date,
    nrw_only: bool,
) -> Tuple[List[Dict[str, Any]], Optional[date], Optional[date]]:
    """Parse JSON into match rows and also return the available date range."""
    if not isinstance(data, list):
        return [], None, None

    rows: List[Dict[str, Any]] = []
    all_dates: List[date] = []

    for m in data:
        t1 = m.get("Team1")
        t2 = m.get("Team2")
        home = (t1 or {}).get("TeamName") if isinstance(t1, dict) else None
        away = (t2 or {}).get("TeamName") if isinstance(t2, dict) else None

        match_dt = _to_iso(m.get("MatchDateTime"))
        if not match_dt:
            continue
        match_date = pd.to_datetime(match_dt).date()
        all_dates.append(match_date)

        # Filter to the requested window.
        if match_date < start or match_date > end:
            continue

        is_nrw = _is_nrw_match(home or "", away or "")
        if nrw_only and not is_nrw:
            continue

        loc = m.get("Location") if isinstance(m.get("Location"), dict) else {}
        rows.append(
            {
                "match_id": m.get("MatchID"),
                "league": league,
                "season": str(season),
                "match_datetime": match_dt,
                "date": match_date.isoformat(),
                "home_team": home,
                "away_team": away,
                "match_is_finished": m.get("MatchIsFinished"),
                "location_city": (loc or {}).get("LocationCity"),
                "location_stadium": (loc or {}).get("LocationStadium"),
                "is_nrw_involved": bool(is_nrw),
                "last_update": m.get("LastUpdateDateTime"),
            }
        )

    min_d = min(all_dates) if all_dates else None
    max_d = max(all_dates) if all_dates else None
    return rows, min_d, max_d


def _daily_agg(df_matches: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate match-level rows into daily tables."""
    overall = (
        df_matches.groupby("date")
        .agg(
            n_matches_total=("match_id", "count"),
            n_matches_finished=("match_is_finished", lambda x: int(pd.Series(x).fillna(False).astype(bool).sum())),
            n_matches_nrw_involved=("is_nrw_involved", lambda x: int(pd.Series(x).fillna(False).astype(bool).sum())),
        )
        .reset_index()
        .sort_values("date")
    )

    city = (
        df_matches.groupby(["date", "location_city"], dropna=False)
        .agg(
            n_matches_total=("match_id", "count"),
            n_matches_finished=("match_is_finished", lambda x: int(pd.Series(x).fillna(False).astype(bool).sum())),
            n_matches_nrw_involved=("is_nrw_involved", lambda x: int(pd.Series(x).fillna(False).astype(bool).sum())),
        )
        .reset_index()
        .sort_values(["date", "location_city"])
    )

    return overall, city


def _write_parquet_or_csv(df: pd.DataFrame, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        df.to_csv(out_path, index=False)
        return 0
    try:
        df.to_parquet(out_path, index=False)
        return 0
    except Exception as e:
        csv_fallback = out_path.with_suffix(".csv")
        df.to_csv(csv_fallback, index=False)
        print(
            "Parquet write failed (missing pyarrow/fastparquet?). "
            f"Wrote CSV instead: {csv_fallback}\nError: {e}",
            file=sys.stderr,
        )
        return 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=None, help="ABSOLUTE base directory for scripts + outputs")
    ap.add_argument("--league", default="bl1", help="League shortcut (default: bl1)")
    ap.add_argument(
        "--season",
        required=True,
        type=int,
        help=(
            "Season year (e.g., 2024). The script will auto-try adjacent seasons if needed "
            "because providers may label seasons by start or end year."
        ),
    )
    ap.add_argument("--start", default=DEFAULT_START, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", default=DEFAULT_END, help="End date (YYYY-MM-DD)")
    ap.add_argument("--nrw-only", action="store_true", help="Keep only matches involving NRW teams (name heuristic)")
    ap.add_argument("--write-matches", action="store_true", help="Also write match-level table (debug/traceability)")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="ABSOLUTE output directory. If omitted, a default path under <base-dir> is used.",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if end < start:
        raise SystemExit("--end must be >= --start")

    base_dir = get_base_dir(args.base_dir)
    out_dir = require_absolute_path(args.out_dir, "--out-dir") if args.out_dir else get_output_dir(base_dir, "openligadb")
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "nrw" if args.nrw_only else "all"
    out_overall = out_dir / f"openligadb_daily_overall_{args.league}_{args.season}_{suffix}_{start.isoformat()}_{end.isoformat()}.parquet"
    out_city = out_dir / f"openligadb_daily_city_{args.league}_{args.season}_{suffix}_{start.isoformat()}_{end.isoformat()}.parquet"
    out_matches = out_dir / f"openligadb_matches_{args.league}_{args.season}_{suffix}_{start.isoformat()}_{end.isoformat()}.parquet"

    if not args.force:
        if out_overall.exists() and out_city.exists() and (not args.write_matches or out_matches.exists()):
            print("Outputs exist, skipping (use --force).")
            return 0

    seasons_to_try = [args.season, args.season + 1, args.season - 1]
    chosen_season: Optional[int] = None
    chosen_rows: List[Dict[str, Any]] = []
    diag_ranges: Dict[int, Tuple[Optional[date], Optional[date]]] = {}

    for season in seasons_to_try:
        url = f"{API_BASE}/getmatchdata/{args.league}/{season}"
        data = _http_get_json(url)
        rows, min_d, max_d = _extract_rows(data, league=args.league, season=season, start=start, end=end, nrw_only=args.nrw_only)
        diag_ranges[season] = (min_d, max_d)
        if rows:
            chosen_season = season
            chosen_rows = rows
            break

    if chosen_season is None or not chosen_rows:
        # Print diagnostics so you can see which seasons overlap your date window.
        parts = []
        for s, (mn, mx) in diag_ranges.items():
            parts.append(f"season={s}: available={mn}..{mx}")
        print(
            "No matches found in requested window. "
            "Try a different --season (e.g., 2024 vs 2025). "
            f"Diagnostics: {', '.join(parts)}",
            file=sys.stderr,
        )
        return 1

    if chosen_season != args.season:
        print(f"NOTE: No matches for season={args.season}. Used season={chosen_season} instead.")

    df = pd.DataFrame(chosen_rows).sort_values(["date", "match_datetime"]).reset_index(drop=True)
    df_overall, df_city = _daily_agg(df)

    rc1 = _write_parquet_or_csv(df_overall, out_overall)
    rc2 = _write_parquet_or_csv(df_city, out_city)
    rc3 = 0
    if args.write_matches:
        rc3 = _write_parquet_or_csv(df, out_matches)

    print("Wrote:")
    print(f"- {out_overall} ({len(df_overall):,} rows)")
    print(f"- {out_city} ({len(df_city):,} rows)")
    if args.write_matches:
        print(f"- {out_matches} ({len(df):,} matches)")

    return max(rc1, rc2, rc3)


if __name__ == "__main__":
    raise SystemExit(main())
