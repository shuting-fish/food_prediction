"""Download German school holidays for NRW via ferien-api.de.

Outputs:
  1) ranges table (start_date, end_date, name)
  2) daily table (date, is_school_holiday, holiday_name)

Why two outputs?
- Ranges are the raw truth.
- Daily table is what you typically merge into a daily forecasting dataset.

Usage (from repo root)
---------------------
python raw_data/code_external_data/download_02_school_holidays_ferien_api.py --start 2025-04-01 --end 2025-06-30 --state NW

Dependencies: requests, pandas
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from abs_path_utils import get_base_dir, get_output_dir, require_absolute_path

import requests

API_BASE = "https://ferien-api.de/api/v1"



def _parse_date(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError as e:
        raise SystemExit(f"Invalid date '{s}'. Expected YYYY-MM-DD.") from e


def _iter_years(d1: date, d2: date) -> List[int]:
    return list(range(d1.year, d2.year + 1))


def _http_get_json(url: str, timeout_s: int = 30, max_retries: int = 5) -> Any:
    backoff = 2
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, timeout=timeout_s)
            if r.status_code == 429:
                time.sleep(min(60, backoff))
                backoff *= 2
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == max_retries:
                raise
            time.sleep(min(60, backoff))
            backoff *= 2
    raise RuntimeError("Unreachable")


def _expand_ranges_to_daily(ranges_df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    # Inclusive start/end
    days = pd.date_range(start=start, end=end, freq="D")
    out = pd.DataFrame({"date": days.date.astype(str)})
    out["is_school_holiday"] = False
    out["holiday_name"] = None

    # Apply ranges
    for _, r in ranges_df.iterrows():
        s = _parse_date(str(r["start_date"]))
        e = _parse_date(str(r["end_date"]))
        name = r.get("name")
        mask = (out["date"] >= s.isoformat()) & (out["date"] <= e.isoformat())
        out.loc[mask, "is_school_holiday"] = True
        # Keep last name if overlaps (rare); could also concatenate.
        out.loc[mask, "holiday_name"] = name

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=None, help="ABSOLUTE base directory for scripts + outputs")
    ap.add_argument("--start", default="2025-04-01", help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", default="2025-06-30", help="End date (YYYY-MM-DD)")
    ap.add_argument("--state", default="NW", help="German state code (default: NW for NRW)")
    ap.add_argument("--out-dir", default=None, help="ABSOLUTE output directory. If omitted, a default path under <base-dir> is used.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    base_dir = get_base_dir(args.base_dir)

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if end < start:
        raise SystemExit("--end must be >= --start")

    out_dir = require_absolute_path(args.out_dir, "--out-dir") if args.out_dir else get_output_dir(base_dir, "school_holidays_ferien_api")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_ranges = out_dir / f"school_holidays_ranges_{args.state}_{start.isoformat()}_{end.isoformat()}.parquet"
    out_daily = out_dir / f"school_holidays_daily_{args.state}_{start.isoformat()}_{end.isoformat()}.parquet"

    if (out_ranges.exists() or out_daily.exists()) and not args.force:
        print(f"Outputs exist, skipping (use --force):\n- {out_ranges}\n- {out_daily}")
        return 0

    rows: List[Dict[str, Any]] = []

    # Prefer year-specific endpoint if available: /holidays/{state}/{year}
    for y in _iter_years(start, end):
        url_year = f"{API_BASE}/holidays/{args.state}/{y}"
        data = _http_get_json(url_year)
        if data is None:
            # Fallback: /holidays/{state} returns a list; filter by year
            url_all = f"{API_BASE}/holidays/{args.state}"
            data = _http_get_json(url_all)
        if not isinstance(data, list):
            raise SystemExit(f"Unexpected API response type: {type(data)}")

        for item in data:
            # Typical fields: start, end, name
            s = _parse_date(item.get("start"))
            e = _parse_date(item.get("end"))
            if e < start or s > end:
                continue
            rows.append(
                {
                    "state": args.state,
                    "name": item.get("name"),
                    "start_date": max(s, start).isoformat(),
                    "end_date": min(e, end).isoformat(),
                    "raw_start_date": s.isoformat(),
                    "raw_end_date": e.isoformat(),
                }
            )
        time.sleep(0.2)

    # Deduplicate identical ranges
    df_ranges = pd.DataFrame(rows).drop_duplicates(subset=["state", "name", "raw_start_date", "raw_end_date"]).sort_values(
        ["raw_start_date", "name"]
    )

    df_daily = _expand_ranges_to_daily(df_ranges, start, end)

    # Write
    try:
        df_ranges.to_parquet(out_ranges, index=False)
        df_daily.to_parquet(out_daily, index=False)
    except Exception as e:
        # Parquet requires pyarrow or fastparquet
        out_ranges_csv = out_ranges.with_suffix(".csv")
        out_daily_csv = out_daily.with_suffix(".csv")
        df_ranges.to_csv(out_ranges_csv, index=False)
        df_daily.to_csv(out_daily_csv, index=False)
        print(
            "Parquet write failed (missing pyarrow/fastparquet?). "
            f"Wrote CSV instead:\n- {out_ranges_csv}\n- {out_daily_csv}\nError: {e}",
            file=sys.stderr,
        )
        return 2

    print(f"Wrote:\n- {out_ranges} ({len(df_ranges):,} ranges)\n- {out_daily} ({len(df_daily):,} days)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
