"""Download German public holidays (incl. NRW relevance) via Nager.Date.

Outputs a tidy table suitable for later merging:
  - date (YYYY-MM-DD)
  - local_name, name
  - global (bool)
  - counties (list or null)
  - is_holiday_nrw (bool)

Notes
-----
- Nager.Date is public and free, but may enforce rate limits.
- Germany has state-specific holidays. We mark NRW via county code 'DE-NW' where provided.

Usage (from repo root)
---------------------
python raw_data/code_external_data/download_01_holidays_nagerdate.py --start 2025-04-01 --end 2025-06-30

Dependencies: requests, pandas
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from abs_path_utils import get_base_dir, get_output_dir, require_absolute_path

import requests


DEFAULT_COUNTRY = "DE"
DEFAULT_STATE_COUNTY = "DE-NW"  # Nager.Date county code for NRW
API_BASE = "https://date.nager.at/api/v3"



def _parse_date(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError as e:
        raise SystemExit(f"Invalid date '{s}'. Expected YYYY-MM-DD.") from e


def _iter_years(d1: date, d2: date) -> List[int]:
    y1, y2 = d1.year, d2.year
    return list(range(y1, y2 + 1))


def _http_get_json(url: str, timeout_s: int = 30, max_retries: int = 5) -> Any:
    backoff = 2
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, timeout=timeout_s)
            if r.status_code == 429:
                # Rate limited
                sleep_s = min(60, backoff)
                time.sleep(sleep_s)
                backoff *= 2
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == max_retries:
                raise
            time.sleep(min(60, backoff))
            backoff *= 2
    raise RuntimeError("Unreachable")


def _nrw_flag(global_holiday: bool, counties: Optional[List[str]]) -> bool:
    if global_holiday:
        return True
    if not counties:
        return False
    return DEFAULT_STATE_COUNTY in counties


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=None, help="ABSOLUTE base directory for scripts + outputs")
    ap.add_argument("--start", default="2025-04-01", help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", default="2025-06-30", help="End date (YYYY-MM-DD)")
    ap.add_argument("--country", default=DEFAULT_COUNTRY, help="Country code (default: DE)")
    ap.add_argument(
        "--out",
        default=None,
        help="ABSOLUTE output FILE path (parquet/csv). If omitted, a default path under <base-dir> is used.",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help=(
            "ABSOLUTE output DIRECTORY. If provided, the script writes its default file name into this directory. "
            "This is convenient when you want all sources under one folder."
        ),
    )
    ap.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = ap.parse_args()

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if end < start:
        raise SystemExit("--end must be >= --start")

    base_dir = get_base_dir(args.base_dir)

    out_path: Path
    if args.out:
        out_path = require_absolute_path(args.out, "--out")
    elif args.out_dir:
        out_dir = require_absolute_path(args.out_dir, "--out-dir")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = (out_dir / f"holidays_{start.isoformat()}_{end.isoformat()}.parquet").resolve()
    else:
        out_dir = get_output_dir(base_dir, "holidays_nagerdate")
        out_path = (out_dir / f"holidays_{start.isoformat()}_{end.isoformat()}.parquet").resolve()

    if out_path.exists() and not args.force:
        print(f"Output exists, skipping: {out_path}")
        return 0

    rows: List[Dict[str, Any]] = []
    for y in _iter_years(start, end):
        url = f"{API_BASE}/PublicHolidays/{y}/{args.country}"
        data = _http_get_json(url)
        if not isinstance(data, list):
            raise SystemExit(f"Unexpected API response for year {y}: {type(data)}")
        for item in data:
            d = _parse_date(item["date"])  # type: ignore[index]
            if d < start or d > end:
                continue
            counties = item.get("counties")
            rows.append(
                {
                    "date": d.isoformat(),
                    "local_name": item.get("localName"),
                    "name": item.get("name"),
                    "country_code": item.get("countryCode"),
                    "fixed": bool(item.get("fixed")),
                    "global": bool(item.get("global")),
                    "counties": counties,
                    "launch_year": item.get("launchYear"),
                    "types": item.get("types"),
                    "is_holiday_nrw": _nrw_flag(bool(item.get("global")), counties),
                }
            )
        time.sleep(0.2)  # Be polite

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        df.to_csv(out_path, index=False)
    else:
        try:
            df.to_parquet(out_path, index=False)
        except Exception as e:
            # Parquet requires pyarrow or fastparquet
            csv_fallback = out_path.with_suffix(".csv")
            df.to_csv(csv_fallback, index=False)
            print(
                "Parquet write failed (missing pyarrow/fastparquet?). "
                f"Wrote CSV instead: {csv_fallback}\nError: {e}",
                file=sys.stderr,
            )
            return 2

    print(f"Wrote: {out_path} ({len(df):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
