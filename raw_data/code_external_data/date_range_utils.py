from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


DEFAULT_SALES_FILE_NAME = "20260218_144523_sales_data.parquet"


def get_repo_root() -> Path:
    """
    Return the repository root based on this file location.
    """
    return Path(__file__).resolve().parents[2]


def get_raw_data_dir() -> Path:
    """
    Return the raw_data directory.
    """
    return get_repo_root() / "raw_data"


def get_default_sales_path() -> Path:
    """
    Return the default canonical sales parquet path.
    """
    return get_raw_data_dir() / DEFAULT_SALES_FILE_NAME


def normalize_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    """
    Convert an input value to a normalized pandas Timestamp.
    """
    ts = pd.to_datetime(value, errors="raise")
    return pd.Timestamp(ts).normalize()


def validate_date_column(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Ensure the date column exists and can be parsed as datetime.
    """
    if date_col not in df.columns:
        raise KeyError(f"Missing required date column: '{date_col}'")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()

    if out[date_col].isna().all():
        raise ValueError(f"Column '{date_col}' contains no valid datetime values.")

    return out


def infer_date_range_from_sales(
    sales_path: str | Path | None = None,
    date_col: str = "date",
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Infer min and max date from the canonical sales parquet.
    """
    resolved_sales_path = (
        Path(sales_path) if sales_path is not None else get_default_sales_path()
    )

    if not resolved_sales_path.exists():
        raise FileNotFoundError(f"Sales parquet not found: {resolved_sales_path}")

    sales_df = pd.read_parquet(resolved_sales_path)
    sales_df = validate_date_column(sales_df, date_col=date_col)

    min_date = sales_df[date_col].min()
    max_date = sales_df[date_col].max()

    if pd.isna(min_date) or pd.isna(max_date):
        raise ValueError(
            f"Could not infer a valid date range from '{resolved_sales_path}'."
        )

    return pd.Timestamp(min_date), pd.Timestamp(max_date)


def resolve_date_range(
    start_date: Optional[str | pd.Timestamp] = None,
    end_date: Optional[str | pd.Timestamp] = None,
    sales_path: str | Path | None = None,
    date_col: str = "date",
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Resolve a final date range.

    Rules:
    - if both start_date and end_date are provided, use them
    - if one or both are missing, infer the missing values from sales parquet
    """
    inferred_min_date, inferred_max_date = infer_date_range_from_sales(
        sales_path=sales_path,
        date_col=date_col,
    )

    resolved_start = (
        normalize_timestamp(start_date) if start_date is not None else inferred_min_date
    )
    resolved_end = (
        normalize_timestamp(end_date) if end_date is not None else inferred_max_date
    )

    if resolved_start > resolved_end:
        raise ValueError(
            f"Invalid date range: start_date ({resolved_start.date()}) "
            f"is after end_date ({resolved_end.date()})."
        )

    return resolved_start, resolved_end


def format_date_range_for_logs(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> str:
    """
    Return a compact log string for a resolved date range.
    """
    return f"{start_date.date()} -> {end_date.date()}"


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build a small CLI for quick testing.
    """
    parser = argparse.ArgumentParser(
        description="Infer or resolve a date range from canonical sales data."
    )
    parser.add_argument(
        "--sales-path",
        type=Path,
        default=get_default_sales_path(),
        help=f"Path to sales parquet. Default: {get_default_sales_path()}",
    )
    parser.add_argument(
        "--date-col",
        default="date",
        help="Date column name in the sales parquet.",
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
    return parser


def main() -> int:
    """
    Run a small CLI smoke test.
    """
    parser = build_argument_parser()
    args = parser.parse_args()

    start_date, end_date = resolve_date_range(
        start_date=args.start_date,
        end_date=args.end_date,
        sales_path=args.sales_path,
        date_col=args.date_col,
    )

    print(f"sales_path: {Path(args.sales_path).resolve()}")
    print(f"resolved_date_range: {format_date_range_for_logs(start_date, end_date)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())