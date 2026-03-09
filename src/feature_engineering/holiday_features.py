from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


PUBLIC_HOLIDAY_TYPE = "holiday"
SCHOOL_HOLIDAY_TYPE = "school_holiday"
SPECIAL_DAY_TYPE = "special_day"
BRIDGE_DAY_NAME = "brückentag"


def _normalize_datetime_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Normalize a date column to midnight timestamps.
    """
    if column not in df.columns:
        raise KeyError(f"Missing required column: '{column}'")

    df = df.copy()
    df[column] = pd.to_datetime(df[column], errors="coerce").dt.normalize()

    if df[column].isna().all():
        raise ValueError(f"Column '{column}' could not be parsed to valid datetimes.")

    return df


def _normalize_text(value: object) -> str:
    """
    Normalize text for robust comparisons.
    """
    if pd.isna(value):
        return ""
    return str(value).strip().casefold()


def load_holidays_from_parquet(
    parquet_path: str | Path,
    date_col: str = "date",
    state_col: Optional[str] = None,
    state_value: Optional[str] = None,
    zipcode_col: Optional[str] = None,
    zipcode_values: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Load holiday data from a parquet file and optionally filter by state and zip code.

    Expected minimum:
    - a date column

    Recommended columns for this project:
    - holiday_type
    - holiday_name
    - subdivision_code
    - zipcode
    """
    parquet_path = Path(parquet_path)

    if not parquet_path.exists():
        raise FileNotFoundError(f"Holiday parquet not found: {parquet_path}")

    holidays_df = pd.read_parquet(parquet_path)
    holidays_df = _normalize_datetime_column(holidays_df, date_col)

    if state_col is not None and state_value is not None:
        if state_col not in holidays_df.columns:
            raise KeyError(f"State column '{state_col}' not found in holiday file.")
        holidays_df = holidays_df.loc[holidays_df[state_col] == state_value].copy()

    if zipcode_col is not None and zipcode_values is not None:
        if zipcode_col not in holidays_df.columns:
            raise KeyError(f"Zip code column '{zipcode_col}' not found in holiday file.")
        zipcode_values = {str(value) for value in zipcode_values}
        holidays_df = holidays_df.loc[holidays_df[zipcode_col].astype(str).isin(zipcode_values)].copy()

    holidays_df = holidays_df.dropna(subset=[date_col]).reset_index(drop=True)

    if holidays_df.empty:
        raise ValueError("No holiday rows available after filtering.")

    return holidays_df


def _unique_dates(series: pd.Series) -> pd.DatetimeIndex:
    """
    Convert a datetime-like series to a unique, sorted DatetimeIndex.
    """
    values = pd.to_datetime(series.dropna().unique(), errors="coerce")
    values = pd.Series(values).dropna().sort_values().unique()
    return pd.DatetimeIndex(values)


def _compute_bridge_days(public_holiday_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Compute bridge days from public holidays.

    A bridge day is defined as a non-weekend workday between a public holiday and a weekend.
    """
    holiday_set = set(public_holiday_dates)
    bridge_days: list[pd.Timestamp] = []

    for holiday_date in public_holiday_dates:
        next_day = holiday_date + pd.Timedelta(days=1)
        prev_day = holiday_date - pd.Timedelta(days=1)

        if next_day.weekday() < 5 and (next_day + pd.Timedelta(days=1)).weekday() >= 5:
            if next_day not in holiday_set:
                bridge_days.append(next_day)

        if prev_day.weekday() < 5 and (prev_day - pd.Timedelta(days=1)).weekday() >= 5:
            if prev_day not in holiday_set:
                bridge_days.append(prev_day)

    if not bridge_days:
        return pd.DatetimeIndex([])

    return pd.DatetimeIndex(sorted(pd.Series(bridge_days).drop_duplicates().tolist()))


def build_holiday_feature_table(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    holidays_df: pd.DataFrame,
    holiday_date_col: str = "date",
    holiday_type_col: str = "holiday_type",
    holiday_name_col: str = "holiday_name",
) -> pd.DataFrame:
    """
    Build a daily holiday feature table for a full date range.

    Features:
    - is_public_holiday
    - is_school_holiday
    - is_special_day
    - is_day_before_public_holiday
    - is_bridge_day
    """
    required_columns = [holiday_date_col]
    for column in required_columns:
        if column not in holidays_df.columns:
            raise KeyError(f"Required holiday column '{column}' not found.")

    start_date = pd.to_datetime(start_date).normalize()
    end_date = pd.to_datetime(end_date).normalize()

    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")

    work_df = holidays_df.copy()
    work_df = _normalize_datetime_column(work_df, holiday_date_col)

    if holiday_type_col in work_df.columns:
        work_df[holiday_type_col] = work_df[holiday_type_col].map(_normalize_text)
    else:
        work_df[holiday_type_col] = ""

    if holiday_name_col in work_df.columns:
        work_df[holiday_name_col] = work_df[holiday_name_col].map(_normalize_text)
    else:
        work_df[holiday_name_col] = ""

    public_holiday_dates = _unique_dates(
        work_df.loc[work_df[holiday_type_col] == PUBLIC_HOLIDAY_TYPE, holiday_date_col]
    )
    school_holiday_dates = _unique_dates(
        work_df.loc[work_df[holiday_type_col] == SCHOOL_HOLIDAY_TYPE, holiday_date_col]
    )
    special_day_dates = _unique_dates(
        work_df.loc[work_df[holiday_type_col] == SPECIAL_DAY_TYPE, holiday_date_col]
    )
    provided_bridge_dates = _unique_dates(
        work_df.loc[work_df[holiday_name_col] == BRIDGE_DAY_NAME, holiday_date_col]
    )

    computed_bridge_dates = _compute_bridge_days(public_holiday_dates)
    bridge_dates = provided_bridge_dates.union(computed_bridge_dates)
    day_before_public_holiday_dates = public_holiday_dates - pd.Timedelta(days=1)

    calendar = pd.DataFrame(
        {"date": pd.date_range(start=start_date, end=end_date, freq="D")}
    )

    calendar["is_public_holiday"] = calendar["date"].isin(public_holiday_dates).astype("int8")
    calendar["is_school_holiday"] = calendar["date"].isin(school_holiday_dates).astype("int8")
    calendar["is_special_day"] = calendar["date"].isin(special_day_dates).astype("int8")
    calendar["is_day_before_public_holiday"] = (
        calendar["date"].isin(day_before_public_holiday_dates).astype("int8")
    )
    calendar["is_bridge_day"] = calendar["date"].isin(bridge_dates).astype("int8")

    return calendar[
        [
            "date",
            "is_public_holiday",
            "is_school_holiday",
            "is_special_day",
            "is_day_before_public_holiday",
            "is_bridge_day",
        ]
    ].copy()


def merge_holiday_features(
    sales_df: pd.DataFrame,
    holidays_df: pd.DataFrame,
    sales_date_col: str = "date",
    holiday_date_col: str = "date",
    holiday_type_col: str = "holiday_type",
    holiday_name_col: str = "holiday_name",
) -> pd.DataFrame:
    """
    Merge holiday features into a sales dataframe.
    """
    sales_df = _normalize_datetime_column(sales_df, sales_date_col).copy()

    feature_table = build_holiday_feature_table(
        start_date=sales_df[sales_date_col].min(),
        end_date=sales_df[sales_date_col].max(),
        holidays_df=holidays_df,
        holiday_date_col=holiday_date_col,
        holiday_type_col=holiday_type_col,
        holiday_name_col=holiday_name_col,
    )

    feature_table = feature_table.rename(columns={"date": "__holiday_date"})

    merged = sales_df.merge(
        feature_table,
        how="left",
        left_on=sales_date_col,
        right_on="__holiday_date",
        validate="m:1",
    )

    merged = merged.drop(columns=["__holiday_date"])

    feature_columns = [
        "is_public_holiday",
        "is_school_holiday",
        "is_special_day",
        "is_day_before_public_holiday",
        "is_bridge_day",
    ]
    for column in feature_columns:
        merged[column] = merged[column].fillna(0).astype("int8")

    return merged


def validate_holiday_coverage(
    sales_df: pd.DataFrame,
    holidays_df: pd.DataFrame,
    sales_date_col: str = "date",
    holiday_date_col: str = "date",
    holiday_type_col: str = "holiday_type",
    holiday_name_col: str = "holiday_name",
) -> pd.DataFrame:
    """
    Return a compact validation table for dates flagged as holiday-related.
    Useful for QA after merging.
    """
    merged = merge_holiday_features(
        sales_df=sales_df,
        holidays_df=holidays_df,
        sales_date_col=sales_date_col,
        holiday_date_col=holiday_date_col,
        holiday_type_col=holiday_type_col,
        holiday_name_col=holiday_name_col,
    )

    feature_columns = [
        "is_public_holiday",
        "is_school_holiday",
        "is_special_day",
        "is_day_before_public_holiday",
        "is_bridge_day",
    ]

    qa_view = merged.loc[
        merged[feature_columns].any(axis=1),
        [sales_date_col, *feature_columns],
    ].drop_duplicates().sort_values(sales_date_col)

    return qa_view.reset_index(drop=True)


def _build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser.
    """
    parser = argparse.ArgumentParser(description="Build and validate holiday features.")
    parser.add_argument("--sales-path", required=True, help="Path to the sales parquet file.")
    parser.add_argument("--holidays-path", required=True, help="Path to the holidays parquet file.")
    parser.add_argument("--sales-date-col", default="date", help="Sales date column name.")
    parser.add_argument("--holiday-date-col", default="date", help="Holiday date column name.")
    parser.add_argument("--holiday-type-col", default="holiday_type", help="Holiday type column name.")
    parser.add_argument("--holiday-name-col", default="holiday_name", help="Holiday name column name.")
    parser.add_argument("--state-col", default="subdivision_code", help="Optional state column for filtering.")
    parser.add_argument("--state-value", default="DE-NW", help="Optional state value for filtering.")
    parser.add_argument("--output-path", default=None, help="Optional output parquet path.")
    parser.add_argument("--print-rows", type=int, default=30, help="Number of QA rows to print.")
    return parser


def main() -> None:
    """
    Run a simple smoke test from the command line.
    """
    parser = _build_argument_parser()
    args = parser.parse_args()

    sales_df = pd.read_parquet(args.sales_path)
    holidays_df = load_holidays_from_parquet(
        parquet_path=args.holidays_path,
        date_col=args.holiday_date_col,
        state_col=args.state_col,
        state_value=args.state_value,
    )

    merged = merge_holiday_features(
        sales_df=sales_df,
        holidays_df=holidays_df,
        sales_date_col=args.sales_date_col,
        holiday_date_col=args.holiday_date_col,
        holiday_type_col=args.holiday_type_col,
        holiday_name_col=args.holiday_name_col,
    )
    qa_view = validate_holiday_coverage(
        sales_df=sales_df,
        holidays_df=holidays_df,
        sales_date_col=args.sales_date_col,
        holiday_date_col=args.holiday_date_col,
        holiday_type_col=args.holiday_type_col,
        holiday_name_col=args.holiday_name_col,
    )

    print("Merged dataframe shape:", merged.shape)
    print("\nQA holiday coverage preview:")
    print(qa_view.head(args.print_rows).to_string(index=False))

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(output_path, index=False)
        print(f"\nSaved merged dataset to: {output_path}")


if __name__ == "__main__":
    main()