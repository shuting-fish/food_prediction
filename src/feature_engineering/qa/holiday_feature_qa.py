from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering.holiday_features import (  # noqa: E402
    load_holidays_from_parquet,
    merge_holiday_features,
)


FEATURE_COLUMNS = [
    "is_public_holiday",
    "is_school_holiday",
    "is_special_day",
    "is_day_before_public_holiday",
    "is_bridge_day",
]


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


def aggregate_sales_to_store_day(
    sales_df: pd.DataFrame,
    date_col: str = "date",
    store_col: str = "store_id",
    target_col: str = "sold_quantity",
) -> pd.DataFrame:
    """
    Aggregate item-level sales to store-day level.
    """
    required_columns = {date_col, store_col, target_col}
    missing_columns = required_columns - set(sales_df.columns)
    if missing_columns:
        raise KeyError(f"Missing required sales columns: {sorted(missing_columns)}")

    sales_df = _normalize_datetime_column(sales_df, date_col).copy()

    aggregation = {
        target_col: (target_col, "sum"),
        "n_item_rows": (target_col, "size"),
    }

    if "item_id" in sales_df.columns:
        aggregation["n_unique_items"] = ("item_id", "nunique")

    if "price" in sales_df.columns:
        sales_df["revenue_proxy"] = sales_df[target_col] * sales_df["price"]
        aggregation["revenue_proxy"] = ("revenue_proxy", "sum")

    store_day_df = (
        sales_df.groupby([store_col, date_col], as_index=False)
        .agg(**aggregation)
        .sort_values([store_col, date_col])
        .reset_index(drop=True)
    )

    return store_day_df


def add_store_relative_target(
    df: pd.DataFrame,
    store_col: str = "store_id",
    target_col: str = "sold_quantity",
) -> pd.DataFrame:
    """
    Add store-relative target columns to control for scale differences between stores.
    """
    df = df.copy()

    store_median = df.groupby(store_col)[target_col].median()
    store_mean = df.groupby(store_col)[target_col].mean()
    store_std = df.groupby(store_col)[target_col].std().replace(0, np.nan)

    df["store_median_target"] = df[store_col].map(store_median)
    df["store_mean_target"] = df[store_col].map(store_mean)
    df["store_std_target"] = df[store_col].map(store_std)

    df["target_vs_store_median"] = df[target_col] / df["store_median_target"].replace(0, np.nan)
    df["target_store_zscore"] = (
        (df[target_col] - df["store_mean_target"]) / df["store_std_target"]
    )
    df["target_log1p"] = np.log1p(df[target_col])

    return df


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Compute Cliff's delta from two samples via Mann-Whitney U.

    Range:
    -1.0 ... 1.0
    """
    x = pd.Series(x).dropna().to_numpy()
    y = pd.Series(y).dropna().to_numpy()

    if len(x) == 0 or len(y) == 0:
        return np.nan

    x_len = len(x)
    y_len = len(y)

    combined = np.concatenate([x, y])
    ranks = pd.Series(combined).rank(method="average").to_numpy()
    rank_sum_x = ranks[:x_len].sum()

    u_x = rank_sum_x - (x_len * (x_len + 1) / 2)
    delta = (2 * u_x / (x_len * y_len)) - 1

    return float(delta)


def build_feature_test_summary(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_col: str = "sold_quantity",
    relative_target_col: str = "target_vs_store_median",
) -> pd.DataFrame:
    """
    Build a statistical screening summary for each holiday feature.

    This is a screening step, not a causal analysis.
    """
    rows: list[dict[str, float | int | str]] = []

    for feature_col in feature_columns:
        if feature_col not in df.columns:
            raise KeyError(f"Feature column '{feature_col}' not found.")

        flagged = df.loc[df[feature_col] == 1].copy()
        baseline = df.loc[df[feature_col] == 0].copy()

        if flagged.empty or baseline.empty:
            rows.append(
                {
                    "feature": feature_col,
                    "n_flagged": len(flagged),
                    "n_baseline": len(baseline),
                    "share_flagged": len(flagged) / len(df),
                    "median_target_flagged": np.nan,
                    "median_target_baseline": np.nan,
                    "median_ratio_flagged_vs_baseline": np.nan,
                    "mean_target_flagged": np.nan,
                    "mean_target_baseline": np.nan,
                    "mean_ratio_flagged_vs_baseline": np.nan,
                    "median_relative_target_flagged": np.nan,
                    "median_relative_target_baseline": np.nan,
                    "median_relative_ratio_flagged_vs_baseline": np.nan,
                    "mannwhitney_pvalue": np.nan,
                    "cliffs_delta_relative_target": np.nan,
                }
            )
            continue

        x = flagged[relative_target_col]
        y = baseline[relative_target_col]

        _, p_value = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
        effect_size = cliffs_delta(x, y)

        median_target_flagged = flagged[target_col].median()
        median_target_baseline = baseline[target_col].median()
        mean_target_flagged = flagged[target_col].mean()
        mean_target_baseline = baseline[target_col].mean()

        median_relative_flagged = flagged[relative_target_col].median()
        median_relative_baseline = baseline[relative_target_col].median()

        rows.append(
            {
                "feature": feature_col,
                "n_flagged": len(flagged),
                "n_baseline": len(baseline),
                "share_flagged": len(flagged) / len(df),
                "median_target_flagged": median_target_flagged,
                "median_target_baseline": median_target_baseline,
                "median_ratio_flagged_vs_baseline": (
                    median_target_flagged / median_target_baseline
                    if median_target_baseline not in [0, np.nan]
                    else np.nan
                ),
                "mean_target_flagged": mean_target_flagged,
                "mean_target_baseline": mean_target_baseline,
                "mean_ratio_flagged_vs_baseline": (
                    mean_target_flagged / mean_target_baseline
                    if mean_target_baseline not in [0, np.nan]
                    else np.nan
                ),
                "median_relative_target_flagged": median_relative_flagged,
                "median_relative_target_baseline": median_relative_baseline,
                "median_relative_ratio_flagged_vs_baseline": (
                    median_relative_flagged / median_relative_baseline
                    if median_relative_baseline not in [0, np.nan]
                    else np.nan
                ),
                "mannwhitney_pvalue": p_value,
                "cliffs_delta_relative_target": effect_size,
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(
        ["mannwhitney_pvalue", "median_relative_ratio_flagged_vs_baseline"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return summary_df


def build_special_day_name_summary(
    store_day_with_holidays_df: pd.DataFrame,
    holidays_df: pd.DataFrame,
    sales_date_col: str = "date",
    target_col: str = "sold_quantity",
) -> pd.DataFrame:
    """
    Break down special days by holiday_name to avoid treating them as one homogeneous bucket.
    """
    required_columns = {"date", "holiday_name", "holiday_type"}
    missing_columns = required_columns - set(holidays_df.columns)
    if missing_columns:
        raise KeyError(f"Missing required holiday columns: {sorted(missing_columns)}")

    special_days_df = holidays_df.copy()
    special_days_df["holiday_type"] = (
        special_days_df["holiday_type"].astype(str).str.strip().str.casefold()
    )
    special_days_df = special_days_df.loc[special_days_df["holiday_type"] == "special_day"].copy()

    if special_days_df.empty:
        return pd.DataFrame(
            columns=[
                "holiday_name",
                "n_store_days",
                "median_target",
                "median_relative_target",
                "mean_target",
                "mean_relative_target",
            ]
        )

    special_days_df = special_days_df[["date", "holiday_name"]].drop_duplicates()

    merged = store_day_with_holidays_df.merge(
        special_days_df,
        how="left",
        left_on=sales_date_col,
        right_on="date",
        validate="m:1",
    ).drop(columns=["date"])

    special_day_summary = (
        merged.loc[merged["holiday_name"].notna()]
        .groupby("holiday_name", as_index=False)
        .agg(
            n_store_days=(target_col, "size"),
            median_target=(target_col, "median"),
            mean_target=(target_col, "mean"),
            median_relative_target=("target_vs_store_median", "median"),
            mean_relative_target=("target_vs_store_median", "mean"),
        )
        .sort_values(["median_relative_target", "n_store_days"], ascending=[False, False])
        .reset_index(drop=True)
    )

    return special_day_summary


def _build_argument_parser() -> argparse.ArgumentParser:
    """
    Build CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Run store-day holiday feature QA.")
    parser.add_argument("--sales-path", required=True, help="Path to the raw sales parquet file.")
    parser.add_argument("--holidays-path", required=True, help="Path to the holidays parquet file.")
    parser.add_argument(
        "--state-col",
        default="subdivision_code",
        help="Holiday state column for filtering.",
    )
    parser.add_argument(
        "--state-value",
        default="DE-NW",
        help="Holiday state value for filtering.",
    )
    parser.add_argument(
        "--store-day-output-path",
        default=None,
        help="Optional output parquet path for store-day sales with holiday features.",
    )
    parser.add_argument(
        "--summary-output-path",
        default=None,
        help="Optional CSV output path for the holiday feature summary.",
    )
    parser.add_argument(
        "--special-day-output-path",
        default=None,
        help="Optional CSV output path for the special-day name summary.",
    )
    return parser


def main() -> None:
    """
    Run holiday feature QA on store-day aggregated sales.
    """
    parser = _build_argument_parser()
    args = parser.parse_args()

    sales_df = pd.read_parquet(args.sales_path)
    holidays_df = load_holidays_from_parquet(
        parquet_path=args.holidays_path,
        date_col="date",
        state_col=args.state_col,
        state_value=args.state_value,
    )

    store_day_df = aggregate_sales_to_store_day(
        sales_df=sales_df,
        date_col="date",
        store_col="store_id",
        target_col="sold_quantity",
    )

    store_day_with_holidays_df = merge_holiday_features(
        sales_df=store_day_df,
        holidays_df=holidays_df,
        sales_date_col="date",
        holiday_date_col="date",
        holiday_type_col="holiday_type",
        holiday_name_col="holiday_name",
    )

    store_day_with_holidays_df = add_store_relative_target(
        store_day_with_holidays_df,
        store_col="store_id",
        target_col="sold_quantity",
    )

    summary_df = build_feature_test_summary(
        df=store_day_with_holidays_df,
        feature_columns=FEATURE_COLUMNS,
        target_col="sold_quantity",
        relative_target_col="target_vs_store_median",
    )

    special_day_summary_df = build_special_day_name_summary(
        store_day_with_holidays_df=store_day_with_holidays_df,
        holidays_df=holidays_df,
        sales_date_col="date",
        target_col="sold_quantity",
    )

    print("\nStore-day dataset shape:", store_day_with_holidays_df.shape)
    print("\nHoliday feature QA summary:")
    print(summary_df.to_string(index=False))

    print("\nSpecial-day breakdown:")
    if special_day_summary_df.empty:
        print("No special-day rows found.")
    else:
        print(special_day_summary_df.to_string(index=False))

    if args.store_day_output_path:
        output_path = Path(args.store_day_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        store_day_with_holidays_df.to_parquet(output_path, index=False)
        print(f"\nSaved store-day holiday dataset to: {output_path}")

    if args.summary_output_path:
        output_path = Path(args.summary_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_path, index=False)
        print(f"Saved holiday feature summary to: {output_path}")

    if args.special_day_output_path:
        output_path = Path(args.special_day_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        special_day_summary_df.to_csv(output_path, index=False)
        print(f"Saved special-day summary to: {output_path}")


if __name__ == "__main__":
    main()