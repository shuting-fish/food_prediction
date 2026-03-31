from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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

    store_day_df = (
        sales_df.groupby([store_col, date_col], as_index=False)
        .agg(**aggregation)
        .sort_values([store_col, date_col])
        .reset_index(drop=True)
    )

    return store_day_df


def validate_hourly_weather_coverage(
    weather_df: pd.DataFrame,
    date_col: str = "date",
    zipcode_col: str = "zipcode",
) -> pd.DataFrame:
    """
    Validate hourly weather row counts per zipcode-date.
    """
    weather_df = _normalize_datetime_column(weather_df, date_col).copy()

    coverage_df = (
        weather_df.groupby([zipcode_col, date_col], as_index=False)
        .size()
        .rename(columns={"size": "hourly_rows"})
        .sort_values([zipcode_col, date_col])
        .reset_index(drop=True)
    )

    return coverage_df


def _safe_mode(series: pd.Series) -> float:
    """
    Return the first mode if available.
    """
    mode = series.mode(dropna=True)
    if mode.empty:
        return np.nan
    return mode.iloc[0]


def aggregate_weather_to_daily(
    weather_df: pd.DataFrame,
    date_col: str = "date",
    zipcode_col: str = "zipcode",
) -> pd.DataFrame:
    """
    Aggregate hourly weather to zipcode-day level.

    Only stable, mostly numeric daily features are generated here.
    """
    required_columns = {
        date_col,
        zipcode_col,
        "temperature",
        "feelslike",
        "precip",
        "humidity",
        "wind_speed",
        "windgust",
        "cloudcover",
        "pressure",
        "visibility",
        "uv_index",
        "weather_code",
    }
    missing_columns = required_columns - set(weather_df.columns)
    if missing_columns:
        raise KeyError(f"Missing required weather columns: {sorted(missing_columns)}")

    weather_df = _normalize_datetime_column(weather_df, date_col).copy()
    weather_df[zipcode_col] = weather_df[zipcode_col].astype(str)

    weather_df["is_rain_hour"] = (weather_df["precip"] > 0).astype("int8")
    weather_df["is_heavy_rain_hour"] = (weather_df["precip"] >= 1.0).astype("int8")
    weather_df["is_frost_hour"] = (weather_df["temperature"] <= 0).astype("int8")

    daily_weather_df = (
        weather_df.groupby([zipcode_col, date_col], as_index=False)
        .agg(
            temperature_mean=("temperature", "mean"),
            temperature_min=("temperature", "min"),
            temperature_max=("temperature", "max"),
            feelslike_mean=("feelslike", "mean"),
            feelslike_min=("feelslike", "min"),
            feelslike_max=("feelslike", "max"),
            precip_sum=("precip", "sum"),
            precip_max=("precip", "max"),
            rain_hours=("is_rain_hour", "sum"),
            heavy_rain_hours=("is_heavy_rain_hour", "sum"),
            frost_hours=("is_frost_hour", "sum"),
            humidity_mean=("humidity", "mean"),
            humidity_max=("humidity", "max"),
            wind_speed_mean=("wind_speed", "mean"),
            wind_speed_max=("wind_speed", "max"),
            windgust_max=("windgust", "max"),
            cloudcover_mean=("cloudcover", "mean"),
            cloudcover_max=("cloudcover", "max"),
            pressure_mean=("pressure", "mean"),
            pressure_min=("pressure", "min"),
            pressure_max=("pressure", "max"),
            visibility_mean=("visibility", "mean"),
            visibility_min=("visibility", "min"),
            uv_index_mean=("uv_index", "mean"),
            uv_index_max=("uv_index", "max"),
            dominant_weather_code=("weather_code", _safe_mode),
            hourly_rows=("weather_code", "size"),
        )
        .sort_values([zipcode_col, date_col])
        .reset_index(drop=True)
    )

    daily_weather_df["temperature_range"] = (
        daily_weather_df["temperature_max"] - daily_weather_df["temperature_min"]
    )
    daily_weather_df["pressure_range"] = (
        daily_weather_df["pressure_max"] - daily_weather_df["pressure_min"]
    )
    daily_weather_df["rain_day_flag"] = (daily_weather_df["precip_sum"] > 0).astype("int8")
    daily_weather_df["heavy_rain_day_flag"] = (
        daily_weather_df["heavy_rain_hours"] > 0
    ).astype("int8")
    daily_weather_df["frost_day_flag"] = (daily_weather_df["frost_hours"] > 0).astype("int8")

    return daily_weather_df


def merge_daily_weather_to_base_dataset(
    base_df: pd.DataFrame,
    stores_df: pd.DataFrame,
    daily_weather_df: pd.DataFrame,
    store_col: str = "store_id",
    date_col: str = "date",
    zipcode_col: str = "zipcode",
) -> pd.DataFrame:
    """
    Merge zipcode-day weather onto a store-day base dataset.
    """
    required_base_columns = {store_col, date_col}
    missing_base_columns = required_base_columns - set(base_df.columns)
    if missing_base_columns:
        raise KeyError(f"Missing required base dataset columns: {sorted(missing_base_columns)}")

    required_store_columns = {store_col, zipcode_col}
    missing_store_columns = required_store_columns - set(stores_df.columns)
    if missing_store_columns:
        raise KeyError(f"Missing required store columns: {sorted(missing_store_columns)}")

    base_df = _normalize_datetime_column(base_df, date_col).copy()
    stores_df = stores_df.copy()
    stores_df[zipcode_col] = stores_df[zipcode_col].astype(str)

    merged_df = base_df.merge(
        stores_df[[store_col, zipcode_col]].drop_duplicates(),
        how="left",
        on=store_col,
        validate="m:1",
    )

    if merged_df[zipcode_col].isna().any():
        missing_store_ids = merged_df.loc[merged_df[zipcode_col].isna(), store_col].drop_duplicates()
        raise ValueError(
            f"Missing zipcode mapping for store_ids: {missing_store_ids.tolist()}"
        )

    merged_df = merged_df.merge(
        daily_weather_df,
        how="left",
        on=[zipcode_col, date_col],
        validate="m:1",
    )

    return merged_df


def _build_argument_parser() -> argparse.ArgumentParser:
    """
    Build CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Aggregate hourly weather to daily features.")
    parser.add_argument("--weather-path", required=True, help="Path to hourly weather parquet.")
    parser.add_argument("--stores-path", required=True, help="Path to stores parquet.")
    parser.add_argument(
        "--base-dataset-path",
        default=None,
        help="Optional path to a store-day base parquet to enrich with weather.",
    )
    parser.add_argument(
        "--sales-path",
        default=None,
        help="Optional raw sales parquet. Used only if base-dataset-path is not provided.",
    )
    parser.add_argument(
        "--daily-weather-output-path",
        default=None,
        help="Optional output path for zipcode-day weather features.",
    )
    parser.add_argument(
        "--merged-output-path",
        default=None,
        help="Optional output path for the base dataset enriched with weather.",
    )
    return parser


def main() -> None:
    """
    Run daily weather aggregation and optional merge onto a store-day dataset.
    """
    parser = _build_argument_parser()
    args = parser.parse_args()

    weather_df = pd.read_parquet(args.weather_path)
    stores_df = pd.read_parquet(args.stores_path)

    coverage_df = validate_hourly_weather_coverage(
        weather_df=weather_df,
        date_col="date",
        zipcode_col="zipcode",
    )

    daily_weather_df = aggregate_weather_to_daily(
        weather_df=weather_df,
        date_col="date",
        zipcode_col="zipcode",
    )

    print("\nWeather coverage summary:")
    print(coverage_df["hourly_rows"].describe().to_string())
    print("\nDaily weather feature dataset shape:", daily_weather_df.shape)

    if args.daily_weather_output_path:
        output_path = Path(args.daily_weather_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        daily_weather_df.to_parquet(output_path, index=False)
        print(f"\nSaved daily weather dataset to: {output_path}")

    if args.base_dataset_path is None and args.sales_path is None:
        return

    if args.base_dataset_path is not None:
        base_df = pd.read_parquet(args.base_dataset_path)
    else:
        sales_df = pd.read_parquet(args.sales_path)
        base_df = aggregate_sales_to_store_day(
            sales_df=sales_df,
            date_col="date",
            store_col="store_id",
            target_col="sold_quantity",
        )

    merged_df = merge_daily_weather_to_base_dataset(
        base_df=base_df,
        stores_df=stores_df,
        daily_weather_df=daily_weather_df,
        store_col="store_id",
        date_col="date",
        zipcode_col="zipcode",
    )

    weather_columns = [column for column in daily_weather_df.columns if column not in ["zipcode", "date"]]
    missing_rate = merged_df[weather_columns].isna().mean().sort_values(ascending=False)

    print("\nMerged dataset shape:", merged_df.shape)
    print("\nWeather feature missing-rate preview:")
    print(missing_rate.head(20).to_string())

    if args.merged_output_path:
        output_path = Path(args.merged_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_parquet(output_path, index=False)
        print(f"\nSaved merged dataset with weather to: {output_path}")


if __name__ == "__main__":
    main()