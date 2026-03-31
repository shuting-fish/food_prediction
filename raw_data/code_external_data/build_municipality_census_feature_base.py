from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("build_municipality_census_feature_base")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
EXTERNAL_CODE_DIR = RAW_DATA_DIR / "code_external_data"

STORE_REFERENCE_PARQUET_PATH = (
    EXTERNAL_CODE_DIR
    / "_external_data"
    / "store_geography"
    / "store_municipality_reference.parquet"
)
STORE_REFERENCE_CSV_PATH = (
    EXTERNAL_CODE_DIR
    / "_external_data"
    / "store_geography"
    / "store_municipality_reference.csv"
)

MUNICIPALITY_CENSUS_RAW_PATH = (
    EXTERNAL_CODE_DIR
    / "census_raw"
    / "municipality_census_raw.csv"
)

OUTPUT_DIR = EXTERNAL_CODE_DIR / "_external_data" / "census_features"

# Edit these two metadata fields for the exact source version you use.
RAW_SOURCE_NAME = "Official municipality census / demography extract"
RAW_SOURCE_REFERENCE_DATE = "2022-05-15"

# This prevents accidental leakage from future-dated static files.
MAX_ALLOWED_REFERENCE_DATE = "2025-06-30"

REQUIRED_STORE_REFERENCE_COLUMNS = {
    "store_id",
    "store_zipcode",
    "municipality_ags",
    "municipality_name",
    "district_name",
    "federal_state_name",
    "qa_unassigned",
}

REQUIRED_MUNICIPALITY_RAW_COLUMNS = {
    "municipality_ags",
    "municipality_name",
    "total_population",
    "area_sq_km",
}

OPTIONAL_MUNICIPALITY_RAW_COLUMNS = {
    "population_age_0_17",
    "population_age_18_64",
    "population_age_65_plus",
    "foreign_population_total",
    "households_total",
    "average_household_size",
    "purchasing_power_index",
    "purchasing_power_per_capita_eur",
}

NUMERIC_COLUMNS = {
    "total_population",
    "area_sq_km",
    "population_age_0_17",
    "population_age_18_64",
    "population_age_65_plus",
    "foreign_population_total",
    "households_total",
    "average_household_size",
    "purchasing_power_index",
    "purchasing_power_per_capita_eur",
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_columns(df: pd.DataFrame, required_columns: Iterable[str], df_name: str) -> None:
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(f"{df_name} is missing required columns: {missing_columns}")


def normalize_ags(series: pd.Series) -> pd.Series:
    normalized = (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.extract(r"(\d{1,8})", expand=False)
        .astype("string")
        .str.zfill(8)
    )
    normalized = normalized.where(normalized.str.fullmatch(r"\d{8}"), pd.NA)
    return normalized


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    result = numerator / denominator
    result = result.where(denominator > 0)
    return result


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, comment="#")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file format: {path}")


def load_store_reference() -> pd.DataFrame:
    if STORE_REFERENCE_PARQUET_PATH.exists():
        LOGGER.info("Loading store reference parquet: %s", STORE_REFERENCE_PARQUET_PATH)
        store_reference = pd.read_parquet(STORE_REFERENCE_PARQUET_PATH)
    elif STORE_REFERENCE_CSV_PATH.exists():
        LOGGER.info("Loading store reference csv: %s", STORE_REFERENCE_CSV_PATH)
        store_reference = pd.read_csv(STORE_REFERENCE_CSV_PATH)
    else:
        raise FileNotFoundError(
            "Store municipality reference not found. Expected one of:\n"
            f"- {STORE_REFERENCE_PARQUET_PATH}\n"
            f"- {STORE_REFERENCE_CSV_PATH}"
        )

    ensure_columns(
        store_reference,
        REQUIRED_STORE_REFERENCE_COLUMNS,
        "store municipality reference",
    )

    store_reference = store_reference.copy()
    store_reference["store_id"] = pd.to_numeric(store_reference["store_id"], errors="raise").astype("int64")
    store_reference["municipality_ags"] = normalize_ags(store_reference["municipality_ags"])
    store_reference["municipality_name"] = store_reference["municipality_name"].astype("string")
    store_reference["district_name"] = store_reference["district_name"].astype("string")
    store_reference["federal_state_name"] = store_reference["federal_state_name"].astype("string")
    store_reference["store_zipcode"] = (
        store_reference["store_zipcode"]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )

    if store_reference["municipality_ags"].isna().any():
        raise ValueError("store municipality reference contains missing or invalid municipality_ags")

    if store_reference["qa_unassigned"].astype(bool).any():
        unresolved_count = int(store_reference["qa_unassigned"].astype(bool).sum())
        raise ValueError(
            f"store municipality reference still contains {unresolved_count} unassigned stores"
        )

    return store_reference


def build_store_municipality_universe(store_reference: pd.DataFrame) -> pd.DataFrame:
    municipality_store_counts = (
        store_reference.groupby("municipality_ags", as_index=False)
        .agg(
            municipality_name_store_reference=("municipality_name", "first"),
            district_name=("district_name", "first"),
            federal_state_name=("federal_state_name", "first"),
            store_count_in_municipality=("store_id", "nunique"),
        )
    )

    municipality_store_ids = (
        store_reference.sort_values(["municipality_ags", "store_id"])
        .groupby("municipality_ags", as_index=False)["store_id"]
        .apply(lambda values: ",".join(str(v) for v in values))
        .rename(columns={"store_id": "store_ids_csv"})
    )

    universe = municipality_store_counts.merge(
        municipality_store_ids,
        on="municipality_ags",
        how="left",
        validate="one_to_one",
    )

    return universe


def load_municipality_census_raw() -> pd.DataFrame:
    LOGGER.info("Loading municipality census raw file: %s", MUNICIPALITY_CENSUS_RAW_PATH)
    raw_df = read_table(MUNICIPALITY_CENSUS_RAW_PATH)
    ensure_columns(raw_df, REQUIRED_MUNICIPALITY_RAW_COLUMNS, "municipality census raw")

    raw_df = raw_df.copy()
    raw_df["municipality_ags"] = normalize_ags(raw_df["municipality_ags"])
    raw_df["municipality_name"] = raw_df["municipality_name"].astype("string")

    for column in OPTIONAL_MUNICIPALITY_RAW_COLUMNS:
        if column not in raw_df.columns:
            raw_df[column] = pd.Series(pd.NA, index=raw_df.index, dtype="float64")

    for column in NUMERIC_COLUMNS:
        raw_df[column] = pd.to_numeric(raw_df[column], errors="coerce")

    if raw_df["municipality_ags"].isna().any():
        invalid_rows = raw_df.loc[raw_df["municipality_ags"].isna()].copy()
        invalid_rows_path = OUTPUT_DIR / "municipality_census_raw_invalid_ags.csv"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        invalid_rows.to_csv(invalid_rows_path, index=False)
        raise ValueError(
            "municipality census raw contains missing or invalid municipality_ags. "
            f"See: {invalid_rows_path}"
        )

    duplicate_mask = raw_df["municipality_ags"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicates = raw_df.loc[duplicate_mask].sort_values("municipality_ags").copy()
        duplicate_path = OUTPUT_DIR / "municipality_census_raw_duplicate_ags.csv"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        duplicates.to_csv(duplicate_path, index=False)
        raise ValueError(
            "municipality census raw contains duplicate municipality_ags values. "
            "Standardize the raw file to exactly one row per municipality before building features. "
            f"See: {duplicate_path}"
        )

    reference_date = pd.to_datetime(RAW_SOURCE_REFERENCE_DATE, errors="coerce")
    max_allowed_date = pd.to_datetime(MAX_ALLOWED_REFERENCE_DATE, errors="coerce")
    if pd.isna(reference_date):
        raise ValueError(f"RAW_SOURCE_REFERENCE_DATE is invalid: {RAW_SOURCE_REFERENCE_DATE}")
    if reference_date > max_allowed_date:
        raise ValueError(
            "RAW_SOURCE_REFERENCE_DATE is later than the allowed causal cutoff. "
            f"Got {RAW_SOURCE_REFERENCE_DATE}, max allowed is {MAX_ALLOWED_REFERENCE_DATE}."
        )

    return raw_df


def build_municipality_feature_base(
    store_municipality_universe: pd.DataFrame,
    municipality_raw: pd.DataFrame,
) -> pd.DataFrame:
    feature_base = store_municipality_universe.merge(
        municipality_raw,
        on="municipality_ags",
        how="left",
        validate="one_to_one",
        suffixes=("_store_reference", ""),
    )

    feature_base["data_source_name"] = RAW_SOURCE_NAME
    feature_base["data_reference_date"] = RAW_SOURCE_REFERENCE_DATE

    feature_base["qa_coverage_missing_for_store_municipality"] = feature_base["municipality_name"].isna()
    feature_base["qa_name_mismatch_store_vs_raw"] = (
        feature_base["municipality_name"].notna()
        & feature_base["municipality_name_store_reference"].notna()
        & (
            feature_base["municipality_name"].str.strip().str.casefold()
            != feature_base["municipality_name_store_reference"].str.strip().str.casefold()
        )
    )
    feature_base["qa_population_missing"] = feature_base["total_population"].isna()
    feature_base["qa_area_missing"] = feature_base["area_sq_km"].isna()
    feature_base["qa_area_non_positive"] = feature_base["area_sq_km"].fillna(-1) <= 0
    feature_base["qa_population_non_positive"] = feature_base["total_population"].fillna(-1) <= 0

    feature_base["qa_feature_row_incomplete"] = (
        feature_base["qa_coverage_missing_for_store_municipality"]
        | feature_base["qa_population_missing"]
        | feature_base["qa_area_missing"]
        | feature_base["qa_area_non_positive"]
        | feature_base["qa_population_non_positive"]
    )

    feature_base["population_density_per_sq_km"] = safe_divide(
        feature_base["total_population"],
        feature_base["area_sq_km"],
    )

    age_columns = [
        "population_age_0_17",
        "population_age_18_64",
        "population_age_65_plus",
    ]
    feature_base["age_population_sum"] = feature_base[age_columns].sum(axis=1, min_count=1)

    feature_base["share_age_0_17"] = safe_divide(
        feature_base["population_age_0_17"],
        feature_base["total_population"],
    )
    feature_base["share_age_18_64"] = safe_divide(
        feature_base["population_age_18_64"],
        feature_base["total_population"],
    )
    feature_base["share_age_65_plus"] = safe_divide(
        feature_base["population_age_65_plus"],
        feature_base["total_population"],
    )
    feature_base["share_foreign_population"] = safe_divide(
        feature_base["foreign_population_total"],
        feature_base["total_population"],
    )
    feature_base["household_density_per_sq_km"] = safe_divide(
        feature_base["households_total"],
        feature_base["area_sq_km"],
    )

    age_data_present = feature_base[age_columns].notna().all(axis=1)
    feature_base["qa_age_bucket_sum_exceeds_total_population"] = (
        age_data_present
        & feature_base["total_population"].notna()
        & (feature_base["age_population_sum"] > feature_base["total_population"])
    )

    feature_base["qa_feature_row_incomplete"] = (
        feature_base["qa_coverage_missing_for_store_municipality"]
        | feature_base["qa_population_missing"]
        | feature_base["qa_area_missing"]
        | feature_base["qa_area_non_positive"]
        | feature_base["qa_population_non_positive"]
    )

    feature_base["municipality_name"] = feature_base["municipality_name"].fillna(
        feature_base["municipality_name_store_reference"]
    )

    output_columns = [
        "municipality_ags",
        "municipality_name",
        "municipality_name_store_reference",
        "district_name",
        "federal_state_name",
        "store_count_in_municipality",
        "store_ids_csv",
        "data_source_name",
        "data_reference_date",
        "total_population",
        "area_sq_km",
        "population_density_per_sq_km",
        "population_age_0_17",
        "population_age_18_64",
        "population_age_65_plus",
        "share_age_0_17",
        "share_age_18_64",
        "share_age_65_plus",
        "foreign_population_total",
        "share_foreign_population",
        "households_total",
        "average_household_size",
        "household_density_per_sq_km",
        "purchasing_power_index",
        "purchasing_power_per_capita_eur",
        "qa_coverage_missing_for_store_municipality",
        "qa_name_mismatch_store_vs_raw",
        "qa_population_missing",
        "qa_area_missing",
        "qa_area_non_positive",
        "qa_population_non_positive",
        "qa_age_bucket_sum_exceeds_total_population",
        "qa_feature_row_incomplete",
    ]

    return feature_base[output_columns].sort_values("municipality_ags").reset_index(drop=True)


def build_store_level_feature_base(
    store_reference: pd.DataFrame,
    municipality_feature_base: pd.DataFrame,
) -> pd.DataFrame:
    municipality_feature_columns = [
        "municipality_ags",
        "data_source_name",
        "data_reference_date",
        "total_population",
        "area_sq_km",
        "population_density_per_sq_km",
        "population_age_0_17",
        "population_age_18_64",
        "population_age_65_plus",
        "share_age_0_17",
        "share_age_18_64",
        "share_age_65_plus",
        "foreign_population_total",
        "share_foreign_population",
        "households_total",
        "average_household_size",
        "household_density_per_sq_km",
        "purchasing_power_index",
        "purchasing_power_per_capita_eur",
        "qa_coverage_missing_for_store_municipality",
        "qa_name_mismatch_store_vs_raw",
        "qa_population_missing",
        "qa_area_missing",
        "qa_area_non_positive",
        "qa_population_non_positive",
        "qa_age_bucket_sum_exceeds_total_population",
        "qa_feature_row_incomplete",
    ]

    store_feature_base = store_reference.merge(
        municipality_feature_base[municipality_feature_columns],
        on="municipality_ags",
        how="left",
        validate="many_to_one",
    )

    return store_feature_base.sort_values("store_id").reset_index(drop=True)


def build_qa_summary(
    store_reference: pd.DataFrame,
    municipality_feature_base: pd.DataFrame,
    store_feature_base: pd.DataFrame,
) -> pd.DataFrame:
    summary_rows = [
        ("store_count", int(len(store_reference))),
        ("municipality_count_in_store_universe", int(store_reference["municipality_ags"].nunique())),
        ("municipality_feature_rows", int(len(municipality_feature_base))),
        (
            "municipality_rows_missing_coverage",
            int(municipality_feature_base["qa_coverage_missing_for_store_municipality"].sum()),
        ),
        (
            "municipality_rows_with_name_mismatch",
            int(municipality_feature_base["qa_name_mismatch_store_vs_raw"].sum()),
        ),
        (
            "municipality_rows_with_missing_population",
            int(municipality_feature_base["qa_population_missing"].sum()),
        ),
        (
            "municipality_rows_with_missing_area",
            int(municipality_feature_base["qa_area_missing"].sum()),
        ),
        (
            "municipality_rows_with_incomplete_feature_row",
            int(municipality_feature_base["qa_feature_row_incomplete"].sum()),
        ),
        (
            "store_rows_with_missing_municipality_features",
            int(store_feature_base["qa_feature_row_incomplete"].fillna(True).sum()),
        ),
    ]
    return pd.DataFrame(summary_rows, columns=["metric", "value"])


def write_outputs(
    municipality_feature_base: pd.DataFrame,
    store_feature_base: pd.DataFrame,
    qa_summary: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    municipality_parquet_path = OUTPUT_DIR / "municipality_census_feature_base.parquet"
    municipality_csv_path = OUTPUT_DIR / "municipality_census_feature_base.csv"
    store_parquet_path = OUTPUT_DIR / "store_census_feature_base.parquet"
    store_csv_path = OUTPUT_DIR / "store_census_feature_base.csv"
    qa_summary_path = OUTPUT_DIR / "census_feature_base_qa_summary.csv"

    municipality_feature_base.to_parquet(municipality_parquet_path, index=False)
    municipality_feature_base.to_csv(municipality_csv_path, index=False)
    store_feature_base.to_parquet(store_parquet_path, index=False)
    store_feature_base.to_csv(store_csv_path, index=False)
    qa_summary.to_csv(qa_summary_path, index=False)

    missing_coverage = municipality_feature_base.loc[
        municipality_feature_base["qa_coverage_missing_for_store_municipality"]
    ].copy()
    if not missing_coverage.empty:
        missing_coverage_path = OUTPUT_DIR / "municipality_census_feature_base_missing_coverage.csv"
        missing_coverage.to_csv(missing_coverage_path, index=False)
        raise RuntimeError(
            "Municipality census feature base is incomplete for the store municipality universe. "
            f"See: {missing_coverage_path}"
        )

    LOGGER.info("Wrote municipality feature base: %s", municipality_parquet_path)
    LOGGER.info("Wrote store-level feature base: %s", store_parquet_path)
    LOGGER.info("Wrote QA summary: %s", qa_summary_path)


def main() -> None:
    setup_logging()

    store_reference = load_store_reference()
    store_municipality_universe = build_store_municipality_universe(store_reference)
    municipality_raw = load_municipality_census_raw()

    municipality_feature_base = build_municipality_feature_base(
        store_municipality_universe=store_municipality_universe,
        municipality_raw=municipality_raw,
    )

    store_feature_base = build_store_level_feature_base(
        store_reference=store_reference,
        municipality_feature_base=municipality_feature_base,
    )

    qa_summary = build_qa_summary(
        store_reference=store_reference,
        municipality_feature_base=municipality_feature_base,
        store_feature_base=store_feature_base,
    )

    write_outputs(
        municipality_feature_base=municipality_feature_base,
        store_feature_base=store_feature_base,
        qa_summary=qa_summary,
    )


if __name__ == "__main__":
    main()