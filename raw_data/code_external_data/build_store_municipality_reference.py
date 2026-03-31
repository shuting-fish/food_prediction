from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("build_store_municipality_reference")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
EXTERNAL_CODE_DIR = RAW_DATA_DIR / "code_external_data"
REFERENCE_GEO_DIR = EXTERNAL_CODE_DIR / "_reference_geo"
OUTPUT_DIR = EXTERNAL_CODE_DIR / "_external_data" / "store_geography"

CANONICAL_STORES_PATH = RAW_DATA_DIR / "20260218_144523_stores.parquet"
ZIPCODE_REFERENCE_PATH = REFERENCE_GEO_DIR / "zipcode_to_municipality_nrw.csv"
STORE_COORDINATES_HELPER_PATH = REFERENCE_GEO_DIR / "store_coordinates.csv"
MUNICIPALITY_GEOMETRIES_PATH = REFERENCE_GEO_DIR / "vg250_nrw_municipalities.gpkg"

REQUIRED_STORE_COLUMNS = {
    "store_id",
    "zipcode",
    "country_code",
    "subdivision_code",
}

REQUIRED_ZIP_REFERENCE_COLUMNS = {
    "zipcode",
    "municipality_ags",
    "municipality_name",
    "district_name",
    "federal_state_name",
}

OPTIONAL_ZIP_REFERENCE_COLUMNS = {
    "district_ags",
    "federal_state_ags",
}

REQUIRED_MUNICIPALITY_COLUMNS = {
    "municipality_ags",
    "municipality_name",
    "district_ags",
    "district_name",
    "federal_state_ags",
    "federal_state_name",
    "geometry",
}

GERMANY_LAT_RANGE = (47.0, 56.0)
GERMANY_LON_RANGE = (5.0, 16.0)
NRW_LAT_RANGE = (50.0, 52.7)
NRW_LON_RANGE = (5.4, 9.7)
COORDINATE_DUPLICATE_ROUNDING = 7


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_columns(df: pd.DataFrame, required_columns: Iterable[str], df_name: str) -> None:
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(f"{df_name} is missing required columns: {missing_columns}")


def normalize_zipcode(series: pd.Series) -> pd.Series:
    normalized = (
        series.astype("string")
        .str.strip()
        .str.extract(r"(\d{1,5})", expand=False)
        .astype("string")
        .str.zfill(5)
    )
    normalized = normalized.where(normalized.str.fullmatch(r"\d{5}"), pd.NA)
    return normalized


def first_present_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def optional_store_name_series(df: pd.DataFrame) -> pd.Series:
    store_name_column = first_present_column(df, ["store_name", "name", "branch_name"])
    if store_name_column is None:
        return pd.Series(pd.NA, index=df.index, dtype="string")
    return df[store_name_column].astype("string")


def optional_coordinate_frame(df: pd.DataFrame) -> pd.DataFrame:
    latitude_column = first_present_column(df, ["store_latitude", "latitude", "lat", "store_lat"])
    longitude_column = first_present_column(df, ["store_longitude", "longitude", "lon", "lng", "store_lon"])

    output = pd.DataFrame(index=df.index)
    output["store_latitude"] = pd.to_numeric(df[latitude_column], errors="coerce") if latitude_column else np.nan
    output["store_longitude"] = pd.to_numeric(df[longitude_column], errors="coerce") if longitude_column else np.nan
    output["has_coordinate_columns"] = latitude_column is not None and longitude_column is not None
    return output


def read_optional_store_coordinates_helper(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        LOGGER.info("Optional store coordinate helper not found: %s", path)
        return None

    helper_df = pd.read_csv(path)
    ensure_columns(helper_df, {"store_id", "store_latitude", "store_longitude"}, "store coordinate helper")

    if helper_df["store_id"].duplicated().any():
        duplicated_ids = sorted(helper_df.loc[helper_df["store_id"].duplicated(), "store_id"].tolist())
        raise ValueError(f"store coordinate helper contains duplicated store_id values: {duplicated_ids}")

    helper_output = helper_df.copy()
    helper_output["store_id"] = pd.to_numeric(helper_output["store_id"], errors="raise").astype("int64")
    helper_output["store_latitude"] = pd.to_numeric(helper_output["store_latitude"], errors="coerce")
    helper_output["store_longitude"] = pd.to_numeric(helper_output["store_longitude"], errors="coerce")
    if "store_name" in helper_output.columns:
        helper_output["store_name"] = helper_output["store_name"].astype("string")
    return helper_output


def validate_nrw_scope(stores_df: pd.DataFrame) -> None:
    invalid_country = stores_df.loc[stores_df["country_code"].astype("string") != "DE", ["store_id", "country_code"]]
    invalid_subdivision = stores_df.loc[
        stores_df["subdivision_code"].astype("string") != "DE-NW",
        ["store_id", "subdivision_code"],
    ]

    if not invalid_country.empty:
        raise ValueError(
            "Canonical stores contain non-DE country codes. This NRW-only reference cannot continue. "
            f"Affected rows: {invalid_country.to_dict(orient='records')}"
        )

    if not invalid_subdivision.empty:
        raise ValueError(
            "Canonical stores contain non-DE-NW subdivision codes. This NRW-only reference cannot continue. "
            f"Affected rows: {invalid_subdivision.to_dict(orient='records')}"
        )


def build_store_base(stores_df: pd.DataFrame, helper_df: pd.DataFrame | None) -> pd.DataFrame:
    ensure_columns(stores_df, REQUIRED_STORE_COLUMNS, "canonical stores")

    if not stores_df["store_id"].is_unique:
        duplicated_store_ids = stores_df.loc[stores_df["store_id"].duplicated(), "store_id"].tolist()
        raise ValueError(f"Canonical stores contain duplicated store_id values: {duplicated_store_ids}")

    validate_nrw_scope(stores_df)

    base = stores_df.copy()
    base["store_id"] = pd.to_numeric(base["store_id"], errors="raise").astype("int64")
    base["store_zipcode"] = normalize_zipcode(base["zipcode"])
    base["store_name"] = optional_store_name_series(base)

    canonical_coordinates = optional_coordinate_frame(base)
    base["canonical_store_latitude"] = canonical_coordinates["store_latitude"]
    base["canonical_store_longitude"] = canonical_coordinates["store_longitude"]
    base["qa_coordinate_columns_present_in_canonical_stores"] = canonical_coordinates["has_coordinate_columns"].astype(bool)

    if helper_df is not None:
        base = base.merge(
            helper_df.rename(
                columns={
                    "store_latitude": "helper_store_latitude",
                    "store_longitude": "helper_store_longitude",
                    "store_name": "helper_store_name",
                }
            ),
            on="store_id",
            how="left",
            validate="one_to_one",
        )
    else:
        base["helper_store_latitude"] = np.nan
        base["helper_store_longitude"] = np.nan
        base["helper_store_name"] = pd.Series(pd.NA, index=base.index, dtype="string")

    base["store_name"] = base["store_name"].fillna(base["helper_store_name"])
    base["store_latitude"] = base["canonical_store_latitude"].fillna(base["helper_store_latitude"])
    base["store_longitude"] = base["canonical_store_longitude"].fillna(base["helper_store_longitude"])

    base["qa_coordinates_from_helper_file"] = (
        base["canonical_store_latitude"].isna()
        & base["canonical_store_longitude"].isna()
        & base["helper_store_latitude"].notna()
        & base["helper_store_longitude"].notna()
    )

    lat_conflict = (
        (base["canonical_store_latitude"] - base["helper_store_latitude"]).abs() > 1e-7
    )
    lon_conflict = (
        (base["canonical_store_longitude"] - base["helper_store_longitude"]).abs() > 1e-7
    )

    base["qa_coordinate_source_conflict"] = (
        base["canonical_store_latitude"].notna()
        & base["canonical_store_longitude"].notna()
        & base["helper_store_latitude"].notna()
        & base["helper_store_longitude"].notna()
        & (lat_conflict | lon_conflict)
    )

    base["qa_zipcode_missing_or_invalid"] = base["store_zipcode"].isna()
    base["qa_coordinates_missing"] = base["store_latitude"].isna() | base["store_longitude"].isna()
    base["qa_coordinates_out_of_germany_range"] = (
        ~base["qa_coordinates_missing"]
        & ~(
            base["store_latitude"].between(*GERMANY_LAT_RANGE)
            & base["store_longitude"].between(*GERMANY_LON_RANGE)
        )
    )
    base["qa_coordinates_outside_nrw_bbox"] = (
        ~base["qa_coordinates_missing"]
        & ~base["qa_coordinates_out_of_germany_range"]
        & ~(
            base["store_latitude"].between(*NRW_LAT_RANGE)
            & base["store_longitude"].between(*NRW_LON_RANGE)
        )
    )
    base["qa_has_valid_coordinates"] = (
        ~base["qa_coordinates_missing"] & ~base["qa_coordinates_out_of_germany_range"]
    )

    valid_coordinate_mask = base["qa_has_valid_coordinates"]
    rounded_lat = base.loc[valid_coordinate_mask, "store_latitude"].round(COORDINATE_DUPLICATE_ROUNDING)
    rounded_lon = base.loc[valid_coordinate_mask, "store_longitude"].round(COORDINATE_DUPLICATE_ROUNDING)
    coordinate_key = rounded_lat.astype("string") + "|" + rounded_lon.astype("string")
    coordinate_counts = coordinate_key.value_counts()
    duplicate_keys = set(coordinate_counts[coordinate_counts > 1].index)

    base["qa_duplicate_coordinates_exact"] = False
    base.loc[valid_coordinate_mask, "qa_duplicate_coordinates_exact"] = coordinate_key.isin(duplicate_keys).to_numpy()

    zipcode_counts = base["store_zipcode"].value_counts(dropna=True)
    duplicate_zipcodes = set(zipcode_counts[zipcode_counts > 1].index)
    base["qa_duplicate_zipcode_multiple_stores"] = base["store_zipcode"].isin(duplicate_zipcodes)

    return base


def load_zipcode_reference(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required zipcode reference file not found: {path}")

    zipcode_reference = pd.read_csv(path, dtype="string")
    ensure_columns(zipcode_reference, REQUIRED_ZIP_REFERENCE_COLUMNS, "zipcode reference")
    zipcode_reference = zipcode_reference.copy()

    for optional_column in OPTIONAL_ZIP_REFERENCE_COLUMNS:
        if optional_column not in zipcode_reference.columns:
            zipcode_reference[optional_column] = pd.Series(
                pd.NA,
                index=zipcode_reference.index,
                dtype="string",
            )

    zipcode_reference["zipcode"] = normalize_zipcode(zipcode_reference["zipcode"])

    if zipcode_reference["zipcode"].duplicated().any():
        duplicated_zipcodes = zipcode_reference.loc[
            zipcode_reference["zipcode"].duplicated(),
            "zipcode",
        ].tolist()
        raise ValueError(f"zipcode reference contains duplicated zipcode values: {duplicated_zipcodes}")

    return zipcode_reference


def load_municipality_geometries(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(
            "Valid store coordinates are available, but the municipality geometry file is missing: "
            f"{path}"
        )

    municipalities = gpd.read_file(path)
    missing_columns = sorted(REQUIRED_MUNICIPALITY_COLUMNS - set(municipalities.columns))
    if missing_columns:
        raise ValueError(
            "Municipality geometry file must already be standardized with these columns: "
            f"{sorted(REQUIRED_MUNICIPALITY_COLUMNS)}. Missing: {missing_columns}"
        )

    municipalities = municipalities[list(REQUIRED_MUNICIPALITY_COLUMNS)].copy()
    municipalities = municipalities.to_crs("EPSG:4326")
    municipalities["municipality_ags"] = municipalities["municipality_ags"].astype("string")
    municipalities["district_ags"] = municipalities["district_ags"].astype("string")
    municipalities["federal_state_ags"] = municipalities["federal_state_ags"].astype("string")
    municipalities["municipality_name"] = municipalities["municipality_name"].astype("string")
    municipalities["district_name"] = municipalities["district_name"].astype("string")
    municipalities["federal_state_name"] = municipalities["federal_state_name"].astype("string")
    return municipalities


def spatial_assign_municipalities(stores_df: pd.DataFrame, municipalities_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    valid_coordinate_stores = stores_df.loc[stores_df["qa_has_valid_coordinates"]].copy()
    if valid_coordinate_stores.empty:
        return pd.DataFrame(columns=[
            "store_id",
            "spatial_municipality_ags",
            "spatial_municipality_name",
            "spatial_district_ags",
            "spatial_district_name",
            "spatial_federal_state_ags",
            "spatial_federal_state_name",
            "qa_spatial_match_found",
            "qa_spatial_match_ambiguous",
        ])

    store_points = gpd.GeoDataFrame(
        valid_coordinate_stores[["store_id", "store_latitude", "store_longitude"]].copy(),
        geometry=gpd.points_from_xy(
            valid_coordinate_stores["store_longitude"],
            valid_coordinate_stores["store_latitude"],
        ),
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(
        store_points,
        municipalities_gdf,
        how="left",
        predicate="intersects",
    )

    non_null_matches = joined.loc[joined["municipality_ags"].notna()].copy()
    match_counts = (
        non_null_matches.groupby("store_id", as_index=False)
        .size()
        .rename(columns={"size": "spatial_match_count"})
    )

    single_matches = non_null_matches.merge(match_counts, on="store_id", how="left")
    single_matches = single_matches.loc[single_matches["spatial_match_count"] == 1].copy()
    single_matches = single_matches.drop_duplicates(subset=["store_id"])

    assignment = valid_coordinate_stores[["store_id"]].copy()
    assignment = assignment.merge(
        single_matches[
            [
                "store_id",
                "municipality_ags",
                "municipality_name",
                "district_ags",
                "district_name",
                "federal_state_ags",
                "federal_state_name",
            ]
        ].rename(
            columns={
                "municipality_ags": "spatial_municipality_ags",
                "municipality_name": "spatial_municipality_name",
                "district_ags": "spatial_district_ags",
                "district_name": "spatial_district_name",
                "federal_state_ags": "spatial_federal_state_ags",
                "federal_state_name": "spatial_federal_state_name",
            }
        ),
        on="store_id",
        how="left",
        validate="one_to_one",
    )

    ambiguous_store_ids = set(match_counts.loc[match_counts["spatial_match_count"] > 1, "store_id"].tolist())
    matched_store_ids = set(match_counts["store_id"].tolist())

    assignment["qa_spatial_match_found"] = assignment["store_id"].isin(matched_store_ids)
    assignment["qa_spatial_match_ambiguous"] = assignment["store_id"].isin(ambiguous_store_ids)
    return assignment


def combine_assignments(
    store_base_df: pd.DataFrame,
    zipcode_reference_df: pd.DataFrame,
    spatial_assignment_df: pd.DataFrame,
) -> pd.DataFrame:
    output = store_base_df.copy()

    output = output.merge(
        zipcode_reference_df.rename(
            columns={
                "zipcode": "store_zipcode",
                "municipality_ags": "zip_municipality_ags",
                "municipality_name": "zip_municipality_name",
                "district_ags": "zip_district_ags",
                "district_name": "zip_district_name",
                "federal_state_ags": "zip_federal_state_ags",
                "federal_state_name": "zip_federal_state_name",
            }
        ),
        on="store_zipcode",
        how="left",
        validate="many_to_one",
    )

    output = output.merge(
        spatial_assignment_df,
        on="store_id",
        how="left",
        validate="one_to_one",
    )

    output["qa_zipcode_reference_found"] = output["zip_municipality_ags"].notna()
    output["qa_spatial_match_found"] = output["qa_spatial_match_found"].fillna(False)
    output["qa_spatial_match_ambiguous"] = output["qa_spatial_match_ambiguous"].fillna(False)
    output["qa_spatial_assignment_available"] = output["spatial_municipality_ags"].notna()
    output["qa_spatial_vs_zip_mismatch"] = (
        output["qa_spatial_assignment_available"]
        & output["qa_zipcode_reference_found"]
        & (output["spatial_municipality_ags"] != output["zip_municipality_ags"])
    )

    output["assignment_method"] = np.select(
        [
            output["qa_spatial_assignment_available"],
            ~output["qa_has_valid_coordinates"] & output["qa_zipcode_reference_found"],
            output["qa_has_valid_coordinates"] & ~output["qa_spatial_match_found"] & output["qa_zipcode_reference_found"],
            output["qa_has_valid_coordinates"] & output["qa_spatial_match_ambiguous"] & output["qa_zipcode_reference_found"],
        ],
        [
            "store_coordinates_spatial_join",
            "zipcode_fallback_no_valid_coordinates",
            "zipcode_fallback_spatial_no_match",
            "zipcode_fallback_spatial_ambiguous",
        ],
        default="unassigned",
    )

    output["municipality_ags"] = np.where(
        output["qa_spatial_assignment_available"],
        output["spatial_municipality_ags"],
        output["zip_municipality_ags"],
    )
    output["municipality_name"] = np.where(
        output["qa_spatial_assignment_available"],
        output["spatial_municipality_name"],
        output["zip_municipality_name"],
    )
    output["district_ags"] = np.where(
        output["qa_spatial_assignment_available"],
        output["spatial_district_ags"],
        output["zip_district_ags"],
    )
    output["district_name"] = np.where(
        output["qa_spatial_assignment_available"],
        output["spatial_district_name"],
        output["zip_district_name"],
    )
    output["federal_state_ags"] = np.where(
        output["qa_spatial_assignment_available"],
        output["spatial_federal_state_ags"],
        output["zip_federal_state_ags"],
    )
    output["federal_state_name"] = np.where(
        output["qa_spatial_assignment_available"],
        output["spatial_federal_state_name"],
        output["zip_federal_state_name"],
    )

    output["qa_assignment_used_fallback"] = output["assignment_method"] != "store_coordinates_spatial_join"
    output["qa_unassigned"] = output["assignment_method"] == "unassigned"

    final_columns = [
        "store_id",
        "store_name",
        "store_latitude",
        "store_longitude",
        "store_zipcode",
        "country_code",
        "subdivision_code",
        "average_weekly_revenue_Q1",
        "municipality_ags",
        "municipality_name",
        "district_ags",
        "district_name",
        "federal_state_ags",
        "federal_state_name",
        "assignment_method",
        "qa_coordinate_columns_present_in_canonical_stores",
        "qa_has_valid_coordinates",
        "qa_coordinates_missing",
        "qa_coordinates_from_helper_file",
        "qa_coordinate_source_conflict",
        "qa_coordinates_out_of_germany_range",
        "qa_coordinates_outside_nrw_bbox",
        "qa_duplicate_coordinates_exact",
        "qa_duplicate_zipcode_multiple_stores",
        "qa_zipcode_missing_or_invalid",
        "qa_zipcode_reference_found",
        "qa_spatial_match_found",
        "qa_spatial_match_ambiguous",
        "qa_spatial_assignment_available",
        "qa_spatial_vs_zip_mismatch",
        "qa_assignment_used_fallback",
        "qa_unassigned",
    ]

    return output[final_columns].sort_values("store_id").reset_index(drop=True)


def build_qa_summary(final_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = [
        ("store_count", int(len(final_df))),
        ("stores_with_valid_coordinates", int(final_df["qa_has_valid_coordinates"].sum())),
        ("stores_assigned_by_spatial_join", int((final_df["assignment_method"] == "store_coordinates_spatial_join").sum())),
        ("stores_assigned_by_zip_fallback", int(final_df["qa_assignment_used_fallback"].sum())),
        ("stores_unassigned", int(final_df["qa_unassigned"].sum())),
        ("stores_with_duplicate_coordinates_exact", int(final_df["qa_duplicate_coordinates_exact"].sum())),
        ("stores_with_spatial_zip_mismatch", int(final_df["qa_spatial_vs_zip_mismatch"].sum())),
        ("stores_with_invalid_zipcode", int(final_df["qa_zipcode_missing_or_invalid"].sum())),
    ]
    return pd.DataFrame(summary_rows, columns=["metric", "value"])


def write_outputs(final_df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parquet_path = OUTPUT_DIR / "store_municipality_reference.parquet"
    csv_path = OUTPUT_DIR / "store_municipality_reference.csv"
    qa_summary_path = OUTPUT_DIR / "store_municipality_reference_qa_summary.csv"

    final_df.to_parquet(parquet_path, index=False)
    final_df.to_csv(csv_path, index=False)
    build_qa_summary(final_df).to_csv(qa_summary_path, index=False)

    duplicate_coordinates = final_df.loc[final_df["qa_duplicate_coordinates_exact"]].copy()
    if not duplicate_coordinates.empty:
        duplicate_coordinates.to_csv(
            OUTPUT_DIR / "store_municipality_reference_duplicate_coordinates.csv",
            index=False,
        )

    mismatch_df = final_df.loc[final_df["qa_spatial_vs_zip_mismatch"]].copy()
    if not mismatch_df.empty:
        mismatch_df.to_csv(
            OUTPUT_DIR / "store_municipality_reference_spatial_zip_mismatch.csv",
            index=False,
        )

    unresolved_df = final_df.loc[final_df["qa_unassigned"]].copy()
    if not unresolved_df.empty:
        unresolved_path = OUTPUT_DIR / "store_municipality_reference_unresolved.csv"
        unresolved_df.to_csv(unresolved_path, index=False)
        raise RuntimeError(
            "Store municipality reference contains unassigned stores. "
            f"See: {unresolved_path}"
        )

    LOGGER.info("Wrote store municipality reference to: %s", parquet_path)
    LOGGER.info("Wrote QA summary to: %s", qa_summary_path)


def main() -> None:
    setup_logging()

    LOGGER.info("Loading canonical stores: %s", CANONICAL_STORES_PATH)
    stores_df = pd.read_parquet(CANONICAL_STORES_PATH)

    helper_df = read_optional_store_coordinates_helper(STORE_COORDINATES_HELPER_PATH)
    store_base_df = build_store_base(stores_df=stores_df, helper_df=helper_df)

    zipcode_reference_df = load_zipcode_reference(ZIPCODE_REFERENCE_PATH)

    if store_base_df["qa_has_valid_coordinates"].any():
        municipalities_gdf = load_municipality_geometries(MUNICIPALITY_GEOMETRIES_PATH)
        spatial_assignment_df = spatial_assign_municipalities(
            stores_df=store_base_df,
            municipalities_gdf=municipalities_gdf,
        )
    else:
        LOGGER.info("No valid store coordinates available. Running zipcode-based fallback only.")
        spatial_assignment_df = pd.DataFrame(
            columns=[
                "store_id",
                "spatial_municipality_ags",
                "spatial_municipality_name",
                "spatial_district_ags",
                "spatial_district_name",
                "spatial_federal_state_ags",
                "spatial_federal_state_name",
                "qa_spatial_match_found",
                "qa_spatial_match_ambiguous",
            ]
        )

    final_df = combine_assignments(
        store_base_df=store_base_df,
        zipcode_reference_df=zipcode_reference_df,
        spatial_assignment_df=spatial_assignment_df,
    )
    write_outputs(final_df)


if __name__ == "__main__":
    main()