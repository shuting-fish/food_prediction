"""
Build zipcode_to_municipality_nrw.csv by spatially assigning NRW ZIP centroids
to official NRW municipalities.

This script replaces brittle postal-code lookup logic with a geospatial workflow
based on official municipality boundaries from the German Federal Agency for
Cartography and Geodesy (BKG) dataset VG250.

What this script does
---------------------
1. Reads the local NRW ZIP centroid CSV.
2. Downloads the current official BKG VG250 GeoPackage ZIP if needed.
3. Extracts the GeoPackage locally.
4. Loads the municipality polygon layer from the GeoPackage.
5. Filters municipalities to Nordrhein-Westfalen using the official AGS prefix.
6. Spatially assigns each ZIP centroid to exactly one official municipality.
7. Enriches the result with official municipality and, if available, district
   and state names from the VG250 administrative assignment table.
8. Writes a clean CSV with self-explanatory column names.

Why this script exists
----------------------
The previous approach used postal-code lookup data and produced mixed
administrative levels such as municipalities, districts, and city-counties.
That is not acceptable when the target is true municipality-level assignment.

This script uses:
- official municipality polygons (`vg250_gem`) as the primary geometry source
- official municipality keys (AGS)
- a real spatial join from ZIP centroids to municipality polygons

Input requirements
------------------
The default input file is:

    raw_data/code_external_data/_reference_geo/plz_centroids_nrw.csv

The input CSV must contain:
- one ZIP code column, auto-detected from:
    zip, zipcode, postal_code, plz
- one longitude column, auto-detected from:
    longitude, lon, lng, x, centroid_longitude, centroid_lon
- one latitude column, auto-detected from:
    latitude, lat, y, centroid_latitude, centroid_lat

The input coordinates are assumed to be WGS84 / EPSG:4326 by default.
You can override this with --input-crs.

Output
------
The default output file is:

    raw_data/code_external_data/_reference_geo/zipcode_to_municipality_nrw.csv

The output CSV is one row per ZIP code and contains self-explanatory columns:
- zipcode
- centroid_longitude
- centroid_latitude
- municipality_ags
- municipality_ars
- municipality_name
- municipality_designation
- municipality_display_name
- district_ars
- district_name
- district_designation
- district_display_name
- federal_state_code
- federal_state_name
- assignment_method
- distance_to_municipality_m

Assignment logic
----------------
Primary method:
- point-in-polygon using official municipality polygons

Fallback method:
- nearest municipality polygon if a centroid is not matched exactly
  (this can happen due to boundary generalization or centroids lying exactly on
  a boundary); such cases are explicitly flagged in assignment_method

Dependencies
------------
Required Python packages:
- pandas
- geopandas
- shapely
- pyogrio

Typical install:
    pip install pandas geopandas shapely pyogrio

Usage
-----
Default run:
    python raw_data/code_external_data/build_zipcode_to_municipality_nrw_csv.py

Force re-download of the official boundary dataset:
    python raw_data/code_external_data/build_zipcode_to_municipality_nrw_csv.py --refresh-boundary-cache

Explicit input CRS:
    python raw_data/code_external_data/build_zipcode_to_municipality_nrw_csv.py --input-crs EPSG:4326

Custom input/output files:
    python raw_data/code_external_data/build_zipcode_to_municipality_nrw_csv.py ^
        --plz-csv raw_data/code_external_data/_reference_geo/plz_centroids_nrw.csv ^
        --out-csv raw_data/code_external_data/_reference_geo/zipcode_to_municipality_nrw.csv
"""

from __future__ import annotations

import argparse
import sqlite3
import zipfile
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

try:
    import geopandas as gpd
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'geopandas'. Install required geospatial packages first:\n"
        "pip install pandas geopandas shapely pyogrio"
    ) from exc


VG250_GPKG_ZIP_URL = (
    "https://daten.gdz.bkg.bund.de/produkte/vg/vg250_ebenen_0101/aktuell/"
    "vg250_01-01.utm32s.gpkg.ebenen.zip"
)

FEDERAL_STATE_CODE_NRW = "05"
FEDERAL_STATE_NAME_NRW = "Nordrhein-Westfalen"

CODE_DIR = Path(__file__).resolve().parent
REFERENCE_GEO_DIR = CODE_DIR / "_reference_geo"
VG250_CACHE_DIR = REFERENCE_GEO_DIR / "vg250_cache"

DEFAULT_PLZ_CSV = REFERENCE_GEO_DIR / "plz_centroids_nrw.csv"
DEFAULT_OUT_CSV = REFERENCE_GEO_DIR / "zipcode_to_municipality_nrw.csv"
VG250_CACHE_ZIP_PATH = VG250_CACHE_DIR / "vg250_01-01.utm32s.gpkg.ebenen.zip"

ZIP_COLUMN_CANDIDATES = ("zipcode", "zip", "postal_code", "plz")
LONGITUDE_COLUMN_CANDIDATES = (
    "centroid_longitude",
    "centroid_lon",
    "longitude",
    "lon",
    "lng",
    "x",
)
LATITUDE_COLUMN_CANDIDATES = (
    "centroid_latitude",
    "centroid_lat",
    "latitude",
    "lat",
    "y",
)

GPKG_MUNICIPALITY_LAYER = "vg250_gem"
GPKG_ADMIN_ASSIGNMENT_TABLE = "vgtb_vz_gem"


def normalize_zipcode_series(series: pd.Series) -> pd.Series:
    """
    Normalize ZIP-like values to 5-digit strings.
    """
    out = series.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    out = out.str.replace(r"\D", "", regex=True)
    out = out.str.zfill(5)
    return out


def normalize_admin_key(series: pd.Series, width: int) -> pd.Series:
    """
    Normalize official administrative keys by keeping digits only and zero-padding.
    """
    out = series.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    out = out.str.replace(r"\D", "", regex=True)
    out = out.str.zfill(width)
    return out


def format_display_name(name: object, designation: object) -> str:
    """
    Build a human-readable display name such as:
    - 'Düsseldorf, Stadt'
    - 'Köln, Stadt'
    - 'Aachen'
    """
    name_str = "" if pd.isna(name) else str(name).strip()
    designation_str = "" if pd.isna(designation) else str(designation).strip()

    if not name_str:
        return ""

    if not designation_str:
        return name_str

    if designation_str.lower() in name_str.lower():
        return name_str

    return f"{name_str}, {designation_str}"


def find_column_case_insensitive(
    columns: Iterable[str],
    candidates: Iterable[str],
    required: bool = True,
) -> str | None:
    """
    Find the first matching column name ignoring case.
    """
    lookup = {str(col).lower(): str(col) for col in columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    if required:
        raise ValueError(
            f"Could not find a required column. Expected one of: {list(candidates)}. "
            f"Available columns: {list(columns)}"
        )

    return None


def download_file(url: str, target_path: Path) -> None:
    """
    Download a remote file to a local cache path.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)

    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; food_prediction/1.0)",
        },
    )

    try:
        with urlopen(request, timeout=120) as response:
            data = response.read()
    except HTTPError as exc:
        raise RuntimeError(f"Boundary dataset URL returned HTTP error: {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Boundary dataset could not be downloaded: {url}") from exc

    target_path.write_bytes(data)


def prepare_vg250_gpkg(
    zip_url: str = VG250_GPKG_ZIP_URL,
    cache_zip_path: Path = VG250_CACHE_ZIP_PATH,
    cache_dir: Path = VG250_CACHE_DIR,
    refresh: bool = False,
) -> Path:
    """
    Download the official VG250 GeoPackage ZIP if needed and extract the contained
    .gpkg file to a stable local path.

    Returns the local path to the extracted GeoPackage.
    """
    if refresh or not cache_zip_path.exists():
        download_file(url=zip_url, target_path=cache_zip_path)

    cache_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(cache_zip_path, "r") as zf:
        gpkg_members = [name for name in zf.namelist() if name.lower().endswith(".gpkg")]

        if not gpkg_members:
            raise RuntimeError(
                f"No .gpkg file found inside boundary ZIP archive: {cache_zip_path}"
            )

        gpkg_member = gpkg_members[0]
        gpkg_filename = Path(gpkg_member).name
        extracted_gpkg_path = cache_dir / gpkg_filename

        if refresh or not extracted_gpkg_path.exists():
            zf.extract(gpkg_member, path=cache_dir)

            extracted_from_zip = cache_dir / gpkg_member
            if extracted_from_zip != extracted_gpkg_path:
                extracted_gpkg_path.parent.mkdir(parents=True, exist_ok=True)
                extracted_from_zip.replace(extracted_gpkg_path)

                # Clean up now-empty nested directories if extraction created them.
                parent_dir = extracted_from_zip.parent
                while parent_dir != cache_dir and parent_dir.exists():
                    try:
                        parent_dir.rmdir()
                    except OSError:
                        break
                    parent_dir = parent_dir.parent

    if not extracted_gpkg_path.exists():
        raise FileNotFoundError(f"Extracted GeoPackage not found: {extracted_gpkg_path}")

    return extracted_gpkg_path


def parse_numeric_coordinate(series: pd.Series) -> pd.Series:
    """
    Parse coordinates robustly, including decimal commas.
    """
    return pd.to_numeric(series.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")


def read_zip_centroids(
    path: Path,
    input_crs: str,
    zipcode_column: str | None = None,
    longitude_column: str | None = None,
    latitude_column: str | None = None,
) -> gpd.GeoDataFrame:
    """
    Read and validate the NRW ZIP centroid CSV and return a GeoDataFrame of points.
    """
    if not path.exists():
        raise FileNotFoundError(f"ZIP centroid CSV not found: {path}")

    df = pd.read_csv(path, dtype=str)

    zip_col = zipcode_column or find_column_case_insensitive(df.columns, ZIP_COLUMN_CANDIDATES)
    lon_col = longitude_column or find_column_case_insensitive(df.columns, LONGITUDE_COLUMN_CANDIDATES)
    lat_col = latitude_column or find_column_case_insensitive(df.columns, LATITUDE_COLUMN_CANDIDATES)

    out = df.copy()
    out = out.rename(
        columns={
            zip_col: "zipcode",
            lon_col: "centroid_longitude",
            lat_col: "centroid_latitude",
        }
    )

    out["zipcode"] = normalize_zipcode_series(out["zipcode"])
    out["centroid_longitude"] = parse_numeric_coordinate(out["centroid_longitude"])
    out["centroid_latitude"] = parse_numeric_coordinate(out["centroid_latitude"])

    out = out[out["zipcode"].str.fullmatch(r"\d{5}", na=False)].copy()
    out = out.dropna(subset=["centroid_longitude", "centroid_latitude"]).copy()

    if out.empty:
        raise RuntimeError(f"No valid ZIP centroids found in input CSV: {path}")

    # A ZIP centroid mapping must be one ZIP -> one coordinate pair.
    duplicate_check = (
        out.groupby("zipcode")[["centroid_longitude", "centroid_latitude"]]
        .nunique(dropna=False)
        .reset_index()
    )
    inconsistent_duplicates = duplicate_check[
        (duplicate_check["centroid_longitude"] > 1) | (duplicate_check["centroid_latitude"] > 1)
    ]

    if not inconsistent_duplicates.empty:
        bad_zips = inconsistent_duplicates["zipcode"].head(20).tolist()
        raise RuntimeError(
            "The centroid CSV contains ZIP codes with multiple different coordinate pairs. "
            f"First problematic ZIPs: {bad_zips}"
        )

    out = out.drop_duplicates(subset=["zipcode"], keep="first").reset_index(drop=True)
    out["point_row_id"] = range(len(out))

    gdf = gpd.GeoDataFrame(
        out,
        geometry=gpd.points_from_xy(out["centroid_longitude"], out["centroid_latitude"]),
        crs=input_crs,
    )

    return gdf


def read_admin_assignment_table(gpkg_path: Path) -> pd.DataFrame:
    """
    Read the optional administrative assignment table from the official GeoPackage.

    This table can enrich municipality rows with superior administrative names such
    as district and federal state names.

    If the table or expected columns are not available, an empty standardized
    DataFrame is returned.
    """
    standardized_empty = pd.DataFrame(
        columns=[
            "municipality_ars",
            "municipality_ags_from_table",
            "district_ars",
            "district_name",
            "district_designation",
            "district_display_name",
            "federal_state_name_from_table",
            "federal_state_designation",
        ]
    )

    with sqlite3.connect(gpkg_path) as conn:
        objects = pd.read_sql_query(
            "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view')",
            conn,
        )

        object_lookup = {name.lower(): name for name in objects["name"].astype(str).tolist()}
        table_name = object_lookup.get(GPKG_ADMIN_ASSIGNMENT_TABLE.lower())

        if table_name is None:
            return standardized_empty

        df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)

    if df.empty:
        return standardized_empty

    municipality_ars_col = find_column_case_insensitive(df.columns, ("ARS_G",), required=False)
    municipality_ags_col = find_column_case_insensitive(df.columns, ("AGS_G",), required=False)
    district_ars_col = find_column_case_insensitive(df.columns, ("ARS_K",), required=False)
    district_name_col = find_column_case_insensitive(df.columns, ("GEN_K",), required=False)
    district_designation_col = find_column_case_insensitive(df.columns, ("BEZ_K",), required=False)
    federal_state_name_col = find_column_case_insensitive(df.columns, ("GEN_L",), required=False)
    federal_state_designation_col = find_column_case_insensitive(df.columns, ("BEZ_L",), required=False)

    if municipality_ars_col is None:
        return standardized_empty

    out = pd.DataFrame(
        {
            "municipality_ars": normalize_admin_key(df[municipality_ars_col], width=12),
            "municipality_ags_from_table": (
                normalize_admin_key(df[municipality_ags_col], width=8)
                if municipality_ags_col is not None
                else pd.Series(pd.NA, index=df.index, dtype="object")
            ),
            "district_ars": (
                normalize_admin_key(df[district_ars_col], width=12)
                if district_ars_col is not None
                else pd.Series(pd.NA, index=df.index, dtype="object")
            ),
            "district_name": (
                df[district_name_col].astype(str).str.strip()
                if district_name_col is not None
                else pd.Series(pd.NA, index=df.index, dtype="object")
            ),
            "district_designation": (
                df[district_designation_col].astype(str).str.strip()
                if district_designation_col is not None
                else pd.Series(pd.NA, index=df.index, dtype="object")
            ),
            "federal_state_name_from_table": (
                df[federal_state_name_col].astype(str).str.strip()
                if federal_state_name_col is not None
                else pd.Series(pd.NA, index=df.index, dtype="object")
            ),
            "federal_state_designation": (
                df[federal_state_designation_col].astype(str).str.strip()
                if federal_state_designation_col is not None
                else pd.Series(pd.NA, index=df.index, dtype="object")
            ),
        }
    )

    out["district_display_name"] = out.apply(
        lambda row: format_display_name(row["district_name"], row["district_designation"]),
        axis=1,
    )

    out = out.replace({"": pd.NA})
    out = out.drop_duplicates(subset=["municipality_ars"], keep="first").reset_index(drop=True)

    return out


def read_nrw_municipality_polygons(gpkg_path: Path) -> gpd.GeoDataFrame:
    """
    Read the official municipality polygon layer, standardize relevant attributes,
    and filter to NRW municipalities.
    """
    municipality_gdf = gpd.read_file(gpkg_path, layer=GPKG_MUNICIPALITY_LAYER)

    ags_col = find_column_case_insensitive(municipality_gdf.columns, ("AGS",))
    ars_col = find_column_case_insensitive(municipality_gdf.columns, ("ARS",))
    municipality_name_col = find_column_case_insensitive(municipality_gdf.columns, ("GEN",))
    municipality_designation_col = find_column_case_insensitive(
        municipality_gdf.columns,
        ("BEZ",),
        required=False,
    )

    out = municipality_gdf.copy()
    out = out.rename(
        columns={
            ags_col: "municipality_ags",
            ars_col: "municipality_ars",
            municipality_name_col: "municipality_name",
        }
    )

    if municipality_designation_col is not None:
        out = out.rename(columns={municipality_designation_col: "municipality_designation"})
    else:
        out["municipality_designation"] = pd.NA

    out["municipality_ags"] = normalize_admin_key(out["municipality_ags"], width=8)
    out["municipality_ars"] = normalize_admin_key(out["municipality_ars"], width=12)
    out["municipality_name"] = out["municipality_name"].astype(str).str.strip()
    out["municipality_designation"] = out["municipality_designation"].astype(str).str.strip()

    out = out.replace({"": pd.NA})
    out = out[out["municipality_ags"].str.startswith(FEDERAL_STATE_CODE_NRW, na=False)].copy()

    if out.empty:
        raise RuntimeError("No NRW municipalities found in the official municipality layer.")

    out["federal_state_code"] = FEDERAL_STATE_CODE_NRW
    out["federal_state_name"] = FEDERAL_STATE_NAME_NRW
    out["municipality_display_name"] = out.apply(
        lambda row: format_display_name(row["municipality_name"], row["municipality_designation"]),
        axis=1,
    )

    admin_assignment_df = read_admin_assignment_table(gpkg_path=gpkg_path)

    if not admin_assignment_df.empty:
        out = out.merge(
            admin_assignment_df,
            how="left",
            on="municipality_ars",
            validate="1:1",
        )

        # Keep geometry-derived AGS as primary, but fill gaps if the table has values.
        out["municipality_ags"] = out["municipality_ags"].fillna(out["municipality_ags_from_table"])
        out["federal_state_name"] = out["federal_state_name_from_table"].fillna(out["federal_state_name"])
    else:
        out["district_ars"] = pd.NA
        out["district_name"] = pd.NA
        out["district_designation"] = pd.NA
        out["district_display_name"] = pd.NA
        out["federal_state_name_from_table"] = pd.NA
        out["federal_state_designation"] = pd.NA
        out["municipality_ags_from_table"] = pd.NA

    keep_columns = [
        "municipality_ags",
        "municipality_ars",
        "municipality_name",
        "municipality_designation",
        "municipality_display_name",
        "district_ars",
        "district_name",
        "district_designation",
        "district_display_name",
        "federal_state_code",
        "federal_state_name",
        "geometry",
    ]

    out = out[keep_columns].copy()

    return out


def resolve_multiple_matches(joined: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Resolve cases where one centroid intersects multiple municipality polygons by
    choosing the lexicographically smallest municipality_ags.

    Such cases should be rare. They are explicitly flagged later.
    """
    if joined.empty:
        return joined

    joined = joined.copy()
    match_counts = joined.groupby("point_row_id").size().rename("match_count")
    joined = joined.merge(match_counts, how="left", on="point_row_id")

    joined = joined.sort_values(
        by=["point_row_id", "municipality_ags"],
        ascending=[True, True],
        kind="mergesort",
    )

    joined = joined.drop_duplicates(subset=["point_row_id"], keep="first").reset_index(drop=True)

    joined["assignment_method"] = joined["match_count"].map(
        lambda n: "polygon_intersects_multiple_resolved_lexicographically" if n and n > 1 else "polygon_intersects"
    )

    joined = joined.drop(columns=["match_count"])

    return joined


def assign_centroids_to_municipalities(
    centroid_gdf: gpd.GeoDataFrame,
    municipality_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Assign each ZIP centroid to exactly one official NRW municipality.

    Primary method:
    - spatial intersection with municipality polygons

    Fallback:
    - nearest municipality polygon for unmatched centroids
    """
    centroids_projected = centroid_gdf.to_crs(municipality_gdf.crs)

    exact_join = gpd.sjoin(
        centroids_projected,
        municipality_gdf,
        how="left",
        predicate="intersects",
    )

    exact_join = resolve_multiple_matches(exact_join)

    matched_exact = exact_join[exact_join["municipality_ags"].notna()].copy()
    matched_exact["distance_to_municipality_m"] = 0.0

    unmatched = exact_join[exact_join["municipality_ags"].isna()].copy()

    if not unmatched.empty:
        unmatched_points = centroids_projected[
            centroids_projected["point_row_id"].isin(unmatched["point_row_id"])
        ].copy()

        try:
            nearest_join = gpd.sjoin_nearest(
                unmatched_points,
                municipality_gdf,
                how="left",
                distance_col="distance_to_municipality_m",
            )
        except Exception as exc:
            first_missing = unmatched["zipcode"].head(20).tolist()
            raise RuntimeError(
                "Some ZIP centroids could not be matched by polygon intersection, and nearest "
                f"fallback also failed. First unmatched ZIPs: {first_missing}"
            ) from exc

        nearest_join = resolve_multiple_matches(nearest_join)
        nearest_join["assignment_method"] = "nearest_municipality_fallback"

        assigned = pd.concat([matched_exact, nearest_join], ignore_index=True, sort=False)
    else:
        assigned = matched_exact.copy()

    if assigned.empty:
        raise RuntimeError("No ZIP centroids could be assigned to NRW municipalities.")

    assigned = assigned.sort_values("zipcode").drop_duplicates(subset=["zipcode"], keep="first")

    missing_mask = assigned["municipality_ags"].isna()
    if missing_mask.any():
        missing_zips = assigned.loc[missing_mask, "zipcode"].head(20).tolist()
        raise RuntimeError(
            f"Missing municipality assignment for {int(missing_mask.sum())} ZIP codes. "
            f"First missing ZIPs: {missing_zips}"
        )

    output_columns = [
        "zipcode",
        "centroid_longitude",
        "centroid_latitude",
        "municipality_ags",
        "municipality_ars",
        "municipality_name",
        "municipality_designation",
        "municipality_display_name",
        "district_ars",
        "district_name",
        "district_designation",
        "district_display_name",
        "federal_state_code",
        "federal_state_name",
        "assignment_method",
        "distance_to_municipality_m",
    ]

    out = assigned[output_columns].copy()

    # Final sanity checks.
    out["zipcode"] = normalize_zipcode_series(out["zipcode"])
    out["municipality_ags"] = normalize_admin_key(out["municipality_ags"], width=8)
    out["municipality_ars"] = normalize_admin_key(out["municipality_ars"], width=12)
    out["federal_state_code"] = FEDERAL_STATE_CODE_NRW
    out["federal_state_name"] = FEDERAL_STATE_NAME_NRW

    duplicate_zipcodes = out["zipcode"].duplicated()
    if duplicate_zipcodes.any():
        bad_zips = out.loc[duplicate_zipcodes, "zipcode"].head(20).tolist()
        raise RuntimeError(
            f"Output contains duplicate ZIP codes, which should not happen. First duplicates: {bad_zips}"
        )

    non_nrw = ~out["municipality_ags"].astype(str).str.startswith(FEDERAL_STATE_CODE_NRW, na=False)
    if non_nrw.any():
        bad_ags = out.loc[non_nrw, "municipality_ags"].head(20).tolist()
        raise RuntimeError(
            "Output contains non-NRW municipality keys, which should not happen. "
            f"First problematic AGS values: {bad_ags}"
        )

    out = out.sort_values("zipcode").reset_index(drop=True)

    return out


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Spatially assign NRW ZIP centroids to official NRW municipalities."
    )
    parser.add_argument(
        "--plz-csv",
        type=Path,
        default=DEFAULT_PLZ_CSV,
        help="Path to the NRW ZIP centroid CSV.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_OUT_CSV,
        help="Path to the output CSV.",
    )
    parser.add_argument(
        "--input-crs",
        type=str,
        default="EPSG:4326",
        help="CRS of the input centroid coordinates. Default: EPSG:4326",
    )
    parser.add_argument(
        "--zipcode-column",
        type=str,
        default=None,
        help="Optional explicit ZIP code column name in the input CSV.",
    )
    parser.add_argument(
        "--longitude-column",
        type=str,
        default=None,
        help="Optional explicit longitude column name in the input CSV.",
    )
    parser.add_argument(
        "--latitude-column",
        type=str,
        default=None,
        help="Optional explicit latitude column name in the input CSV.",
    )
    parser.add_argument(
        "--refresh-boundary-cache",
        action="store_true",
        help="Force re-download and re-extraction of the official VG250 GeoPackage ZIP.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point.
    """
    args = parse_args()

    vg250_gpkg_path = prepare_vg250_gpkg(refresh=args.refresh_boundary_cache)

    centroid_gdf = read_zip_centroids(
        path=args.plz_csv,
        input_crs=args.input_crs,
        zipcode_column=args.zipcode_column,
        longitude_column=args.longitude_column,
        latitude_column=args.latitude_column,
    )

    municipality_gdf = read_nrw_municipality_polygons(gpkg_path=vg250_gpkg_path)

    out = assign_centroids_to_municipalities(
        centroid_gdf=centroid_gdf,
        municipality_gdf=municipality_gdf,
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    fallback_count = int((out["assignment_method"] == "nearest_municipality_fallback").sum())

    print(f"Wrote {len(out):,} rows to: {args.out_csv}")
    print(f"Nearest-fallback assignments: {fallback_count}")


if __name__ == "__main__":
    main()