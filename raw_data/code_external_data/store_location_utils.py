"""Helpers to derive store coordinates for location-based feature downloads.

Problem this module solves
--------------------------
Your current stores table contains a ZIP code but no latitude/longitude. Several external
sources (weather, OSM POIs, air quality) require coordinates.

This module provides a single, defensive code path:
- If the stores file already contains lat/lon: use them.
- Else: derive lat/lon by joining ZIP codes to a centroid reference file.

All paths must be absolute (enforced by the calling scripts).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


def autodetect_column(cols: List[str], preferred: List[str], contains_any: List[str]) -> str:
    """Pick a column name from `cols` using a simple heuristic.

    This is intentionally minimal: we only need robust detection for common column naming
    variations (e.g., store_id vs StoreID).
    """
    lower = {c.lower(): c for c in cols}
    for p in preferred:
        if p.lower() in lower:
            return lower[p.lower()]
    for c in cols:
        cl = c.lower()
        if any(tok in cl for tok in contains_any):
            return c
    raise SystemExit(f"Could not autodetect required column. Available columns: {cols}")


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV or Parquet file into a DataFrame."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def normalize_zip(series: pd.Series) -> pd.Series:
    """Normalize German ZIP codes to a 5-character string.

    We treat ZIP codes as strings because leading zeros are meaningful.
    """
    s = series.astype(str).str.strip()
    # Remove trailing .0 from Excel-like numeric conversions.
    s = s.str.replace(r"\.0$", "", regex=True)
    # Keep digits only where possible.
    s = s.str.replace(r"\D", "", regex=True)
    return s.str.zfill(5)


def load_plz_centroids(plz_centroids_path: Path) -> pd.DataFrame:
    """Load ZIP centroids and standardize to columns: zipcode, lat, lon."""
    df = read_table(plz_centroids_path)
    cols = list(df.columns)

    zip_col = autodetect_column(cols, ["zipcode", "plz", "postal_code"], ["zip", "plz", "postal"])
    lat_col = autodetect_column(cols, ["lat", "latitude"], ["lat"])
    lon_col = autodetect_column(cols, ["lon", "lng", "longitude"], ["lon", "lng", "long"])

    out = df[[zip_col, lat_col, lon_col]].copy()
    out.rename(columns={zip_col: "zipcode", lat_col: "lat", lon_col: "lon"}, inplace=True)

    out["zipcode"] = normalize_zip(out["zipcode"])
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")

    out = out.dropna(subset=["zipcode", "lat", "lon"]).drop_duplicates(subset=["zipcode"]).reset_index(drop=True)
    if out.empty:
        raise SystemExit(
            "ZIP centroid file does not contain usable rows after normalization. "
            "Expected at least columns for ZIP + lat + lon."
        )
    return out


def load_store_locations(
    stores_path: Path,
    plz_centroids_path: Optional[Path],
    *,
    require_complete: bool = True,
) -> pd.DataFrame:
    """Return store locations as: store_id, zipcode, lat, lon.

    Parameters
    ----------
    stores_path:
        Absolute path to your stores table.
    plz_centroids_path:
        Absolute path to `plz_centroids_nrw.csv` (or a compatible file). Required if the stores
        table does not have lat/lon.
    require_complete:
        If True, raise an error if any store ends up without coordinates.

    Notes
    -----
    - We keep `zipcode` in the output because it is still useful for debugging/QA.
    - The calling scripts typically need lat/lon only, but the ZIP is the bridge key.
    """
    df = read_table(stores_path)
    cols = list(df.columns)

    store_id_col = autodetect_column(cols, ["store_id", "storeid", "id"], ["store", "filiale", "shop"])

    # Try lat/lon first.
    lat_candidates = [c for c in cols if c.lower() in ["lat", "latitude"] or "lat" in c.lower()]
    lon_candidates = [c for c in cols if c.lower() in ["lon", "lng", "longitude"] or any(t in c.lower() for t in ["lon", "lng", "long"]) ]

    if lat_candidates and lon_candidates:
        lat_col = lat_candidates[0]
        lon_col = lon_candidates[0]
        out = df[[store_id_col, lat_col, lon_col]].copy()
        out.rename(columns={store_id_col: "store_id", lat_col: "lat", lon_col: "lon"}, inplace=True)
        out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
        out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
        out["zipcode"] = pd.NA
        out = out[["store_id", "zipcode", "lat", "lon"]]
    else:
        # Fallback: use ZIP centroids.
        if plz_centroids_path is None:
            raise SystemExit(
                "Stores file has no lat/lon columns. Provide --plz-centroids-file with an absolute path."
            )

        zip_col = autodetect_column(cols, ["zipcode", "plz", "postal_code"], ["zip", "plz", "postal"])
        out = df[[store_id_col, zip_col]].copy()
        out.rename(columns={store_id_col: "store_id", zip_col: "zipcode"}, inplace=True)
        out["zipcode"] = normalize_zip(out["zipcode"])

        cent = load_plz_centroids(plz_centroids_path)
        out = out.merge(cent, on="zipcode", how="left")

    if require_complete:
        missing = out["lat"].isna() | out["lon"].isna()
        if missing.any():
            n_missing = int(missing.sum())
            examples = out.loc[missing, ["store_id", "zipcode"]].head(10).to_dict(orient="records")
            raise SystemExit(
                f"{n_missing} store(s) have no coordinates after ZIP centroid join. "
                f"Examples: {examples}"
            )

    return out.reset_index(drop=True)

