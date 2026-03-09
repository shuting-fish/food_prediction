"""Utilities to enforce ABSOLUTE paths in all external downloaders.

Design goals
------------
- Every path written to disk must be absolute.
- The default base directory is explicitly configured for your machine.
- Scripts must work even if you run them from an arbitrary working directory.

Important
---------
This module intentionally raises hard errors for non-absolute paths. This is to prevent
silent miswrites into unexpected folders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


# Base directory requested by the user. All scripts and outputs live underneath this folder.
# Note: forward slashes work on Windows and avoid escaping issues.
DEFAULT_BASE_DIR = Path(r"C:/Users/simon/food_prediction/raw_data/code_external_data")

# Subfolder (under DEFAULT_BASE_DIR) used for all downloaded external datasets.
DEFAULT_DATA_SUBDIR = "_external_data"

# Subfolder (under DEFAULT_BASE_DIR) for small reference files required to compute features.
# Example: ZIP -> centroid mapping used to derive store coordinates.
DEFAULT_REFERENCE_SUBDIR = "_reference_geo"

# Default expected location of the ZIP centroid file.
DEFAULT_PLZ_CENTROIDS_PATH = (DEFAULT_BASE_DIR / DEFAULT_REFERENCE_SUBDIR / "plz_centroids_nrw.csv").resolve()


def require_absolute_path(path_str: str, flag_name: str) -> Path:
    """Validate a CLI path argument.

    Parameters
    ----------
    path_str:
        Path string passed by the user (e.g., from argparse).
    flag_name:
        CLI flag name (used for clear error messages).

    Returns
    -------
    Path
        A resolved absolute Path.

    Raises
    ------
    SystemExit:
        If the provided path is not absolute.
    """
    p = Path(path_str)
    if not p.is_absolute():
        raise SystemExit(f"{flag_name} must be an absolute path. Got: {path_str}")
    return p.resolve()


def get_base_dir(base_dir_arg: Optional[str]) -> Path:
    """Return the absolute base directory used by the downloaders."""
    if base_dir_arg:
        base = require_absolute_path(base_dir_arg, "--base-dir")
    else:
        base = DEFAULT_BASE_DIR.resolve()

    # Create it if it doesn't exist yet (first-time setup).
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_repo_root(base_dir: Path) -> Path:
    """Infer the repository root from the absolute base directory.

    Assumption: <repo_root>/raw_data/code_external_data == base_dir
    """
    return base_dir.parents[1].resolve()


def get_output_dir(base_dir: Path, source_name: str) -> Path:
    """Create and return the output directory for a specific data source."""
    out_dir = (base_dir / DEFAULT_DATA_SUBDIR / source_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_reference_dir(base_dir: Path) -> Path:
    """Create and return the directory for reference files (e.g., ZIP centroids)."""
    ref_dir = (base_dir / DEFAULT_REFERENCE_SUBDIR).resolve()
    ref_dir.mkdir(parents=True, exist_ok=True)
    return ref_dir


def get_default_plz_centroids_path(base_dir: Path) -> Path:
    """Return the default expected path of `plz_centroids_nrw.csv` for this project."""
    return (base_dir / DEFAULT_REFERENCE_SUBDIR / "plz_centroids_nrw.csv").resolve()

