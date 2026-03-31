from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
CENSUS_RAW_DIR = SCRIPT_DIR / "census_raw"
DEFAULT_DESTATIS_XLSX = CENSUS_RAW_DIR / "destatis_gvisys_31122024.xlsx"
DEFAULT_SOURCE_CSV = CENSUS_RAW_DIR / "municipality_population_area_source.csv"
DEFAULT_CENSUS_RAW_CSV = CENSUS_RAW_DIR / "municipality_census_raw.csv"
DEFAULT_UNRESOLVED_CSV = CENSUS_RAW_DIR / "municipality_population_area_unresolved.csv"
DEFAULT_FEATURE_BASE_SCRIPT = SCRIPT_DIR / "build_municipality_census_feature_base.py"

REQUIRED_RAW_COLUMNS = [
    "municipality_ags",
    "municipality_name",
    "area_sq_km",
    "total_population",
]
REQUIRED_SOURCE_COLUMNS = [
    "municipality_ags",
    "municipality_name",
    "area_sq_km",
    "total_population",
]


class PipelineError(RuntimeError):
    """Hard failure for deterministic pipeline errors."""


def log(message: str) -> None:
    print(f"[fill_municipality_census_population_area] {message}")


def normalize_text(value: object) -> str:
    if pd.isna(value):
        raise PipelineError("Encountered missing municipality_name in Destatis workbook.")
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text:
        raise PipelineError("Encountered empty municipality_name in Destatis workbook.")
    return text


def normalize_code(value: object, width: int, field_name: str) -> str:
    if pd.isna(value):
        raise PipelineError(f"Encountered missing {field_name} value.")

    text = str(value).strip()
    if not text:
        raise PipelineError(f"Encountered empty {field_name} value.")

    text = text.replace("\xa0", "").replace(" ", "")

    if re.fullmatch(r"\d+", text):
        number_text = text
    elif re.fullmatch(r"\d+\.0+", text):
        number_text = text.split(".", 1)[0]
    else:
        raise PipelineError(
            f"Invalid {field_name} value '{value}' encountered while building municipality_ags."
        )

    if len(number_text) > width:
        raise PipelineError(
            f"Invalid {field_name} value '{value}': exceeds expected width {width}."
        )

    return number_text.zfill(width)


def parse_numeric(value: object, field_name: str, integer: bool) -> float | int:
    if pd.isna(value):
        raise PipelineError(f"Encountered missing {field_name} value.")

    if isinstance(value, bool):
        raise PipelineError(f"Invalid boolean {field_name} value '{value}'.")

    if isinstance(value, int):
        return int(value) if integer else float(value)

    if isinstance(value, float):
        if integer and not float(value).is_integer():
            raise PipelineError(f"Invalid non-integer {field_name} value '{value}'.")
        return int(value) if integer else float(value)

    text = str(value).strip()
    if not text:
        raise PipelineError(f"Encountered empty {field_name} value.")

    text = text.replace("\xa0", "").replace(" ", "")

    if integer:
        if re.fullmatch(r"\d+", text):
            return int(text)
        if re.fullmatch(r"\d+\.0+", text):
            return int(text.split(".", 1)[0])
        if re.fullmatch(r"\d{1,3}(?:[\.,]\d{3})+", text):
            return int(text.replace(".", "").replace(",", ""))
        raise PipelineError(f"Invalid integer {field_name} value '{value}'.")

    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            normalized = text.replace(".", "").replace(",", ".")
        else:
            normalized = text.replace(",", "")
    elif "," in text:
        normalized = text.replace(",", ".")
    else:
        normalized = text

    try:
        return float(normalized)
    except ValueError as exc:
        raise PipelineError(f"Invalid float {field_name} value '{value}'.") from exc


def read_csv_with_string_ags(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise PipelineError(f"Required CSV file does not exist: {path}")
    return pd.read_csv(path, dtype={"municipality_ags": "string"})


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        delete=False,
        dir=path.parent,
        suffix=".tmp",
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        df.to_csv(tmp_file, index=False)
    tmp_path.replace(path)


def detect_destatis_sheet(xlsx_path: Path, explicit_sheet_name: str | None) -> str:
    workbook = pd.ExcelFile(xlsx_path, engine="openpyxl")

    if explicit_sheet_name is not None:
        if explicit_sheet_name not in workbook.sheet_names:
            raise PipelineError(
                f"Sheet '{explicit_sheet_name}' not found in workbook. Available sheets: {workbook.sheet_names}"
            )
        return explicit_sheet_name

    candidates: list[tuple[str, int]] = []
    for sheet_name in workbook.sheet_names:
        probe_df = pd.read_excel(
            workbook,
            sheet_name=sheet_name,
            header=None,
            usecols=[0],
            engine="openpyxl",
        )
        satzart_numeric = pd.to_numeric(probe_df.iloc[:, 0], errors="coerce")
        municipality_row_count = int((satzart_numeric == 60).sum())
        if municipality_row_count > 0:
            candidates.append((sheet_name, municipality_row_count))

    if not candidates:
        raise PipelineError("No worksheet with Satzart == 60 rows found in Destatis workbook.")

    if len(candidates) > 1:
        raise PipelineError(
            "Auto sheet detection is ambiguous. Multiple sheets contain Satzart == 60 rows: "
            f"{candidates}. Pass --sheet-name explicitly."
        )

    return candidates[0][0]


def load_destatis_municipalities(xlsx_path: Path, sheet_name: str | None) -> pd.DataFrame:
    if not xlsx_path.exists():
        raise PipelineError(f"Destatis workbook does not exist: {xlsx_path}")

    resolved_sheet_name = detect_destatis_sheet(xlsx_path, sheet_name)
    log(f"Using Destatis worksheet: {resolved_sheet_name}")

    raw_df = pd.read_excel(
        xlsx_path,
        sheet_name=resolved_sheet_name,
        header=None,
        usecols=list(range(10)),
        engine="openpyxl",
    )

    if raw_df.shape[1] < 10:
        raise PipelineError(
            f"Destatis worksheet has only {raw_df.shape[1]} columns. Expected at least 10 columns."
        )

    # Forward-fill hierarchical AGS components to compensate for Excel merged-cell exports.
    raw_df[[2, 3, 4]] = raw_df[[2, 3, 4]].ffill()

    satzart_numeric = pd.to_numeric(raw_df[0], errors="coerce")
    municipality_df = raw_df.loc[satzart_numeric == 60, list(range(10))].copy()

    if municipality_df.empty:
        raise PipelineError("No municipality rows (Satzart == 60) found after worksheet load.")

    municipality_df = municipality_df.rename(
        columns={
            0: "satzart",
            2: "land",
            3: "rb",
            4: "kreis",
            5: "vb",
            6: "gemeinde",
            7: "municipality_name",
            8: "area_sq_km",
            9: "total_population",
        }
    )

    municipality_df["municipality_ags"] = (
        municipality_df["land"].map(lambda value: normalize_code(value, 2, "Land"))
        + municipality_df["rb"].map(lambda value: normalize_code(value, 1, "RB"))
        + municipality_df["kreis"].map(lambda value: normalize_code(value, 2, "Kreis"))
        + municipality_df["gemeinde"].map(lambda value: normalize_code(value, 3, "Gemeinde"))
    )
    municipality_df["municipality_name"] = municipality_df["municipality_name"].map(normalize_text)
    municipality_df["area_sq_km"] = municipality_df["area_sq_km"].map(
        lambda value: parse_numeric(value, "area_sq_km", integer=False)
    )
    municipality_df["total_population"] = municipality_df["total_population"].map(
        lambda value: parse_numeric(value, "total_population", integer=True)
    )

    municipality_df = municipality_df[
        ["municipality_ags", "municipality_name", "area_sq_km", "total_population"]
    ].copy()

    municipality_df = municipality_df.sort_values("municipality_ags").reset_index(drop=True)
    log(f"Loaded {len(municipality_df)} municipality rows from Destatis workbook.")
    return municipality_df


def normalize_target_ags(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if "municipality_ags" not in df.columns:
        raise PipelineError(f"{source_name} is missing required column 'municipality_ags'.")

    normalized_df = df.copy()
    normalized_df["municipality_ags"] = normalized_df["municipality_ags"].map(
        lambda value: normalize_code(value, 8, f"{source_name}.municipality_ags")
    )
    return normalized_df


def fail_on_ambiguous_target_ags(
    target_df: pd.DataFrame,
    destatis_df: pd.DataFrame,
    unresolved_csv: Path,
) -> None:
    duplicate_counts = destatis_df["municipality_ags"].value_counts()
    ambiguous_ags = duplicate_counts[duplicate_counts > 1].index.tolist()
    if not ambiguous_ags:
        return

    ambiguous_target_df = target_df[target_df["municipality_ags"].isin(ambiguous_ags)].copy()
    if ambiguous_target_df.empty:
        raise PipelineError(
            "Duplicate municipality_ags values exist in Destatis municipality rows. "
            "The workbook is structurally ambiguous and must be fixed before import. "
            f"Affected AGS examples: {ambiguous_ags[:10]}"
        )

    ambiguous_target_df = ambiguous_target_df.drop_duplicates("municipality_ags")
    ambiguous_target_df["reason"] = "ambiguous_duplicate_in_destatis"
    ambiguous_target_df["candidate_count"] = ambiguous_target_df["municipality_ags"].map(duplicate_counts)
    ambiguous_target_df = ambiguous_target_df.sort_values("municipality_ags").reset_index(drop=True)
    write_csv_atomic(ambiguous_target_df, unresolved_csv)
    raise PipelineError(
        "Ambiguous duplicate municipality_ags values found in Destatis for target municipalities. "
        f"Unresolved file written to: {unresolved_csv}"
    )


def prepare_target_municipalities(census_raw_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = read_csv_with_string_ags(census_raw_csv)

    missing_columns = [column for column in REQUIRED_RAW_COLUMNS if column not in raw_df.columns]
    if missing_columns:
        raise PipelineError(
            f"municipality_census_raw.csv is missing required columns: {missing_columns}"
        )

    raw_df = normalize_target_ags(raw_df, "municipality_census_raw.csv")
    target_df = raw_df[["municipality_ags", "municipality_name"]].drop_duplicates("municipality_ags")

    log(
        "Loaded municipality_census_raw.csv with "
        f"{len(raw_df)} rows and {len(target_df)} unique municipality_ags values."
    )
    return raw_df, target_df


def resolve_targets_against_destatis(
    target_df: pd.DataFrame,
    destatis_df: pd.DataFrame,
    unresolved_csv: Path,
) -> pd.DataFrame:
    merged_df = target_df.merge(destatis_df, on="municipality_ags", how="left", suffixes=("_target", ""))

    unresolved_mask = merged_df[["municipality_name", "area_sq_km", "total_population"]].isna().any(axis=1)
    if unresolved_mask.any():
        unresolved_df = merged_df.loc[unresolved_mask, ["municipality_ags", "municipality_name_target"]].rename(
            columns={"municipality_name_target": "municipality_name"}
        )
        unresolved_df["reason"] = "missing_in_destatis"
        unresolved_df = unresolved_df.sort_values("municipality_ags").reset_index(drop=True)
        write_csv_atomic(unresolved_df, unresolved_csv)
        raise PipelineError(
            "Not all municipalities from municipality_census_raw.csv could be resolved from Destatis. "
            f"Unresolved file written to: {unresolved_csv}"
        )

    resolved_df = merged_df[
        ["municipality_ags", "municipality_name", "area_sq_km", "total_population"]
    ].copy()
    resolved_df = resolved_df.sort_values("municipality_ags").reset_index(drop=True)

    if unresolved_csv.exists():
        unresolved_csv.unlink()

    log(f"Resolved {len(resolved_df)} of {len(target_df)} target municipalities from Destatis.")
    return resolved_df


def build_source_output(source_csv: Path, resolved_df: pd.DataFrame) -> pd.DataFrame:
    if source_csv.exists():
        existing_source_df = read_csv_with_string_ags(source_csv)
        existing_source_df = normalize_target_ags(existing_source_df, "municipality_population_area_source.csv")

        if existing_source_df["municipality_ags"].duplicated().any():
            duplicates = existing_source_df.loc[
                existing_source_df["municipality_ags"].duplicated(keep=False), "municipality_ags"
            ].tolist()
            raise PipelineError(
                "municipality_population_area_source.csv contains duplicate municipality_ags values: "
                f"{duplicates}"
            )

        final_columns = list(existing_source_df.columns)
        for column in REQUIRED_SOURCE_COLUMNS:
            if column not in final_columns:
                final_columns.append(column)

        retained_extra_columns = [
            column for column in existing_source_df.columns if column not in REQUIRED_SOURCE_COLUMNS
        ]
        source_out_df = resolved_df.merge(
            existing_source_df[["municipality_ags", *retained_extra_columns]],
            on="municipality_ags",
            how="left",
        )
        source_out_df = source_out_df[final_columns]
    else:
        source_out_df = resolved_df[REQUIRED_SOURCE_COLUMNS].copy()

    source_out_df = source_out_df.sort_values("municipality_ags").reset_index(drop=True)
    return source_out_df


def update_census_raw(raw_df: pd.DataFrame, resolved_df: pd.DataFrame) -> pd.DataFrame:
    resolved_lookup = resolved_df.set_index("municipality_ags")
    updated_raw_df = raw_df.copy()

    updated_raw_df["municipality_name"] = updated_raw_df["municipality_ags"].map(
        resolved_lookup["municipality_name"]
    )
    updated_raw_df["area_sq_km"] = updated_raw_df["municipality_ags"].map(
        resolved_lookup["area_sq_km"]
    )
    updated_raw_df["total_population"] = updated_raw_df["municipality_ags"].map(
        resolved_lookup["total_population"]
    )

    if updated_raw_df[["municipality_name", "area_sq_km", "total_population"]].isna().any().any():
        raise PipelineError(
            "Internal error while updating municipality_census_raw.csv: unexpected missing values after AGS mapping."
        )

    return updated_raw_df


def maybe_run_feature_base(build_script: Path) -> None:
    if not build_script.exists():
        raise PipelineError(f"Feature base build script does not exist: {build_script}")

    log(f"Running downstream script: {build_script}")
    subprocess.run([sys.executable, str(build_script)], check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill municipality population and area fields from a local Destatis GV workbook."
    )
    parser.add_argument("--destatis-xlsx", type=Path, default=DEFAULT_DESTATIS_XLSX)
    parser.add_argument("--sheet-name", type=str, default=None)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--census-raw-csv", type=Path, default=DEFAULT_CENSUS_RAW_CSV)
    parser.add_argument("--unresolved-csv", type=Path, default=DEFAULT_UNRESOLVED_CSV)
    parser.add_argument("--run-feature-base", action="store_true")
    parser.add_argument("--feature-base-script", type=Path, default=DEFAULT_FEATURE_BASE_SCRIPT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_df, target_df = prepare_target_municipalities(args.census_raw_csv)
    destatis_df = load_destatis_municipalities(args.destatis_xlsx, args.sheet_name)
    fail_on_ambiguous_target_ags(target_df, destatis_df, args.unresolved_csv)
    resolved_df = resolve_targets_against_destatis(target_df, destatis_df, args.unresolved_csv)

    source_out_df = build_source_output(args.source_csv, resolved_df)
    updated_raw_df = update_census_raw(raw_df, resolved_df)

    write_csv_atomic(source_out_df, args.source_csv)
    log(f"Wrote {len(source_out_df)} rows to {args.source_csv}")

    write_csv_atomic(updated_raw_df, args.census_raw_csv)
    log(f"Updated {len(updated_raw_df)} rows in {args.census_raw_csv}")

    if args.run_feature_base:
        maybe_run_feature_base(args.feature_base_script)

    log("Done.")


if __name__ == "__main__":
    try:
        main()
    except PipelineError as exc:
        log(f"ERROR: {exc}")
        raise SystemExit(1) from exc
    except subprocess.CalledProcessError as exc:
        log(f"ERROR: Downstream script failed with exit code {exc.returncode}.")
        raise SystemExit(exc.returncode) from exc