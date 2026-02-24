#!/usr/bin/env python3
"""
RAW data EDA (single PDF) for daily store and product sales forecasting.

Output:
- RAW_DATA_EDA.pdf in <raw_data>/visualized_raw_data_analysis/

Key behavior:
- Streams the sales parquet by row groups (RAM-sparing).
- Strict centroid mode if geo/plz_centroids.csv exists:
  must match store ZIPs, otherwise hard fail.
- one chart/table per page.
- Each page is logged with page number and title.
- Each page has a footer stamp with script version + timestamp.

"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import fastparquet  # required
import openpyxl  # noqa: F401
import xlsxwriter  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    sns = None
    HAS_SEABORN = False


# ============================
# Constants
# ============================
SCRIPT_VERSION = "v3.1-weekday-fix-group-normalized"
EXTRACT_PREFIX = "20260218_144523"

THEME = {
    "fig_bg": "#F4F7FB",
    "panel_bg": "#E8F0FF",      # tinted plot background for ALL pages
    "grid": "#D0D7DE",
    "text": "#1F2328",
    "marine": "#0B3D91",
    "blue": "#1F77B4",
    "orange": "#E69F00",
    "black": "#111111",
    "grey": "#6E7781",
}

LOG = logging.getLogger("raw_eda")


# ============================
# Dataclasses
# ============================
@dataclass(frozen=True)
class SalesColumns:
    date: str
    store_id: str
    product_id: str
    target: str


@dataclass(frozen=True)
class WeatherColumns:
    date: str
    temp: Optional[str]
    precip: Optional[str]


@dataclass(frozen=True)
class HolidayColumns:
    date: str
    holiday_name: Optional[str]


@dataclass(frozen=True)
class StoreColumns:
    store_id: str
    zip_code: Optional[str]


@dataclass
class Aggregates:
    daily_total: Dict[pd.Timestamp, float]
    store_total: Dict[str, float]
    product_total: Dict[str, float]
    weekday_total: Dict[int, float]
    weekday_count: Dict[int, int]
    month_total: Dict[pd.Timestamp, float]
    missing_target_rows: int
    total_rows: int


# ============================
# Logging
# ============================
def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOG.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    LOG.handlers.clear()
    LOG.addHandler(ch)
    LOG.addHandler(fh)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)


# ============================
# Helpers
# ============================
def _pick_first_existing(cols: List[str], candidates: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def _coerce_date(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s, errors="coerce")


def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def normalize_zip(x: object) -> Optional[str]:
    """Normalize German ZIP / PLZ to 5 digits (keeps leading zeros)."""
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return None
    if len(digits) < 5:
        digits = digits.zfill(5)
    if len(digits) > 5:
        digits = digits[:5]
    return digits


def _dict_add(d: Dict, k, v) -> None:
    d[k] = d.get(k, 0) + v


def mpl_version_tuple() -> Tuple[int, int, int]:
    parts = (matplotlib.__version__.split("+", 1)[0]).split(".")
    nums: List[int] = []
    for p in parts[:3]:
        try:
            nums.append(int("".join(ch for ch in p if ch.isdigit()) or "0"))
        except Exception:
            nums.append(0)
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums)  # type: ignore[return-value]


def boxplot_with_labels(ax: plt.Axes, groups: List[pd.Series], labels: List[str], showfliers: bool = False) -> dict:
    """Matplotlib 3.9 renamed labels -> tick_labels; keep compatibility."""
    if mpl_version_tuple() >= (3, 9, 0):
        return ax.boxplot(groups, tick_labels=labels, showfliers=showfliers, patch_artist=True)
    return ax.boxplot(groups, labels=labels, showfliers=showfliers, patch_artist=True)


def linear_trend(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Simple linear trend line (robust to NaNs)."""
    m = np.isfinite(x) & np.isfinite(y)
    x2 = x[m]
    y2 = y[m]
    if x2.size < 2:
        return np.array([]), np.array([])
    a, b = np.polyfit(x2, y2, 1)
    xs = np.linspace(float(x2.min()), float(x2.max()), 200)
    ys = a * xs + b
    return xs, ys


# ============================
# Robust path resolving
# ============================
def resolve_raw_data_dir(script_path: Path, override: Optional[str]) -> Path:
    if override:
        p = Path(override).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"--raw-data-dir does not exist: {p}")
        return p

    expected = f"{EXTRACT_PREFIX}_sales_data.parquet"

    candidates: List[Path] = []

    for parent in script_path.parents:
        if parent.name.lower() == "raw_data":
            candidates.append(parent)
            break

    if len(script_path.parents) > 1:
        candidates.append(script_path.parents[1])

    candidates.append(Path.cwd())

    seen = set()
    uniq: List[Path] = []
    for c in candidates:
        c = c.resolve()
        if c not in seen:
            uniq.append(c)
            seen.add(c)

    for c in uniq:
        if (c / expected).exists():
            return c
        if list(c.glob("*sales_data.parquet")):
            return c

    raise FileNotFoundError(
        "Could not locate raw_data automatically. "
        "Run with: --raw-data-dir \"C:\\Users\\simon\\Food_Prediction\\raw_data\""
    )


def resolve_input_files(raw_dir: Path) -> Dict[str, Path]:
    exact = {
        "sales": raw_dir / f"{EXTRACT_PREFIX}_sales_data.parquet",
        "weather": raw_dir / f"{EXTRACT_PREFIX}_weather.parquet",
        "holidays": raw_dir / f"{EXTRACT_PREFIX}_holidays.parquet",
        "stores": raw_dir / f"{EXTRACT_PREFIX}_stores.parquet",
    }
    if all(p.exists() for p in exact.values()):
        return exact

    patterns = {
        "sales": "*sales_data.parquet",
        "weather": "*weather.parquet",
        "holidays": "*holidays.parquet",
        "stores": "*stores.parquet",
    }
    out: Dict[str, Path] = {}
    for key, pat in patterns.items():
        matches = sorted(raw_dir.glob(pat))
        if not matches:
            raise FileNotFoundError(f"Missing parquet for '{key}' with pattern {pat} in {raw_dir}")
        out[key] = matches[-1]
    return out


# ============================
# Parquet helpers
# ============================
def _parquetfile(path: Path) -> fastparquet.ParquetFile:
    return fastparquet.ParquetFile(str(path))


def parquet_row_groups(path: Path, columns: Optional[List[str]] = None) -> Iterable[pd.DataFrame]:
    pf = _parquetfile(path)

    if hasattr(pf, "iter_row_groups"):
        for rg_df in pf.iter_row_groups(columns=columns):
            yield rg_df
        return

    rgs = getattr(pf, "row_groups", None)
    if rgs is not None:
        for i in range(len(rgs)):
            yield pf.to_pandas(row_group=i, columns=columns)
        return

    yield pf.to_pandas(columns=columns)


def parquet_columns(path: Path) -> List[str]:
    pf = _parquetfile(path)
    cols = getattr(pf, "columns", None)
    if cols:
        return list(cols)

    schema = getattr(pf, "schema", None)
    if schema is not None:
        names = getattr(schema, "names", None)
        if names:
            return list(names)

    first = next(parquet_row_groups(path, columns=None))
    return list(first.columns)


def read_parquet_sample(path: Path, candidate_cols: List[str], n_rows: int = 50_000) -> pd.DataFrame:
    schema_cols = parquet_columns(path)
    schema_low = {c.lower(): c for c in schema_cols}
    desired = [schema_low[c.lower()] for c in candidate_cols if c.lower() in schema_low]
    chunk = next(parquet_row_groups(path, columns=desired if desired else None))
    return chunk.head(n_rows).copy()


# ============================
# Column inference
# ============================
def infer_sales_columns(sample: pd.DataFrame) -> SalesColumns:
    cols = list(sample.columns)

    date_col = _pick_first_existing(cols, ["date", "day", "sales_date", "ds"])
    store_col = _pick_first_existing(cols, ["store_id", "store", "shop_id", "branch_id", "filiale_id"])
    prod_col = _pick_first_existing(cols, ["item_id", "product_id", "sku", "article_id", "goods_id", "item", "product"])
    if date_col is None or store_col is None or prod_col is None:
        raise ValueError(f"Could not infer sales keys. Found columns: {cols}")

    target = _pick_first_existing(cols, ["sold_quantity", "qty", "quantity", "units", "sales_qty", "y", "target", "demand", "sales"])
    if target is None:
        numeric_cols = [c for c in cols if _is_numeric_series(sample[c]) and c not in {store_col, prod_col}]
        if not numeric_cols:
            raise ValueError(f"No numeric target candidate found in sales. Columns: {cols}")
        scores: List[Tuple[float, str]] = []
        for c in numeric_cols:
            s = sample[c].dropna()
            if s.empty:
                continue
            scores.append((float(np.nanvar(s.values)) * float(len(s)), c))
        scores.sort(reverse=True)
        target = scores[0][1]

    return SalesColumns(date=date_col, store_id=store_col, product_id=prod_col, target=target)


def infer_weather_columns(sample: pd.DataFrame) -> WeatherColumns:
    cols = list(sample.columns)
    date_col = _pick_first_existing(cols, ["date", "day", "weather_date", "ds"])
    if date_col is None:
        raise ValueError(f"Could not infer weather date column. Found columns: {cols}")
    temp_col = _pick_first_existing(cols, ["temperature", "temp", "tavg", "tmean", "t"])
    precip_col = _pick_first_existing(cols, ["precip", "precipitation", "rain", "prcp"])
    return WeatherColumns(date=date_col, temp=temp_col, precip=precip_col)


def infer_holiday_columns(sample: pd.DataFrame) -> HolidayColumns:
    cols = list(sample.columns)
    date_col = _pick_first_existing(cols, ["date", "day", "holiday_date", "ds"])
    if date_col is None:
        raise ValueError(f"Could not infer holidays date column. Found columns: {cols}")
    name_col = _pick_first_existing(cols, ["holiday_name", "name", "holiday", "event"])
    return HolidayColumns(date=date_col, holiday_name=name_col)


def infer_store_columns(sample: pd.DataFrame) -> StoreColumns:
    cols = list(sample.columns)
    store_col = _pick_first_existing(cols, ["store_id", "store", "shop_id", "branch_id", "filiale_id"])
    if store_col is None:
        raise ValueError(f"Could not infer store_id column in stores. Found columns: {cols}")
    zip_col = _pick_first_existing(cols, ["zipcode", "zip_code", "plz", "postal_code"])
    return StoreColumns(store_id=store_col, zip_code=zip_col)


# ============================
# Load small tables
# ============================
def load_holidays(path: Path, cols: HolidayColumns) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="fastparquet").copy()
    df["_date"] = _coerce_date(df[cols.date]).dt.normalize()
    df = df.dropna(subset=["_date"])

    if cols.holiday_name and cols.holiday_name in df.columns:
        name = df[cols.holiday_name].astype(str).str.strip()
        df["is_holiday"] = (name.notna()) & (name != "") & (name.str.lower() != "nan")
    else:
        df["is_holiday"] = True

    return df[["_date", "is_holiday"]].drop_duplicates()


def load_weather(path: Path, cols: WeatherColumns) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="fastparquet").copy()
    df["_date"] = _coerce_date(df[cols.date]).dt.normalize()
    df = df.dropna(subset=["_date"])

    out = pd.DataFrame({"_date": df["_date"]})
    if cols.temp and cols.temp in df.columns:
        out["_temp"] = pd.to_numeric(df[cols.temp], errors="coerce")
    if cols.precip and cols.precip in df.columns:
        out["_precip"] = pd.to_numeric(df[cols.precip], errors="coerce")

    return out.groupby("_date", as_index=False).mean(numeric_only=True)


def load_stores(path: Path, cols: StoreColumns) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="fastparquet").copy()
    df[cols.store_id] = df[cols.store_id].astype(str)

    out = pd.DataFrame({cols.store_id: df[cols.store_id]})
    if cols.zip_code and cols.zip_code in df.columns:
        out["_zip"] = df[cols.zip_code].map(normalize_zip)
    return out.drop_duplicates(subset=[cols.store_id])


def load_plz_centroids_strict(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=",", encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=",", encoding="cp1252")

    if df.shape[1] == 1:
        try:
            df = pd.read_csv(path, sep=";", encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, sep=";", encoding="cp1252")

    cols = list(df.columns)
    zip_col = _pick_first_existing(cols, ["plz", "zip", "zipcode", "zip_code", "postal_code"])
    lat_col = _pick_first_existing(cols, ["lat", "latitude", "y"])
    lon_col = _pick_first_existing(cols, ["lon", "lng", "longitude", "x"])

    if zip_col is None or lat_col is None or lon_col is None:
        raise ValueError(f"Centroids CSV invalid. Need zip+lat+lng/lon. Found columns: {cols}")

    out = pd.DataFrame()
    out["_zip"] = df[zip_col].map(normalize_zip)
    out["_lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    out["_lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    out = out.dropna(subset=["_zip", "_lat", "_lon"]).drop_duplicates(subset=["_zip"])

    if out.shape[0] < 1000:
        raise ValueError(f"Centroids CSV looks too small after cleaning (rows={out.shape[0]}).")

    return out


def enforce_centroids_match_strict(stores: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    if stores is None or stores.empty:
        raise ValueError("Centroids file exists, but stores data is missing/empty.")
    if "_zip" not in stores.columns or stores["_zip"].isna().all():
        raise ValueError("Centroids file exists, but stores have no usable ZIP codes in stores.parquet.")

    merged = stores.merge(centroids[["_zip", "_lat", "_lon"]], on="_zip", how="left")
    matched = int(merged["_lat"].notna().sum())
    total = int(merged.shape[0])

    if matched < total:
        missing = total - matched
        raise ValueError(f"Centroids strict mode failed: {missing} stores have no coordinates. Fix ZIP mismatch.")

    return merged


# ============================
# Aggregations
# ============================
def first_pass_aggregates(sales_path: Path, cols: SalesColumns, max_rowgroups: Optional[int] = None) -> Aggregates:
    daily_total: Dict[pd.Timestamp, float] = {}
    store_total: Dict[str, float] = {}
    product_total: Dict[str, float] = {}
    weekday_total: Dict[int, float] = {i: 0.0 for i in range(7)}
    weekday_count: Dict[int, int] = {i: 0 for i in range(7)}
    month_total: Dict[pd.Timestamp, float] = {}
    missing_target_rows = 0
    total_rows = 0

    use_cols = [cols.date, cols.store_id, cols.product_id, cols.target]

    for rg_i, chunk in enumerate(parquet_row_groups(sales_path, columns=use_cols)):
        if max_rowgroups is not None and rg_i >= max_rowgroups:
            break

        total_rows += int(chunk.shape[0])

        dt = _coerce_date(chunk[cols.date])
        y = pd.to_numeric(chunk[cols.target], errors="coerce")
        missing_target_rows += int(y.isna().sum())

        chunk = chunk.assign(_date=dt, _y=y).dropna(subset=["_date", "_y"])
        if chunk.empty:
            continue

        chunk["_date"] = chunk["_date"].dt.normalize()
        chunk[cols.store_id] = chunk[cols.store_id].astype(str)
        chunk[cols.product_id] = chunk[cols.product_id].astype(str)

        for k, v in chunk.groupby("_date")["_y"].sum().items():
            _dict_add(daily_total, pd.Timestamp(k), float(v))

        for k, v in chunk.groupby(cols.store_id)["_y"].sum().items():
            _dict_add(store_total, str(k), float(v))

        for k, v in chunk.groupby(cols.product_id)["_y"].sum().items():
            _dict_add(product_total, str(k), float(v))

        # kept for diagnostics but NOT used for page 3 anymore
        w = chunk["_date"].dt.weekday
        for wd, val in chunk.groupby(w)["_y"].sum().items():
            weekday_total[int(wd)] += float(val)
        for wd, cnt in chunk.groupby(w)["_y"].size().items():
            weekday_count[int(wd)] += int(cnt)

        months = chunk["_date"].dt.to_period("M").dt.to_timestamp()
        for m, val in chunk.groupby(months)["_y"].sum().items():
            _dict_add(month_total, pd.Timestamp(m), float(val))

    return Aggregates(
        daily_total=daily_total,
        store_total=store_total,
        product_total=product_total,
        weekday_total=weekday_total,
        weekday_count=weekday_count,
        month_total=month_total,
        missing_target_rows=missing_target_rows,
        total_rows=total_rows,
    )


def make_store_groups(store_total: pd.Series) -> Dict[str, str]:
    """Hotspot >= 80th percentile; Quiet Zone <= 20th percentile; else Middle."""
    if store_total.empty:
        return {}
    q80 = float(store_total.quantile(0.80))
    q20 = float(store_total.quantile(0.20))
    out: Dict[str, str] = {}
    for sid, val in store_total.items():
        if float(val) >= q80:
            out[str(sid)] = "Hotspot"
        elif float(val) <= q20:
            out[str(sid)] = "Quiet Zone"
        else:
            out[str(sid)] = "Middle"
    return out


def count_store_groups(store_group_map: Dict[str, str]) -> Dict[str, int]:
    counts = {"Hotspot": 0, "Middle": 0, "Quiet Zone": 0}
    for g in store_group_map.values():
        if g in counts:
            counts[g] += 1
    return counts


def second_pass_store_group_timeseries(
    sales_path: Path,
    cols: SalesColumns,
    store_group_map: Dict[str, str],
    max_rowgroups: Optional[int] = None,
) -> pd.DataFrame:
    """Daily SUM sales for 3 store groups. Normalization happens later in the PDF build."""
    series: Dict[Tuple[pd.Timestamp, str], float] = {}
    use_cols = [cols.date, cols.store_id, cols.target]

    for rg_i, chunk in enumerate(parquet_row_groups(sales_path, columns=use_cols)):
        if max_rowgroups is not None and rg_i >= max_rowgroups:
            break

        dt = _coerce_date(chunk[cols.date])
        y = pd.to_numeric(chunk[cols.target], errors="coerce")
        chunk = chunk.assign(_date=dt, _y=y).dropna(subset=["_date", "_y"])
        if chunk.empty:
            continue

        chunk["_date"] = chunk["_date"].dt.normalize()
        chunk[cols.store_id] = chunk[cols.store_id].astype(str)
        chunk["_grp"] = chunk[cols.store_id].map(lambda s: store_group_map.get(str(s), "Middle"))

        g = chunk.groupby(["_date", "_grp"])["_y"].sum()
        for (d, grp), v in g.items():
            _dict_add(series, (pd.Timestamp(d), str(grp)), float(v))

    if not series:
        return pd.DataFrame()

    idx = pd.MultiIndex.from_tuples(series.keys(), names=["date", "group"])
    s = pd.Series(list(series.values()), index=idx, name="sales").sort_index()
    df = s.unstack("group").fillna(0.0)
    df.index = pd.to_datetime(df.index)

    for col in ["Hotspot", "Middle", "Quiet Zone"]:
        if col not in df.columns:
            df[col] = 0.0
    return df[["Hotspot", "Middle", "Quiet Zone"]].sort_index()


def second_pass_top_items_timeseries(
    sales_path: Path,
    cols: SalesColumns,
    top_products: List[str],
    max_rowgroups: Optional[int] = None,
) -> pd.DataFrame:
    series: Dict[Tuple[pd.Timestamp, str], float] = {}
    use_cols = [cols.date, cols.product_id, cols.target]
    top_set = set(top_products)

    for rg_i, chunk in enumerate(parquet_row_groups(sales_path, columns=use_cols)):
        if max_rowgroups is not None and rg_i >= max_rowgroups:
            break

        dt = _coerce_date(chunk[cols.date])
        y = pd.to_numeric(chunk[cols.target], errors="coerce")
        chunk = chunk.assign(_date=dt, _y=y).dropna(subset=["_date", "_y"])
        if chunk.empty:
            continue

        chunk["_date"] = chunk["_date"].dt.normalize()
        chunk[cols.product_id] = chunk[cols.product_id].astype(str)

        cp = chunk[chunk[cols.product_id].isin(top_set)]
        if cp.empty:
            continue

        g = cp.groupby(["_date", cols.product_id])["_y"].sum()
        for (d, pid), v in g.items():
            _dict_add(series, (pd.Timestamp(d), str(pid)), float(v))

    if not series:
        return pd.DataFrame()

    idx = pd.MultiIndex.from_tuples(series.keys(), names=["date", "product_id"])
    s = pd.Series(list(series.values()), index=idx, name="sales").sort_index()
    df = s.unstack("product_id").fillna(0.0)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


# ============================
# PDF helpers
# ============================
def apply_style() -> None:
    if HAS_SEABORN:
        sns.set_theme(style="whitegrid")

    plt.rcParams.update({
        "figure.dpi": 170,
        "savefig.dpi": 170,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.edgecolor": THEME["grid"],
        "grid.color": THEME["grid"],
        "text.color": THEME["text"],
        "axes.labelcolor": THEME["text"],
        "xtick.color": THEME["text"],
        "ytick.color": THEME["text"],
    })


def new_page_fig() -> Tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    fig.patch.set_facecolor(THEME["fig_bg"])
    ax = fig.add_subplot(111)
    ax.set_facecolor(THEME["panel_bg"])
    return fig, ax


def format_date_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def polish_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(THEME["grid"])


def _wrap_text(text: str, width: int = 105) -> str:
    words = text.split()
    lines: List[str] = []
    cur: List[str] = []
    n = 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if n + add > width:
            lines.append(" ".join(cur))
            cur = [w]
            n = len(w)
        else:
            cur.append(w)
            n += add
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)


def add_summary(fig: plt.Figure, text: str) -> None:
    fig.text(
        0.02, 0.02,
        "Management summary:\n" + _wrap_text(text),
        ha="left", va="bottom",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.98, edgecolor=THEME["grid"]),
    )


def add_footer_stamp(fig: plt.Figure, run_stamp: str) -> None:
    fig.text(
        0.985, 0.015,
        f"{SCRIPT_VERSION} | {run_stamp}",
        ha="right", va="bottom",
        fontsize=8, color=THEME["grey"],
    )


def save_page(pdf: PdfPages, fig: plt.Figure) -> None:
    fig.tight_layout(rect=[0, 0.12, 1, 0.98])
    pdf.savefig(fig)
    plt.close(fig)


def log_page(page_no: int, title: str) -> None:
    LOG.info("PAGE %02d | %s", page_no, title)


def build_quality_table(agg: Aggregates, store_total: pd.Series, prod_total: pd.Series) -> Tuple[List[str], List[List[str]]]:
    dates = pd.to_datetime(list(agg.daily_total.keys())) if agg.daily_total else pd.to_datetime([])
    date_min = str(dates.min().date()) if len(dates) else "-"
    date_max = str(dates.max().date()) if len(dates) else "-"

    miss = agg.missing_target_rows
    tot = max(agg.total_rows, 1)
    miss_pct = f"{(100.0 * miss / tot):.2f}%"

    headers = ["Metric", "Value"]
    rows = [
        ["Rows in sales file", f"{agg.total_rows:,}"],
        ["Missing sales values", f"{agg.missing_target_rows:,} ({miss_pct})"],
        ["Date range", f"{date_min} to {date_max}"],
        ["Number of days", f"{len(agg.daily_total):,}"],
        ["Number of stores in sales", f"{len(store_total):,}"],
        ["Number of products in sales", f"{len(prod_total):,}"],
    ]
    return headers, rows


# ============================
# Build PDF (13 pages)
# ============================
def build_pdf(
    out_pdf: Path,
    agg: Aggregates,
    daily_store_groups_ts: pd.DataFrame,
    store_group_sizes: Dict[str, int],
    daily_top_items_ts: pd.DataFrame,
    holidays: Optional[pd.DataFrame],
    weather: Optional[pd.DataFrame],
    stores_with_coords: Optional[pd.DataFrame],
    run_stamp: str,
) -> None:
    apply_style()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    daily_total = pd.Series(agg.daily_total).sort_index()
    daily_total.index = pd.to_datetime(daily_total.index)

    month_total = pd.Series(agg.month_total).sort_index()
    month_total.index = pd.to_datetime(month_total.index)

    # v3.1 FIX: weekday averages from DAILY TOTALS (not from raw row counts)
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday_avg = (
        pd.DataFrame({"date": daily_total.index, "sales": daily_total.values})
        .assign(weekday=lambda d: d["date"].dt.weekday)
        .groupby("weekday")["sales"]
        .mean()
        .reindex(range(7), fill_value=0.0)
        .tolist()
    )

    store_total = pd.Series(agg.store_total).sort_values(ascending=False)
    prod_total = pd.Series(agg.product_total).sort_values(ascending=False)

    with PdfPages(out_pdf) as pdf:
        # 1
        title = "Daily total sales over time"
        log_page(1, title)
        fig, ax = new_page_fig()
        ax.plot(daily_total.index, daily_total.values, linewidth=2.0, color=THEME["marine"])
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Sold quantity")
        format_date_axis(ax)
        polish_axes(ax)
        add_summary(fig, "This is the overall trend. Big jumps should be checked (holiday, promo, or data issue).")
        add_footer_stamp(fig, run_stamp)
        save_page(pdf, fig)

        # 2
        title = "Sales baseline (7-day average) and top days"
        log_page(2, title)
        fig, ax = new_page_fig()
        s = daily_total.copy()
        roll7 = s.rolling(7, min_periods=1).mean()
        ax.plot(s.index, s.values, linewidth=1.2, color=THEME["grey"], alpha=0.65, label="Daily sales")
        ax.plot(roll7.index, roll7.values, linewidth=2.3, color=THEME["marine"], label="7-day average")
        if len(s) > 0:
            topk = s.sort_values(ascending=False).head(5)
            ax.scatter(topk.index, topk.values, s=70, color=THEME["orange"],
                       edgecolor=THEME["black"], linewidth=0.6, zorder=4, label="Top days")
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Sold quantity")
        format_date_axis(ax)
        polish_axes(ax)
        ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.95)
        add_summary(fig, "The dark line is the baseline. Orange dots are the biggest days to investigate.")
        add_footer_stamp(fig, run_stamp)
        save_page(pdf, fig)

        # 3 (FIXED)
        title = "Average sales by weekday"
        log_page(3, title)
        fig, ax = new_page_fig()
        ax.bar(weekday_names, weekday_avg, color=THEME["blue"])
        ax.set_title(title)
        ax.set_xlabel("Weekday")
        ax.set_ylabel("Average sold quantity per day")
        polish_axes(ax)
        add_summary(fig, "This is based on daily totals (not per row). Weekday flags should improve forecasting.")
        add_footer_stamp(fig, run_stamp)
        save_page(pdf, fig)

        # 4
        title = "Monthly sales pattern (sum per month)"
        log_page(4, title)
        fig, ax = new_page_fig()
        ax.plot(month_total.index, month_total.values, marker="o", linewidth=2.0, color=THEME["marine"])
        ax.set_title(title)
        ax.set_xlabel("Month")
        ax.set_ylabel("Sold quantity")
        format_date_axis(ax)
        polish_axes(ax)
        add_summary(fig, "This shows longer cycles. With only a few months, do not over-interpret seasonality.")
        add_footer_stamp(fig, run_stamp)
        save_page(pdf, fig)

        # 5
        title = "Store concentration (cumulative share of total sales)"
        log_page(5, title)
        fig, ax = new_page_fig()
        st = store_total.values
        cs = np.cumsum(st) / (st.sum() if st.sum() > 0 else 1.0)
        ax.plot(np.arange(1, len(cs) + 1), cs, linewidth=2.2, color=THEME["marine"])
        ax.axhline(0.8, color=THEME["black"], linewidth=1.2, linestyle="--")
        x80 = int(np.searchsorted(cs, 0.8) + 1)
        ax.axvline(x80, color=THEME["black"], linewidth=1.2, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Number of stores (sorted by sales)")
        ax.set_ylabel("Cumulative share")
        polish_axes(ax)
        add_summary(fig, f"About 80% of sales come from the top ~{x80} stores. Focus forecast quality there.")
        add_footer_stamp(fig, run_stamp)
        save_page(pdf, fig)

        # 6
        title = "Product concentration (cumulative share of total sales)"
        log_page(6, title)
        fig, ax = new_page_fig()
        pt = prod_total.values
        cs = np.cumsum(pt) / (pt.sum() if pt.sum() > 0 else 1.0)
        ax.plot(np.arange(1, len(cs) + 1), cs, linewidth=2.2, color=THEME["blue"])
        ax.axhline(0.8, color=THEME["black"], linewidth=1.2, linestyle="--")
        x80 = int(np.searchsorted(cs, 0.8) + 1)
        ax.axvline(x80, color=THEME["black"], linewidth=1.2, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Number of products (sorted by sales)")
        ax.set_ylabel("Cumulative share")
        polish_axes(ax)
        add_summary(fig, f"About 80% of sales come from the top ~{x80} products. Many products are low-volume.")
        add_footer_stamp(fig, run_stamp)
        save_page(pdf, fig)

        # 7 (FIXED: normalize to avg per store)
        title = "Store groups over time (Hotspot vs Middle vs Quiet Zone)"
        log_page(7, title)
        fig, ax = new_page_fig()

        if not daily_store_groups_ts.empty:
            norm = daily_store_groups_ts.copy()
            for g in ["Hotspot", "Middle", "Quiet Zone"]:
                denom = max(int(store_group_sizes.get(g, 0)), 1)
                norm[g] = norm[g] / denom

            ax.plot(norm.index, norm["Hotspot"], color=THEME["black"], linewidth=2.2, label="Hotspot")
            ax.plot(norm.index, norm["Middle"], color=THEME["blue"], linewidth=2.2, label="Middle")
            ax.plot(norm.index, norm["Quiet Zone"], color=THEME["orange"], linewidth=2.2, label="Quiet Zone")
            format_date_axis(ax)
            ax.legend(loc="upper left", frameon=True, ncols=3, facecolor="white", framealpha=0.95)

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Sold quantity per day (avg per store)")
        polish_axes(ax)

        hs = store_group_sizes.get("Hotspot", 0)
        ms = store_group_sizes.get("Middle", 0)
        qs = store_group_sizes.get("Quiet Zone", 0)
        add_summary(fig, f"Normalized per store (Hotspot n={hs}, Middle n={ms}, Quiet Zone n={qs}). This compares behavior, not group size.")
        add_footer_stamp(fig, run_stamp)
        save_page(pdf, fig)

        # 8
        title = "Daily sales for the 4 top products"
        log_page(8, title)
        fig, ax = new_page_fig()
        if not daily_top_items_ts.empty:
            cols4 = list(daily_top_items_ts.columns)[:4]
            palette = [THEME["marine"], THEME["blue"], THEME["orange"], THEME["black"]]
            for i, c in enumerate(cols4):
                ax.plot(daily_top_items_ts.index, daily_top_items_ts[c], linewidth=2.0, label=f"Item {c}", color=palette[i % len(palette)])
            format_date_axis(ax)
            ax.legend(loc="upper left", frameon=True, ncols=2, facecolor="white", framealpha=0.95)
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Sold quantity per day")
        polish_axes(ax)
        add_summary(fig, "These items are the main drivers. Different patterns suggest item-specific features.")
        add_footer_stamp(fig, run_stamp)
        save_page(pdf, fig)

        # 9
        title = "Holiday vs non-holiday days (sales)"
        log_page(9, title)
        if holidays is not None and not holidays.empty and len(daily_total) > 0:
            dfh = pd.DataFrame({"_date": daily_total.index, "sales": daily_total.values})
            dfh = dfh.merge(holidays[["_date", "is_holiday"]], on="_date", how="left")
            dfh["is_holiday"] = dfh["is_holiday"].fillna(False)

            fig, ax = new_page_fig()
            groups = [
                dfh.loc[dfh["is_holiday"] == False, "sales"].dropna(),  # noqa: E712
                dfh.loc[dfh["is_holiday"] == True, "sales"].dropna(),   # noqa: E712
            ]
            bp = boxplot_with_labels(ax, groups, ["No holiday", "Holiday"], showfliers=False)
            for box in bp["boxes"]:
                box.set(facecolor=THEME["marine"], edgecolor=THEME["black"], alpha=0.70, linewidth=1.2)
            for k in ["whiskers", "caps", "medians"]:
                for line in bp[k]:
                    line.set(color=THEME["black"], linewidth=1.4)

            ax.set_title(title)
            ax.set_ylabel("Sold quantity per day")
            polish_axes(ax)
            n0 = int(len(groups[0]))
            n1 = int(len(groups[1]))
            add_summary(fig, f"Holiday flag may help. Sample sizes: non-holiday n={n0}, holiday n={n1}.")
            add_footer_stamp(fig, run_stamp)
            save_page(pdf, fig)
        else:
            fig, ax = new_page_fig()
            ax.set_axis_off()
            ax.set_title(title)
            add_summary(fig, "Holiday data not available. This page is skipped in content.")
            add_footer_stamp(fig, run_stamp)
            save_page(pdf, fig)

        # 10
        title = "Temperature vs daily sales (with trend)"
        log_page(10, title)
        fig, ax = new_page_fig()
        if weather is not None and "_temp" in weather.columns and len(daily_total) > 0:
            dfw = pd.DataFrame({"_date": daily_total.index, "sales": daily_total.values})
            dfw = dfw.merge(weather[["_date", "_temp"]], on="_date", how="left")
            ax.scatter(dfw["_temp"], dfw["sales"], s=22, alpha=0.55, color=THEME["blue"], edgecolor="none")
            xs, ys = linear_trend(dfw["_temp"].to_numpy(dtype=float), dfw["sales"].to_numpy(dtype=float))
            if xs.size:
                ax.plot(xs, ys, color=THEME["black"], linewidth=2.2, label="Trend")
                ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.95)
            ax.set_xlabel("Temperature")
            ax.set_ylabel("Sold quantity per day")
            polish_axes(ax)
            add_summary(fig, "The black line shows the general direction. Use temperature if the slope is clear.")
        else:
            ax.set_axis_off()
            add_summary(fig, "Temperature data not available.")
        ax.set_title(title)
        add_footer_stamp(fig, run_stamp)
        save_page(pdf, fig)

        # 11
        title = "Rain vs daily sales (with trend)"
        log_page(11, title)
        fig, ax = new_page_fig()
        if weather is not None and "_precip" in weather.columns and len(daily_total) > 0:
            dfw = pd.DataFrame({"_date": daily_total.index, "sales": daily_total.values})
            dfw = dfw.merge(weather[["_date", "_precip"]], on="_date", how="left")
            ax.scatter(dfw["_precip"], dfw["sales"], s=22, alpha=0.55, color=THEME["orange"], edgecolor="none")
            xs, ys = linear_trend(dfw["_precip"].to_numpy(dtype=float), dfw["sales"].to_numpy(dtype=float))
            if xs.size:
                ax.plot(xs, ys, color=THEME["black"], linewidth=2.2, label="Trend")
                ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.95)
            ax.set_xlabel("Precipitation")
            ax.set_ylabel("Sold quantity per day")
            polish_axes(ax)
            add_summary(fig, "If the black line slopes down, rain may reduce sales. If flat, rain may not matter much.")
        else:
            ax.set_axis_off()
            add_summary(fig, "Rain data not available.")
        ax.set_title(title)
        add_footer_stamp(fig, run_stamp)
        save_page(pdf, fig)

        # 12
        title = "Shopping Hotspots vs. Quiet Zones (stores + sales)"
        log_page(12, title)
        fig, ax = new_page_fig()
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        polish_axes(ax)

        if stores_with_coords is None or stores_with_coords.empty:
            ax.text(
                0.5, 0.5,
                "No store coordinates available.\nAdd geo/plz_centroids.csv to enable the map view.",
                ha="center", va="center", fontsize=14, color=THEME["text"]
            )
            ax.set_axis_off()
            add_summary(fig, "A map needs coordinates. With ZIP centroids, this page becomes a true map.")
        else:
            st = store_total.rename("sales_total").reset_index()
            st.columns = ["store_id", "sales_total"]

            sdf = stores_with_coords.copy()
            store_id_col = sdf.columns[0]
            sdf[store_id_col] = sdf[store_id_col].astype(str)
            sdf = sdf.merge(st, left_on=store_id_col, right_on="store_id", how="left")
            sdf["sales_total"] = sdf["sales_total"].fillna(0.0)

            q_hi = sdf["sales_total"].quantile(0.80)
            q_lo = sdf["sales_total"].quantile(0.20)
            sdf["zone"] = np.where(
                sdf["sales_total"] >= q_hi, "Hotspot",
                np.where(sdf["sales_total"] <= q_lo, "Quiet Zone", "Middle"),
            )

            colors = {"Hotspot": THEME["black"], "Middle": THEME["blue"], "Quiet Zone": THEME["orange"]}
            markers = {"Hotspot": "o", "Middle": "s", "Quiet Zone": "^"}  # circle / square / triangle

            size = np.sqrt(sdf["sales_total"].clip(lower=0.0))
            size = 25 + 95 * (size / (size.max() if size.max() > 0 else 1.0))

            for zone in ["Hotspot", "Middle", "Quiet Zone"]:
                sub = sdf[sdf["zone"] == zone]
                ax.scatter(
                    sub["_lon"], sub["_lat"],
                    s=size.loc[sub.index],
                    alpha=0.86,
                    label=zone,
                    color=colors[zone],
                    marker=markers[zone],
                    edgecolor="white",
                    linewidth=0.35,
                )
            ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95)
            add_summary(fig, "Markers + colors separate zones: Hotspot (● black), Middle (■ blue), Quiet Zone (▲ orange).")

        add_footer_stamp(fig, run_stamp)
        save_page(pdf, fig)

        # 13
        title = "Data quality summary (sales file)"
        log_page(13, title)
        fig, ax = new_page_fig()
        ax.set_axis_off()
        headers, rows = build_quality_table(agg, store_total, prod_total)
        tbl = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="left", colLoc="left")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        tbl.scale(1, 1.6)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor(THEME["grid"])
            if r == 0:
                cell.set_facecolor(THEME["marine"])
                cell.get_text().set_color("white")
                cell.get_text().set_weight("bold")
            else:
                cell.set_facecolor("#FFFFFF" if r % 2 == 0 else "#F7FAFF")
        ax.set_title(title, pad=18, color=THEME["text"])
        add_summary(fig, "Quick check: date range, missing values, and how many stores/products are included.")
        add_footer_stamp(fig, run_stamp)
        save_page(pdf, fig)


# ============================
# Main
# ============================
def main() -> int:
    parser = argparse.ArgumentParser(description="RAW data EDA PDF generator (robust, 13 pages).")
    parser.add_argument("--raw-data-dir", type=str, default=None, help="Path to raw_data directory (recommended).")
    parser.add_argument("--plz-centroids", type=str, default=None, help="Optional: override centroid CSV path.")
    parser.add_argument("--max-rowgroups", type=int, default=None, help="Optional: limit row groups for quick tests.")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    raw_dir = resolve_raw_data_dir(script_path, args.raw_data_dir)

    reports_dir = raw_dir / "visualized_raw_data_analysis"
    reports_dir.mkdir(parents=True, exist_ok=True)
    log_path = reports_dir / "viz_raw_data_analysis.log"
    setup_logging(log_path)

    run_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    LOG.info("Script: %s | %s", SCRIPT_VERSION, run_stamp)
    LOG.info("RAW data directory: %s", raw_dir)
    LOG.info("Reports directory: %s", reports_dir)

    files = resolve_input_files(raw_dir)
    for k, p in files.items():
        LOG.info("Input resolved: %s -> %s", k, p)

    sales_sample = read_parquet_sample(
        files["sales"],
        [
            "date", "day", "sales_date", "ds",
            "store_id", "store", "shop_id", "branch_id", "filiale_id",
            "item_id", "product_id",
            "sold_quantity", "qty", "quantity", "units", "sales_qty", "y", "target", "demand", "sales",
        ],
        n_rows=50_000,
    )
    sales_cols = infer_sales_columns(sales_sample)
    LOG.info("Sales columns inferred: %s", sales_cols)

    weather_cols = None
    holiday_cols = None
    store_cols = None

    try:
        weather_sample = read_parquet_sample(
            files["weather"],
            ["date", "day", "weather_date", "ds", "temperature", "temp", "precip", "precipitation", "rain", "prcp"],
            n_rows=50_000,
        )
        weather_cols = infer_weather_columns(weather_sample)
        LOG.info("Weather columns inferred: %s", weather_cols)
    except Exception as e:
        LOG.warning("Weather inference failed (weather pages may be empty): %s", e)

    try:
        holiday_sample = read_parquet_sample(
            files["holidays"],
            ["date", "day", "holiday_date", "ds", "holiday_name", "name", "holiday", "event"],
            n_rows=50_000,
        )
        holiday_cols = infer_holiday_columns(holiday_sample)
        LOG.info("Holiday columns inferred: %s", holiday_cols)
    except Exception as e:
        LOG.warning("Holiday inference failed (holiday page may be empty): %s", e)

    try:
        store_sample = read_parquet_sample(
            files["stores"],
            ["store_id", "zipcode", "zip_code", "plz", "postal_code"],
            n_rows=50_000,
        )
        store_cols = infer_store_columns(store_sample)
        LOG.info("Store columns inferred: %s", store_cols)
    except Exception as e:
        LOG.warning("Store inference failed (map may be empty): %s", e)

    agg = first_pass_aggregates(files["sales"], sales_cols, max_rowgroups=args.max_rowgroups)

    holidays = load_holidays(files["holidays"], holiday_cols) if holiday_cols is not None else None
    weather = load_weather(files["weather"], weather_cols) if weather_cols is not None else None
    stores = load_stores(files["stores"], store_cols) if store_cols is not None else None

    store_total = pd.Series(agg.store_total).sort_values(ascending=False)
    store_group_map = make_store_groups(store_total)
    store_group_sizes = count_store_groups(store_group_map)

    daily_store_groups_ts = second_pass_store_group_timeseries(
        files["sales"], sales_cols, store_group_map, max_rowgroups=args.max_rowgroups
    )

    top_products = [k for k, _ in sorted(agg.product_total.items(), key=lambda x: x[1], reverse=True)[:4]]
    daily_top_items_ts = second_pass_top_items_timeseries(
        files["sales"], sales_cols, top_products, max_rowgroups=args.max_rowgroups
    )

    stores_with_coords = None
    centroids_path = Path(args.plz_centroids).resolve() if args.plz_centroids else (raw_dir / "geo" / "plz_centroids.csv")
    if centroids_path.exists():
        LOG.info("Centroids file found (strict mode): %s", centroids_path)
        centroids = load_plz_centroids_strict(centroids_path)
        if stores is None or stores.empty:
            raise SystemExit("Centroids file exists, but stores could not be loaded.")
        stores_with_coords = enforce_centroids_match_strict(stores, centroids)
        LOG.info("Centroids match OK. Stores with coords: %d", int(stores_with_coords["_lat"].notna().sum()))
    else:
        LOG.info("Centroids file not found at: %s (map page will be informational)", centroids_path)

    out_pdf = reports_dir / "RAW_DATA_EDA.pdf"
    LOG.info("Writing EDA PDF: %s", out_pdf)

    build_pdf(
        out_pdf=out_pdf,
        agg=agg,
        daily_store_groups_ts=daily_store_groups_ts,
        store_group_sizes=store_group_sizes,
        daily_top_items_ts=daily_top_items_ts,
        holidays=holidays,
        weather=weather,
        stores_with_coords=stores_with_coords,
        run_stamp=run_stamp,
    )

    LOG.info("Done. Generated: %s", out_pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())