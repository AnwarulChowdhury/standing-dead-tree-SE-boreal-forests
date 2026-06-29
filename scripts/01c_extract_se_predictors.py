"""Extract plot-level AlphaEarth satellite-embedding predictors only.

This script extracts area-weighted means for 64 AlphaEarth embedding bands for
2025 and a previous year, then computes temporal-change predictors dSE_B01_mean
... dSE_B64_mean. CHM and MSI predictors are extracted by separate scripts.

"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import windows
from shapely.geometry import box
from tqdm import tqdm

PLOT_ID_COL_DEFAULT = "Name"
N_EMBEDDING_BANDS = 64
RASTER_EXTENSIONS = ("*.tif", "*.tiff", "*.vrt", "*.img", "*.jp2", "*.TIF", "*.TIFF", "*.VRT", "*.IMG", "*.JP2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract plot-level AlphaEarth satellite-embedding predictors only.")
    parser.add_argument("--plots-vector", required=True, help="Plot-buffer vector file, e.g. shapefile, GeoPackage, or GeoJSON.")
    parser.add_argument("--se-2025-folder", required=True, help="Folder containing 2025 AlphaEarth embedding rasters.")
    parser.add_argument("--se-previous-folder", required=True, help="Folder containing earlier-year AlphaEarth embedding rasters used for dSE predictors.")
    parser.add_argument("--previous-year-label", default="previous", help="Label for earlier SE year used in metadata only, e.g. 2023.")
    parser.add_argument("--plot-id-col", default=PLOT_ID_COL_DEFAULT, help="Plot ID column in the plot vector file.")
    parser.add_argument("--study-area-col", default=None, help="Optional study-area column in the plot vector file.")
    parser.add_argument("--output-csv", default="results/00_extracted_predictors/se_predictors.csv", help="Output CSV path for SE predictors.")
    return parser.parse_args()


def find_rasters(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Raster folder does not exist: {folder}")
    files: List[Path] = []
    for pattern in RASTER_EXTENSIONS:
        files.extend(folder.rglob(pattern))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No raster files found in: {folder}")
    return files


def clean_array(arr: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    out = arr.astype("float64", copy=True)
    if nodata is not None and np.isfinite(nodata):
        out[out == nodata] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out


def raster_intersects_geometry(src: rasterio.io.DatasetReader, geom) -> bool:
    raster_box = box(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
    return geom.intersects(raster_box)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def pixel_overlap_weights(src: rasterio.io.DatasetReader, geom, window: windows.Window) -> np.ndarray:
    transform = src.window_transform(window)
    n_rows = int(window.height)
    n_cols = int(window.width)
    weights = np.zeros((n_rows, n_cols), dtype="float64")
    for row in range(n_rows):
        for col in range(n_cols):
            left, top = transform * (col, row)
            right, bottom = transform * (col + 1, row + 1)
            pix = box(min(left, right), min(bottom, top), max(left, right), max(bottom, top))
            inter_area = pix.intersection(geom).area
            if inter_area > 0:
                weights[row, col] = inter_area / pix.area
    return weights


def read_weighted_pixels(src: rasterio.io.DatasetReader, geom, band_indexes: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    if not raster_intersects_geometry(src, geom):
        return np.empty((len(band_indexes), 0)), np.empty(0)
    try:
        window = windows.from_bounds(*geom.bounds, transform=src.transform)
        window = window.round_offsets().round_lengths()
        window = window.intersection(windows.Window(0, 0, src.width, src.height))
    except Exception:
        return np.empty((len(band_indexes), 0)), np.empty(0)
    if window.width <= 0 or window.height <= 0:
        return np.empty((len(band_indexes), 0)), np.empty(0)
    arr = src.read(indexes=list(band_indexes), window=window, boundless=False)
    arr = clean_array(arr, src.nodata)
    weights_2d = pixel_overlap_weights(src, geom, window)
    weights_flat = weights_2d.ravel()
    valid_weight = weights_flat > 0
    if not np.any(valid_weight):
        return np.empty((len(band_indexes), 0)), np.empty(0)
    values = arr.reshape((len(band_indexes), -1))[:, valid_weight]
    weights = weights_flat[valid_weight]
    return values, weights


def prepare_plots_for_rasters(plots: gpd.GeoDataFrame, raster_files: Sequence[Path]) -> gpd.GeoDataFrame:
    with rasterio.open(raster_files[0]) as src:
        raster_crs = src.crs
    return plots.to_crs(raster_crs) if plots.crs != raster_crs else plots.copy()


def standardize_plot_ids(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def extract_se_band_means(plots: gpd.GeoDataFrame, se_files: Sequence[Path], plot_id_col: str, prefix: str) -> pd.DataFrame:
    plots_raster_crs = prepare_plots_for_rasters(plots, se_files)
    records = []
    band_indexes = list(range(1, N_EMBEDDING_BANDS + 1))
    for _, row in tqdm(plots_raster_crs.iterrows(), total=len(plots_raster_crs), desc=f"Extracting {prefix}"):
        plot_id = row[plot_id_col]
        geom = row.geometry
        band_values: List[List[np.ndarray]] = [[] for _ in range(N_EMBEDDING_BANDS)]
        band_weights: List[List[np.ndarray]] = [[] for _ in range(N_EMBEDDING_BANDS)]
        for path in se_files:
            with rasterio.open(path) as src:
                if src.count < N_EMBEDDING_BANDS:
                    raise ValueError(f"{path} has {src.count} bands; expected at least {N_EMBEDDING_BANDS}.")
                values, weights = read_weighted_pixels(src, geom, band_indexes)
            if values.shape[1] == 0:
                continue
            for i in range(N_EMBEDDING_BANDS):
                band_values[i].append(values[i])
                band_weights[i].append(weights)
        rec = {plot_id_col: plot_id}
        band_means = []
        for i in range(N_EMBEDDING_BANDS):
            col = f"{prefix}_B{i + 1:02d}_mean"
            if band_values[i]:
                vals = np.concatenate(band_values[i])
                weights = np.concatenate(band_weights[i])
                rec[col] = weighted_mean(vals, weights)
            else:
                rec[col] = np.nan
            band_means.append(rec[col])
        band_means_arr = np.asarray(band_means, dtype="float64")
        rec[f"{prefix}_mean_across_bands"] = float(np.nanmean(band_means_arr)) if np.any(np.isfinite(band_means_arr)) else np.nan
        rec[f"{prefix}_std_across_bands"] = float(np.nanstd(band_means_arr)) if np.any(np.isfinite(band_means_arr)) else np.nan
        rec[f"{prefix}_min_band_mean"] = float(np.nanmin(band_means_arr)) if np.any(np.isfinite(band_means_arr)) else np.nan
        rec[f"{prefix}_max_band_mean"] = float(np.nanmax(band_means_arr)) if np.any(np.isfinite(band_means_arr)) else np.nan
        records.append(rec)
    return pd.DataFrame(records)


def add_se_change_predictors(df: pd.DataFrame, previous_prefix: str = "SE_previous", current_prefix: str = "SE2025") -> pd.DataFrame:
    out = df.copy()
    dse_values = []
    for i in range(1, N_EMBEDDING_BANDS + 1):
        current_col = f"{current_prefix}_B{i:02d}_mean"
        previous_col = f"{previous_prefix}_B{i:02d}_mean"
        dse_col = f"dSE_B{i:02d}_mean"
        out[dse_col] = out[current_col] - out[previous_col] if current_col in out.columns and previous_col in out.columns else np.nan
        dse_values.append(dse_col)
    out["dSE_mean_across_bands"] = out[dse_values].mean(axis=1, skipna=True)
    out["dSE_std_across_bands"] = out[dse_values].std(axis=1, skipna=True)
    return out


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = parse_args()
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    plots = gpd.read_file(args.plots_vector)
    if args.plot_id_col not in plots.columns:
        raise ValueError(f"Plot ID column '{args.plot_id_col}' was not found in {args.plots_vector}.")
    if plots.crs is None:
        raise ValueError("Plot vector file has no CRS. Please define the CRS before extraction.")
    plots[args.plot_id_col] = standardize_plot_ids(plots[args.plot_id_col])
    se_2025_files = find_rasters(args.se_2025_folder)
    se_previous_files = find_rasters(args.se_previous_folder)
    plot_meta_cols = [args.plot_id_col]
    if args.study_area_col and args.study_area_col in plots.columns:
        plot_meta_cols.append(args.study_area_col)
    plot_meta = plots.drop(columns="geometry")[plot_meta_cols].copy()
    se_2025_df = extract_se_band_means(plots, se_2025_files, args.plot_id_col, "SE2025")
    se_previous_df = extract_se_band_means(plots, se_previous_files, args.plot_id_col, "SE_previous")
    out = plot_meta.merge(se_2025_df, on=args.plot_id_col, how="left")
    out = out.merge(se_previous_df, on=args.plot_id_col, how="left")
    out = add_se_change_predictors(out, previous_prefix="SE_previous", current_prefix="SE2025")
    previous_cols = [c for c in out.columns if c.startswith("SE_previous")]
    out = out.drop(columns=previous_cols)
    rename = {args.plot_id_col: "plot_id"}
    if args.study_area_col and args.study_area_col in out.columns:
        rename[args.study_area_col] = "study_area_id"
    out = out.rename(columns=rename)
    out.to_csv(output_csv, index=False)
    print(f"Saved SE predictors: {output_csv}")
    print(f"Rows: {len(out)}; Columns: {len(out.columns)}")


if __name__ == "__main__":
    main()
