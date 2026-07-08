"""Extract plot-level aerial MSI predictors only.

This script extracts MSI predictors from aerial RGB-NIR rasters using exact
partial-pixel weighting over 15 m plot-buffer polygons. 

Expected band order for MSI rasters: Red, Green, Blue, NIR.

"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import windows
from shapely.geometry import box
from tqdm import tqdm

try:
    from skimage.feature import graycomatrix, graycoprops
    SKIMAGE_AVAILABLE = True
except Exception:  # pragma: no cover
    SKIMAGE_AVAILABLE = False

PLOT_ID_COL_DEFAULT = "Name"
EPS = 1e-12
MSI_BAND_ORDER = {"R": 1, "G": 2, "B": 3, "NIR": 4}
RASTER_EXTENSIONS = ("*.tif", "*.tiff", "*.vrt", "*.img", "*.jp2", "*.TIF", "*.TIFF", "*.VRT", "*.IMG", "*.JP2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract plot-level aerial MSI/RGB-NIR predictors only.")
    parser.add_argument("--plots-vector", required=True, help="Plot-buffer vector file, e.g. shapefile, GeoPackage, or GeoJSON.")
    parser.add_argument("--msi-folder", required=True, help="Folder containing aerial RGB-NIR/MSI rasters.")
    parser.add_argument("--plot-id-col", default=PLOT_ID_COL_DEFAULT, help="Plot ID column in the plot vector file.")
    parser.add_argument("--study-area-col", default=None, help="Optional study-area column in the plot vector file.")
    parser.add_argument("--output-csv", default="results/00_extracted_predictors/msi_predictors.csv", help="Output CSV path for MSI predictors.")
    parser.add_argument("--no-texture", action="store_true", help="Skip GLCM texture predictors for faster extraction.")
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


def safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.full_like(a, np.nan, dtype="float64"), where=np.isfinite(a) & np.isfinite(b) & (np.abs(b) > EPS))


def raster_intersects_geometry(src: rasterio.io.DatasetReader, geom) -> bool:
    raster_box = box(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
    return geom.intersects(raster_box)


def weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.nan
    vals = values[mask]
    w = weights[mask]
    mean = np.average(vals, weights=w)
    var = np.average((vals - mean) ** 2, weights=w)
    return float(np.sqrt(max(var, 0.0)))


def weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.nan
    vals = values[mask]
    w = weights[mask]
    order = np.argsort(vals)
    vals = vals[order]
    w = w[order]
    cum_w = np.cumsum(w)
    cutoff = percentile / 100.0 * cum_w[-1]
    return float(vals[np.searchsorted(cum_w, cutoff, side="left")])


def weighted_summary(values: np.ndarray, weights: np.ndarray, prefix: str, stats: Sequence[str]) -> Dict[str, float]:
    values = np.asarray(values, dtype="float64")
    weights = np.asarray(weights, dtype="float64")
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    out: Dict[str, float] = {}
    for stat in stats:
        key = f"{prefix}_{stat}"
        if not np.any(mask):
            out[key] = np.nan
            continue
        vals = values[mask]
        w = weights[mask]
        if stat == "mean":
            out[key] = float(np.average(vals, weights=w))
        elif stat in ("std", "sd"):
            out[key] = weighted_std(vals, w)
        elif stat == "cv":
            mean = np.average(vals, weights=w)
            out[key] = float(weighted_std(vals, w) / mean) if abs(mean) > EPS else np.nan
        elif stat == "min":
            out[key] = float(np.nanmin(vals))
        elif stat == "max":
            out[key] = float(np.nanmax(vals))
        elif stat == "p10":
            out[key] = weighted_percentile(vals, w, 10)
        elif stat == "p50":
            out[key] = weighted_percentile(vals, w, 50)
        elif stat == "p90":
            out[key] = weighted_percentile(vals, w, 90)
        else:
            raise ValueError(f"Unsupported statistic: {stat}")
    return out


def weighted_fraction(values: np.ndarray, weights: np.ndarray, condition) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.nan
    vals = values[mask]
    w = weights[mask]
    return float(np.sum(w[condition(vals)]) / np.sum(w))


def pixel_overlap_weights(src: rasterio.io.DatasetReader, geom, window: windows.Window) -> np.ndarray:
    """Exact proportional pixel-overlap weights for a raster window."""
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


def make_texture(values_2d: np.ndarray, prefix: str) -> Dict[str, float]:
    if not SKIMAGE_AVAILABLE:
        return {}
    img = values_2d.astype("float64", copy=True)
    if np.sum(np.isfinite(img)) < 20:
        return {f"{prefix}_texture_contrast": np.nan, f"{prefix}_texture_homogeneity": np.nan}
    med = np.nanmedian(img)
    img[~np.isfinite(img)] = med
    p02 = np.nanpercentile(img, 2)
    p98 = np.nanpercentile(img, 98)
    if not np.isfinite(p02) or not np.isfinite(p98) or p98 <= p02:
        return {f"{prefix}_texture_contrast": np.nan, f"{prefix}_texture_homogeneity": np.nan}
    img_q = np.clip((img - p02) / (p98 - p02), 0, 1)
    img_q = (img_q * 31).astype("uint8")
    glcm = graycomatrix(img_q, distances=[1], angles=[0], levels=32, symmetric=True, normed=True)
    return {f"{prefix}_texture_contrast": float(graycoprops(glcm, "contrast")[0, 0]), f"{prefix}_texture_homogeneity": float(graycoprops(glcm, "homogeneity")[0, 0])}


def standardize_plot_ids(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def prepare_plots_for_rasters(plots: gpd.GeoDataFrame, raster_files: Sequence[Path]) -> gpd.GeoDataFrame:
    with rasterio.open(raster_files[0]) as src:
        raster_crs = src.crs
    return plots.to_crs(raster_crs) if plots.crs != raster_crs else plots.copy()


def extract_msi_predictors(plots: gpd.GeoDataFrame, msi_files: Sequence[Path], plot_id_col: str, compute_texture: bool = True) -> pd.DataFrame:
    plots_raster_crs = prepare_plots_for_rasters(plots, msi_files)
    records: List[Dict[str, float]] = []
    band_indexes = [MSI_BAND_ORDER[k] for k in ("R", "G", "B", "NIR")]
    for _, row in tqdm(plots_raster_crs.iterrows(), total=len(plots_raster_crs), desc="Extracting MSI"):
        plot_id = row[plot_id_col]
        geom = row.geometry
        arrays: Dict[str, List[np.ndarray]] = {k: [] for k in ["R", "G", "B", "NIR", "NDVI", "GNDVI", "NDWI", "VARI", "ExG", "brightness"]}
        weights_all: List[np.ndarray] = []
        texture_rows: List[Dict[str, float]] = []
        for path in msi_files:
            with rasterio.open(path) as src:
                values, weights = read_weighted_pixels(src, geom, band_indexes)
                if values.shape[1] == 0:
                    continue
                R, G, B, NIR = values
                NDVI = safe_divide(NIR - R, NIR + R)
                GNDVI = safe_divide(NIR - G, NIR + G)
                NDWI = safe_divide(G - NIR, G + NIR)
                VARI = safe_divide(G - R, G + R - B)
                ExG = 2 * G - R - B
                brightness = (R + G + B + NIR) / 4.0
                tile_arrays = {"R": R, "G": G, "B": B, "NIR": NIR, "NDVI": NDVI, "GNDVI": GNDVI, "NDWI": NDWI, "VARI": VARI, "ExG": ExG, "brightness": brightness}
                for key, arr in tile_arrays.items():
                    arrays[key].append(arr)
                weights_all.append(weights)
                if compute_texture and SKIMAGE_AVAILABLE:
                    try:
                        window = windows.from_bounds(*geom.bounds, transform=src.transform)
                        window = window.round_offsets().round_lengths()
                        window = window.intersection(windows.Window(0, 0, src.width, src.height))
                        img = src.read(indexes=band_indexes, window=window, boundless=False)
                        img = clean_array(img, src.nodata)
                        weights_2d = pixel_overlap_weights(src, geom, window)
                        R2, G2, B2, NIR2 = img
                        NDVI2 = safe_divide(NIR2 - R2, NIR2 + R2)
                        bright2 = (R2 + G2 + B2 + NIR2) / 4.0
                        NDVI2[weights_2d <= 0] = np.nan
                        bright2[weights_2d <= 0] = np.nan
                        tex = {}
                        tex.update(make_texture(NDVI2, "NDVI"))
                        tex.update(make_texture(bright2, "brightness"))
                        texture_rows.append(tex)
                    except Exception:
                        pass
        rec: Dict[str, float] = {plot_id_col: plot_id}
        if weights_all:
            weights_combined = np.concatenate(weights_all)
            def concat(name: str) -> np.ndarray:
                return np.concatenate(arrays[name]) if arrays[name] else np.array([], dtype="float64")
            rec.update(weighted_summary(concat("R"), weights_combined, "R", ["mean", "std", "max", "p50", "p90"]))
            rec.update(weighted_summary(concat("G"), weights_combined, "G", ["mean", "std", "max", "p50", "p90"]))
            rec.update(weighted_summary(concat("B"), weights_combined, "B", ["mean", "std", "max", "p50", "p90"]))
            rec.update(weighted_summary(concat("NIR"), weights_combined, "NIR", ["mean", "std"]))
            rec.update(weighted_summary(concat("NDVI"), weights_combined, "NDVI", ["mean", "std", "min", "max", "p90"]))
            rec.update(weighted_summary(concat("GNDVI"), weights_combined, "GNDVI", ["mean", "std", "max", "p50", "p90"]))
            rec.update(weighted_summary(concat("NDWI"), weights_combined, "NDWI", ["mean", "std", "max", "p50", "p90"]))
            rec.update(weighted_summary(concat("VARI"), weights_combined, "VARI", ["mean", "std", "max", "p50", "p90"]))
            rec.update(weighted_summary(concat("ExG"), weights_combined, "ExG", ["mean", "std", "min", "max", "p50", "p90"]))
            rec.update(weighted_summary(concat("brightness"), weights_combined, "brightness", ["mean", "std", "max", "p50", "p90"]))
            ndvi = concat("NDVI")
            bright = concat("brightness")
            rec["NDVI_fraction_above_0_5"] = weighted_fraction(ndvi, weights_combined, lambda x: x > 0.5)
            rec["NDVI_fraction_above_0_6"] = weighted_fraction(ndvi, weights_combined, lambda x: x > 0.6)
            rec["brightness_p90_minus_p10"] = weighted_percentile(bright, weights_combined, 90) - weighted_percentile(bright, weights_combined, 10)
        if texture_rows:
            tex_df = pd.DataFrame(texture_rows)
            for col in tex_df.columns:
                rec[col] = tex_df[col].mean(skipna=True)
        records.append(rec)
    out = pd.DataFrame(records)
    for col in ["NDVI_texture_contrast", "NDVI_texture_homogeneity", "brightness_texture_homogeneity"]:
        if col not in out.columns:
            out[col] = np.nan
    if "brightness_texture_contrast" in out.columns:
        out = out.drop(columns=["brightness_texture_contrast"])
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
    msi_files = find_rasters(args.msi_folder)
    plot_meta_cols = [args.plot_id_col]
    if args.study_area_col and args.study_area_col in plots.columns:
        plot_meta_cols.append(args.study_area_col)
    plot_meta = plots.drop(columns="geometry")[plot_meta_cols].copy()
    msi_df = extract_msi_predictors(plots, msi_files, args.plot_id_col, compute_texture=not args.no_texture)
    out = plot_meta.merge(msi_df, on=args.plot_id_col, how="left")
    rename = {args.plot_id_col: "plot_id"}
    if args.study_area_col and args.study_area_col in out.columns:
        rename[args.study_area_col] = "study_area_id"
    out = out.rename(columns=rename)
    out.to_csv(output_csv, index=False)
    print(f"Saved MSI predictors: {output_csv}")
    print(f"Rows: {len(out)}; Columns: {len(out.columns)}")


if __name__ == "__main__":
    main()
