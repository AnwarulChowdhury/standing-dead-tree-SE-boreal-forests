r"""

Outputs
-------
CHM_mean, CHM_sd, CHM_cv, CHM_max, CHM_p90, n_chm_pixels,
chm_tiles_used, chm_periods_used, chm_status

"""

from __future__ import annotations

import argparse
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import box
from tqdm import tqdm

INDEX_URL = "https://avoin.metsakeskus.fi/aineistot/Latvusmalli/Latvusmalli_indeksi/Latvusmalli_indeksi.zip"
CHM_ROOT_URL = "https://avoin.metsakeskus.fi/aineistot/Latvusmalli/Karttalehti/"
PERIOD_PRIORITY = ["2025"] + [str(y) for y in range(2024, 2010, -1)]


def download_file(url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, out_path)
    return out_path


def url_exists(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=20) as response:
            return response.status == 200
    except Exception:
        try:
            req = urllib.request.Request(url)
            req.add_header("Range", "bytes=0-10")
            with urllib.request.urlopen(req, timeout=20) as response:
                return response.status in (200, 206)
        except Exception:
            return False


def load_chm_index(cache_dir: Path) -> gpd.GeoDataFrame:
    index_zip = cache_dir / "Latvusmalli_indeksi.zip"
    index_dir = cache_dir / "Latvusmalli_indeksi"
    index_dir.mkdir(parents=True, exist_ok=True)

    download_file(INDEX_URL, index_zip)

    if not any(index_dir.iterdir()):
        with zipfile.ZipFile(index_zip, "r") as z:
            z.extractall(index_dir)

    candidates = (
        list(index_dir.rglob("*.gpkg"))
        + list(index_dir.rglob("*.shp"))
        + list(index_dir.rglob("*.geojson"))
    )
    if not candidates:
        raise FileNotFoundError("No vector file found in Latvusmalli index zip.")

    print(f"Reading index vector: {candidates[0]}")
    return gpd.read_file(candidates[0])


def extract_map_sheet_id(value: object) -> str | None:
    """Extract map-sheet ID such as M4334E from strings like CHM_M4334E_2018."""
    s = str(value).strip().upper()
    match = re.search(r"([A-Z]\d{4}[A-H])", s)
    if match:
        return match.group(1)
    return None


def detect_tile_column(index_gdf: gpd.GeoDataFrame) -> str:
    best_col = None
    best_count = 0
    for col in index_gdf.columns:
        if col == "geometry":
            continue
        count = index_gdf[col].apply(extract_map_sheet_id).notna().sum()
        if count > best_count:
            best_col = col
            best_count = count
    if best_col is None or best_count == 0:
        raise ValueError("Could not detect map-sheet column in index.")
    print(f"Using index tile/map-sheet column: {best_col}")
    return best_col


def chm_url(tile_id: str, period: str) -> str:
    return f"{CHM_ROOT_URL}{period}/CHM_{tile_id}_{period}.tif"


def choose_id_column(gdf: gpd.GeoDataFrame, requested: str | None) -> str:
    if requested:
        if requested not in gdf.columns:
            raise ValueError(f"Column '{requested}' not found. Available columns: {list(gdf.columns)}")
        return requested
    for col in ["plot_id", "Plot_ID", "PLOT_ID", "Name", "NAME", "PlotName", "plot_name", "OBJECTID", "fid", "ID", "id", "FOREST_NAM"]:
        if col in gdf.columns:
            return col
    gdf["plot_id_auto"] = [f"plot_{i+1:03d}" for i in range(len(gdf))]
    return "plot_id_auto"


def weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.nan
    v = values[mask]
    w = weights[mask]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cw = np.cumsum(w)
    cutoff = percentile / 100.0 * cw[-1]
    return float(v[np.searchsorted(cw, cutoff)])


def chm_metrics(values: list[float], weights: list[float]) -> dict[str, float | int]:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return {
            "CHM_mean": np.nan,
            "CHM_sd": np.nan,
            "CHM_cv": np.nan,
            "CHM_max": np.nan,
            "CHM_p90": np.nan,
            "n_chm_pixels": 0,
        }
    v = v[mask]
    w = w[mask]
    mean = float(np.average(v, weights=w))
    sd = float(np.sqrt(np.average((v - mean) ** 2, weights=w)))
    return {
        "CHM_mean": mean,
        "CHM_sd": sd,
        "CHM_cv": float(sd / mean) if mean > 0 else np.nan,
        "CHM_max": float(np.nanmax(v)),
        "CHM_p90": weighted_percentile(v, w, 90),
        "n_chm_pixels": int(len(v)),
    }


def extract_weighted_values_from_raster(raster_path: Path, polygon) -> tuple[list[float], list[float]]:
    values: list[float] = []
    weights: list[float] = []
    with rasterio.open(raster_path) as src:
        raster_box = box(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        if not polygon.intersects(raster_box):
            return values, weights

        geom = polygon.intersection(raster_box)
        if geom.is_empty:
            return values, weights

        window = from_bounds(*geom.bounds, transform=src.transform).round_offsets().round_lengths()
        arr = src.read(1, window=window, masked=True)
        if arr.size == 0:
            return values, weights

        transform = src.window_transform(window)
        nrows, ncols = arr.shape
        for row in range(nrows):
            for col in range(ncols):
                val = arr[row, col]
                if np.ma.is_masked(val):
                    continue
                val = float(val)
                if not np.isfinite(val) or val < 0:
                    continue

                x0, y0 = transform * (col, row)
                x1, y1 = transform * (col + 1, row + 1)
                pix = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
                overlap = pix.intersection(polygon).area
                if overlap <= 0:
                    continue
                values.append(val)
                weights.append(overlap / pix.area)
    return values, weights


class TileFetcher:
    """Cache URL checks and downloaded rasters during one run."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.exists_cache: dict[tuple[str, str], bool] = {}
        self.path_cache: dict[tuple[str, str], Path | None] = {}

    def get_path(self, tile_id: str, period: str) -> Path | None:
        key = (tile_id, period)
        if key in self.path_cache:
            return self.path_cache[key]

        url = chm_url(tile_id, period)
        if key not in self.exists_cache:
            self.exists_cache[key] = url_exists(url)
        if not self.exists_cache[key]:
            self.path_cache[key] = None
            return None

        local_path = self.cache_dir / "chm_tiles" / period / f"CHM_{tile_id}_{period}.tif"
        path = download_file(url, local_path)
        self.path_cache[key] = path
        return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Finnish Forest Centre 1 m CHM metrics with partial pixel weighting. "
            "For each plot/map sheet, try 2025 first; if zero valid pixels, fallback to the newest older annual file."
        )
    )
    parser.add_argument("--polygons", required=True, help="Input 15 m plot-buffer polygons, e.g. shapefile.")
    parser.add_argument("--output", required=True, help="Output CSV for CHM metrics.")
    parser.add_argument("--cache-dir", default="temporary_chm_cache", help="Temporary folder for index and downloaded CHM files.")
    parser.add_argument("--plot-id-col", default=None, help="Polygon ID column, e.g. Name, OBJECTID, plot_id.")
    parser.add_argument("--delete-cache-after-run", action="store_true", help="Delete temporary downloaded CHM files after successful extraction.")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Reading input polygons...")
    polygons = gpd.read_file(args.polygons)
    if polygons.crs is None:
        raise ValueError("Input polygons have no CRS.")
    id_col = choose_id_column(polygons, args.plot_id_col)
    print(f"Using polygon ID column: {id_col}")

    print("Loading Finnish Forest Centre Latvusmalli index...")
    index = load_chm_index(cache_dir)
    if index.crs is None:
        raise ValueError("CHM index has no CRS.")
    tile_col = detect_tile_column(index)
    index["map_sheet_id"] = index[tile_col].apply(extract_map_sheet_id)
    index = index.dropna(subset=["map_sheet_id"]).drop_duplicates(subset=["map_sheet_id"])

    polygons_index_crs = polygons.to_crs(index.crs)
    joined = gpd.sjoin(
        polygons_index_crs[[id_col, "geometry"]],
        index[["map_sheet_id", "geometry"]],
        how="left",
        predicate="intersects",
    )
    needed_tiles = sorted(joined["map_sheet_id"].dropna().unique())
    print(f"Unique intersecting CHM map sheets: {len(needed_tiles)}")
    print(", ".join(needed_tiles[:20]) + (" ..." if len(needed_tiles) > 20 else ""))
    print("Period order: " + ", ".join(PERIOD_PRIORITY[:8]) + ", ...")

    # Find a raster CRS from the first available tile among needed map sheets.
    fetcher = TileFetcher(cache_dir)
    first_raster = None
    first_raster_period = None
    print("Finding one available CHM raster to get raster CRS...")
    for tile_id in needed_tiles:
        for period in PERIOD_PRIORITY:
            candidate = fetcher.get_path(tile_id, period)
            if candidate is not None:
                first_raster = candidate
                first_raster_period = period
                break
        if first_raster is not None:
            break
    if first_raster is None:
        raise RuntimeError("No CHM tiles could be found for the input polygons.")

    with rasterio.open(first_raster) as src:
        raster_crs = src.crs
    print(f"Using raster CRS from first available tile ({first_raster_period}): {raster_crs}")

    polygons_raster_crs = polygons.to_crs(raster_crs)
    index_raster_crs = index.to_crs(raster_crs)

    large_area = polygons_raster_crs.geometry.area.max()
    if large_area > 10000:
        print("WARNING: At least one polygon is larger than 10,000 m². Use 15 m plot buffers for manuscript predictors.")

    records = []
    detailed_rows = []

    for _, row in tqdm(polygons_raster_crs.iterrows(), total=len(polygons_raster_crs), desc="Extracting CHM metrics"):
        polygon = row.geometry
        plot_id = row[id_col]
        possible_tiles = index_raster_crs[index_raster_crs.geometry.intersects(polygon)]["map_sheet_id"].tolist()

        all_values: list[float] = []
        all_weights: list[float] = []
        used_tiles: list[str] = []
        used_periods: list[str] = []

        for tile_id in possible_tiles:
            found_values_for_this_tile = False
            for period in PERIOD_PRIORITY:
                raster_path = fetcher.get_path(tile_id, period)
                if raster_path is None:
                    continue

                vals, wts = extract_weighted_values_from_raster(raster_path, polygon)
                detailed_rows.append(
                    {
                        "plot_id": plot_id,
                        "map_sheet_id": tile_id,
                        "period_tested": period,
                        "file_exists": True,
                        "valid_pixels_found": len(vals),
                        "used_for_metrics": len(vals) > 0,
                    }
                )

                if vals:
                    all_values.extend(vals)
                    all_weights.extend(wts)
                    used_tiles.append(tile_id)
                    used_periods.append(period)
                    found_values_for_this_tile = True
                    break

            if not found_values_for_this_tile:
                detailed_rows.append(
                    {
                        "plot_id": plot_id,
                        "map_sheet_id": tile_id,
                        "period_tested": "all_available_checked",
                        "file_exists": False,
                        "valid_pixels_found": 0,
                        "used_for_metrics": False,
                    }
                )

        metrics = chm_metrics(all_values, all_weights)
        status = "ok" if metrics["n_chm_pixels"] > 0 else "no_valid_chm_pixels_after_fallback"
        rec = {
            "plot_id": plot_id,
            **metrics,
            "chm_tiles_used": ";".join(sorted(set(used_tiles))),
            "chm_periods_used": ";".join(sorted(set(used_periods))),
            "chm_status": status,
        }
        records.append(rec)

    out = pd.DataFrame(records)
    out.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Saved CHM metrics: {output}")

    detail_path = output.with_name(output.stem + "_fallback_detail.csv")
    pd.DataFrame(detailed_rows).to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"Saved fallback detail table: {detail_path}")

    summary = (
        out.groupby(["chm_status", "chm_periods_used"], dropna=False)
        .size()
        .reset_index(name="n_plots")
    )
    summary_path = output.with_name(output.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Saved summary table: {summary_path}")

    print("\nCHM status counts:")
    print(out["chm_status"].value_counts(dropna=False))
    print("\nPeriods used:")
    print(out["chm_periods_used"].value_counts(dropna=False))

    if args.delete_cache_after_run:
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"Deleted temporary cache folder: {cache_dir}")


if __name__ == "__main__":
    main()
