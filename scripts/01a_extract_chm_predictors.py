"""
Outputs
-------
plot_id, CHM_mean, CHM_sd, CHM_cv, CHM_max, CHM_p90, n_chm_pixels,
chm_rasters_used, chm_status

Example
-------
python extract_chm_metrics_local_only.py \
    --polygons "plots_15m_buffers.shp" \
    --imagery-dir "D:/path/to/local/chm_tiles" \ ## need to change directory
    --output "chm_predictors.csv" \
    --plot-id-col "Name" \
    --glob "*.tif"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import box
from tqdm import tqdm


VALID_RASTER_SUFFIXES = {".tif", ".tiff"}


def choose_id_column(gdf: gpd.GeoDataFrame, requested: str | None) -> str:
    """Choose a stable plot ID column, or create one if no known ID field exists."""
    if requested:
        if requested not in gdf.columns:
            raise ValueError(
                f"Column '{requested}' not found. Available columns: {list(gdf.columns)}"
            )
        return requested

    for col in [
        "plot_id", "Plot_ID", "PLOT_ID", "Name", "NAME", "PlotName",
        "plot_name", "OBJECTID", "fid", "ID", "id", "FOREST_NAM",
    ]:
        if col in gdf.columns:
            return col

    gdf["plot_id_auto"] = [f"plot_{i + 1:03d}" for i in range(len(gdf))]
    return "plot_id_auto"


def find_local_rasters(imagery_dir: Path, glob_pattern: str, recursive: bool) -> list[Path]:
    """Return local raster paths matching the requested pattern."""
    if not imagery_dir.exists():
        raise FileNotFoundError(f"Imagery directory does not exist: {imagery_dir}")
    if not imagery_dir.is_dir():
        raise NotADirectoryError(f"Imagery path is not a directory: {imagery_dir}")

    if recursive:
        candidates = list(imagery_dir.rglob(glob_pattern))
    else:
        candidates = list(imagery_dir.glob(glob_pattern))

    rasters = sorted(
        p for p in candidates
        if p.is_file() and p.suffix.lower() in VALID_RASTER_SUFFIXES
    )

    if not rasters:
        raise FileNotFoundError(
            f"No raster files found in {imagery_dir} using pattern '{glob_pattern}'."
        )

    return rasters


def weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    """Weighted percentile for partial pixel overlap weights."""
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.nan

    v = values[mask]
    w = weights[mask]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cumulative_weight = np.cumsum(w)
    cutoff = percentile / 100.0 * cumulative_weight[-1]
    return float(v[np.searchsorted(cumulative_weight, cutoff)])


def chm_metrics(values: list[float], weights: list[float]) -> dict[str, float | int]:
    """Calculate weighted CHM summary metrics."""
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


def extract_weighted_values_from_raster(
    raster_path: Path,
    polygon,
    minimum_valid_value: float,
) -> tuple[list[float], list[float]]:
    """Extract raster values weighted by polygon overlap with each pixel."""
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
                if not np.isfinite(val) or val < minimum_valid_value:
                    continue

                x0, y0 = transform * (col, row)
                x1, y1 = transform * (col + 1, row + 1)
                pixel = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
                overlap_area = pixel.intersection(polygon).area

                if overlap_area <= 0:
                    continue

                values.append(val)
                weights.append(overlap_area / pixel.area)

    return values, weights


def raster_metadata_table(rasters: list[Path]) -> pd.DataFrame:
    """Build a lightweight table of raster path, CRS, and bounds."""
    rows = []
    for path in rasters:
        try:
            with rasterio.open(path) as src:
                if src.crs is None:
                    raise ValueError("Raster has no CRS")
                rows.append(
                    {
                        "path": path,
                        "name": path.name,
                        "crs": src.crs,
                        "left": src.bounds.left,
                        "bottom": src.bounds.bottom,
                        "right": src.bounds.right,
                        "top": src.bounds.top,
                        "geometry": box(
                            src.bounds.left,
                            src.bounds.bottom,
                            src.bounds.right,
                            src.bounds.top,
                        ),
                    }
                )
        except Exception as exc:
            print(f"WARNING: Skipping unreadable raster: {path} ({exc})")

    if not rows:
        raise RuntimeError("No readable rasters with CRS were found.")

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract CHM metrics from local raster imagery only. No downloads and no period fallback."
    )
    parser.add_argument(
        "--polygons",
        required=True,
        help="Input plot-buffer polygons, for example a shapefile or GeoPackage.",
    )
    parser.add_argument(
        "--imagery-dir",
        required=True,
        help="Local directory containing CHM raster tiles.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV for CHM metrics.",
    )
    parser.add_argument(
        "--plot-id-col",
        default=None,
        help="Polygon ID column, for example Name, OBJECTID, or plot_id.",
    )
    parser.add_argument(
        "--glob",
        default="*.tif",
        help="Raster filename pattern. Use this to restrict year/product, for example CHM_*_2025.tif.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for rasters recursively inside imagery-dir.",
    )
    parser.add_argument(
        "--minimum-valid-value",
        type=float,
        default=0.0,
        help="Minimum raster value treated as valid. Default: 0.0.",
    )
    args = parser.parse_args()

    polygons_path = Path(args.polygons).expanduser().resolve()
    imagery_dir = Path(args.imagery_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    print("Reading input polygons...")
    polygons = gpd.read_file(polygons_path)
    if polygons.crs is None:
        raise ValueError("Input polygons have no CRS.")

    id_col = choose_id_column(polygons, args.plot_id_col)
    print(f"Using polygon ID column: {id_col}")

    print("Scanning local imagery directory...")
    raster_paths = find_local_rasters(imagery_dir, args.glob, args.recursive)
    print(f"Found {len(raster_paths)} local raster(s).")

    rasters_df = raster_metadata_table(raster_paths)
    crs_values = list(dict.fromkeys(str(crs) for crs in rasters_df["crs"]))
    print(f"Raster CRS count: {len(crs_values)}")

    if len(crs_values) > 1:
        print(
            "WARNING: Multiple raster CRSs were detected. "
            "The script will reproject polygons separately for each raster CRS."
        )

    large_area_checked = False
    records = []
    detailed_rows = []

    for _, poly_row in tqdm(polygons.iterrows(), total=len(polygons), desc="Extracting CHM metrics"):
        plot_id = poly_row[id_col]
        all_values: list[float] = []
        all_weights: list[float] = []
        used_rasters: list[str] = []

        for _, raster_row in rasters_df.iterrows():
            raster_path = Path(raster_row["path"])
            raster_crs = raster_row["crs"]

            polygon_raster_crs = (
                gpd.GeoSeries([poly_row.geometry], crs=polygons.crs)
                .to_crs(raster_crs)
                .iloc[0]
            )

            if not large_area_checked:
                if polygon_raster_crs.area > 10000:
                    print(
                        "WARNING: At least one polygon is larger than 10,000 m². "
                        "Use 15 m plot buffers for manuscript predictors."
                    )
                large_area_checked = True

            raster_bounds_geom = raster_row["geometry"]
            if not polygon_raster_crs.intersects(raster_bounds_geom):
                continue

            vals, wts = extract_weighted_values_from_raster(
                raster_path,
                polygon_raster_crs,
                minimum_valid_value=args.minimum_valid_value,
            )

            detailed_rows.append(
                {
                    "plot_id": plot_id,
                    "raster": raster_path.name,
                    "raster_path": str(raster_path),
                    "valid_pixels_found": len(vals),
                    "used_for_metrics": len(vals) > 0,
                }
            )

            if vals:
                all_values.extend(vals)
                all_weights.extend(wts)
                used_rasters.append(raster_path.name)

        metrics = chm_metrics(all_values, all_weights)
        status = "ok" if metrics["n_chm_pixels"] > 0 else "no_valid_chm_pixels_in_local_rasters"

        records.append(
            {
                "plot_id": plot_id,
                **metrics,
                "chm_rasters_used": ";".join(sorted(set(used_rasters))),
                "chm_status": status,
            }
        )

    out = pd.DataFrame(records)
    out.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Saved CHM metrics: {output}")

    detail_path = output.with_name(output.stem + "_local_raster_detail.csv")
    pd.DataFrame(detailed_rows).to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"Saved local raster detail table: {detail_path}")

    summary = (
        out.groupby(["chm_status"], dropna=False)
        .size()
        .reset_index(name="n_plots")
    )
    summary_path = output.with_name(output.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Saved summary table: {summary_path}")

    print("\nCHM status counts:")
    print(out["chm_status"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
