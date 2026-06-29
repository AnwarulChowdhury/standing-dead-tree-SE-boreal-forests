"""Summarize spatial sampling design from plot/study-area polygons.
The public repository does not include exact plot geometries. Run this script
only with the local/private plot or study-area spatial file.

"""

from __future__ import annotations

import argparse
import re
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd


VECTOR_EXTENSIONS = (".geojson", ".json", ".gpkg", ".shp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create plot-count and spatial-distance summaries for the sampling design."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input vector file or ZIP containing a GeoJSON, shapefile, or GeoPackage.",
    )
    parser.add_argument(
        "--name-col",
        default="Name",
        help="Plot-name/plot-ID column used for grouping when --study-area-col is not provided.",
    )
    parser.add_argument(
        "--study-area-col",
        default=None,
        help="Optional official study-area column. If provided, this is used instead of name-based grouping.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/08_spatial_sampling",
        help="Output folder for CSV summaries.",
    )
    parser.add_argument(
        "--metric-crs",
        default="EPSG:3067",
        help="Projected CRS for distance calculations in metres. Default: EPSG:3067.",
    )
    parser.add_argument(
        "--keep-extracted",
        action="store_true",
        help="Keep temporary extracted files when --input is a ZIP.",
    )
    return parser.parse_args()


def find_vector_in_zip(zip_path: Path, extract_dir: Path) -> Path:
    """Extract a ZIP and return the first supported vector file found inside."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    candidates = []
    for ext in VECTOR_EXTENSIONS:
        candidates.extend(extract_dir.rglob(f"*{ext}"))

    # Prefer GeoJSON/GPKG over individual shapefile components if several exist.
    priority = {".geojson": 0, ".json": 1, ".gpkg": 2, ".shp": 3}
    candidates = sorted(candidates, key=lambda p: (priority.get(p.suffix.lower(), 99), str(p)))
    if not candidates:
        raise FileNotFoundError(f"No supported vector file found inside ZIP: {zip_path}")
    return candidates[0]


def resolve_input_vector(input_path: Path, out_dir: Path) -> tuple[Path, Optional[Path]]:
    """Return vector path and optional temporary extraction folder."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() == ".zip":
        extract_dir = out_dir / "_extracted_input"
        vector_path = find_vector_in_zip(input_path, extract_dir)
        return vector_path, extract_dir

    if input_path.suffix.lower() not in VECTOR_EXTENSIONS:
        raise ValueError(
            f"Unsupported input extension: {input_path.suffix}. "
            f"Supported: {', '.join(VECTOR_EXTENSIONS)} or .zip"
        )
    return input_path, None


def assign_study_area_from_name(name: object) -> str:
    """Assign study area from original plot/site names.

    This mirrors the original spatial summary script. If an official study-area
    column exists, prefer --study-area-col instead of this name-based grouping.
    """
    s = str(name).lower()

    if "isojärvi" in s or "isojarvi" in s:
        return "Isojärvi"
    if "kaihua" in s or "kivalot" in s:
        return "Kaihua–Kivalot"
    if "aarnikotka" in s:
        return "Aarnikotka"
    if "rohrstrand" in s or re.fullmatch(r"plot_?\d+", s):
        return "Rohrstrand"
    if "ruunaa" in s:
        return "Ruunaa"
    if "kolovesi" in s:
        return "Kolovesi"
    if "martinselkonen" in s:
        return "Martinselkonen"
    if "närängänvaara" in s or "naranganvaara" in s or "virmajoki" in s:
        return "Närängänvaara–Virmajoki"
    if "pilkkakorvenmäki" in s or "pilkkakorvenmaki" in s:
        return "Pilkkakorvenmäki"
    if "salamajärvi" in s or "salamajarvi" in s:
        return "Salamajärvi"
    if (
        "peuratunturi" in s
        or "muotkavaara" in s
        or "kuusivaara" in s
        or "sallatunturi" in s
    ):
        return "Salla area"
    if "ukk_north" in s:
        return "UKK North"
    if "yllas_pallas" in s or "ylläs" in s or "ylläspallas" in s:
        return "Ylläs–Pallas"

    return "Unknown"


def distance_summary(values: np.ndarray, as_km: bool = False) -> tuple[str, str]:
    """Return formatted median and range strings."""
    values = np.asarray(values, dtype="float64")
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return "NA", "NA"
    if as_km:
        return f"{np.median(values) / 1000:.2f} km", f"{values.min() / 1000:.2f}–{values.max() / 1000:.2f} km"
    return f"{np.median(values):.0f} m", f"{values.min():.0f} m–{values.max() / 1000:.2f} km"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vector_path, extract_dir = resolve_input_vector(input_path, out_dir)
    print(f"Reading: {vector_path}")

    gdf = gpd.read_file(vector_path)
    if gdf.crs is None:
        raise ValueError("Input vector file has no CRS. Define CRS before calculating distances.")

    if args.study_area_col and args.study_area_col not in gdf.columns:
        raise ValueError(
            f"Study-area column '{args.study_area_col}' not found. Available columns: {list(gdf.columns)}"
        )
    if not args.study_area_col and args.name_col not in gdf.columns:
        raise ValueError(
            f"Name column '{args.name_col}' not found. Provide --study-area-col or choose a valid --name-col. "
            f"Available columns: {list(gdf.columns)}"
        )

    # Distances in metres.
    plots = gdf.to_crs(args.metric_crs).copy()
    plots["geometry"] = plots.geometry.centroid

    if args.study_area_col:
        plots["study_area"] = plots[args.study_area_col].astype(str).str.strip()
    else:
        plots["study_area"] = plots[args.name_col].apply(assign_study_area_from_name)

    unknown = plots.loc[plots["study_area"] == "Unknown"]
    if len(unknown) > 0:
        print("Warning: some plots were not assigned to a study area:")
        cols = [c for c in [args.name_col, args.study_area_col] if c and c in plots.columns]
        print(unknown[cols].to_string(index=False))

    plot_counts = (
        plots.groupby("study_area")
        .size()
        .reset_index(name="Number of plots")
        .sort_values("study_area")
    )

    nearest_neighbour_distances: list[float] = []
    within_area_pairwise_distances: list[float] = []

    for _, sub in plots.groupby("study_area"):
        coords = np.array([(geom.x, geom.y) for geom in sub.geometry])
        if len(coords) < 2:
            continue
        dist_matrix = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(dist_matrix, np.inf)
        nearest_neighbour_distances.extend(dist_matrix.min(axis=1))
        upper_triangle = np.triu_indices(len(coords), k=1)
        within_area_pairwise_distances.extend(dist_matrix[upper_triangle])

    nearest_neighbour_distances_arr = np.asarray(nearest_neighbour_distances)
    within_area_pairwise_distances_arr = np.asarray(within_area_pairwise_distances)

    site_centroids = plots.groupby("study_area").geometry.apply(lambda geom: geom.union_all().centroid)
    if len(site_centroids) >= 2:
        site_coords = np.array([(geom.x, geom.y) for geom in site_centroids])
        site_dist_matrix = np.sqrt(((site_coords[:, None, :] - site_coords[None, :, :]) ** 2).sum(axis=2))
        upper_triangle = np.triu_indices(len(site_coords), k=1)
        between_area_distances = site_dist_matrix[upper_triangle]
    else:
        between_area_distances = np.array([], dtype="float64")

    nn_median, nn_range = distance_summary(nearest_neighbour_distances_arr, as_km=False)
    pair_median, pair_range = distance_summary(within_area_pairwise_distances_arr, as_km=True)
    between_median, between_range = distance_summary(between_area_distances, as_km=True)

    summary = pd.DataFrame(
        {
            "Spatial characteristic": [
                "Number of field plots",
                "Number of study areas",
                "Plots per study area, median",
                "Plots per study area, range",
                "Within-area nearest-neighbour distance, median",
                "Within-area nearest-neighbour distance, range",
                "Within-area pairwise plot distance, median",
                "Within-area pairwise plot distance, range",
                "Between-area centroid distance, median",
                "Between-area centroid distance, range",
            ],
            "Value": [
                len(plots),
                plots["study_area"].nunique(),
                int(plot_counts["Number of plots"].median()),
                f"{plot_counts['Number of plots'].min()}–{plot_counts['Number of plots'].max()}",
                nn_median,
                nn_range,
                pair_median,
                pair_range,
                between_median,
                between_range,
            ],
        }
    )

    counts_path = out_dir / "study_area_plot_counts.csv"
    summary_path = out_dir / "spatial_sampling_summary.csv"
    plot_counts.to_csv(counts_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nPlot counts per study area:")
    print(plot_counts.to_string(index=False))
    print("\nSpatial sampling summary:")
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(f"- {counts_path}")
    print(f"- {summary_path}")

    if extract_dir is not None and not args.keep_extracted:
        shutil.rmtree(extract_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
