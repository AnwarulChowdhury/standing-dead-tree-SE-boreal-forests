"""Build the final analysis-ready predictor matrix.

This script merges the three predictor-extraction outputs:
1. CHM predictors
2. MSI predictors
3. AlphaEarth SE predictors

Then it joins the standing dead-tree volume fraction response variable.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge CHM, MSI, SE, and response data into the analysis matrix.")
    parser.add_argument("--chm-csv", required=True, help="CSV from 01a_extract_chm_predictors.py")
    parser.add_argument("--msi-csv", required=True, help="CSV from 01b_extract_msi_predictors.py")
    parser.add_argument("--se-csv", required=True, help="CSV from 01c_extract_se_predictors.py")
    parser.add_argument("--volume-csv", required=True, help="CSV containing standing dead-tree volume fraction.")
    parser.add_argument("--plot-id-col", default="plot_id", help="Plot ID column in predictor CSVs. Default: plot_id")
    parser.add_argument("--volume-plot-col", default="Plot", help="Plot ID column in the response CSV. Default: Plot")
    parser.add_argument("--target-col", default="Dead_F", help="Response column in volume CSV. Default: Dead_F")
    parser.add_argument("--output-csv", default="results/00_extracted_predictors/analysis_ready_predictor_matrix.csv", help="Output CSV path.")
    return parser.parse_args()


def clean_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def main() -> None:
    args = parse_args()
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)

    chm = pd.read_csv(args.chm_csv)
    msi = pd.read_csv(args.msi_csv)
    se = pd.read_csv(args.se_csv)
    volume = pd.read_csv(args.volume_csv)

    for name, df in {"CHM": chm, "MSI": msi, "SE": se}.items():
        if args.plot_id_col not in df.columns:
            raise ValueError(f"{name} file does not contain plot ID column '{args.plot_id_col}'.")
        df[args.plot_id_col] = clean_id(df[args.plot_id_col])

    if args.volume_plot_col not in volume.columns:
        raise ValueError(f"Volume CSV does not contain plot ID column '{args.volume_plot_col}'.")
    if args.target_col not in volume.columns:
        raise ValueError(f"Volume CSV does not contain target column '{args.target_col}'.")
    volume[args.volume_plot_col] = clean_id(volume[args.volume_plot_col])

    df = se.merge(msi, on=args.plot_id_col, how="left", suffixes=("", "_msi"))
    df = df.merge(chm, on=args.plot_id_col, how="left", suffixes=("", "_chm"))
    df = df.merge(volume[[args.volume_plot_col, args.target_col]], left_on=args.plot_id_col, right_on=args.volume_plot_col, how="inner")

    if args.volume_plot_col != args.plot_id_col and args.volume_plot_col in df.columns:
        df = df.drop(columns=[args.volume_plot_col])
    if args.target_col != "Dead_F":
        df = df.rename(columns={args.target_col: "Dead_F"})

    first_cols = [c for c in [args.plot_id_col, "study_area_id"] if c in df.columns]
    response_col = "Dead_F"
    predictor_cols = [c for c in df.columns if c not in first_cols + [response_col]]
    df = df[first_cols + predictor_cols + [response_col]]
    if args.plot_id_col != "plot_id":
        df = df.rename(columns={args.plot_id_col: "plot_id"})

    df.to_csv(output, index=False)
    print(f"Saved analysis-ready predictor matrix: {output}")
    print(f"Rows: {len(df)}; Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()
