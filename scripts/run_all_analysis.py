"""Run the modular analysis workflow in order.

This wrapper executes the analysis-stage scripts in one shared Python namespace,
so later stages can reuse objects created by earlier stages. For reviewers who
want the simplest full rerun, this is the recommended entry point.

"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all modelling and figure/table stages.")
    parser.add_argument("--data", required=True, help="Analysis-ready predictor matrix CSV.")
    parser.add_argument("--target-col", default="Dead_F", help="Target column name. Default: Dead_F")
    parser.add_argument("--out-root", default="results/REANALYSIS", help="Output root folder.")
    parser.add_argument("--figures", default="final", help="Figure selection passed to 02_model_nested_cv.py.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing --out-root folder.")
    return parser.parse_args()


def exec_script(path: Path, namespace: dict) -> None:
    print("\n" + "=" * 80)
    print(f"Running: {path.name}")
    print("=" * 80)
    code = path.read_text(encoding="utf-8")
    exec(compile(code, str(path), "exec"), namespace)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    stages = [
    "02_model_nested_cv.py",
    "03_added_value_SE.py",
    "04_training_sample_size.py",
    "05_shap_analysis.py",
    "06_plot_specific_analysis.py",
    "07_make_final_figures_tables.py",
]

    namespace = {"__name__": "__main__"}

    # The first analysis stage has its own argparse interface. We temporarily
    # provide the appropriate command-line arguments to it.
    original_argv = sys.argv[:]
    sys.argv = [
        str(script_dir / "02_model_nested_cv.py"),
        "--data", args.data,
        "--target-col", args.target_col,
        "--out-root", args.out_root,
        "--figures", args.figures,
    ]
    if args.overwrite:
        sys.argv.append("--overwrite")

    try:
        for i, stage in enumerate(stages):
            if i > 0:
                # Later stages do not parse command-line args; keep sys.argv simple.
                sys.argv = [str(script_dir / stage)]
            exec_script(script_dir / stage, namespace)
    finally:
        sys.argv = original_argv

    print("\nAll analysis stages completed.")


if __name__ == "__main__":
    main()
