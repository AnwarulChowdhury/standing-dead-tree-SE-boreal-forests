# Objective 2:
# Added-value assessment of SE predictors
# Comparing paired outer-CV performance with and without SE
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from pathlib import Path


# ------------------------------------------------------------
# 1. Output folder
# ------------------------------------------------------------

try:
    OUT_DIR
except NameError:
    OUT_DIR = Path(r"C:/Users/Dead_tree_fraction")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

O2_DIR = OUT_DIR / "O2_SE_added_value"
O2_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Detect outer CV split column
# ------------------------------------------------------------

if "Outer_Split" in results_df.columns:
    fold_col = "Outer_Split"
elif "Outer_CV_Run" in results_df.columns:
    fold_col = "Outer_CV_Run"
elif "CV_Fold" in results_df.columns:
    fold_col = "CV_Fold"
else:
    raise ValueError(
        "No outer CV split column found. Expected one of: "
        "'Outer_Split', 'Outer_CV_Run', or 'CV_Fold'. "
        f"Available columns are: {results_df.columns.tolist()}"
    )

print("Using outer CV split column:", fold_col)


# ------------------------------------------------------------
# 3. Define model order
# ------------------------------------------------------------

if "model_order" not in globals():
    model_order = ["RF", "GradientBoosting", "XGBoost"]


# ------------------------------------------------------------
# 4. Define SE added-value comparisons
# ------------------------------------------------------------

se_comparisons = {
    "CHM_to_CHM+SE": {
        "Without_SE": "CHM",
        "With_SE": "CHM+SE"
    },
    "RGBNIR_to_RGBNIR+SE": {
        "Without_SE": "RGBNIR",
        "With_SE": "RGBNIR+SE"
    },
    "CHM+RGBNIR_to_CHM+RGBNIR+SE": {
        "Without_SE": "CHM+RGBNIR",
        "With_SE": "CHM+RGBNIR+SE"
    }
}


# ------------------------------------------------------------
# 5. Paired SE added-value calculation
# ------------------------------------------------------------

added_value_rows = []
paired_delta_rows = []

for model_name in model_order:

    for comparison_name, comparison in se_comparisons.items():

        without_set = comparison["Without_SE"]
        with_set = comparison["With_SE"]

        without_df = results_df[
            (results_df["Feature_Set"] == without_set) &
            (results_df["Model"] == model_name)
        ][[fold_col, "R2", "RMSE", "MAE"]].copy()

        with_df = results_df[
            (results_df["Feature_Set"] == with_set) &
            (results_df["Model"] == model_name)
        ][[fold_col, "R2", "RMSE", "MAE"]].copy()

        if without_df.empty or with_df.empty:
            print(f"Skipping missing comparison: {model_name} - {comparison_name}")
            continue

        paired_df = without_df.merge(
            with_df,
            on=fold_col,
            suffixes=("_without_SE", "_with_SE")
        )

        if paired_df.shape[0] != without_df.shape[0] or paired_df.shape[0] != with_df.shape[0]:
            raise ValueError(
                f"Fold pairing mismatch for {model_name} - {comparison_name}: "
                f"without SE = {without_df.shape[0]}, "
                f"with SE = {with_df.shape[0]}, "
                f"paired = {paired_df.shape[0]}"
            )

        paired_df["Model"] = model_name
        paired_df["Comparison"] = comparison_name
        paired_df["Without_SE"] = without_set
        paired_df["With_SE"] = with_set

        paired_df["Delta_R2"] = (
            paired_df["R2_with_SE"] -
            paired_df["R2_without_SE"]
        )

        paired_df["Delta_RMSE"] = (
            paired_df["RMSE_without_SE"] -
            paired_df["RMSE_with_SE"]
        )

        paired_df["Delta_MAE"] = (
            paired_df["MAE_without_SE"] -
            paired_df["MAE_with_SE"]
        )

        paired_df["RMSE_percent_improvement"] = (
            paired_df["Delta_RMSE"] /
            paired_df["RMSE_without_SE"]
        ) * 100

        paired_df["MAE_percent_improvement"] = (
            paired_df["Delta_MAE"] /
            paired_df["MAE_without_SE"]
        ) * 100

        # Store split-level deltas
        paired_delta_rows.append(
            paired_df[
                [
                    fold_col,
                    "Model",
                    "Comparison",
                    "Without_SE",
                    "With_SE",
                    "R2_without_SE",
                    "R2_with_SE",
                    "Delta_R2",
                    "RMSE_without_SE",
                    "RMSE_with_SE",
                    "Delta_RMSE",
                    "RMSE_percent_improvement",
                    "MAE_without_SE",
                    "MAE_with_SE",
                    "Delta_MAE",
                    "MAE_percent_improvement"
                ]
            ]
        )

        # Wilcoxon paired tests
        try:
            p_r2 = wilcoxon(
                paired_df["R2_without_SE"],
                paired_df["R2_with_SE"],
                zero_method="wilcox"
            ).pvalue
        except ValueError:
            p_r2 = np.nan

        try:
            p_rmse = wilcoxon(
                paired_df["RMSE_without_SE"],
                paired_df["RMSE_with_SE"],
                zero_method="wilcox"
            ).pvalue
        except ValueError:
            p_rmse = np.nan

        try:
            p_mae = wilcoxon(
                paired_df["MAE_without_SE"],
                paired_df["MAE_with_SE"],
                zero_method="wilcox"
            ).pvalue
        except ValueError:
            p_mae = np.nan

        added_value_rows.append({
            "Model": model_name,
            "Comparison": comparison_name,
            "Without_SE": without_set,
            "With_SE": with_set,

            "N_Paired_Outer_Splits": paired_df.shape[0],

            "R2_without_SE_mean": paired_df["R2_without_SE"].mean(),
            "R2_with_SE_mean": paired_df["R2_with_SE"].mean(),
            "Delta_R2_mean": paired_df["Delta_R2"].mean(),
            "Delta_R2_sd": paired_df["Delta_R2"].std(ddof=1),
            "Delta_R2_median": paired_df["Delta_R2"].median(),
            "Wilcoxon_p_R2": p_r2,

            "RMSE_without_SE_mean": paired_df["RMSE_without_SE"].mean(),
            "RMSE_with_SE_mean": paired_df["RMSE_with_SE"].mean(),
            "Delta_RMSE_mean": paired_df["Delta_RMSE"].mean(),
            "Delta_RMSE_sd": paired_df["Delta_RMSE"].std(ddof=1),
            "Delta_RMSE_median": paired_df["Delta_RMSE"].median(),
            "RMSE_percent_improvement_mean": paired_df["RMSE_percent_improvement"].mean(),
            "RMSE_percent_improvement_median": paired_df["RMSE_percent_improvement"].median(),
            "Wilcoxon_p_RMSE": p_rmse,

            "MAE_without_SE_mean": paired_df["MAE_without_SE"].mean(),
            "MAE_with_SE_mean": paired_df["MAE_with_SE"].mean(),
            "Delta_MAE_mean": paired_df["Delta_MAE"].mean(),
            "Delta_MAE_sd": paired_df["Delta_MAE"].std(ddof=1),
            "Delta_MAE_median": paired_df["Delta_MAE"].median(),
            "MAE_percent_improvement_mean": paired_df["MAE_percent_improvement"].mean(),
            "MAE_percent_improvement_median": paired_df["MAE_percent_improvement"].median(),
            "Wilcoxon_p_MAE": p_mae
        })


# ------------------------------------------------------------
# 6. Save Objective 2 results
# ------------------------------------------------------------

added_value_df = pd.DataFrame(added_value_rows)

if len(paired_delta_rows) > 0:
    paired_delta_df = pd.concat(paired_delta_rows, ignore_index=True)
else:
    paired_delta_df = pd.DataFrame()

print("\nObjective 2: SE added-value summary")
print(added_value_df)

added_value_df.to_csv(
    O2_DIR / "O2_SE_added_value_summary.csv",
    index=False
)

paired_delta_df.to_csv(
    O2_DIR / "O2_SE_added_value_paired_outer_split_deltas.csv",
    index=False
)

with pd.ExcelWriter(O2_DIR / "O2_SE_added_value_results.xlsx") as writer:
    added_value_df.to_excel(writer, sheet_name="Summary", index=False)
    paired_delta_df.to_excel(writer, sheet_name="Paired_Deltas", index=False)

print("\nSaved Objective 2 files to:")
print(O2_DIR)

# CELL

# ============================================================
# Figures: Added value of SE predictors
# ============================================================

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Output folder
try:
    O2_DIR
except NameError:
    O2_DIR = Path(".")
else:
    O2_DIR = Path(O2_DIR)

# Detect correct RMSE improvement column
if "RMSE_percent_improvement_mean" in added_value_df.columns:
    rmse_improvement_col = "RMSE_percent_improvement_mean"
elif "RMSE_percent_improvement" in added_value_df.columns:
    rmse_improvement_col = "RMSE_percent_improvement"
else:
    raise ValueError(
        "No RMSE improvement column found. Expected "
        "'RMSE_percent_improvement_mean' or 'RMSE_percent_improvement'."
    )

# Sort for cleaner plotting
plot_df = added_value_df.copy()
plot_df["Plot_Label"] = plot_df["Comparison"] + "\n" + plot_df["Model"]

model_order = ["RF", "GradientBoosting", "XGBoost"]
plot_df["Model"] = pd.Categorical(plot_df["Model"], categories=model_order, ordered=True)

plot_df = plot_df.sort_values(["Comparison", "Model"]).reset_index(drop=True)

x = np.arange(len(plot_df))


# ============================================================
# Figure 1: RMSE improvement from adding SE predictors
# ============================================================

plt.figure(figsize=(12, 6))

plt.bar(
    x,
    plot_df[rmse_improvement_col]
)

plt.axhline(0, linestyle="--", linewidth=1)

plt.xticks(x, plot_df["Plot_Label"], rotation=45, ha="right")
plt.ylabel("RMSE improvement after adding SE (%)")
plt.xlabel("Comparison and model")
plt.title("Added value of SE predictors based on RMSE reduction")

plt.tight_layout()

plt.savefig(
    O2_DIR / "Figure_O2_SE_added_value_RMSE_improvement.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close("all")


# ============================================================
# Figure 2: R² improvement from adding SE predictors
# ============================================================

if "Delta_R2_mean" not in plot_df.columns:
    raise ValueError("Column 'Delta_R2_mean' was not found in added_value_df.")

plt.figure(figsize=(12, 6))

plt.bar(
    x,
    plot_df["Delta_R2_mean"]
)

plt.axhline(0, linestyle="--", linewidth=1)

plt.xticks(x, plot_df["Plot_Label"], rotation=45, ha="right")
plt.ylabel("Change in cross-validated R² after adding SE")
plt.xlabel("Comparison and model")
plt.title("Added value of SE predictors based on R² improvement")

plt.tight_layout()

plt.savefig(
    O2_DIR / "Figure_O2_SE_added_value_R2_improvement.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close("all")

# CELL

# ============================================================
