# Task 3:
# Label-efficiency evaluation
# Assess whether SE predictors improve performance
# under reduced training-data scenarios
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ------------------------------------------------------------
# 1. Output folder
# ------------------------------------------------------------

try:
    OUT_DIR
except NameError:
    OUT_DIR = Path(r"C:/Users/Dead_tree_fraction")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

O3_DIR = OUT_DIR / "O3_label_efficiency"
O3_DIR.mkdir(parents=True, exist_ok=True)


def display_feature_set_name(name):
    """Use manuscript labels in figures only; keep internal names unchanged."""
    return str(name).replace("RGBNIR", "MSI")


def display_model_name(name):
    """Use compact manuscript model labels in figures only."""
    return str(name).replace("GradientBoosting", "GB")


# ------------------------------------------------------------
# 2. Training-data proportions
# ------------------------------------------------------------

train_proportions = [0.2, 0.4, 0.6, 0.8, 1.0]
n_repeats = 30


# ------------------------------------------------------------
# 3. Create target bins for stratified splitting
# ------------------------------------------------------------

def make_regression_bins(y, n_bins=5):
    """
    Create quantile bins for stratified regression splitting.
    Automatically reduces number of bins if needed.
    """

    y_series = pd.Series(y).reset_index(drop=True)

    max_bins = min(n_bins, y_series.nunique())

    for q in range(max_bins, 1, -1):
        try:
            bins = pd.qcut(
                y_series,
                q=q,
                labels=False,
                duplicates="drop"
            )

            if pd.Series(bins).isna().any():
                continue

            return pd.Series(bins, index=y.index)

        except Exception:
            continue

    raise ValueError("Could not create valid target bins for stratified splitting.")


y_bins = make_regression_bins(y, n_bins=5)


# ------------------------------------------------------------
# 4. Main comparison sets for label efficiency
# ------------------------------------------------------------

label_efficiency_feature_sets = {
    "CHM+RGBNIR": feature_sets["CHM+RGBNIR"],
    "CHM+RGBNIR+SE": feature_sets["CHM+RGBNIR+SE"]
}


# ------------------------------------------------------------
# 5. Select models
# ------------------------------------------------------------

# Use the same base models from Objective 1
label_efficiency_models = {
    model_name: model_info["model"]
    for model_name, model_info in models_and_params.items()
}

# Option: only use the best overall model
# label_efficiency_models = {
#     best_model_name: models_and_params[best_model_name]["model"]
# }


# ------------------------------------------------------------
# 6. Fixed k-per-group for label-efficiency experiment
# ------------------------------------------------------------

# Recommended: use a fixed selector size to isolate the effect of sample size.
# You can change this to another entry from k_grid.
fixed_k_per_group = {"CHM": 5, "RGBNIR": 25, "SE": 25}

# Alternative:
# fixed_k_per_group = k_grid[-1]


# ------------------------------------------------------------
# 7. Prepare fixed independent test split
# ------------------------------------------------------------

test_splitter = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_full_idx, test_idx = next(
    test_splitter.split(X, y_bins)
)

X_train_full = X.iloc[train_full_idx].copy()
y_train_full = y.iloc[train_full_idx].copy()
y_bins_train_full = y_bins.iloc[train_full_idx].copy()

X_test_full = X.iloc[test_idx].copy()
y_test = y.iloc[test_idx].copy()

print("Full training pool size:", X_train_full.shape[0])
print("Independent test size:", X_test_full.shape[0])


# ------------------------------------------------------------
# 8. Run label-efficiency experiment
# ------------------------------------------------------------

label_eff_rows = []

for train_prop in train_proportions:

    print("\nTraining proportion:", train_prop)

    # At 100%, there is only one unique training subset.
    # Repeating it 30 times would produce duplicate results for deterministic models.
    if train_prop == 1.0:
        repeat_range = range(1)
    else:
        repeat_range = range(n_repeats)

    for repeat in repeat_range:

        if train_prop < 1.0:

            sub_splitter = StratifiedShuffleSplit(
                n_splits=1,
                train_size=train_prop,
                random_state=1000 + repeat
            )

            sub_train_idx_relative, _ = next(
                sub_splitter.split(X_train_full, y_bins_train_full)
            )

            sub_train_idx = X_train_full.index[sub_train_idx_relative]

        else:
            sub_train_idx = X_train_full.index

        for feature_name, groups in label_efficiency_feature_sets.items():

            selected_input_cols = []

            for cols in groups.values():
                selected_input_cols.extend(cols)

            selected_input_cols = list(dict.fromkeys(selected_input_cols))

            X_train_sub = X.loc[sub_train_idx, selected_input_cols].copy()
            y_train_sub = y.loc[sub_train_idx].copy()

            X_test_sub = X_test_full[selected_input_cols].copy()

            for model_name, model in label_efficiency_models.items():

                selector = GroupWiseSelectKBest(
                    groups=groups,
                    k_per_group=fixed_k_per_group
                )

                pipe = Pipeline([
                    ("selector", selector),
                    ("model", clone(model))
                ])

                pipe.fit(X_train_sub, y_train_sub)

                y_pred = pipe.predict(X_test_sub)

                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)

                label_eff_rows.append({
                    "Objective": "O3_Label_efficiency",
                    "Train_Proportion": train_prop,
                    "Train_N": len(y_train_sub),
                    "Feature_Set": feature_name,
                    "Model": model_name,
                    "Repeat": repeat + 1,
                    "R2": r2,
                    "RMSE": rmse,
                    "MAE": mae
                })

                print(
                    f"{feature_name} | {model_name} | "
                    f"Train prop={train_prop} | Repeat={repeat + 1} | "
                    f"RMSE={rmse:.4f}, R2={r2:.4f}"
                )


# ------------------------------------------------------------
# 9. Save raw label-efficiency results
# ------------------------------------------------------------

label_eff_df = pd.DataFrame(label_eff_rows)

label_eff_df.to_csv(
    O3_DIR / "O3_label_efficiency_results.csv",
    index=False
)

print("\nLabel-efficiency results:")
print(label_eff_df.head())

print("\nSaved:")
print(O3_DIR / "O3_label_efficiency_results.csv")


# ------------------------------------------------------------
# 10. Summary table
# ------------------------------------------------------------

label_eff_summary_df = (
    label_eff_df
    .groupby(["Train_Proportion", "Feature_Set", "Model"])
    .agg(
        R2_mean=("R2", "mean"),
        R2_sd=("R2", "std"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_sd=("RMSE", "std"),
        MAE_mean=("MAE", "mean"),
        MAE_sd=("MAE", "std"),
        Train_N_mean=("Train_N", "mean")
    )
    .reset_index()
)

label_eff_summary_df.to_csv(
    O3_DIR / "O3_label_efficiency_summary.csv",
    index=False
)

print("\nLabel-efficiency summary:")
print(label_eff_summary_df)


# ------------------------------------------------------------
# 11. Calculate SE added value at each training proportion
# ------------------------------------------------------------

label_eff_added_rows = []

for model_name in label_eff_df["Model"].unique():

    for train_prop in train_proportions:

        without_se_df = label_eff_df[
            (label_eff_df["Model"] == model_name) &
            (label_eff_df["Train_Proportion"] == train_prop) &
            (label_eff_df["Feature_Set"] == "CHM+RGBNIR")
        ].copy()

        with_se_df = label_eff_df[
            (label_eff_df["Model"] == model_name) &
            (label_eff_df["Train_Proportion"] == train_prop) &
            (label_eff_df["Feature_Set"] == "CHM+RGBNIR+SE")
        ].copy()

        paired_df = without_se_df.merge(
            with_se_df,
            on=["Model", "Train_Proportion", "Repeat"],
            suffixes=("_without_SE", "_with_SE")
        )

        if paired_df.empty:
            continue

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

        label_eff_added_rows.append({
            "Model": model_name,
            "Train_Proportion": train_prop,
            "N_Paired_Repeats": paired_df.shape[0],

            "Delta_R2_mean": paired_df["Delta_R2"].mean(),
            "Delta_R2_sd": paired_df["Delta_R2"].std(ddof=1),

            "Delta_RMSE_mean": paired_df["Delta_RMSE"].mean(),
            "Delta_RMSE_sd": paired_df["Delta_RMSE"].std(ddof=1),
            "RMSE_percent_improvement_mean": paired_df["RMSE_percent_improvement"].mean(),
            "RMSE_percent_improvement_sd": paired_df["RMSE_percent_improvement"].std(ddof=1),

            "Delta_MAE_mean": paired_df["Delta_MAE"].mean(),
            "Delta_MAE_sd": paired_df["Delta_MAE"].std(ddof=1),
            "MAE_percent_improvement_mean": paired_df["MAE_percent_improvement"].mean(),
            "MAE_percent_improvement_sd": paired_df["MAE_percent_improvement"].std(ddof=1)
        })

label_eff_added_df = pd.DataFrame(label_eff_added_rows)

label_eff_added_df.to_csv(
    O3_DIR / "O3_label_efficiency_SE_added_value_by_training_size.csv",
    index=False
)

print("\nSE added value by training proportion:")
print(label_eff_added_df)


# ------------------------------------------------------------
# 12. Plot RMSE learning curves
# ------------------------------------------------------------

for model_name in label_eff_summary_df["Model"].unique():

    plot_df = label_eff_summary_df[
        label_eff_summary_df["Model"] == model_name
    ].copy()

    plt.figure(figsize=(8, 6))

    for feature_name in ["CHM+RGBNIR", "CHM+RGBNIR+SE"]:

        data = plot_df[
            plot_df["Feature_Set"] == feature_name
        ].sort_values("Train_Proportion")

        plt.errorbar(
            data["Train_Proportion"] * 100,
            data["RMSE_mean"],
            yerr=data["RMSE_sd"],
            marker="o",
            capsize=4,
            label=display_feature_set_name(feature_name)
        )

    plt.xlabel("Training data used (%)")
    plt.ylabel("Test RMSE")
    plt.title(f"Training-size sensitivity: RMSE\n{display_model_name(model_name)}")
    plt.legend(title="Predictor set")
    plt.tight_layout()

    safe_model_name = model_name.replace(" ", "_").replace("+", "_plus_")

    plt.savefig(
        O3_DIR / f"Figure_O3_label_efficiency_RMSE_{safe_model_name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close("all")


# ------------------------------------------------------------
# 13. Plot SE added-value by training size
# ------------------------------------------------------------

for model_name in label_eff_added_df["Model"].unique():

    plot_df = label_eff_added_df[
        label_eff_added_df["Model"] == model_name
    ].sort_values("Train_Proportion")

    plt.figure(figsize=(8, 6))

    plt.errorbar(
        plot_df["Train_Proportion"] * 100,
        plot_df["RMSE_percent_improvement_mean"],
        yerr=plot_df["RMSE_percent_improvement_sd"],
        marker="o",
        capsize=4
    )

    plt.axhline(0, linestyle="--", linewidth=1)

    plt.xlabel("Training data used (%)")
    plt.ylabel("RMSE improvement from adding SE (%)")
    plt.title(f"SE added value under reduced training data\n{display_model_name(model_name)}")

    plt.tight_layout()

    safe_model_name = model_name.replace(" ", "_").replace("+", "_plus_")

    plt.savefig(
        O3_DIR / f"Figure_O3_SE_added_value_by_training_size_{safe_model_name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close("all")

# CELL

# ============================================================
# O3 summary table
# ============================================================

label_eff_summary = (
    label_eff_df
    .groupby(["Train_Proportion", "Feature_Set", "Model"])
    .agg(
        Train_N_mean=("Train_N", "mean"),
        R2_mean=("R2", "mean"),
        R2_sd=("R2", "std"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_sd=("RMSE", "std"),
        MAE_mean=("MAE", "mean"),
        MAE_sd=("MAE", "std")
    )
    .reset_index()
)

label_eff_summary.to_csv(
    O3_DIR / "O3_label_efficiency_summary.csv",
    index=False
)

print(label_eff_summary)


excel_path = O3_DIR / "O3_label_efficiency_summary.xlsx"

label_eff_summary.to_excel(excel_path, index=False)

print("Saved Excel file:")
print(excel_path)

# CELL

# ============================================================
# Figure: label efficiency based on RMSE
# ============================================================

for model_name in label_efficiency_models.keys():

    plt.figure(figsize=(7, 5))

    for feature_name in label_efficiency_feature_sets.keys():

        plot_df = label_eff_summary[
            (label_eff_summary["Model"] == model_name) &
            (label_eff_summary["Feature_Set"] == feature_name)
        ].sort_values("Train_Proportion")

        plt.errorbar(
            plot_df["Train_Proportion"] * 100,
            plot_df["RMSE_mean"],
            yerr=plot_df["RMSE_sd"],
            marker="o",
            capsize=4,
            label=display_feature_set_name(feature_name)
        )

    plt.xlabel("Training data used (%)")
    plt.ylabel("RMSE")
    plt.title(f"Training-size sensitivity based on RMSE: {display_model_name(model_name)}")
    plt.legend(title="Predictor set")
    plt.tight_layout()

    safe_model_name = model_name.replace(" ", "_").replace("+", "_plus_")

    plt.savefig(
        O3_DIR / f"O3_label_efficiency_RMSE_{safe_model_name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close("all")

# CELL

# ============================================================
# Figure: label efficiency based on R2
# ============================================================

from pathlib import Path
import matplotlib.pyplot as plt

# Output folder
try:
    O3_DIR
except NameError:
    O3_DIR = Path(".")
else:
    O3_DIR = Path(O3_DIR)

for model_name in label_efficiency_models.keys():

    plt.figure(figsize=(7, 5))

    has_data = False

    for feature_name in label_efficiency_feature_sets.keys():

        plot_df = label_eff_summary_df[
            (label_eff_summary_df["Model"] == model_name) &
            (label_eff_summary_df["Feature_Set"] == feature_name)
        ].sort_values("Train_Proportion")

        if plot_df.empty:
            continue

        has_data = True

        plt.errorbar(
            plot_df["Train_Proportion"] * 100,
            plot_df["R2_mean"],
            yerr=plot_df["R2_sd"],
            marker="o",
            capsize=4,
            label=display_feature_set_name(feature_name)
        )

    if not has_data:
        plt.close()
        continue

    plt.xlabel("Training data used (%)")
    plt.ylabel("Test R²")
    plt.title(f"Training-size sensitivity based on R²\n{display_model_name(model_name)}")
    plt.legend(title="Predictor set")
    plt.tight_layout()

    safe_model_name = model_name.replace(" ", "_").replace("+", "_plus_")

    plt.savefig(
        O3_DIR / f"Figure_O3_label_efficiency_R2_{safe_model_name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close("all")

# CELL

# ============================================================
# SE gain by training-data proportion
# ============================================================

gain_rows = []

for model_name in label_efficiency_models.keys():

    for train_prop in train_proportions:

        base_df = label_eff_df[
            (label_eff_df["Model"] == model_name) &
            (label_eff_df["Train_Proportion"] == train_prop) &
            (label_eff_df["Feature_Set"] == "CHM+RGBNIR")
        ].copy()

        se_df = label_eff_df[
            (label_eff_df["Model"] == model_name) &
            (label_eff_df["Train_Proportion"] == train_prop) &
            (label_eff_df["Feature_Set"] == "CHM+RGBNIR+SE")
        ].copy()

        paired_df = base_df.merge(
            se_df,
            on=["Model", "Train_Proportion", "Repeat"],
            suffixes=("_without_SE", "_with_SE")
        )

        if paired_df.empty:
            print(f"Skipping missing gain comparison: {model_name}, train proportion {train_prop}")
            continue

        delta_rmse = paired_df["RMSE_without_SE"] - paired_df["RMSE_with_SE"]
        delta_r2 = paired_df["R2_with_SE"] - paired_df["R2_without_SE"]
        delta_mae = paired_df["MAE_without_SE"] - paired_df["MAE_with_SE"]

        gain_rows.append({
            "Model": model_name,
            "Train_Proportion": train_prop,
            "N_Paired_Repeats": paired_df.shape[0],
            "Delta_RMSE_mean": delta_rmse.mean(),
            "Delta_RMSE_sd": delta_rmse.std(ddof=1) if paired_df.shape[0] > 1 else 0.0,
            "RMSE_improvement_percent": (
                delta_rmse.mean() / paired_df["RMSE_without_SE"].mean()
            ) * 100,
            "Delta_R2_mean": delta_r2.mean(),
            "Delta_R2_sd": delta_r2.std(ddof=1) if paired_df.shape[0] > 1 else 0.0,
            "Delta_MAE_mean": delta_mae.mean(),
            "Delta_MAE_sd": delta_mae.std(ddof=1) if paired_df.shape[0] > 1 else 0.0
        })

label_eff_gain_df = pd.DataFrame(gain_rows)

label_eff_gain_df.to_csv(
    O3_DIR / "O3_SE_gain_by_training_fraction.csv",
    index=False
)

print(label_eff_gain_df)

# CELL


# ============================================================
# Figure: SE improvement under reduced labels
# ============================================================

plt.figure(figsize=(8, 5))

for model_name in label_efficiency_models.keys():

    plot_df = label_eff_gain_df[
        label_eff_gain_df["Model"] == model_name
    ].sort_values("Train_Proportion")

    plt.plot(
        plot_df["Train_Proportion"] * 100,
        plot_df["RMSE_improvement_percent"],
        marker="o",
        label=display_model_name(model_name)
    )

plt.axhline(0, linestyle="--", linewidth=1)

plt.xlabel("Training data used (%)", fontsize=10)
plt.ylabel("RMSE improvement from SE (%)", fontsize=10)
plt.title("", fontsize=10)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.legend(title="Model", fontsize=10, title_fontsize=10)

plt.tight_layout()

plt.savefig(
    O3_DIR / "O3_SE_improvement_under_reduced_labels.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close("all")

# CELL
