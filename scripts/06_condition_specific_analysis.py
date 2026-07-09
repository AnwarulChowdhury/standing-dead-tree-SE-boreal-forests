# ============================================================
# Objective 7:
# Plot-specific analysis
# Under which plot-level forest structural or spectral variables
# do SE predictors provide the greatest improvement?
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib.lines import Line2D


# ------------------------------------------------------------
# 0. Output folder
# ------------------------------------------------------------

try:
    OUT_DIR
except NameError:
    OUT_DIR = Path(r"C:\Users\Dead_tree_fraction\O1_nested_tuning_outputs")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

O7_DIR = OUT_DIR / "O7_plot_specific_SE_benefit"
O7_DIR.mkdir(parents=True, exist_ok=True)


def display_feature_set_name(name):
    """Use manuscript labels in figures only; keep internal names unchanged."""
    return str(name).replace("RGBNIR", "MSI")


def display_model_name(name):
    """Use compact manuscript model labels in figures only."""
    return str(name).replace("GradientBoosting", "GB")


def safe_name(name):
    """Make a string safe for filenames."""
    return (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("+", "_plus_")
    )


def pretty_name(x):
    """Make variable names cleaner for manuscript figures."""
    return str(x).replace("RGBNIR", "MSI").replace("_", " ")


def make_bin_labels(n_bins):
    """Return readable bin labels for qcut output."""
    if n_bins == 4:
        return ["Low", "Medium-low", "Medium-high", "High"]
    elif n_bins == 3:
        return ["Low", "Medium", "High"]
    elif n_bins == 2:
        return ["Low", "High"]
    else:
        return [f"Bin {i + 1}" for i in range(n_bins)]


# ------------------------------------------------------------
# 1. Choose baseline and SE-enhanced feature sets
# ------------------------------------------------------------

baseline_feature_set = "CHM+RGBNIR"
se_feature_set = "CHM+RGBNIR+SE"

if baseline_feature_set not in feature_sets:
    raise ValueError(f"{baseline_feature_set} not found in feature_sets.")

if se_feature_set not in feature_sets:
    raise ValueError(f"{se_feature_set} not found in feature_sets.")


# ------------------------------------------------------------
# 2. Select models
# ------------------------------------------------------------

# Use all models from Objective 1
plot_models = {
    model_name: model_info["model"]
    for model_name, model_info in models_and_params.items()
}

# Option: only use best overall model
# plot_models = {
#     best_model_name: models_and_params[best_model_name]["model"]
# }


# ------------------------------------------------------------
# 3. Fixed k-per-group for this analysis
# ------------------------------------------------------------

# Use tuned k_per_group if available; otherwise use a reasonable default.
if "best_final_params" in globals() and "selector__k_per_group" in best_final_params:
    plot_k_per_group = best_final_params["selector__k_per_group"]
else:
    plot_k_per_group = k_grid[-1]

print("Using k_per_group:", plot_k_per_group)


# ------------------------------------------------------------
# 4. Create or reuse repeated CV splitter
# ------------------------------------------------------------

# Use the same outer CV logic from Objective 1 if available.
try:
    plot_cv = outer_cv
except NameError:
    plot_cv = RepeatedStratifiedKFoldReg(
        n_splits=5,
        n_repeats=10,
        n_bins=5,
        random_state=42
    )


# ------------------------------------------------------------
# 5. Create repeated-CV predictions for baseline vs SE model
# ------------------------------------------------------------

plot_prediction_rows = []

for model_name, model in plot_models.items():

    print("\nModel:", model_name)

    for feature_name in [baseline_feature_set, se_feature_set]:

        print("Feature set:", feature_name)

        groups = feature_sets[feature_name]

        selected_input_cols = []

        for cols in groups.values():
            selected_input_cols.extend(cols)

        selected_input_cols = list(dict.fromkeys(selected_input_cols))

        X_sub = X[selected_input_cols].copy()

        selector = GroupWiseSelectKBest(
            groups=groups,
            k_per_group=plot_k_per_group
        )

        pipe = Pipeline([
            ("selector", selector),
            ("model", clone(model))
        ])

        for fold_id, (train_idx, test_idx) in enumerate(
            plot_cv.split(X_sub, y),
            start=1
        ):

            pipe_fold = clone(pipe)

            X_train = X_sub.iloc[train_idx].copy()
            X_test = X_sub.iloc[test_idx].copy()

            y_train = y.iloc[train_idx].copy()
            y_test = y.iloc[test_idx].copy()

            pipe_fold.fit(X_train, y_train)

            y_pred = pipe_fold.predict(X_test)

            for idx, obs, pred in zip(X_test.index, y_test, y_pred):

                plot_prediction_rows.append({
                    "Index": idx,
                    "CV_Run": fold_id,
                    "Model": model_name,
                    "Feature_Set": feature_name,
                    "Observed": obs,
                    "Predicted": pred,
                    "Abs_Error": abs(obs - pred),
                    "Squared_Error": (obs - pred) ** 2
                })


plot_pred_df = pd.DataFrame(plot_prediction_rows)

plot_pred_df.to_csv(
    O7_DIR / "O7_plot_specific_predictions.csv",
    index=False
)


# ------------------------------------------------------------
# 6. Convert long predictions to baseline vs SE comparison
# ------------------------------------------------------------

plot_compare_df = plot_pred_df.pivot_table(
    index=["Index", "CV_Run", "Model", "Observed"],
    columns="Feature_Set",
    values=["Predicted", "Abs_Error", "Squared_Error"],
    aggfunc="first"
)

plot_compare_df.columns = [
    "_".join(col).strip()
    for col in plot_compare_df.columns.values
]

plot_compare_df = plot_compare_df.reset_index()

required_cols = [
    f"Abs_Error_{baseline_feature_set}",
    f"Abs_Error_{se_feature_set}",
    f"Squared_Error_{baseline_feature_set}",
    f"Squared_Error_{se_feature_set}"
]

missing_cols = [
    col for col in required_cols
    if col not in plot_compare_df.columns
]

if missing_cols:
    raise ValueError(f"Missing columns after pivot: {missing_cols}")

plot_compare_df["SE_Abs_Error_Improvement"] = (
    plot_compare_df[f"Abs_Error_{baseline_feature_set}"]
    - plot_compare_df[f"Abs_Error_{se_feature_set}"]
)

plot_compare_df["SE_Squared_Error_Improvement"] = (
    plot_compare_df[f"Squared_Error_{baseline_feature_set}"]
    - plot_compare_df[f"Squared_Error_{se_feature_set}"]
)

plot_compare_df["SE_Better"] = (
    plot_compare_df["SE_Abs_Error_Improvement"] > 0
)

plot_compare_df.to_csv(
    O7_DIR / "O7_SE_error_improvement_per_plot_CV.csv",
    index=False
)


# ------------------------------------------------------------
# 7. Average repeated-CV values to one record per plot and model
# ------------------------------------------------------------

# This avoids treating repeated predictions for the same plot
# as independent observations in plot-level analyses.
plot_summary_df = (
    plot_compare_df
    .groupby(["Index", "Model"], as_index=False)
    .agg(
        Observed=("Observed", "first"),
        Mean_SE_Abs_Error_Improvement=("SE_Abs_Error_Improvement", "mean"),
        Mean_SE_Squared_Error_Improvement=("SE_Squared_Error_Improvement", "mean"),
        Percent_CV_Runs_SE_Better=("SE_Better", "mean"),
        N_CV_Runs=("SE_Better", "count")
    )
)

plot_summary_df["SE_Abs_Error_Improvement"] = (
    plot_summary_df["Mean_SE_Abs_Error_Improvement"]
)

plot_summary_df["SE_Squared_Error_Improvement"] = (
    plot_summary_df["Mean_SE_Squared_Error_Improvement"]
)

plot_summary_df["SE_Better"] = (
    plot_summary_df["Percent_CV_Runs_SE_Better"] > 0.5
)

plot_summary_df.to_csv(
    O7_DIR / "O7_SE_error_improvement_per_plot_mean.csv",
    index=False
)

# Use the plot-level aggregated table for all analyses below.
plot_compare_df = plot_summary_df.copy()


# ------------------------------------------------------------
# 8. Add plot-level variables
# ------------------------------------------------------------

# Use all CHM and MSI/RGB-NIR predictors as potential plot-level variables.
# These describe each plot structurally or spectrally.
plot_vars = list(dict.fromkeys(CHM_cols + RGBNIR_cols))

# Add observed target as a plot-level variable.
plot_compare_df["Dead_F_observed"] = plot_compare_df["Observed"]
plot_vars.append("Dead_F_observed")

plot_vars = list(dict.fromkeys(plot_vars))

# Add plot-level variable values from the original X matrix.
for col in plot_vars:
    if col in X.columns:
        plot_compare_df[col] = plot_compare_df["Index"].map(X[col])
    elif col == "Dead_F_observed":
        pass


# ------------------------------------------------------------
# 9. Correlation between plot-level variables and SE improvement
# ------------------------------------------------------------

plot_corr_rows = []

for model_name in plot_compare_df["Model"].unique():

    model_df = plot_compare_df[
        plot_compare_df["Model"] == model_name
    ].copy()

    for var in plot_vars:

        if var not in model_df.columns:
            continue

        temp = model_df[[var, "SE_Abs_Error_Improvement"]].dropna()

        if temp[var].nunique() < 3:
            continue

        corr = temp[var].corr(
            temp["SE_Abs_Error_Improvement"],
            method="spearman"
        )

        if pd.isna(corr):
            continue

        plot_corr_rows.append({
            "Model": model_name,
            "Plot_Variable": var,
            "Spearman_Correlation_with_SE_Improvement": corr,
            "Abs_Correlation": abs(corr)
        })


plot_corr_df = pd.DataFrame(plot_corr_rows)

if plot_corr_df.empty:
    raise ValueError("No valid plot-level correlations were calculated.")

plot_corr_df = plot_corr_df.sort_values(
    "Abs_Correlation",
    ascending=False
)

plot_corr_df.to_csv(
    O7_DIR / "O7_plot_variable_correlations_with_SE_improvement.csv",
    index=False
)

print("\nTop plot-level variables associated with SE improvement:")
print(plot_corr_df.head(20))


# ------------------------------------------------------------
# 10. Bin-based plot-variable analysis
# ------------------------------------------------------------

top_plot_vars = (
    plot_corr_df
    .head(6)["Plot_Variable"]
    .tolist()
)

bin_summary_rows = []

for model_name in plot_compare_df["Model"].unique():

    model_df = plot_compare_df[
        plot_compare_df["Model"] == model_name
    ].copy()

    for var in top_plot_vars:

        if var not in model_df.columns:
            continue

        temp = model_df[
            [
                var,
                "SE_Abs_Error_Improvement",
                "SE_Better",
                "Percent_CV_Runs_SE_Better"
            ]
        ].dropna().copy()

        if temp[var].nunique() < 5:
            continue

        try:
            preliminary_bins = pd.qcut(
                temp[var],
                q=4,
                duplicates="drop"
            )
        except ValueError:
            continue

        n_bins = len(preliminary_bins.cat.categories)

        if n_bins < 2:
            continue

        bin_labels = make_bin_labels(n_bins)

        try:
            temp["Plot_Variable_Bin"] = pd.qcut(
                temp[var],
                q=4,
                labels=bin_labels,
                duplicates="drop"
            )
        except ValueError:
            continue

        summary = (
            temp
            .groupby("Plot_Variable_Bin", observed=False)
            .agg(
                Mean_SE_Abs_Error_Improvement=("SE_Abs_Error_Improvement", "mean"),
                SD_SE_Abs_Error_Improvement=("SE_Abs_Error_Improvement", "std"),
                Percent_Plots_SE_Better_Most_CV_Runs=("SE_Better", "mean"),
                Mean_Percent_CV_Runs_SE_Better=("Percent_CV_Runs_SE_Better", "mean"),
                N_Plots=("SE_Better", "count")
            )
            .reset_index()
        )

        summary["Percent_Plots_SE_Better_Most_CV_Runs"] = (
            summary["Percent_Plots_SE_Better_Most_CV_Runs"] * 100
        )

        summary["Mean_Percent_CV_Runs_SE_Better"] = (
            summary["Mean_Percent_CV_Runs_SE_Better"] * 100
        )

        summary["Model"] = model_name
        summary["Plot_Variable"] = var

        bin_summary_rows.append(summary)


if len(bin_summary_rows) > 0:
    bin_summary_df = pd.concat(bin_summary_rows, ignore_index=True)
else:
    bin_summary_df = pd.DataFrame()

bin_summary_df.to_csv(
    O7_DIR / "O7_plot_variable_bin_summary.csv",
    index=False
)

print("\nPlot-variable bin summary:")
print(bin_summary_df.head())


# ------------------------------------------------------------
# 11. Figure: SE improvement across plot-variable bins
# ------------------------------------------------------------

if not bin_summary_df.empty:

    for model_name in bin_summary_df["Model"].unique():

        model_bin_df = bin_summary_df[
            bin_summary_df["Model"] == model_name
        ]

        for var in model_bin_df["Plot_Variable"].unique():

            plot_df = model_bin_df[
                model_bin_df["Plot_Variable"] == var
            ].copy()

            plt.figure(figsize=(6.5, 5))

            plt.bar(
                plot_df["Plot_Variable_Bin"].astype(str),
                plot_df["Mean_SE_Abs_Error_Improvement"],
                yerr=plot_df["SD_SE_Abs_Error_Improvement"],
                capsize=4
            )

            plt.axhline(0, linestyle="--", linewidth=1)

            plt.xlabel(f"{pretty_name(var)} quartile")
            plt.ylabel("Mean absolute-error improvement from SE")
            plt.title(
                f"SE benefit across plot-level {pretty_name(var)}\n"
                f"{display_model_name(model_name)}"
            )

            plt.tight_layout()

            safe_model = safe_name(model_name)
            safe_var = safe_name(var)

            plt.savefig(
                O7_DIR / f"Figure_O7_SE_improvement_plot_bins_{safe_model}_{safe_var}.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close("all")


# ------------------------------------------------------------
# 12. Figure: top plot-variable correlations
# ------------------------------------------------------------

top_corr_plot = plot_corr_df.head(20).copy()

top_corr_plot["Model_plot"] = top_corr_plot["Model"].apply(display_model_name)
top_corr_plot["Plot_Label"] = top_corr_plot["Plot_Variable"].apply(pretty_name)

top_corr_plot["Label"] = (
    top_corr_plot["Model_plot"] + " | " + top_corr_plot["Plot_Label"]
)

top_corr_plot = top_corr_plot.sort_values(
    "Abs_Correlation",
    ascending=True
)

plt.figure(figsize=(9, 8))

plt.barh(
    top_corr_plot["Label"],
    top_corr_plot["Spearman_Correlation_with_SE_Improvement"]
)

plt.axvline(0, linestyle="--", linewidth=1)

plt.xlabel("Spearman correlation with SE error improvement")
plt.ylabel("Model and plot-level variable")
plt.title("Plot-level variables associated with stronger SE benefit")

plt.tight_layout()

plt.savefig(
    O7_DIR / "Figure_O7_top_plot_variable_correlations_SE_improvement.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close("all")


# ------------------------------------------------------------
# 13. Best plot-level SE benefit table
# ------------------------------------------------------------

if not bin_summary_df.empty:

    best_plot_variable_table = (
        bin_summary_df
        .sort_values("Mean_SE_Abs_Error_Improvement", ascending=False)
    )

    best_plot_variable_table.to_csv(
        O7_DIR / "O7_best_plot_variables_for_SE_benefit.csv",
        index=False
    )

    print("\nBest plot-level variables for SE benefit:")
    print(best_plot_variable_table.head(20))

else:
    print("\nNo bin-based plot-variable table was created.")


print("\nObjective 7 outputs saved to:")
print(O7_DIR)


# ============================================================
# Objective 7 alternative manuscript figure:
# Split dot plot of negative and positive plot-level SE-benefit associations
# ============================================================

# ------------------------------------------------------------
# 14. File paths
# ------------------------------------------------------------

base_dir = O7_DIR

corr_file = base_dir / "O7_plot_variable_correlations_with_SE_improvement.csv"

outdir = base_dir / "manuscript_figures"
outdir.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 15. Load plot-level correlation data
# ------------------------------------------------------------

corr_df = pd.read_csv(corr_file)
corr_df.columns = corr_df.columns.str.strip()


# ------------------------------------------------------------
# 16. Clean model names and labels
# ------------------------------------------------------------

corr_df["Model"] = corr_df["Model"].replace({
    "GradientBoosting": "GB",
    "Gradient_Boosting": "GB"
})

model_order = ["RF", "GB", "XGBoost"]

model_colors = {
    "RF": "#1f77b4",
    "GB": "#ff7f0e",
    "XGBoost": "#2ca02c"
}

model_markers = {
    "RF": "o",
    "GB": "s",
    "XGBoost": "^"
}

corr_col = "Spearman_Correlation_with_SE_Improvement"

corr_df["Plot_Label"] = corr_df["Plot_Variable"].apply(pretty_name)
corr_df["Full_Label"] = corr_df["Model"] + " | " + corr_df["Plot_Label"]


# ------------------------------------------------------------
# 17. Select strongest positive and negative associations
# ------------------------------------------------------------

n_show = 10

neg_df = (
    corr_df[corr_df[corr_col] < 0]
    .sort_values(corr_col, ascending=True)
    .head(n_show)
    .copy()
)

pos_df = (
    corr_df[corr_df[corr_col] > 0]
    .sort_values(corr_col, ascending=False)
    .head(n_show)
    .copy()
)

# Sort for display.
neg_df = neg_df.sort_values(corr_col, ascending=False).reset_index(drop=True)
pos_df = pos_df.sort_values(corr_col, ascending=True).reset_index(drop=True)


# ------------------------------------------------------------
# 18. Plot split dot plot
# ------------------------------------------------------------

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(12.5, 5.8),
    sharex=False
)


def plot_correlation_side(ax, side_df, panel_label, side_title, value_side):
    """Plot one side of the split dot plot."""

    y = np.arange(len(side_df))

    if side_df.empty:
        ax.axvline(0, linestyle="--", linewidth=1, color="black")
        ax.set_yticks([])
        ax.set_xlabel("Spearman correlation", fontsize=9)
        ax.text(
            0.5,
            0.5,
            "No associations",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9
        )
        ax.set_xlim(-0.1, 0.1)

    else:
        for i, row in side_df.iterrows():

            model_key = row["Model"]

            ax.scatter(
                row[corr_col],
                i,
                color=model_colors.get(model_key, "black"),
                marker=model_markers.get(model_key, "o"),
                s=55
            )

            if value_side == "left":
                text_x = row[corr_col] - 0.006
                ha = "right"
            else:
                text_x = row[corr_col] + 0.006
                ha = "left"

            ax.text(
                text_x,
                i,
                f"{row[corr_col]:.2f}",
                ha=ha,
                va="center",
                fontsize=8
            )

        ax.axvline(0, linestyle="--", linewidth=1, color="black")
        ax.set_yticks(y)
        ax.set_yticklabels(side_df["Full_Label"], fontsize=8.5)
        ax.set_xlabel("Spearman correlation", fontsize=9)
        ax.grid(axis="x", alpha=0.25)

        if value_side == "left":
            ax.set_xlim(min(side_df[corr_col].min() - 0.04, -0.22), 0.02)
        else:
            ax.set_xlim(-0.02, max(side_df[corr_col].max() + 0.04, 0.14))

    ax.text(
        0.01,
        1.03,
        panel_label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom"
    )

    ax.text(
        1.08,
        0.5,
        side_title,
        transform=ax.transAxes,
        rotation=-90,
        fontsize=9,
        ha="center",
        va="center"
    )


plot_correlation_side(
    axes[0],
    neg_df,
    "(a)",
    "Plot-level variables associated with weaker SE benefit",
    "left"
)

plot_correlation_side(
    axes[1],
    pos_df,
    "(b)",
    "Plot-level variables associated with stronger SE benefit",
    "right"
)


# ------------------------------------------------------------
# 19. Shared legend
# ------------------------------------------------------------

legend_handles = [
    Line2D(
        [0], [0],
        marker=model_markers[m],
        color="w",
        markerfacecolor=model_colors[m],
        markeredgecolor=model_colors[m],
        markersize=7,
        label=m
    )
    for m in model_order
    if m in model_markers
]

fig.legend(
    handles=legend_handles,
    loc="upper center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 1.02),
    fontsize=9
)


# ------------------------------------------------------------
# 20. Footnote and save
# ------------------------------------------------------------

fig.text(
    0.5,
    0.01,
    (
        "Positive correlations indicate greater SE benefit at higher plot-level variable values; "
        "negative correlations indicate weaker SE benefit."
    ),
    ha="center",
    fontsize=8
)

plt.tight_layout(rect=(0, 0.05, 0.95, 0.96))

fig.savefig(
    outdir / "Figure_O7_split_dotplot_plot_variable_associations.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    outdir / "Figure_O7_split_dotplot_plot_variable_associations.svg",
    bbox_inches="tight"
)

fig.savefig(
    outdir / "Figure_O7_split_dotplot_plot_variable_associations.pdf",
    bbox_inches="tight"
)

# Additional fixed-name copies for final manuscript figure selection.
fig.savefig(
    outdir / "Figure_O7_split_dotplot_plot_variable_associations_FIXED.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    outdir / "Figure_O7_split_dotplot_plot_variable_associations_FIXED.svg",
    bbox_inches="tight"
)

fig.savefig(
    outdir / "Figure_O7_split_dotplot_plot_variable_associations_FIXED.pdf",
    bbox_inches="tight"
)

plt.close("all")

print("\nObjective 7 manuscript figures saved to:")
print(outdir)
