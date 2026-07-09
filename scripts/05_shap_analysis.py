
O4_DIR = OUT_DIR / "O4_SHAP_and_complementarity"
O4_DIR.mkdir(parents=True, exist_ok=True)
print("Objective 4/5 outputs will be saved to:", O4_DIR)

# ============================================================
# Task 4:
# Model interpretability using SHAP
# Evaluate importance of individual predictors and groups
# ============================================================

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV


# ------------------------------------------------------------
# 1. Use the final tuned best pipeline from Objective 1
# ------------------------------------------------------------

best_groups = feature_sets[best_feature_set]

best_input_cols = []
for cols in best_groups.values():
    best_input_cols.extend(cols)

best_input_cols = list(dict.fromkeys(best_input_cols))

X_best = X[best_input_cols].copy()

# Prefer the already tuned/fitted final pipeline from Objective 1.
# This avoids recalculating SHAP on an untuned/default model.
if "best_pipe" in globals():
    shap_pipe = best_pipe
    print("Using existing fitted best_pipe from Objective 1 for SHAP.")
else:
    candidate_pipeline = OUT_DIR / "O1_best_tuned_final_pipeline.joblib"
    if candidate_pipeline.exists():
        import joblib
        shap_pipe = joblib.load(candidate_pipeline)
        print("Loaded fitted best pipeline for SHAP:", candidate_pipeline)
    else:
        print(
            "Warning: fitted best pipeline was not found. "
            "Refitting the best model using Objective 1 settings."
        )

        if "best_final_params" in globals() and "selector__k_per_group" in best_final_params:
            best_k_per_group = best_final_params["selector__k_per_group"]
        else:
            best_k_per_group = k_grid[0]

        best_selector = GroupWiseSelectKBest(
            groups=best_groups,
            k_per_group=best_k_per_group
        )

        best_model = clone(models_and_params[best_model_name]["model"])

        shap_pipe = Pipeline([
            ("selector", best_selector),
            ("model", best_model)
        ])

        shap_pipe.fit(X_best, y)

# ------------------------------------------------------------
# 2. Get selected predictors
# ------------------------------------------------------------

selected_features = shap_pipe.named_steps["selector"].get_feature_names_out().tolist()

X_selected = shap_pipe.named_steps["selector"].transform(X_best)
X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X_best.index)

final_model = shap_pipe.named_steps["model"]

print("Best model:", best_model_name)
print("Best feature set:", best_feature_set)
print("Number of selected predictors:", len(selected_features))


# ------------------------------------------------------------
# 3. Compute SHAP values
# ------------------------------------------------------------

explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_selected)

if isinstance(shap_values, list):
    shap_matrix = shap_values[0]
else:
    shap_matrix = np.array(shap_values)

if len(shap_matrix.shape) == 3:
    shap_matrix = shap_matrix[:, :, 0]

if shap_matrix.shape[1] != len(selected_features):
    raise ValueError(
        f"SHAP matrix has {shap_matrix.shape[1]} columns but "
        f"{len(selected_features)} selected features were found."
    )


# ------------------------------------------------------------
# 4. SHAP importance table
# ------------------------------------------------------------

shap_importance_df = pd.DataFrame({
    "Predictor": selected_features,
    "Mean_abs_SHAP": np.abs(shap_matrix).mean(axis=0),
    "SD_abs_SHAP": np.abs(shap_matrix).std(axis=0)
})


def get_group_name(feature):
    if feature.startswith("CHM"):
        return "CHM"
    elif feature.startswith("SE2025") or feature.startswith("dSE"):
        return "SE"
    else:
        return "RGB-NIR"


shap_importance_df["Group"] = shap_importance_df["Predictor"].apply(get_group_name)

shap_importance_df = shap_importance_df.sort_values(
    "Mean_abs_SHAP",
    ascending=False
)

shap_importance_df.to_csv(
    O4_DIR / "O4_SHAP_predictor_importance.csv",
    index=False
)

print("\nTop SHAP predictors:")
print(shap_importance_df.head(20))


# ------------------------------------------------------------
# 5. Top 20 SHAP predictors figure
# ------------------------------------------------------------

top_n = min(20, len(shap_importance_df))

plot_df = (
    shap_importance_df
    .head(top_n)
    .sort_values("Mean_abs_SHAP", ascending=True)
)

# Manuscript colours: CHM = blue, MSI/RGB-NIR = orange, SE = green
group_colors = {
    "CHM": "#6aaed6",
    "RGB-NIR": "#e7ad72",
    "SE": "#7bd88f"
}

plt.figure(figsize=(9, 8))

plt.barh(
    plot_df["Predictor"],
    plot_df["Mean_abs_SHAP"],
    xerr=plot_df["SD_abs_SHAP"],
    color=plot_df["Group"].map(group_colors),
    capsize=3
)

plt.xlabel("Mean absolute SHAP value")
plt.ylabel("Predictor")
plt.title(f"Top {top_n} SHAP predictors\n{best_model_name} | {best_feature_set}")

from matplotlib.patches import Patch

present_groups = plot_df["Group"].unique()

legend_elements = [
    Patch(facecolor=group_colors[g], label=("MSI" if g == "RGB-NIR" else g))
    for g in present_groups
    if g in group_colors
]

plt.legend(handles=legend_elements, title="Predictor group")

plt.tight_layout()
plt.savefig(
    O4_DIR / "Figure_O4_top20_SHAP_predictors.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close("all")


# ------------------------------------------------------------
# 6. Group-level SHAP importance
# ------------------------------------------------------------

group_shap_df = (
    shap_importance_df
    .groupby("Group", as_index=False)
    .agg(
        Total_Mean_abs_SHAP=("Mean_abs_SHAP", "sum"),
        Mean_Mean_abs_SHAP=("Mean_abs_SHAP", "mean"),
        N_Predictors=("Predictor", "count")
    )
    .sort_values("Total_Mean_abs_SHAP", ascending=False)
)

group_shap_df.to_csv(
    O4_DIR / "O4_SHAP_group_importance.csv",
    index=False
)

print("\nGroup-level SHAP importance:")
print(group_shap_df)


plt.figure(figsize=(6, 5))

plt.bar(
    group_shap_df["Group"],
    group_shap_df["Total_Mean_abs_SHAP"],
    color=group_shap_df["Group"].map(group_colors)
)

plt.xlabel("Predictor group")
plt.ylabel("Total mean absolute SHAP value")
plt.title(f"Group-level SHAP importance\n{best_model_name} | {best_feature_set}")

plt.tight_layout()
plt.savefig(
    O4_DIR / "Figure_O4_group_level_SHAP_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close("all")


# ------------------------------------------------------------
# 7. SHAP summary bar plot
# ------------------------------------------------------------

plt.figure()

shap.summary_plot(
    shap_matrix,
    X_selected,
    plot_type="bar",
    max_display=20,
    show=False
)

plt.title(f"SHAP summary bar plot\n{best_model_name} | {best_feature_set}")

plt.tight_layout()

plt.savefig(
    O4_DIR / "Figure_O4_SHAP_summary_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close("all")


# CELL

# ============================================================
# Objective 4B:
# Algorithm-specific SHAP importance for RF, GradientBoosting,
# and XGBoost using CHM + RGBNIR + SE models
# ============================================================

import joblib
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

ALGO_SHAP_DIR = O4_DIR / "O4_algorithm_specific_SHAP"
ALGO_SHAP_DIR.mkdir(parents=True, exist_ok=True)

print("Algorithm-specific SHAP outputs will be saved to:", ALGO_SHAP_DIR)

pipeline_dir = OUT_DIR / "ImpPred_BestByAlgorithm"


def clean_model_name_for_plot(model_name):
    if model_name == "GradientBoosting":
        return "GB"
    return model_name


def get_group_name_for_algorithm_shap(feature):
    if str(feature).startswith("CHM"):
        return "CHM"
    elif str(feature).startswith("SE2025") or str(feature).startswith("dSE"):
        return "SE"
    else:
        return "RGB-NIR"


def clean_group_for_plot(group):
    if str(group) in ["RGB-NIR", "RGBNIR", "RGB_NIR", "MSI"]:
        return "MSI"
    return str(group)


def safe_feature_set_name_for_file(feature_set):
    return (
        str(feature_set)
        .replace(" ", "_")
        .replace("+", "_plus_")
        .replace("/", "_")
    )


def load_or_refit_full_algorithm_pipeline(algorithm_name):
    """
    Prefer the already saved CHM+RGBNIR+SE pipeline.
    If it is not available, refit CHM+RGBNIR+SE on the full dataset
    using the same selector and randomized-search framework.
    """
    full_feature_set_name = "CHM+RGBNIR+SE"
    safe_algorithm_name = algorithm_name.replace(" ", "_").replace("+", "_plus_")
    safe_full_feature_set_name = safe_feature_set_name_for_file(full_feature_set_name)

    expected_pipeline = (
        pipeline_dir /
        f"Final_pipeline_best_{safe_algorithm_name}_{safe_full_feature_set_name}.joblib"
    )

    if expected_pipeline.exists():
        print(f"Loading existing full predictor-set pipeline for {algorithm_name}:")
        print(expected_pipeline)
        return joblib.load(expected_pipeline), expected_pipeline

    print(
        f"Full predictor-set pipeline for {algorithm_name} was not found. "
        f"Refitting {algorithm_name} with CHM+RGBNIR+SE for SHAP interpretation."
    )

    if full_feature_set_name not in feature_sets:
        raise ValueError("Feature set CHM+RGBNIR+SE is not available in feature_sets.")

    algorithm_groups = feature_sets[full_feature_set_name]

    algorithm_input_cols = []
    for cols in algorithm_groups.values():
        algorithm_input_cols.extend(cols)
    algorithm_input_cols = list(dict.fromkeys(algorithm_input_cols))

    X_algorithm_full = X[algorithm_input_cols].copy()

    algorithm_selector = GroupWiseSelectKBest(
        groups=algorithm_groups,
        k_per_group=k_grid[0]
    )

    algorithm_pipe = Pipeline([
        ("selector", algorithm_selector),
        ("model", clone(models_and_params[algorithm_name]["model"]))
    ])

    algorithm_search = RandomizedSearchCV(
        estimator=algorithm_pipe,
        param_distributions=models_and_params[algorithm_name]["params"],
        n_iter=N_ITER,
        scoring="r2",
        cv=inner_cv,
        n_jobs=-1,
        random_state=3000 + ["RF", "GradientBoosting", "XGBoost"].index(algorithm_name),
        refit=True,
        error_score="raise"
    )

    algorithm_search.fit(X_algorithm_full, y)

    fitted_pipe = algorithm_search.best_estimator_

    saved_pipeline = (
        ALGO_SHAP_DIR /
        f"Final_pipeline_SHAP_{safe_algorithm_name}_{safe_full_feature_set_name}.joblib"
    )

    joblib.dump(fitted_pipe, saved_pipeline)

    return fitted_pipe, saved_pipeline


all_algorithm_shap_tables = []
all_algorithm_shap_group_tables = []
all_algorithm_shap_model_info = []

for algorithm_name in ["RF", "GradientBoosting", "XGBoost"]:

    print("\n=====================================")
    print("Algorithm-specific SHAP:", algorithm_name)
    print("=====================================")

    fitted_pipe, pipeline_file = load_or_refit_full_algorithm_pipeline(algorithm_name)

    selector = fitted_pipe.named_steps["selector"]
    fitted_model = fitted_pipe.named_steps["model"]

    selected_features_algorithm = selector.get_feature_names_out().tolist()

    full_feature_set_name = "CHM+RGBNIR+SE"
    algorithm_groups = feature_sets[full_feature_set_name]
    algorithm_input_cols = []
    for cols in algorithm_groups.values():
        algorithm_input_cols.extend(cols)
    algorithm_input_cols = list(dict.fromkeys(algorithm_input_cols))

    X_algorithm = X[algorithm_input_cols].copy()

    X_selected_algorithm = selector.transform(X_algorithm)
    X_selected_algorithm = pd.DataFrame(
        X_selected_algorithm,
        columns=selected_features_algorithm,
        index=X_algorithm.index
    )

    explainer_algorithm = shap.TreeExplainer(fitted_model)
    shap_values_algorithm = explainer_algorithm.shap_values(X_selected_algorithm)

    if isinstance(shap_values_algorithm, list):
        shap_matrix_algorithm = shap_values_algorithm[0]
    else:
        shap_matrix_algorithm = np.array(shap_values_algorithm)

    if len(shap_matrix_algorithm.shape) == 3:
        shap_matrix_algorithm = shap_matrix_algorithm[:, :, 0]

    if shap_matrix_algorithm.shape[1] != len(selected_features_algorithm):
        raise ValueError(
            f"SHAP matrix has {shap_matrix_algorithm.shape[1]} columns but "
            f"{len(selected_features_algorithm)} selected features were found for {algorithm_name}."
        )

    model_plot_name = clean_model_name_for_plot(algorithm_name)

    shap_algorithm_df = pd.DataFrame({
        "Model": algorithm_name,
        "Model_plot": model_plot_name,
        "Feature_Set": "CHM+RGBNIR+SE",
        "Predictor": selected_features_algorithm,
        "Mean_abs_SHAP": np.abs(shap_matrix_algorithm).mean(axis=0),
        "SD_abs_SHAP": np.abs(shap_matrix_algorithm).std(axis=0)
    })

    shap_algorithm_df["Group"] = shap_algorithm_df["Predictor"].apply(
        get_group_name_for_algorithm_shap
    )

    shap_algorithm_df = shap_algorithm_df.sort_values(
        "Mean_abs_SHAP",
        ascending=False
    )

    shap_algorithm_df.to_csv(
        ALGO_SHAP_DIR / f"SHAP_predictor_importance_{model_plot_name}_CHM_plus_RGBNIR_plus_SE.csv",
        index=False
    )

    shap_algorithm_df.head(20).to_csv(
        ALGO_SHAP_DIR / f"SHAP_top20_predictor_importance_{model_plot_name}_CHM_plus_RGBNIR_plus_SE.csv",
        index=False
    )

    group_algorithm_df = (
        shap_algorithm_df
        .groupby(["Model", "Model_plot", "Feature_Set", "Group"], as_index=False)
        .agg(
            Total_Mean_abs_SHAP=("Mean_abs_SHAP", "sum"),
            Mean_Mean_abs_SHAP=("Mean_abs_SHAP", "mean"),
            N_Predictors=("Predictor", "count")
        )
        .sort_values("Total_Mean_abs_SHAP", ascending=False)
    )

    group_algorithm_df.to_csv(
        ALGO_SHAP_DIR / f"SHAP_group_importance_{model_plot_name}_CHM_plus_RGBNIR_plus_SE.csv",
        index=False
    )

    all_algorithm_shap_tables.append(shap_algorithm_df)
    all_algorithm_shap_group_tables.append(group_algorithm_df)

    all_algorithm_shap_model_info.append({
        "Model": algorithm_name,
        "Model_plot": model_plot_name,
        "Feature_Set": "CHM+RGBNIR+SE",
        "Selected_Features": ", ".join(selected_features_algorithm),
        "N_Selected_Features": len(selected_features_algorithm),
        "Pipeline": str(pipeline_file)
    })


all_algorithm_shap_df = pd.concat(
    all_algorithm_shap_tables,
    ignore_index=True
)

all_algorithm_shap_df.to_csv(
    ALGO_SHAP_DIR / "All_algorithm_specific_SHAP_predictor_importance.csv",
    index=False
)

all_algorithm_group_shap_df = pd.concat(
    all_algorithm_shap_group_tables,
    ignore_index=True
)

all_algorithm_group_shap_df.to_csv(
    ALGO_SHAP_DIR / "All_algorithm_specific_SHAP_group_importance.csv",
    index=False
)

pd.DataFrame(all_algorithm_shap_model_info).to_csv(
    ALGO_SHAP_DIR / "All_algorithm_specific_SHAP_model_information.csv",
    index=False
)


# ------------------------------------------------------------
# Overall algorithm-averaged SHAP ranking
# Missing predictors are treated as zero in algorithms where
# they were not selected.
# ------------------------------------------------------------

model_cols = ["RF", "GB", "XGBoost"]

overall_shap_wide = (
    all_algorithm_shap_df
    .pivot_table(
        index=["Predictor", "Group"],
        columns="Model_plot",
        values="Mean_abs_SHAP",
        aggfunc="mean"
    )
    .reset_index()
)

for model_col in model_cols:
    if model_col not in overall_shap_wide.columns:
        overall_shap_wide[model_col] = 0.0

overall_shap_wide[model_cols] = overall_shap_wide[model_cols].fillna(0.0)

overall_shap_wide["Mean_abs_SHAP_overall"] = overall_shap_wide[model_cols].mean(axis=1)
overall_shap_wide["SD_abs_SHAP_across_models"] = overall_shap_wide[model_cols].std(axis=1)

overall_shap_df = overall_shap_wide.sort_values(
    "Mean_abs_SHAP_overall",
    ascending=False
)

overall_shap_df.to_csv(
    ALGO_SHAP_DIR / "Overall_mean_SHAP_importance_across_algorithms.csv",
    index=False
)

overall_shap_df.head(20).to_csv(
    ALGO_SHAP_DIR / "Table_07a_SHAP_top20_predictor_importance_across_algorithms.csv",
    index=False
)

overall_group_shap_df = (
    overall_shap_df
    .groupby("Group", as_index=False)
    .agg(
        Total_Mean_abs_SHAP_overall=("Mean_abs_SHAP_overall", "sum"),
        Mean_Mean_abs_SHAP_overall=("Mean_abs_SHAP_overall", "mean"),
        N_Predictors=("Predictor", "count")
    )
    .sort_values("Total_Mean_abs_SHAP_overall", ascending=False)
)

overall_group_shap_df.to_csv(
    ALGO_SHAP_DIR / "Overall_SHAP_group_importance_across_algorithms.csv",
    index=False
)


# ------------------------------------------------------------
# Four-panel algorithm-specific SHAP top-predictor figure
# ------------------------------------------------------------

group_colors_plot = {
    "CHM": "#6aaed6",
    "MSI": "#e7ad72",
    "SE": "#7bd88f",
}

all_algorithm_shap_df["Plot_Group"] = all_algorithm_shap_df["Group"].apply(clean_group_for_plot)
overall_shap_df["Plot_Group"] = overall_shap_df["Group"].apply(clean_group_for_plot)

overall_top = overall_shap_df.head(20).copy()

rf_top = (
    all_algorithm_shap_df[all_algorithm_shap_df["Model_plot"] == "RF"]
    .sort_values("Mean_abs_SHAP", ascending=False)
    .head(10)
)

gb_top = (
    all_algorithm_shap_df[all_algorithm_shap_df["Model_plot"] == "GB"]
    .sort_values("Mean_abs_SHAP", ascending=False)
    .head(10)
)

xgb_top = (
    all_algorithm_shap_df[all_algorithm_shap_df["Model_plot"] == "XGBoost"]
    .sort_values("Mean_abs_SHAP", ascending=False)
    .head(10)
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.9,
})

fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), dpi=300)
axes = axes.flatten()


def plot_shap_barh(ax, df, value_col, title, panel_label, tick_step=0.02):
    df_plot = df.sort_values(value_col, ascending=True).copy()
    colors = df_plot["Plot_Group"].map(group_colors_plot).fillna("#bdbdbd")
    xmax = float(df[value_col].max()) * 1.10 if len(df) else 0.1
    xmax = max(xmax, 0.01)

    ax.barh(
        df_plot["Predictor"],
        df_plot[value_col],
        color=colors,
        edgecolor=colors,
        height=0.70
    )

    ax.set_title(title, fontweight="bold", pad=7)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_locator(MultipleLocator(tick_step))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=8, pad=2)
    ax.tick_params(axis="x", labelsize=8)
    ax.text(
        0.0, 1.045, panel_label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False
    )


plot_shap_barh(axes[0], overall_top, "Mean_abs_SHAP_overall", "Overall Top 20 Predictors", "(a)")
plot_shap_barh(axes[1], rf_top, "Mean_abs_SHAP", "RF | Top 10 Predictors", "(b)")
plot_shap_barh(axes[2], gb_top, "Mean_abs_SHAP", "GB | Top 10 Predictors", "(c)")
plot_shap_barh(axes[3], xgb_top, "Mean_abs_SHAP", "XGBoost | Top 10 Predictors", "(d)")

legend_handles = [
    Patch(facecolor=group_colors_plot["CHM"], edgecolor="black", label="CHM"),
    Patch(facecolor=group_colors_plot["MSI"], edgecolor="black", label="MSI"),
    Patch(facecolor=group_colors_plot["SE"], edgecolor="black", label="SE"),
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 0.035),
    fontsize=8
)

fig.subplots_adjust(
    left=0.185,
    right=0.985,
    top=0.94,
    bottom=0.12,
    wspace=0.32,
    hspace=0.28
)

fig.savefig(
    ALGO_SHAP_DIR / "Top_predictors_A4_exact_style_FIXED.png",
    dpi=300
)

fig.savefig(
    ALGO_SHAP_DIR / "Top_predictors_A4_exact_style_FIXED.pdf"
)

plt.close("all")

print("\nAlgorithm-specific SHAP analysis completed.")
print("Outputs saved to:", ALGO_SHAP_DIR)


# CELL

# ============================================================
# Objective 5:
# Complementarity analysis
# SHAP dependence and interaction analysis
# Do SE predictors complement CHM and MSI predictors?
# ============================================================

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Use outputs from Objective 4
# Required existing objects:
# shap_matrix
# X_selected
# shap_importance_df
# selected_features
# final_model
# best_model_name
# best_feature_set
# ------------------------------------------------------------

print("Running Objective 5 for:")
print("Model:", best_model_name)
print("Feature set:", best_feature_set)


# ------------------------------------------------------------
# 2. Identify top predictors by group
# ------------------------------------------------------------

top_se_features = (
    shap_importance_df[shap_importance_df["Group"] == "SE"]
    .sort_values("Mean_abs_SHAP", ascending=False)
    .head(5)["Predictor"]
    .tolist()
)

top_chm_features = (
    shap_importance_df[shap_importance_df["Group"] == "CHM"]
    .sort_values("Mean_abs_SHAP", ascending=False)
    .head(5)["Predictor"]
    .tolist()
)

top_rgbnir_features = (
    shap_importance_df[shap_importance_df["Group"] == "RGB-NIR"]
    .sort_values("Mean_abs_SHAP", ascending=False)
    .head(5)["Predictor"]
    .tolist()
)

print("Top SE predictors:", top_se_features)
print("Top CHM predictors:", top_chm_features)
print("Top MSI predictors:", top_rgbnir_features)


# ------------------------------------------------------------
# 3. SHAP dependence plots for top SE predictors
# ------------------------------------------------------------

for se_feature in top_se_features:

    plt.figure(figsize=(6.5, 5))

    shap.dependence_plot(
        se_feature,
        shap_matrix,
        X_selected,
        interaction_index="auto",
        show=False
    )

    plt.title(f"SHAP dependence plot\n{se_feature}")
    plt.tight_layout()

    plt.savefig(
        O4_DIR / f"Figure_O5_SHAP_dependence_{se_feature}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close("all")


# ------------------------------------------------------------
# 4. SE × CHM dependence plots
# ------------------------------------------------------------

for se_feature in top_se_features[:3]:

    for chm_feature in top_chm_features[:3]:

        plt.figure(figsize=(6.5, 5))

        shap.dependence_plot(
            se_feature,
            shap_matrix,
            X_selected,
            interaction_index=chm_feature,
            show=False
        )

        plt.title(f"SHAP dependence: {se_feature} × {chm_feature}")
        plt.tight_layout()

        plt.savefig(
            O4_DIR / f"Figure_O5_dependence_{se_feature}_x_{chm_feature}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close("all")


# ------------------------------------------------------------
# 5. SE × RGB-NIR dependence plots
# ------------------------------------------------------------

for se_feature in top_se_features[:3]:

    for rgb_feature in top_rgbnir_features[:3]:

        plt.figure(figsize=(6.5, 5))

        shap.dependence_plot(
            se_feature,
            shap_matrix,
            X_selected,
            interaction_index=rgb_feature,
            show=False
        )

        plt.title(f"SHAP dependence: {se_feature} × {rgb_feature}")
        plt.tight_layout()

        plt.savefig(
            O4_DIR / f"Figure_O5_dependence_{se_feature}_x_{rgb_feature}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close("all")


# ------------------------------------------------------------
# 6. Approximate complementarity using SHAP correlation
# ------------------------------------------------------------

complementarity_rows = []

for se_feature in top_se_features:

    se_idx = list(X_selected.columns).index(se_feature)
    se_shap = shap_matrix[:, se_idx]

    for other_feature in top_chm_features + top_rgbnir_features:

        other_idx = list(X_selected.columns).index(other_feature)
        other_shap = shap_matrix[:, other_idx]

        corr = np.corrcoef(se_shap, other_shap)[0, 1]

        complementarity_rows.append({
            "SE_Feature": se_feature,
            "Other_Feature": other_feature,
            "Other_Group": (
                "CHM" if other_feature in top_chm_features else "RGB-NIR"
            ),
            "SHAP_Correlation": corr,
            "Abs_SHAP_Correlation": abs(corr)
        })


complementarity_df = pd.DataFrame(complementarity_rows)

complementarity_df = complementarity_df.sort_values(
    "Abs_SHAP_Correlation",
    ascending=False
)

complementarity_df.to_csv(
    O4_DIR / "O5_SHAP_complementarity_correlation.csv",
    index=False
)

print("\nSHAP complementarity correlation table:")
print(complementarity_df)


# ------------------------------------------------------------
# 7. Plot SHAP complementarity correlation
# ------------------------------------------------------------

plot_df = complementarity_df.copy()
plot_df["Pair"] = plot_df["SE_Feature"] + " × " + plot_df["Other_Feature"]

plot_df = plot_df.sort_values("Abs_SHAP_Correlation", ascending=True)

plt.figure(figsize=(10, 8))

plt.barh(
    plot_df["Pair"],
    plot_df["Abs_SHAP_Correlation"]
)

plt.xlabel("Absolute correlation between SHAP values")
plt.ylabel("Feature pair")
plt.title("Complementarity between SE and CHM/MSI predictors")

plt.tight_layout()

plt.savefig(
    O4_DIR / "Figure_O5_SHAP_complementarity_correlation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close("all")


# ------------------------------------------------------------
# 8. Optional: SHAP interaction values for tree models
# ------------------------------------------------------------

try:
    explainer_tree = shap.TreeExplainer(final_model)
    shap_interaction_values = explainer_tree.shap_interaction_values(X_selected)

    if isinstance(shap_interaction_values, list):
        shap_interaction_values = shap_interaction_values[0]

    interaction_rows = []

    feature_list = list(X_selected.columns)

    for se_feature in top_se_features:
        se_idx = feature_list.index(se_feature)

        for other_feature in top_chm_features + top_rgbnir_features:
            other_idx = feature_list.index(other_feature)

            mean_abs_interaction = np.abs(
                shap_interaction_values[:, se_idx, other_idx]
            ).mean()

            interaction_rows.append({
                "SE_Feature": se_feature,
                "Other_Feature": other_feature,
                "Other_Group": (
                    "CHM" if other_feature in top_chm_features else "RGB-NIR"
                ),
                "Mean_abs_SHAP_Interaction": mean_abs_interaction
            })

    interaction_df = pd.DataFrame(interaction_rows)

    interaction_df = interaction_df.sort_values(
        "Mean_abs_SHAP_Interaction",
        ascending=False
    )

    interaction_df.to_csv(
        O4_DIR / "O5_SHAP_interaction_values.csv",
        index=False
    )

    print("\nSHAP interaction table:")
    print(interaction_df)


    # --------------------------------------------------------
    # 9. SHAP interaction figure
    # --------------------------------------------------------

    plot_df = interaction_df.copy()
    plot_df["Pair"] = plot_df["SE_Feature"] + " × " + plot_df["Other_Feature"]

    plot_df = plot_df.sort_values(
        "Mean_abs_SHAP_Interaction",
        ascending=True
    )

    plt.figure(figsize=(10, 8))

    plt.barh(
        plot_df["Pair"],
        plot_df["Mean_abs_SHAP_Interaction"]
    )

    plt.xlabel("Mean absolute SHAP interaction value")
    plt.ylabel("Feature pair")
    plt.title("SHAP interaction strength: SE with CHM/MSI")

    plt.tight_layout()

    plt.savefig(
        O4_DIR / "Figure_O5_SHAP_interaction_strength.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close("all")

except Exception as e:
    print("SHAP interaction values could not be calculated.")
    print("Reason:", e)

# CELL

# ============================================================
