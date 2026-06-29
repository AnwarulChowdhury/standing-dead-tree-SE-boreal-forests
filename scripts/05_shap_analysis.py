# Switch bare SHAP/Objective-5 output filenames into their own folder.
O4_DIR = OUT_DIR / "O4_SHAP_and_complementarity"
O4_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(O4_DIR)
print("Objective 4/5 outputs will be saved to:", O4_DIR)

# ============================================================
# Objective 4:
# Model interpretability using SHAP
# Evaluate importance of individual predictors and groups
# ============================================================

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.pipeline import Pipeline


# ------------------------------------------------------------
# 1. Rebuild and fit best model from Objective 1
# ------------------------------------------------------------

best_groups = feature_sets[best_feature_set]

best_input_cols = []
for cols in best_groups.values():
    best_input_cols.extend(cols)

best_input_cols = list(dict.fromkeys(best_input_cols))

X_best = X[best_input_cols]

# Use the best tuned k_per_group if available.
# Otherwise use the first/default k_grid setting.
if "best_final_params" in globals() and "selector__k_per_group" in best_final_params:
    best_k_per_group = best_final_params["selector__k_per_group"]
else:
    best_k_per_group = k_grid[0]

best_selector = GroupWiseSelectKBest(
    groups=best_groups,
    k_per_group=best_k_per_group
)

# Use current Objective 1 model dictionary
best_model = clone(models_and_params[best_model_name]["model"])

best_pipe = Pipeline([
    ("selector", best_selector),
    ("model", best_model)
])

best_pipe.fit(X_best, y)


# ------------------------------------------------------------
# 2. Get selected predictors
# ------------------------------------------------------------

selected_features = best_pipe.named_steps["selector"].get_feature_names_out()

X_selected = best_pipe.named_steps["selector"].transform(X_best)
X_selected = pd.DataFrame(X_selected, columns=selected_features)

final_model = best_pipe.named_steps["model"]

print("Best model:", best_model_name)
print("Best feature set:", best_feature_set)
print("k_per_group used:", best_k_per_group)
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
    "O4_SHAP_predictor_importance.csv",
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

group_colors = {
    "CHM": "tab:blue",
    "RGB-NIR": "tab:green",
    "SE": "tab:orange"
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
    Patch(facecolor=group_colors[g], label=g)
    for g in present_groups
    if g in group_colors
]

plt.legend(handles=legend_elements, title="Predictor group")

plt.tight_layout()
plt.savefig(
    "Figure_O4_top20_SHAP_predictors.png",
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
    "O4_SHAP_group_importance.csv",
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
    "Figure_O4_group_level_SHAP_importance.png",
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
    "Figure_O4_SHAP_summary_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close("all")

# CELL

# ============================================================
# Objective 5:
# Complementarity analysis
# SHAP dependence and interaction analysis
# Do SE predictors complement CHM and RGB-NIR predictors?
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
print("Top RGB-NIR predictors:", top_rgbnir_features)


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
        f"Figure_O5_SHAP_dependence_{se_feature}.png",
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
            f"Figure_O5_dependence_{se_feature}_x_{chm_feature}.png",
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
            f"Figure_O5_dependence_{se_feature}_x_{rgb_feature}.png",
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
    "O5_SHAP_complementarity_correlation.csv",
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
plt.title("Complementarity between SE and CHM/RGB-NIR predictors")

plt.tight_layout()

plt.savefig(
    "Figure_O5_SHAP_complementarity_correlation.png",
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
        "O5_SHAP_interaction_values.csv",
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
        "Figure_O5_SHAP_interaction_strength.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close("all")

except Exception as e:
    print("SHAP interaction values could not be calculated.")
    print("Reason:", e)

# CELL

# ============================================================
