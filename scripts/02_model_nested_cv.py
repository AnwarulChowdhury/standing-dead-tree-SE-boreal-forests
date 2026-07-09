# ============================================================
# This script starts from an already-extracted predictor CSV and reruns:
#   Task 1: nested CV model comparison
# ============================================================

from __future__ import annotations

import argparse
import datetime as _datetime
import html as _html
import json as _json
import os
import shutil
import sys
import traceback
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Redo all modelling analyses from an already extracted predictor CSV. Predictor extraction is not run."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to predictor CSV, e.g. predictor_with_CHM - Copy.csv",
    )
    parser.add_argument(
        "--target-col",
        default="Dead_F",
        help="Target column name. Default: Dead_F",
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help="Output root folder. Default: REANALYSIS_<timestamp> beside the CSV.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing --out-root folder.",
    )
    parser.add_argument(
        "--figures",
        "--selected-figures",
        default="final",
        help=(
            "Comma-separated figure selection. Tables are always generated. "
            "Use final, all, none, model, ovp, top, shap, o3, o3se, o7, "
            "or *_all groups such as shap_all, o3_all, o7_all, importance_all. "
            "Default: final (only final manuscript figures with fixed margins)."
        ),
    )
    return parser.parse_args()


ARGS = _parse_args()
CSV_PATH_FROM_ARGS = Path(ARGS.data).expanduser().resolve()
if not CSV_PATH_FROM_ARGS.exists():
    raise FileNotFoundError(f"Predictor CSV not found: {CSV_PATH_FROM_ARGS}")

_timestamp = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
if ARGS.out_root:
    RUN_ROOT = Path(ARGS.out_root).expanduser().resolve()
else:
    RUN_ROOT = CSV_PATH_FROM_ARGS.parent / f"REANALYSIS_R2_{_timestamp}"

if RUN_ROOT.exists() and not ARGS.overwrite:
    raise FileExistsError(
        f"Output folder already exists: {RUN_ROOT}\n"
        "Use --overwrite or choose a new --out-root."
    )
RUN_ROOT.mkdir(parents=True, exist_ok=True)

# Keep a copy of the exact predictor CSV used for the rerun.
try:
    shutil.copy2(CSV_PATH_FROM_ARGS, RUN_ROOT / "input_predictor_matrix_used.csv")
except Exception as exc:
    print(f"Warning: could not copy input CSV into run folder: {exc}")

# Run all relative outputs inside the new analysis folder.
os.chdir(RUN_ROOT)
print("Input predictor CSV:", CSV_PATH_FROM_ARGS)
print("All new outputs will be saved in:", RUN_ROOT)

# ============================================================
# Figure selection
# ============================================================
# Tables and model outputs are always generated. This selection only controls
# which figure files are written to disk. The default is "final", meaning only
# the final manuscript-style figures with fixed margins are saved.

_FIGURE_SELECTION_RAW = str(getattr(ARGS, "figures", "final") or "final")

def _parse_figure_selection(raw: str) -> set[str]:
    parts = []
    for chunk in str(raw).replace(";", ",").split(","):
        item = chunk.strip().lower().replace("-", "_").replace(" ", "_")
        if item:
            parts.append(item)
    if not parts:
        parts = ["final"]
    aliases = {
        "selected": "final",
        "manuscript": "final",
        "manuscript_final": "final",
        "fixed": "final",
        "no": "none",
        "false": "none",
        "0": "none",
        "yes": "all",
        "true": "all",
        "1": "all",
        "condition": "o7",
        "conditions": "o7",
        "label": "o3",
        "label_efficiency": "o3",
        "se_gain": "o3se",
        "se_training": "o3se",
        "top_predictors": "top",
        "interaction": "shap",
        "complementarity": "shap",
        "observed_predicted": "ovp",
        "obs_pred": "ovp",
    }
    return {aliases.get(p, p) for p in parts}

_SELECTED_FIGURE_GROUPS = _parse_figure_selection(_FIGURE_SELECTION_RAW)
_SKIPPED_FIGURES_BY_SELECTION: list[str] = []
_SAVED_FIGURES_BY_SELECTION: list[str] = []

# Keywords are checked against the lower-case output filename/path.
# The short group names below are intentionally conservative: e.g. "top" saves
# only the final fixed top-predictor figure, whereas "importance_all" saves all
# diagnostic importance figures as well.
_FIGURE_GROUP_KEYWORDS = {
    "final": [
        "top_predictors_a4_exact_style_fixed",
        "shap_interaction_complementarity_a4_landscape_fixed",
        "o3_label_efficiency_3x3_a4_fixed",
        "o3_rmse_improvement_from_se_by_training_size_fixed",
        "figure_o7_split_dotplot_condition_associations_fixed",
        "observed_vs_predicted_grid_a4_fixed",
    ],
    "top": ["top_predictors_a4_exact_style_fixed"],
    "shap": ["shap_interaction_complementarity_a4_landscape_fixed"],
    "o3": ["o3_label_efficiency_3x3_a4_fixed"],
    "o3se": ["o3_rmse_improvement_from_se_by_training_size_fixed"],
    "o7": ["figure_o7_split_dotplot_condition_associations_fixed"],
    "ovp": ["observed_vs_predicted_grid_a4_fixed"],
    "model": [
        "figure_01_nested_tuned_model_comparison",
        "figure_02_nested_tuned_model_comparison",
        "figure_03_nested_tuned_model_comparison",
    ],
    "ovp_all": ["fig_ovp_"],
    "importance_all": [
        "figure_05_best_tuned_model_top_10_predictors",
        "figure_06_best_tuned_model_group_level_importance",
        "figure_top_",
        "figure_group_importance",
        "top_predictors_a4_exact_style",
    ],
    "o2_all": ["figure_o2_", "o2_se_added_value_heatmap"],
    "o3_all": [
        "figure_o3_label_efficiency",
        "o3_label_efficiency_rmse",
        "o3_label_efficiency_3x3_a4",
    ],
    "o3se_all": [
        "figure_o3_se_added_value_by_training_size",
        "o3_se_improvement_under_reduced_labels",
        "o3_rmse_improvement_from_se_by_training_size",
    ],
    "shap_all": [
        "figure_o4_",
        "figure_o5_",
        "algorithm_specific_shap",
        "top_predictors_a4_exact_style",
        "shap_interaction_complementarity_a4_landscape",
    ],
    "o7_all": ["figure_o7_"],
}

def _should_save_figure(fname) -> bool:
    groups = _SELECTED_FIGURE_GROUPS
    if "all" in groups:
        return True
    if "none" in groups:
        return False
    filename = Path(str(fname)).name.lower()
    full = str(fname).lower()
    for group in groups:
        for keyword in _FIGURE_GROUP_KEYWORDS.get(group, [group]):
            key = str(keyword).lower()
            if key in filename or key in full:
                return True
    return False

def _record_figure_decision(fname, saved: bool) -> None:
    try:
        s = str(fname)
        if saved:
            _SAVED_FIGURES_BY_SELECTION.append(s)
        else:
            _SKIPPED_FIGURES_BY_SELECTION.append(s)
    except Exception:
        pass

def _write_figure_selection_report() -> None:
    report = RUN_ROOT / "FIGURE_SELECTION_REPORT.txt"
    lines = [
        f"Requested --figures: {_FIGURE_SELECTION_RAW}",
        f"Normalized groups: {', '.join(sorted(_SELECTED_FIGURE_GROUPS))}",
        "",
        f"Saved figures: {len(_SAVED_FIGURES_BY_SELECTION)}",
        *[f"  SAVED: {p}" for p in _SAVED_FIGURES_BY_SELECTION],
        "",
        f"Skipped figures: {len(_SKIPPED_FIGURES_BY_SELECTION)}",
        *[f"  SKIPPED: {p}" for p in _SKIPPED_FIGURES_BY_SELECTION],
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    print("Figure selection report:", report)

print("Figure selection:", ", ".join(sorted(_SELECTED_FIGURE_GROUPS)))

# ============================================================
# Objective 1: Nested CV model development with group-wise
# feature selection and observed-vs-predicted figures for
# every model using repeated outer-CV test predictions
# ============================================================

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import json
import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensures parent folders exist before every savefig call and applies --figures selection.
_ORIGINAL_PLT_SAVEFIG = plt.savefig
import matplotlib.figure as _mpl_figure
_ORIGINAL_FIGURE_SAVEFIG = _mpl_figure.Figure.savefig

def safe_savefig(fname, *args, **kwargs):
    path = Path(fname)
    if not _should_save_figure(path):
        _record_figure_decision(path, saved=False)
        print(f"Skipping figure by --figures: {path.name}")
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    _record_figure_decision(path, saved=True)
    return _ORIGINAL_PLT_SAVEFIG(str(path), *args, **kwargs)

def safe_figure_savefig(self, fname, *args, **kwargs):
    path = Path(fname)
    if not _should_save_figure(path):
        _record_figure_decision(path, saved=False)
        print(f"Skipping figure by --figures: {path.name}")
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    _record_figure_decision(path, saved=True)
    return _ORIGINAL_FIGURE_SAVEFIG(self, str(path), *args, **kwargs)

plt.savefig = safe_savefig
_mpl_figure.Figure.savefig = safe_figure_savefig

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import f_regression
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    RepeatedKFold,
    RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor


# ============================================================
# 1. Load data
# ============================================================

CSV_PATH = CSV_PATH_FROM_ARGS

OUT_DIR = RUN_ROOT / "O1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_FIG_DIR = OUT_DIR / "OvP"
PRED_FIG_DIR.mkdir(parents=True, exist_ok=True)

target_col = ARGS.target_col

df = pd.read_csv(CSV_PATH)

if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' was not found.")

# Drop target and plot identifier.
drop_cols = [target_col]

if "plot_id" in df.columns:
    drop_cols.append("plot_id")
    sample_names = df["plot_id"].astype(str).copy()
elif "Name" in df.columns:
    drop_cols.append("Name")
    sample_names = df["Name"].astype(str).copy()
else:
    sample_names = pd.Series(df.index.astype(str), index=df.index, name="Sample_Name")

X = df.drop(columns=drop_cols)
y = df[target_col]

# Keep only numeric predictors
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

if len(non_numeric_cols) > 0:
    print("Dropping non-numeric predictor columns:")
    print(non_numeric_cols)
    X = X.drop(columns=non_numeric_cols)

# Basic checks
if y.isna().any():
    raise ValueError("Target column contains missing values.")

if X.isna().all(axis=0).any():
    bad_cols = X.columns[X.isna().all(axis=0)].tolist()
    raise ValueError(f"These predictors are entirely missing: {bad_cols}")

if X.shape[1] == 0:
    raise ValueError("No numeric predictors available.")

print("Dataset shape:", df.shape)
print("Predictor matrix shape:", X.shape)
print("Target:", target_col)


# ============================================================
# 2. Predictor groups
# ============================================================

SE_cols = [
    c for c in X.columns
    if c.startswith("SE2025") or c.startswith("dSE")
]

CHM_cols = [
    c for c in X.columns
    if c.startswith("CHM")
]

RGBNIR_cols = [
    c for c in X.columns
    if c not in SE_cols + CHM_cols
]

print("\nPredictor groups:")
print("CHM predictors:", len(CHM_cols))
print("RGB-NIR predictors:", len(RGBNIR_cols))
print("SE predictors:", len(SE_cols))


# ============================================================
# 3. Robust repeated stratified CV for regression
# ============================================================

class RepeatedStratifiedKFoldReg:
    """
    Repeated stratified K-fold for regression.

    The continuous target is binned using quantiles.
    If valid bins cannot be created, the class falls back to RepeatedKFold.
    """

    def __init__(self, n_splits=5, n_repeats=1, n_bins=5, random_state=42):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.n_bins = n_bins
        self.random_state = random_state

    def _make_bins(self, y):
        y_series = pd.Series(y).reset_index(drop=True)

        max_bins = min(self.n_bins, y_series.nunique())

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

                bin_counts = pd.Series(bins).value_counts()

                if bin_counts.min() >= self.n_splits:
                    return bins

            except Exception:
                continue

        return None

    def split(self, X, y, groups=None):
        y_bins = self._make_bins(y)

        if y_bins is not None:
            cv = RepeatedStratifiedKFold(
                n_splits=self.n_splits,
                n_repeats=self.n_repeats,
                random_state=self.random_state
            )
            return cv.split(X, y_bins)

        print(
            "Warning: valid regression stratification was not possible. "
            "Using RepeatedKFold instead."
        )

        cv = RepeatedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state
        )

        return cv.split(X, y)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits * self.n_repeats


# ============================================================
# 4. Leakage-safe group-wise selector
# ============================================================

class GroupWiseSelectKBest(BaseEstimator, TransformerMixin):
    """
    Selects top k predictors per group using f_regression.

    Leakage-safe because:
    - median imputation is fitted only on training data,
    - f_regression scores are fitted only on training data,
    - selector is inside the sklearn Pipeline.
    """

    def __init__(self, groups, k_per_group):
        self.groups = groups
        self.k_per_group = k_per_group

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame with column names.")

        X = X.copy()

        self.feature_names_in_ = X.columns.to_list()
        self.selected_features_ = []
        self.group_selected_features_ = {}

        self.medians_ = X.median(numeric_only=True)
        X_filled = X.fillna(self.medians_)

        for group_name, cols in self.groups.items():

            cols = [c for c in cols if c in X_filled.columns]

            if len(cols) == 0:
                self.group_selected_features_[group_name] = []
                continue

            requested_k = self.k_per_group.get(group_name, len(cols))
            k = min(requested_k, len(cols))

            scores, _ = f_regression(X_filled[cols], y)
            scores = np.nan_to_num(
                scores,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            ranked_cols = (
                pd.DataFrame({
                    "feature": cols,
                    "score": scores
                })
                .sort_values("score", ascending=False)
                .head(k)["feature"]
                .tolist()
            )

            self.group_selected_features_[group_name] = ranked_cols
            self.selected_features_.extend(ranked_cols)

        if len(self.selected_features_) == 0:
            raise ValueError("No predictors were selected.")

        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame with column names.")

        X = X.copy()
        X = X.fillna(self.medians_)

        return X[self.selected_features_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.selected_features_)


# ============================================================
# 5. k-per-group tuning grid
# ============================================================

k_grid = [
    {"CHM": 1, "RGBNIR": 5,  "SE": 5},
    {"CHM": 2, "RGBNIR": 10, "SE": 10},
    {"CHM": 3, "RGBNIR": 15, "SE": 15},
    {"CHM": 4, "RGBNIR": 20, "SE": 20},
    {"CHM": 5, "RGBNIR": 25, "SE": 25},
]


# ============================================================
# 6. Feature-set combinations
# ============================================================

feature_sets = {
    "CHM": {
        "CHM": CHM_cols
    },
    "RGBNIR": {
        "RGBNIR": RGBNIR_cols
    },
    "SE": {
        "SE": SE_cols
    },
    "CHM+RGBNIR": {
        "CHM": CHM_cols,
        "RGBNIR": RGBNIR_cols
    },
    "CHM+SE": {
        "CHM": CHM_cols,
        "SE": SE_cols
    },
    "RGBNIR+SE": {
        "RGBNIR": RGBNIR_cols,
        "SE": SE_cols
    },
    "CHM+RGBNIR+SE": {
        "CHM": CHM_cols,
        "RGBNIR": RGBNIR_cols,
        "SE": SE_cols
    }
}

# Remove feature sets with zero predictors
feature_sets = {
    feature_name: groups
    for feature_name, groups in feature_sets.items()
    if sum(len(cols) for cols in groups.values()) > 0
}


# ============================================================
# 7. Models and hyperparameter grids
# ============================================================

models_and_params = {
    "RF": {
        "model": RandomForestRegressor(
            random_state=42,
            n_jobs=1
        ),
        "params": {
            "selector__k_per_group": k_grid,
            "model__n_estimators": [300, 500, 800],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 6],
            "model__max_features": ["sqrt", 0.5, 0.7, 1.0],
            "model__bootstrap": [True]
        }
    },

    "GradientBoosting": {
        "model": GradientBoostingRegressor(
            random_state=42
        ),
        "params": {
            "selector__k_per_group": k_grid,
            "model__n_estimators": [200, 300, 500, 800],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 0.5, 0.7, 1.0]
        }
    },

    "XGBoost": {
        "model": XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
            tree_method="hist"
        ),
        "params": {
            "selector__k_per_group": k_grid,
            "model__n_estimators": [200, 300, 500, 800],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4, 5],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__reg_alpha": [0, 0.01, 0.1, 1],
            "model__reg_lambda": [0.1, 1, 5, 10]
        }
    }
}


# ============================================================
# 8. Nested CV settings
# ============================================================

outer_cv = RepeatedStratifiedKFoldReg(
    n_splits=5,
    n_repeats=10,
    n_bins=5,
    random_state=42
)

inner_cv = RepeatedStratifiedKFoldReg(
    n_splits=5,
    n_repeats=3,
    n_bins=5,
    random_state=123
)

N_ITER = 10

all_results = []
all_predictions = []
all_best_params = []
all_selected_features = []


# ============================================================
# 9. Nested CV hyperparameter tuning
# ============================================================

for feature_name, groups in feature_sets.items():

    selected_input_cols = []

    for cols in groups.values():
        selected_input_cols.extend(cols)

    selected_input_cols = list(dict.fromkeys(selected_input_cols))

    X_sub = X[selected_input_cols].copy()

    print("\n=====================================")
    print("Feature set:", feature_name)
    print("Number of predictors:", X_sub.shape[1])
    print("=====================================")

    for model_name, model_info in models_and_params.items():

        print("\nTuning model:", model_name)

        outer_split_id = 0

        for train_idx, test_idx in outer_cv.split(X_sub, y):

            outer_split_id += 1

            X_train = X_sub.iloc[train_idx].copy()
            X_test = X_sub.iloc[test_idx].copy()

            y_train = y.iloc[train_idx].copy()
            y_test = y.iloc[test_idx].copy()

            selector = GroupWiseSelectKBest(
                groups=groups,
                k_per_group=k_grid[0]
            )

            pipe = Pipeline([
                ("selector", selector),
                ("model", clone(model_info["model"]))
            ])

            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=model_info["params"],
                n_iter=N_ITER,
                scoring="r2",
                cv=inner_cv,
                n_jobs=-1,
                random_state=42 + outer_split_id,
                refit=True,
                error_score="raise"
            )

            # Inner CV tuning happens only inside outer training data
            search.fit(X_train, y_train)

            best_pipe = search.best_estimator_

            # Prediction is made only on outer test data
            y_pred = best_pipe.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            selected_features = (
                best_pipe
                .named_steps["selector"]
                .get_feature_names_out()
                .tolist()
            )

            all_results.append({
                "Objective": "O1_Model_development_nested_tuning",
                "Feature_Set": feature_name,
                "Model": model_name,
                "Outer_Split": outer_split_id,
                "R2": r2,
                "RMSE": rmse,
                "MAE": mae,
                "Best_Inner_R2": search.best_score_,
                "N_Selected_Features": len(selected_features)
            })

            all_best_params.append({
                "Feature_Set": feature_name,
                "Model": model_name,
                "Outer_Split": outer_split_id,
                "Best_Params_JSON": json.dumps(search.best_params_, default=str)
            })

            all_selected_features.append({
                "Feature_Set": feature_name,
                "Model": model_name,
                "Outer_Split": outer_split_id,
                "Selected_Features": ", ".join(selected_features)
            })

            # Store outer-CV test predictions
            for row_position, row_index, obs, pred in zip(
                test_idx,
                X_sub.index[test_idx],
                y_test,
                y_pred
            ):
                all_predictions.append({
                    "Feature_Set": feature_name,
                    "Model": model_name,
                    "Outer_Split": outer_split_id,
                    "Row_Position": row_position,
                    "Row_Index": row_index,
                    "Sample_Name": sample_names.iloc[row_position],
                    "Observed": obs,
                    "Predicted": pred
                })

            print(
                f"{feature_name} | {model_name} | Outer split {outer_split_id} | "
                f"RMSE={rmse:.4f}, R2={r2:.4f}, MAE={mae:.4f}"
            )


# ============================================================
# 10. Save nested CV results
# ============================================================

results_df = pd.DataFrame(all_results)
params_df = pd.DataFrame(all_best_params)
selected_features_df = pd.DataFrame(all_selected_features)
pred_all_df = pd.DataFrame(all_predictions)

results_df.to_csv(
    OUT_DIR / "O1_nested_hyperparameter_tuning_CV_results.csv",
    index=False
)

params_df.to_csv(
    OUT_DIR / "O1_nested_hyperparameter_tuning_best_params_by_outer_split.csv",
    index=False
)

selected_features_df.to_csv(
    OUT_DIR / "O1_nested_hyperparameter_tuning_selected_features_by_outer_split.csv",
    index=False
)

pred_all_df.to_csv(
    OUT_DIR / "O1_nested_hyperparameter_tuning_all_outer_test_predictions.csv",
    index=False
)


# ============================================================
# 11. Nested CV summary table
# ============================================================

summary_df = (
    results_df
    .groupby(["Feature_Set", "Model"])
    .agg(
        R2_mean=("R2", "mean"),
        R2_sd=("R2", "std"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_sd=("RMSE", "std"),
        MAE_mean=("MAE", "mean"),
        MAE_sd=("MAE", "std"),
        N_Selected_Features_mean=("N_Selected_Features", "mean"),
        N_Selected_Features_sd=("N_Selected_Features", "std")
    )
    .reset_index()
    .sort_values("R2_mean", ascending=False)
)

summary_df.to_csv(
    OUT_DIR / "O1_nested_hyperparameter_tuning_summary.csv",
    index=False
)

print("\nNested CV tuning summary:")
print(summary_df)


# ============================================================
# 12. Best model from nested CV
# ============================================================

best_row = summary_df.iloc[0]

best_feature_set = best_row["Feature_Set"]
best_model_name = best_row["Model"]

print("\nBest tuned model based on nested CV R2:")
print("Feature set:", best_feature_set)
print("Model:", best_model_name)
print("Nested CV RMSE:", best_row["RMSE_mean"])
print("Nested CV R2:", best_row["R2_mean"])
print("Nested CV MAE:", best_row["MAE_mean"])


# ============================================================
# 13. Model comparison figures
# ============================================================

feature_order = [
    "CHM",
    "RGBNIR",
    "SE",
    "CHM+RGBNIR",
    "CHM+SE",
    "RGBNIR+SE",
    "CHM+RGBNIR+SE"
]

feature_order = [
    f for f in feature_order
    if f in summary_df["Feature_Set"].unique()
]

model_order = ["RF", "GradientBoosting", "XGBoost"]


def plot_metric(summary, metric_mean, metric_sd, ylabel, title, filename):

    x = np.arange(len(feature_order))
    width = 0.25

    plt.figure(figsize=(13, 6))

    for i, model in enumerate(model_order):

        data = (
            summary[summary["Model"] == model]
            .set_index("Feature_Set")
            .reindex(feature_order)
        )

        plt.bar(
            x + i * width,
            data[metric_mean],
            width,
            yerr=data[metric_sd],
            capsize=4,
            label=model
        )

    plt.xticks(x + width, feature_order, rotation=35, ha="right")
    plt.ylabel(ylabel)
    plt.xlabel("Predictor set")
    plt.title(title)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close("all")


plot_metric(
    summary_df,
    "R2_mean",
    "R2_sd",
    "Nested CV R²",
    "Nested CV tuned model comparison: R²",
    "Figure_01_nested_tuned_model_comparison_R2.png"
)

plot_metric(
    summary_df,
    "RMSE_mean",
    "RMSE_sd",
    "Nested CV RMSE",
    "Nested CV tuned model comparison: RMSE",
    "Figure_02_nested_tuned_model_comparison_RMSE.png"
)

plot_metric(
    summary_df,
    "MAE_mean",
    "MAE_sd",
    "Nested CV MAE",
    "Nested CV tuned model comparison: MAE",
    "Figure_03_nested_tuned_model_comparison_MAE.png"
)


# ============================================================
# 14. Observed vs predicted figures for every model
#     Based only on repeated outer-CV test predictions
# ============================================================

all_mean_prediction_summaries = []
all_oop_metrics = []

for feature_name in feature_order:

    for model_name in model_order:

        this_pred_repeated_df = pred_all_df[
            (pred_all_df["Feature_Set"] == feature_name) &
            (pred_all_df["Model"] == model_name)
        ].copy()

        if this_pred_repeated_df.empty:
            continue

        # Each sample has repeated outer-test predictions.
        # Average them to get one plotted prediction per sample.
        this_pred_df = (
            this_pred_repeated_df
            .groupby(["Row_Position", "Row_Index", "Sample_Name"], as_index=False)
            .agg(
                Observed=("Observed", "first"),
                Predicted_Mean=("Predicted", "mean"),
                Predicted_SD=("Predicted", "std"),
                N_Outer_Test_Predictions=("Predicted", "count")
            )
        )

        this_pred_df["Feature_Set"] = feature_name
        this_pred_df["Model"] = model_name

        all_mean_prediction_summaries.append(this_pred_df)

        outer_test_r2 = r2_score(
            this_pred_df["Observed"],
            this_pred_df["Predicted_Mean"]
        )

        outer_test_rmse = np.sqrt(
            mean_squared_error(
                this_pred_df["Observed"],
                this_pred_df["Predicted_Mean"]
            )
        )

        outer_test_mae = mean_absolute_error(
            this_pred_df["Observed"],
            this_pred_df["Predicted_Mean"]
        )

        all_oop_metrics.append({
            "Feature_Set": feature_name,
            "Model": model_name,
            "Mean_Outer_Test_R2": outer_test_r2,
            "Mean_Outer_Test_RMSE": outer_test_rmse,
            "Mean_Outer_Test_MAE": outer_test_mae,
            "N_Samples": this_pred_df.shape[0],
            "Mean_N_Outer_Test_Predictions": this_pred_df["N_Outer_Test_Predictions"].mean()
        })

        safe_feature_name = (
            feature_name
            .replace("+", "_plus_")
            .replace("/", "_")
            .replace(" ", "_")
        )

        safe_model_name = (
            model_name
            .replace("+", "_plus_")
            .replace("/", "_")
            .replace(" ", "_")
        )

        PRED_FIG_DIR.mkdir(parents=True, exist_ok=True)

        this_pred_df.to_csv(
            PRED_FIG_DIR / f"Pred_{safe_feature_name}_{safe_model_name}.csv",
            index=False
        )

        plt.figure(figsize=(6.5, 6.5))

        plt.scatter(
            this_pred_df["Observed"],
            this_pred_df["Predicted_Mean"],
            alpha=0.75,
            edgecolor="k",
            linewidth=0.3
        )

        min_val = min(
            this_pred_df["Observed"].min(),
            this_pred_df["Predicted_Mean"].min()
        )

        max_val = max(
            this_pred_df["Observed"].max(),
            this_pred_df["Predicted_Mean"].max()
        )

        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            linestyle="--",
            linewidth=1.5
        )

        plt.xlabel("Observed dead-tree fraction")
        plt.ylabel("Mean repeated outer-CV test prediction")
        plt.title(f"Observed vs predicted\n{model_name} | {feature_name}")

        plt.text(
            0.05,
            0.95,
            (
                f"R² = {outer_test_r2:.3f}\n"
                f"RMSE = {outer_test_rmse:.3f}\n"
                f"MAE = {outer_test_mae:.3f}"
            ),
            transform=plt.gca().transAxes,
            va="top",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.8
            )
        )

        plt.tight_layout()

        plt.savefig(
            PRED_FIG_DIR / f"Fig_OvP_{safe_feature_name}_{safe_model_name}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close("all")


all_mean_predictions_df = pd.concat(
    all_mean_prediction_summaries,
    ignore_index=True
)

all_mean_predictions_df.to_csv(
    OUT_DIR / "O1_all_models_mean_repeated_outer_test_predictions.csv",
    index=False
)

outer_prediction_metrics_df = pd.DataFrame(all_oop_metrics)

outer_prediction_metrics_df.to_csv(
    OUT_DIR / "O1_all_models_mean_repeated_outer_test_prediction_metrics.csv",
    index=False
)

print("\nObserved vs predicted figures saved to:")
print(PRED_FIG_DIR)


# ============================================================
# 15. Best-model observed vs predicted table
# ============================================================

best_pred_df = all_mean_predictions_df[
    (all_mean_predictions_df["Feature_Set"] == best_feature_set) &
    (all_mean_predictions_df["Model"] == best_model_name)
].copy()

best_pred_df.to_csv(
    OUT_DIR / "O1_best_model_mean_repeated_outer_test_predictions.csv",
    index=False
)

best_pred_metrics = outer_prediction_metrics_df[
    (outer_prediction_metrics_df["Feature_Set"] == best_feature_set) &
    (outer_prediction_metrics_df["Model"] == best_model_name)
].iloc[0]

mean_outer_test_r2 = best_pred_metrics["Mean_Outer_Test_R2"]
mean_outer_test_rmse = best_pred_metrics["Mean_Outer_Test_RMSE"]
mean_outer_test_mae = best_pred_metrics["Mean_Outer_Test_MAE"]


# ============================================================
# 16. Final tuning on full data for final deployable model
# ============================================================

best_groups = feature_sets[best_feature_set]

best_input_cols = []

for cols in best_groups.values():
    best_input_cols.extend(cols)

best_input_cols = list(dict.fromkeys(best_input_cols))

X_best = X[best_input_cols].copy()

final_selector = GroupWiseSelectKBest(
    groups=best_groups,
    k_per_group=k_grid[0]
)

final_pipe = Pipeline([
    ("selector", final_selector),
    ("model", clone(models_and_params[best_model_name]["model"]))
])

final_search = RandomizedSearchCV(
    estimator=final_pipe,
    param_distributions=models_and_params[best_model_name]["params"],
    n_iter=N_ITER,
    scoring="r2",
    cv=inner_cv,
    n_jobs=-1,
    random_state=999,
    refit=True,
    error_score="raise"
)

final_search.fit(X_best, y)

best_pipe = final_search.best_estimator_
best_final_params = final_search.best_params_

print("\nFinal best parameters fitted on full data:")
print(best_final_params)

selected_features = (
    best_pipe
    .named_steps["selector"]
    .get_feature_names_out()
    .tolist()
)

final_model = best_pipe.named_steps["model"]

joblib.dump(
    best_pipe,
    OUT_DIR / "O1_best_tuned_final_pipeline.joblib"
)


# ============================================================
# 17. Predictor importance for final fitted model
# ============================================================

importance_df = None
group_importance_df = None

if hasattr(final_model, "feature_importances_"):

    importance_df = pd.DataFrame({
        "Predictor": selected_features,
        "Importance": final_model.feature_importances_
    })

    def get_group_name(feature):
        if feature.startswith("CHM"):
            return "CHM"
        elif feature.startswith("SE2025") or feature.startswith("dSE"):
            return "SE"
        else:
            return "RGB-NIR"

    importance_df["Group"] = importance_df["Predictor"].apply(get_group_name)

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    importance_df.to_csv(
        OUT_DIR / "O1_best_tuned_model_predictor_importance.csv",
        index=False
    )

    print("\nTop important predictors:")
    print(importance_df.head(20))

    top_n = min(10, len(importance_df))

    plot_df = (
        importance_df
        .head(top_n)
        .sort_values("Importance", ascending=True)
    )

    # Manuscript colours: CHM = blue, MSI/RGB-NIR = orange, SE = green
    group_colors = {
        "CHM": "#6aaed6",
        "RGB-NIR": "#e7ad72",
        "SE": "#7bd88f"
    }

    plt.figure(figsize=(8, 6))

    plt.barh(
        plot_df["Predictor"],
        plot_df["Importance"],
        color=plot_df["Group"].map(group_colors)
    )

    plt.xlabel("Built-in tree feature importance")
    plt.ylabel("Predictor")
    plt.title(
        f"Top {top_n} important predictors\n"
        f"{best_model_name} | {best_feature_set}"
    )

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
        OUT_DIR / "Figure_05_best_tuned_model_top_10_predictors.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close("all")

    group_importance_df = (
        importance_df
        .groupby("Group", as_index=False)
        .agg(Total_Importance=("Importance", "sum"))
        .sort_values("Total_Importance", ascending=False)
    )

    group_importance_df.to_csv(
        OUT_DIR / "O1_best_tuned_model_group_level_importance.csv",
        index=False
    )

    plt.figure(figsize=(6, 5))

    plt.bar(
        group_importance_df["Group"],
        group_importance_df["Total_Importance"]
    )

    plt.xlabel("Predictor group")
    plt.ylabel("Total built-in feature importance")
    plt.title(
        f"Group-level predictor importance\n"
        f"{best_model_name} | {best_feature_set}"
    )

    plt.tight_layout()

    plt.savefig(
        OUT_DIR / "Figure_06_best_tuned_model_group_level_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close("all")

else:
    print("The selected final model does not support built-in feature importance.")


# ============================================================
# 18. Save best model information
# ============================================================

best_info = pd.DataFrame([{
    "Best_Feature_Set": best_feature_set,
    "Best_Model": best_model_name,

    "Nested_CV_R2_mean": best_row["R2_mean"],
    "Nested_CV_R2_sd": best_row["R2_sd"],

    "Nested_CV_RMSE_mean": best_row["RMSE_mean"],
    "Nested_CV_RMSE_sd": best_row["RMSE_sd"],

    "Nested_CV_MAE_mean": best_row["MAE_mean"],
    "Nested_CV_MAE_sd": best_row["MAE_sd"],

    "Mean_Repeated_Outer_Test_R2": mean_outer_test_r2,
    "Mean_Repeated_Outer_Test_RMSE": mean_outer_test_rmse,
    "Mean_Repeated_Outer_Test_MAE": mean_outer_test_mae,

    "Final_Best_Params_JSON": json.dumps(best_final_params, default=str),

    "Selected_Features": ", ".join(selected_features),
    "N_Selected_Features": len(selected_features),

    "Final_Pipeline_File": "O1_best_tuned_final_pipeline.joblib"
}])

best_info.to_csv(
    OUT_DIR / "O1_best_tuned_model_information.csv",
    index=False
)


# ============================================================
# 19. Finish
# ============================================================

print("\nDone. Nested hyperparameter tuning completed.")
print("\nMain output folder:")
print(OUT_DIR)

print("\nObserved-vs-predicted figures:")
print(PRED_FIG_DIR)

print("\nImportant files:")
print("1. O1_nested_hyperparameter_tuning_summary.csv")
print("2. O1_nested_hyperparameter_tuning_all_outer_test_predictions.csv")
print("3. O1_all_models_mean_repeated_outer_test_predictions.csv")
print("4. O1_all_models_mean_repeated_outer_test_prediction_metrics.csv")
print("5. O1_best_tuned_model_information.csv")
print("6. O1_best_tuned_final_pipeline.joblib")


# ============================================================
# Diagnostic section:
# Built-in tree feature_importances_ from the best RF, XGBoost,
# and GradientBoosting models.
#
# These outputs are retained only as model diagnostics.
# Manuscript predictor-importance results should use SHAP outputs
# from Objective 4B below, not these built-in importances.
# ============================================================

MODEL_IMPORTANCE_DIR = OUT_DIR / "ImpPred_BestByAlgorithm"
MODEL_IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)

all_algorithm_importances = []
all_algorithm_best_info = []

for algorithm_name in ["RF", "XGBoost", "GradientBoosting"]:

    print("\n=====================================")
    print("Final importance analysis for:", algorithm_name)
    print("=====================================")

    # --------------------------------------------------------
    # 1. Find best feature set for this algorithm
    # --------------------------------------------------------

    algorithm_summary = summary_df[
        summary_df["Model"] == algorithm_name
    ].copy()

    if algorithm_summary.empty:
        print(f"No results found for {algorithm_name}. Skipping.")
        continue

    algorithm_best_row = algorithm_summary.sort_values(
        "R2_mean",
        ascending=False
    ).iloc[0]

    algorithm_best_feature_set = algorithm_best_row["Feature_Set"]

    print("Best feature set:", algorithm_best_feature_set)
    print("Nested CV RMSE:", algorithm_best_row["RMSE_mean"])
    print("Nested CV R2:", algorithm_best_row["R2_mean"])
    print("Nested CV MAE:", algorithm_best_row["MAE_mean"])

    # --------------------------------------------------------
    # 2. Prepare predictors for this model's best feature set
    # --------------------------------------------------------

    algorithm_best_groups = feature_sets[algorithm_best_feature_set]

    algorithm_input_cols = []

    for cols in algorithm_best_groups.values():
        algorithm_input_cols.extend(cols)

    algorithm_input_cols = list(dict.fromkeys(algorithm_input_cols))

    X_algorithm = X[algorithm_input_cols].copy()

    # --------------------------------------------------------
    # 3. Final tuning on full data for this algorithm
    # --------------------------------------------------------

    algorithm_selector = GroupWiseSelectKBest(
        groups=algorithm_best_groups,
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
        random_state=2025,
        refit=True,
        error_score="raise"
    )

    algorithm_search.fit(X_algorithm, y)

    algorithm_best_pipe = algorithm_search.best_estimator_
    algorithm_best_params = algorithm_search.best_params_

    selected_features_algorithm = (
        algorithm_best_pipe
        .named_steps["selector"]
        .get_feature_names_out()
        .tolist()
    )

    final_algorithm_model = algorithm_best_pipe.named_steps["model"]

    # Save final fitted pipeline for this algorithm
    safe_algorithm_name = algorithm_name.replace(" ", "_").replace("+", "_plus_")
    safe_feature_set_name = (
        algorithm_best_feature_set
        .replace(" ", "_")
        .replace("+", "_plus_")
        .replace("/", "_")
    )

    joblib.dump(
        algorithm_best_pipe,
        MODEL_IMPORTANCE_DIR / f"Final_pipeline_best_{safe_algorithm_name}_{safe_feature_set_name}.joblib"
    )

    all_algorithm_best_info.append({
        "Model": algorithm_name,
        "Best_Feature_Set": algorithm_best_feature_set,
        "Nested_CV_R2_mean": algorithm_best_row["R2_mean"],
        "Nested_CV_R2_sd": algorithm_best_row["R2_sd"],
        "Nested_CV_RMSE_mean": algorithm_best_row["RMSE_mean"],
        "Nested_CV_RMSE_sd": algorithm_best_row["RMSE_sd"],
        "Nested_CV_MAE_mean": algorithm_best_row["MAE_mean"],
        "Nested_CV_MAE_sd": algorithm_best_row["MAE_sd"],
        "Final_Best_Params_JSON": json.dumps(algorithm_best_params, default=str),
        "Selected_Features": ", ".join(selected_features_algorithm),
        "N_Selected_Features": len(selected_features_algorithm),
        "Saved_Pipeline": f"Final_pipeline_best_{safe_algorithm_name}_{safe_feature_set_name}.joblib"
    })

    # --------------------------------------------------------
    # 4. Extract built-in feature importance
    # --------------------------------------------------------

    if not hasattr(final_algorithm_model, "feature_importances_"):
        print(f"{algorithm_name} does not support built-in feature_importances_.")
        continue

    importance_algorithm_df = pd.DataFrame({
        "Model": algorithm_name,
        "Best_Feature_Set": algorithm_best_feature_set,
        "Predictor": selected_features_algorithm,
        "Importance": final_algorithm_model.feature_importances_
    })

    def get_group_name(feature):
        if feature.startswith("CHM"):
            return "CHM"
        elif feature.startswith("SE2025") or feature.startswith("dSE"):
            return "SE"
        else:
            return "RGB-NIR"

    importance_algorithm_df["Group"] = importance_algorithm_df["Predictor"].apply(
        get_group_name
    )

    importance_algorithm_df = importance_algorithm_df.sort_values(
        "Importance",
        ascending=False
    )

    all_algorithm_importances.append(importance_algorithm_df)

    # Save individual importance table
    importance_algorithm_df.to_csv(
        MODEL_IMPORTANCE_DIR / f"Predictor_importance_best_{safe_algorithm_name}_{safe_feature_set_name}.csv",
        index=False
    )

    print("\nTop 20 important predictors:")
    print(importance_algorithm_df.head(20))

    # --------------------------------------------------------
    # 5. Plot top 10 predictors for this algorithm
    # --------------------------------------------------------

    top_n = min(10, len(importance_algorithm_df))

    plot_df = (
        importance_algorithm_df
        .head(top_n)
        .sort_values("Importance", ascending=True)
    )

    # Manuscript colours: CHM = blue, MSI/RGB-NIR = orange, SE = green
    group_colors = {
        "CHM": "#6aaed6",
        "RGB-NIR": "#e7ad72",
        "SE": "#7bd88f"
    }

    plt.figure(figsize=(8, 6))

    plt.barh(
        plot_df["Predictor"],
        plot_df["Importance"],
        color=plot_df["Group"].map(group_colors)
    )

    plt.xlabel("Built-in tree feature importance")
    plt.ylabel("Predictor")
    plt.title(
        f"Top {top_n} important predictors\n"
        f"{algorithm_name} | {algorithm_best_feature_set}"
    )

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
        MODEL_IMPORTANCE_DIR / f"Figure_top_{top_n}_predictors_best_{safe_algorithm_name}_{safe_feature_set_name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close("all")

    # --------------------------------------------------------
    # 6. Group-level importance for this algorithm
    # --------------------------------------------------------

    group_importance_algorithm_df = (
        importance_algorithm_df
        .groupby(["Model", "Best_Feature_Set", "Group"], as_index=False)
        .agg(Total_Importance=("Importance", "sum"))
        .sort_values("Total_Importance", ascending=False)
    )

    group_importance_algorithm_df.to_csv(
        MODEL_IMPORTANCE_DIR / f"Group_importance_best_{safe_algorithm_name}_{safe_feature_set_name}.csv",
        index=False
    )

    plt.figure(figsize=(6, 5))

    plt.bar(
        group_importance_algorithm_df["Group"],
        group_importance_algorithm_df["Total_Importance"]
    )

    plt.xlabel("Predictor group")
    plt.ylabel("Total built-in feature importance")
    plt.title(
        f"Group-level predictor importance\n"
        f"{algorithm_name} | {algorithm_best_feature_set}"
    )

    plt.tight_layout()

    plt.savefig(
        MODEL_IMPORTANCE_DIR / f"Figure_group_importance_best_{safe_algorithm_name}_{safe_feature_set_name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close("all")


# ============================================================
# Save combined importance outputs for all three best models
# ============================================================

if len(all_algorithm_importances) > 0:

    all_algorithm_importance_df = pd.concat(
        all_algorithm_importances,
        ignore_index=True
    )

    all_algorithm_importance_df.to_csv(
        MODEL_IMPORTANCE_DIR / "All_best_algorithm_predictor_importances.csv",
        index=False
    )

    print("\nCombined predictor importance table saved.")

else:
    print("\nNo predictor importance tables were created.")


if len(all_algorithm_best_info) > 0:

    all_algorithm_best_info_df = pd.DataFrame(all_algorithm_best_info)

    all_algorithm_best_info_df.to_csv(
        MODEL_IMPORTANCE_DIR / "Best_model_information_by_algorithm.csv",
        index=False
    )

    print("\nBest model information by algorithm saved.")


print("\nImportant predictor outputs saved to:")
print(MODEL_IMPORTANCE_DIR)


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

try:
    import shap
except ImportError as exc:
    raise ImportError(
        "The shap package is required for Objective 4B. Install it with: pip install shap"
    ) from exc

O4_DIR = OUT_DIR / "O4_SHAP_and_complementarity"
O4_DIR.mkdir(parents=True, exist_ok=True)

ALGO_SHAP_DIR = O4_DIR / "O4_algorithm_specific_SHAP"
ALGO_SHAP_DIR.mkdir(parents=True, exist_ok=True)

print("Algorithm-specific SHAP outputs will be saved to:", ALGO_SHAP_DIR)

pipeline_dir = OUT_DIR / "ImpPred_BestByAlgorithm"


def clean_model_name_for_plot(model_name):
    if model_name == "GradientBoosting":
        return "GB"
    return model_name


def get_group_name_for_shap(feature):
    if str(feature).startswith("CHM"):
        return "CHM"
    elif str(feature).startswith("SE2025") or str(feature).startswith("dSE"):
        return "SE"
    else:
        return "RGB-NIR"


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
    If it is not available, refit the CHM+RGBNIR+SE pipeline on the full dataset.
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

    X_selected_algorithm = selector.transform(X)
    X_selected_algorithm = pd.DataFrame(
        X_selected_algorithm,
        columns=selected_features_algorithm,
        index=X.index
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

    shap_algorithm_df["Group"] = shap_algorithm_df["Predictor"].apply(get_group_name_for_shap)

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

def clean_plot_group(group):
    if group in ["RGB-NIR", "RGBNIR", "RGB_NIR", "MSI"]:
        return "MSI"
    return group


group_colors_plot = {
    "CHM": "#6aaed6",
    "MSI": "#e7ad72",
    "SE": "#7bd88f",
}

all_algorithm_shap_df["Plot_Group"] = all_algorithm_shap_df["Group"].apply(clean_plot_group)
overall_shap_df["Plot_Group"] = overall_shap_df["Group"].apply(clean_plot_group)

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


