# Scripts

The repository separates predictor extraction from modelling analysis.

## Predictor extraction scripts

| Script | Purpose |
|---|---|
| `01a_extract_chm_predictors.py` | Extract CHM predictors from Finnish Forest Centre Latvusmalli data using per-plot fallback and partial-pixel weighting. |
| `01b_extract_msi_predictors.py` | Extract aerial MSI/RGB-NIR predictors and vegetation-index summaries. |
| `01c_extract_se_predictors.py` | Extract AlphaEarth 2025 embedding predictors and temporal-change dSE predictors. |
| `01d_build_analysis_matrix.py` | Merge CHM, MSI, SE, and standing dead-tree fraction into one analysis-ready matrix. |

## Analysis scripts

| Script | Purpose |
|---|---|
| `02_model_nested_cv.py` | Run RF, GB, and XGBoost nested CV model comparison. |
| `03_added_value_SE.py` | Compare models with and without SE predictors. |
| `04_training_sample_size.py` | Run training-size / label-efficiency analysis. |
| `05_shap_analysis.py` | Run SHAP interpretation and complementarity analyses. |
| `06_condition_specific_analysis.py` | Run condition-specific SE-benefit analysis. |
| `07_make_final_figures_tables.py` | Create manuscript-ready figures, tables, and collected outputs. |
| `08_spatial_sampling_summary.py` | Create plot-count and spatial-distance summaries from the private plot/study-area geometry file. |
| `run_all_analysis.py` | Execute scripts `02` through `07` in order. |
| `00_full_analysis_workflow.py` | Original complete workflow retained for traceability. |

For reproducing the manuscript analyses from the public data table, run:

```bash
python scripts/run_all_analysis.py --data data/analysis_ready_predictor_matrix.csv --out-root results/REANALYSIS --figures final --overwrite
```


## Optional spatial sampling script

`08_spatial_sampling_summary.py` is not run by `run_all_analysis.py` because it requires private plot/study-area geometry. It writes two CSV files: `study_area_plot_counts.csv` and `spatial_sampling_summary.csv`.

Example:

```bash
python scripts/08_spatial_sampling_summary.py --input path/to/study_area.zip --name-col Name --out-dir results/08_spatial_sampling
```
