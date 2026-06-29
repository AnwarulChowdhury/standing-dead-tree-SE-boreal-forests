# Reproducibility notes

- The modelling workflow can be reproduced from `data/analysis_ready_predictor_matrix.csv`.
- Raw raster extraction is split into CHM extraction and MSI/SE extraction.
- CHM extraction uses partial pixel-overlap weighting and a per-plot fallback rule: `uusin` first, then the newest available annual CHM layer.
- MSI and SE extraction uses the same proportional pixel-overlap weighting principle for plot-level summaries.
- Generated outputs are written to `results/` and are not tracked by default.
