# V2 Plan

## Current V1 Status

### Implemented in V1

- Observation ingestion for ASOS and optional AWS/local CSV supplementation.
- ERA5 extraction at stations with `nearest` and `bilinear` modes.
- UTC-normalized training table generation.
- Sliding-window dataset generation for direct sequence modeling.
- Baselines: `persistence`, `seasonal_persistence`, `ridge`.
- TFT wrapper with `auto` backend resolution.
- Train / evaluate / predict CLIs with experiment artifacts.
- `latest/` and `best/` experiment aliases.

### Gaps That V2 Must Address

- V1 is still oriented around prototype-scale station runs.
- V1 defaults are not organized around single-target temp/humidity experiments.
- Feature engineering is still light for production-like weather forecasting.
- Experiment tracking exists, but V2 needs stricter summaries and leaderboard updates.
- TFT exists, but V2 should treat real TFT as the main sequence path.

## V2 Scope

- Preserve V1 behavior.
- Add a separate V2 pipeline under `src/weather_korea_forecast/v2/`.
- Use unified V2 experiment configs under `configs/v2/experiments/`.
- Default to direct `24h` forecasting and single-target experiments.
- Support multi-station inputs with station metadata and geographic features.
- Preserve config-driven ASOS primary plus optional supplementary observation sources; `prefer_columns` can prioritize selected AWS variables without replacing the primary target column.

## Files That Needed Refactoring

- `src/weather_korea_forecast/data/station_metadata.py`
- `src/weather_korea_forecast/features/geo_features.py`
- `src/weather_korea_forecast/models/baselines.py`
- `src/weather_korea_forecast/models/registry.py`
- `src/weather_korea_forecast/models/tft_model.py`
- `pyproject.toml`

## New V2 Files

- `src/weather_korea_forecast/v2/data.py`
- `src/weather_korea_forecast/v2/dataset.py`
- `src/weather_korea_forecast/v2/scaling.py`
- `src/weather_korea_forecast/v2/train.py`
- `src/weather_korea_forecast/v2/evaluate.py`
- `src/weather_korea_forecast/v2/predict.py`
- `src/weather_korea_forecast/v2/prepare_data.py`
- `src/weather_korea_forecast/v2/artifacts.py`
- `configs/v2/experiments/*.yaml`
- `configs/v2/templates/station_metadata_template.csv`
- `tests/test_v2_pipeline.py`
- `scripts/run_v2_*.sh`

## V2 Data Flow

```text
observations
-> UTC normalization
-> station metadata join
-> ERA5 join/extraction
-> time features
-> lag / rolling / delta features
-> split assignment
-> train-only scaler fit
-> direct multi-horizon datasets
-> train / evaluate / predict
```

## V2 Defaults

- Single-target experiments:
  - `v2_temp_*`
  - `v2_humidity_*`
- Direct forecast horizon: `24h`
- Recommended encoder lengths:
  - `72h`
  - `168h`
- Geographic static features:
  - `lat`
  - `lon`
  - `elevation`
  - `coastal_distance_km`
  - `region_class`

## Standard V2 Artifacts

- `experiment_config.yaml`
- `model.pt`
- `scaler.json`
  - supports `global`, `station_wise`/`stationwise`, `region_wise`/`regionwise`, and `none`
  - group-wise modes store train-split-only group means/stds
- `training_history.json`
- `predictions_test.csv`
- `metrics_test.json`
- `metrics_summary.json`
- `metrics_target_name.csv`
- `metrics_target_name_horizon_step.csv`
- `metrics_target_name_station_id.csv`
- `metrics_target_name_region.csv`
- `metrics_target_name_season.csv`
- `metrics_raw_target_name.csv`
- `metrics_raw_target_name_horizon_step.csv`
- `metrics_raw_target_name_station_id.csv`
- `metrics_raw_target_name_region.csv`
- `metrics_raw_target_name_season.csv`
- `forecast_vs_actual.png`
- `horizon_error.png`
- `prediction_scatter.png`
- `raw_vs_corrected.png`
- `horizon_station_heatmap.png`
- `station_rmse_bar.png`
- `region_rmse_bar.png`
- `daily_max_min_error.png`
- `extreme_temperature_scatter.png`
- `bias_correction.json`
  - includes calibration/holdout selection evidence when `apply_when: improves_on_holdout`
  - defaults to station+horizon guarded mean-bias correction for multi-station V2 configs
  - supports `method: affine` for guarded horizon-wise `slope * prediction + intercept` calibration
  - supports additional modes: `per_station`, `per_season_horizon`, `per_region_horizon`
  - supports `method: quantile` for empirical quantile correction candidates
  - supports `mode: auto` / `method: auto` candidate comparison for research runs
- Ridge configs can set `model.alpha_selection.metric: bias_corrected_holdout_mse` so `alpha_grid` is selected against the same guarded station+horizon correction objective used in evaluation, instead of raw validation MSE only.
- `worst_case_samples.csv`
- `worst_case_summary.json`
- `feature_importance.csv` for supported baselines
- `horizon_model_metrics.csv` for horizon-wise ridge/lightgbm and residual components
- `predictions_test_components.csv` for residual experiments
- `daily_temperature_errors.csv`
- `metrics_daily_temperature.csv`
- `metrics_target_name_rolling_origin_fold.csv`
- `experiment_summary.json`
- `experiment_summary.md`
- `leaderboard.csv`
- `leaderboard_<target_name>.csv`

## Remaining Follow-ups

- V2 temperature now has a strong official ridge baseline; follow-up experiments should compare against the corrected RMSE/MAE/Bias leaderboard row rather than treating persistence as the main target.
- `horizon_wise_ridge` is available for horizon-specific intercept/alpha learning, with `72->24` and `168->24` configs.
- The generic `residual` wrapper supports baseline + residual learner experiments for baseline/ridge/lightgbm/fallback-torch style models and writes component predictions.
- `fallback_torch` now has a residual shortcut from recent observed targets; true-pytorch-forecasting residual dataset support remains a follow-up item.
- Rolling-origin reporting exists, but true rolling-origin retraining is not yet the default path.
- The canonical 12-20 station benchmark still depends on user-provided local ASOS/ERA5 data files.
- Local real-data bootstrap configs currently cover Seoul smoke runs, not the full multi-station benchmark.
- Multi-station setup now has config-driven raw data bootstrap files, but still depends on successful KMA/CDS downloads in the user environment.
