# Changelog

## 2026-05-10

### Added
- Added V2 `horizon_wise_ridge` with independent ridge heads, horizon-specific alpha search, and `horizon_model_metrics.csv`.
- Added explicit `horizon_wise_lightgbm` and a generic residual forecasting wrapper that trains `baseline + residual` models and writes component predictions.
- Added 168h encoder, horizon-wise ridge, horizon-wise LightGBM, residual LightGBM-on-ridge, station-wise scaling, and region-wise scaling temperature configs.
- Added V2 daily max/min temperature metrics, diurnal range metrics, horizon-station heatmap, station/region bar plots, extreme-temperature scatter, and worst-case summary artifacts.
- Added `docs/V2_STATUS.md` to record the current temperature baseline, diagnosis, experiment order, artifacts, and limitations.
- Added leakage-oriented synthetic tests for lag/rolling/delta features and train-only group scaling.

### Changed
- V2 temp ridge is documented as the official 72->24 temperature baseline and keeps alpha-grid selection aligned with guarded holdout bias correction.
- V2 bias correction supports additional grouped modes (`per_station`, `per_season_horizon`, `per_region_horizon`) and quantile correction candidates.
- V2 scaler supports `stationwise` and `regionwise` aliases and persists group statistics for station/region normalization.
- V2 leaderboard rows now include scaling mode, station count, train/val/test periods, raw/corrected metrics, and best/worst horizons.

### Known Limitations
- Residual forecasting does not yet build true `pytorch_forecasting` residual datasets; use fallback/baseline-style residual learners for now.

## 2026-03-26

### Added
- V2 single-target experiment pipeline under `src/weather_korea_forecast/v2/`.
- Direct multi-horizon dataset flow for station-level `24h` forecasting.
- Geographic metadata support with `lat`, `lon`, `elevation`, `coastal_distance_km`, and `region_class`.
- Stronger V2 baselines: `persistence`, `seasonal_persistence`, `ridge`, and optional `lightgbm`.
- V2 artifact and leaderboard management with config snapshot, scaler metadata, summaries, plots, and alias directories.
- V2 synthetic regression coverage in `tests/test_v2_pipeline.py`.
- Local real-data bootstrap configs under `configs/v2/experiments/real/` for Seoul `Q4 2024 -> Q1 2025` smoke runs.
- KMA station metadata downloader and multi-station raw-data bootstrap configs for V2.
- KMA station metadata downloader now supports a public DataWiki source so multi-station setup does not depend on API Hub auth.
- Multi-station ERA5 bootstrap now uses split `2024Q4` and `2025Q1` download configs plus a concat utility to stay under CDS cost limits.

### Changed
- README now documents V2 single-target temp/humidity strategy, artifact structure, raw vs corrected metrics, and real-data bootstrap execution.
- `docs/V2_plan.md` now records the implemented V2 scope, remaining gaps, and prioritized follow-up items.
- V2 evaluation now exports raw/corrected breakdowns, rolling-origin slice reports, worst-case samples, and additional plots.
- Humidity V2 configs now include dew-point-derived features, humidity-specific lag/rolling/delta features, and prediction clipping.
- V2 LightGBM prediction now keeps consistent feature names to avoid repeated sklearn warning noise during evaluation and inference.
- V2 leaderboard writing now also emits per-target leaderboard files such as `leaderboard_temp.csv` and `leaderboard_humidity.csv`.
- V2 multi-station LightGBM default configs were reduced to lighter tree settings so the direct 24h baseline completes in practical runtime on the current environment.
- V2 multi-station LightGBM configs now also set `n_jobs` explicitly for better runtime on CPU-heavy direct multi-horizon runs.
- The `fallback_torch` TFT substitute now supports residual forecasting from recent observed targets and applies configured gradient clipping during training.
- ASOS/AWS observation merging now supports `prefer_columns` so supplementary AWS variables can improve feature quality without replacing the primary ASOS target column.
- Forecast plots now use a headless matplotlib backend for stable training/evaluation tests in GUI-less environments.
- V2 bias correction now supports calibration/holdout selection so correction is applied only when it improves a held-out validation slice.
- V2 bias correction now supports guarded affine calibration, and ridge baselines can select `alpha` from an `alpha_grid` by validation loss.
- V2 default correction now uses guarded station+horizon mean-bias calibration after evaluation showed it lowered temp ridge RMSE more than the prior horizon-only correction.
- V2 temp ridge now supports bias-corrected holdout `alpha_grid` selection, and the default temp ridge config is strengthened to the best high-regularization alpha from the latest local sweep, lowering the temp benchmark RMSE further.

### Known Limitations
- The canonical multi-station V2 benchmark still depends on user-provided local ASOS/ERA5/metadata files.
- True-TFT-specific residual variants and true rolling-origin retraining remain follow-up items.
