# V2 Status

## Current official temperature baseline

`v2_temp_ridge` is the official V2 temperature baseline for the 72h encoder -> 24h direct forecast track.

Latest local benchmark summary:

| metric | raw | corrected |
| --- | ---: | ---: |
| RMSE | 2.662 | 2.511 |
| MAE | 1.929 | 1.844 |
| Bias | -0.728 | -0.224 |

Bias correction is therefore part of the baseline contract. Temperature model comparisons should focus on corrected RMSE, MAE, Bias, and horizon-wise RMSE/MAE/Bias. MAPE is retained for compatibility, but it is not a primary temperature metric because near-zero or negative temperatures distort it.

## Main diagnosis

- The 1-6h horizons are relatively stable.
- Error and under-forecast bias grow toward 12-24h.
- Ridge is a strong linear baseline but compresses extremes toward the mean.
- The next useful improvements should reduce late-horizon bias and restore high/low temperature extremes without losing the ridge baseline's stability.

## Implemented V2 experiment surfaces

- `ridge`: global direct multi-horizon ridge baseline.
- `horizon_wise_ridge`: independent ridge head and alpha search for each horizon/target output.
- `lightgbm`: optional multi-output LightGBM baseline.
- `horizon_wise_lightgbm`: explicit horizon-wise LightGBM alias with horizon metric artifacts.
- `residual`: trains a baseline model, then trains a residual model on `actual - baseline`; final prediction is `baseline + residual`.

New temperature configs include:

- `configs/v2/experiments/v2_temp_ridge_168to24.yaml`
- `configs/v2/experiments/v2_temp_horizonwise_ridge_72to24.yaml`
- `configs/v2/experiments/v2_temp_horizonwise_ridge_168to24.yaml`
- `configs/v2/experiments/v2_temp_horizonwise_lgbm_168to24.yaml`
- `configs/v2/experiments/v2_temp_residual_lgbm_on_ridge_168to24.yaml`
- `configs/v2/experiments/v2_temp_ridge_168to24_stationwise.yaml`
- `configs/v2/experiments/v2_temp_ridge_168to24_regionwise.yaml`

## Scaling

`data.scaling.mode` supports:

- `global`
- `stationwise` / `station_wise`
- `regionwise` / `region_wise`
- `none`

Group-wise modes fit means/stds on the train split only and persist them in `scaler.json` as `group_means` and `group_stds`.

## Evaluation artifacts

V2 now writes the original metric breakdowns plus:

- `horizon_model_metrics.csv`
- `predictions_test_components.csv` for residual experiments
- `worst_case_summary.json`
- `daily_temperature_errors.csv`
- `metrics_daily_temperature.csv`
- `horizon_station_heatmap.png`
- `station_rmse_bar.png`
- `region_rmse_bar.png`
- `daily_max_min_error.png`
- `extreme_temperature_scatter.png`

The leaderboard includes scaling mode, number of stations, train/val/test periods, raw/corrected metrics, and best/worst horizons.

## Recommended next experiment order

1. Freeze `v2_temp_ridge` as the 72->24 baseline.
2. Compare `v2_temp_ridge_168to24` against 72->24.
3. Compare `v2_temp_horizonwise_ridge_72to24` and `v2_temp_horizonwise_ridge_168to24`.
4. Compare global vs station-wise vs region-wise scaling on the 168->24 ridge track.
5. Run `v2_temp_horizonwise_lgbm_168to24` if LightGBM is installed.
6. Run `v2_temp_residual_lgbm_on_ridge_168to24` to test nonlinear residual gains.
7. Use horizon/station/region/daily max-min artifacts to decide whether to add residual TFT or an ensemble next.

## Known limitations

- True `pytorch_forecasting` residual datasets are not implemented yet; residual experiments currently support baseline/fallback-style learners.
- Canonical multi-station benchmark quality still depends on local ASOS/ERA5/metadata coverage.
- Rolling-origin reporting exists, but retraining per origin is still future work.
