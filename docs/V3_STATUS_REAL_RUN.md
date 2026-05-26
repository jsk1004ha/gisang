# V3 Real-Run Status

Date: 2026-05-25

## Verification performed

```bash
PYTHONPATH=src .venv312/Scripts/python.exe -m pytest -q
PYTHONPATH=src .venv312/Scripts/python.exe -m compileall -q src tests
find configs/v3/experiments -name "*.yaml"
PYTHONPATH=src .venv312/Scripts/python.exe -m weather_korea_forecast.reporting.generate_report \
  --experiments-root data/artifacts --output-dir reports --title "Gisang V1-V3 Experiment Report"
```

Initial baseline before this update: `53 passed, 56 warnings`; compileall passed.
Final verification is recorded in the task report.

## Existing V3 configs

- `diagnostic/v3_humidity_observed_oracle_decoder_feature_72to24.yaml`
- `diagnostic/v3_temp_observed_oracle_decoder_feature_72to24.yaml`
- `v3_humidity_dewpoint_depression_lgbm_72to24.yaml`
- `v3_humidity_dewpoint_depression_lgbm_grid_72to24.yaml`
- `v3_humidity_dewpoint_lgbm_72to24.yaml`
- `v3_humidity_dewpoint_lgbm_grid_72to24.yaml`
- `v3_humidity_direct_lgbm_72to24.yaml`
- `v3_humidity_nwp_mos_lgbm_72to24.yaml`
- `v3_humidity_openmeteo_ecmwf_mos_lgbm_72to24.yaml`
- `v3_temp_mos_ensemble_72to24.yaml`
- `v3_temp_mos_horizonwise_residual_lgbm_72to24.yaml`
- `v3_temp_mos_horizonwise_residual_ridge_168to24.yaml`
- `v3_temp_mos_horizonwise_residual_ridge_72to24.yaml`
- `v3_temp_mos_residual_catboost_72to24.yaml`
- `v3_temp_mos_residual_lgbm_168to24.yaml`
- `v3_temp_mos_residual_lgbm_72to24.yaml`
- `v3_temp_mos_residual_lgbm_grid_72to24.yaml`
- `v3_temp_mos_residual_ridge_168to24.yaml`
- `v3_temp_mos_residual_ridge_72to24.yaml`
- `v3_temp_observation_only_horizonwise_ridge_168to24.yaml`

## Implemented and executable

- Temperature residual MOS Ridge/LightGBM configs through `weather_korea_forecast.v2.train`.
- Horizon-wise Ridge/LightGBM configs through the same V2 train path; horizon summary and feature importance artifacts are exported.
- Artifact-level ensemble config through `weather_korea_forecast.v2.train` when the config contains only an `ensemble:` section.
- Humidity direct/dew-point/dew-point-depression LightGBM configs through `weather_korea_forecast.v2.train`.
- Humidity NWP-MOS issue-time aligned config through `weather_korea_forecast.v3.nwp_mos`.
- Prepared forecast CSV adapter via `load_future_weather_features(source="prepared_forecast_csv", ...)`.
- Unified reporting CLI under `weather_korea_forecast.reporting.generate_report`.

## Skeleton or optional areas

- GFS/ECMWF/KMA live downloaders are not complete operational production downloaders; the stable supported operational handoff is prepared forecast CSV.
- CatBoost experiments require the optional `catboost` dependency.
- Observation-only V3 improvement has a runnable config, but it is not the primary tuned track yet.
- Full RMSE <= 1.0°C confirmation depends on successful real-data training runs and should be read from `reports/experiment_summary.csv` / `docs/V3_REALDATA_RESULTS.md`.

## Next execution order

Use the runner scripts:

```bash
scripts/run_v3_temp_mos_experiments.sh
scripts/run_v3_humidity_experiments.sh
scripts/run_v3_all_core_experiments.sh
```

Core manual order:

1. `v3_temp_mos_residual_ridge_72to24`
2. `v3_temp_mos_residual_ridge_168to24`
3. `v3_temp_mos_horizonwise_residual_ridge_72to24`
4. `v3_temp_mos_horizonwise_residual_ridge_168to24`
5. `v3_temp_mos_residual_lgbm_72to24`
6. `v3_temp_mos_residual_lgbm_168to24`
7. `v3_temp_mos_ensemble_72to24`
8. `v3_humidity_dewpoint_lgbm_72to24`
9. `v3_humidity_dewpoint_depression_lgbm_72to24`
10. `v3_humidity_nwp_mos_lgbm_72to24`

Each script logs to `logs/v3_runs/`, continues past failed experiments, prints RMSE/MAE/Bias when available, and regenerates the report.
