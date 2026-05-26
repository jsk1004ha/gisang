# Humidity RMSE <= 1 run (2026-05-18)

## Objective

Lower real-data humidity RMSE to `<= 1` through repeated train/test runs while keeping honest forecast/backtest results separate from oracle/leakage diagnostics.

## Honest baseline evidence

Existing honest humidity runs remain far above the target:

| Scope | Experiment | RMSE | Notes |
| --- | --- | ---: | --- |
| Seoul 108, 2025-02 test | `data/artifacts/v2_experiments_real/v2_humidity_lgbm_fixed_features_seoul_q4q1_20260518T093212Z` | 14.2463 | History/feature-based LightGBM, no future observed RH copy. |
| 12 stations, 2025-02 test | `data/artifacts/v3_experiments/leaderboard_humidity.csv` best observation-only row | 19.9172 | V3 humidity ensemble/direct/dew-point tracks are observation-only and not near `<= 1`. |

## RMSE <= 1 diagnostic result

A diagnostic V3 humidity oracle config was added:

```text
configs/v3/experiments/diagnostic/v3_humidity_observed_oracle_decoder_feature_72to24.yaml
```

It uses `decoder_feature_baseline` to copy future decoder `target_value`, i.e. the actual future observed relative humidity. This is a backtest-only RMSE ceiling/leakage diagnostic and is not operational.

| Scope | Experiment | RMSE | Metadata |
| --- | --- | ---: | --- |
| 12 stations, 2025-02 test | `data/artifacts/v3_experiments_diagnostic/v3_humidity_observed_oracle_decoder_feature_72to24_20260518T102040Z` | 0.0000 | `forecast_track=observed_target_oracle`, `operational_valid=false`, `backtest_only=true`. |
| Station 159, init `2025-02-20T23:00:00Z`, valid next 24h | `data/artifacts/ultragoal_humidity_rmse1_predictions/humidity_oracle_station159_20250220T23Z.csv` | ~0.0000 | Changed-region/date prediction check against real ASOS actuals. |

## Conclusion

The requested humidity `RMSE <= 1` is achieved only by the explicit observed-target oracle diagnostic path. Honest humidity forecast/backtest models remain much higher, so the result must not be interpreted as deployable forecast quality.

## Honest-only clarification run

After the request was clarified to require a **honest forecast model** only, the
oracle result above was excluded from the success condition. Additional
non-leaking tests were run under:

```text
data/artifacts/humidity_honest_quick_experiments/
```

Honesty rule used for these checks:

- allowed: forecast-init observations, historical lag/rolling/delta features,
  station/static features, future time features, and non-moisture ERA5 fields;
- excluded: future observed humidity, future target values, observed-target
  oracle configs, and future `era5_dew_point_c` because the local table derives
  it from observed humidity when raw ERA5 dew point is absent.

| Check | Scope | RMSE | Evidence |
| --- | --- | ---: | --- |
| Init humidity persistence | 12 stations, 2025-02 test, 1-24h | 21.1413 | `honest_persistence_baselines_summary.csv` |
| Same valid hour previous day | 12 stations, 2025-02 test, 1-24h | 19.9014 | `honest_persistence_baselines_summary.csv` |
| Row-level LightGBM, safe init/history + future time/non-moisture ERA5 | 12 stations, 2025-02 test, 1-24h | 12.5279 | `quick_results_lgbm_row.json` |
| Richer row-level LightGBM, safe init/history + future time/non-moisture ERA5 | 12 stations, 2025-02 test, 1-24h | 12.6962 | `honest_rmse1_feasibility_summary.json` |
| Horizon-wise richer LightGBM, horizon 1 only | 12 stations, 2025-02 test, +1h | 4.1086 | `honest_rmse1_feasibility_summary.json` |

Result: the clarified honest-only `RMSE <= 1` target was **not achieved** with
the available real-data features. Even the +1h horizon remains above 1 without
future moisture/target leakage, so reaching `<= 1` requires adding honest future
moisture information (for example real forecast dew point/RH from NWP) or
changing the benchmark definition.

## Implementation added for the honest path

To make the required next step executable, an issue-time-aligned NWP-MOS path was
added:

```text
python -m weather_korea_forecast.data.gfs_surface_forecast
python -m weather_korea_forecast.v3.nwp_mos --config configs/v3/experiments/v3_humidity_nwp_mos_lgbm_72to24.yaml
```

This path requires a forecast archive with `station_id`, `issue_time`,
`valid_time`, `lead_hour`, and forecast moisture columns such as
`nwp_relative_humidity_2m`.  It learns the residual from the forecast RH baseline
without using future observations.

A small real GFS check was run for station 159, issue `2025-02-20T18:00Z`, lead
1-6h:

| Forecast input | Count | RMSE | Evidence |
| --- | ---: | ---: | --- |
| NOAA GFS 0.25 `RH:2 m above ground` nearest station baseline | 6 | 10.7064 | `data/artifacts/humidity_honest_quick_experiments/gfs_station159_20250220_18_f001_f006_baseline.json` |

This confirms that simply copying a real public NWP RH forecast is still not
near `RMSE <= 1`; the new MOS path can learn calibration when a full historical
forecast archive is available, but the remaining limiter is the quality and
coverage of the honest NWP moisture input.
