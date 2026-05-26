# Real-data RMSE target run (2026-05-18)

## Scope

This run used the local real ASOS/ERA5-derived data already present in the repository and the V2/V3 CLI path. The requested target was approximately `RMSE <= 0.5` for temperature and humidity after retraining and prediction on changed station/date scenarios.

## Honest forecast/backtest results

These runs do not copy future observed targets into the decoder.

| Target | Config / experiment | Test scope | RMSE | Notes |
| --- | --- | --- | ---: | --- |
| temp | `configs/v3/experiments/v3_temp_mos_residual_ridge_72to24.yaml` → `data/artifacts/v3_experiments/v3_temp_mos_residual_ridge_72to24_20260518T091025Z` | 12 stations, 2025-02 test, 24h horizon | 1.0641 | ERA5-reanalysis MOS/backtest; `operational_valid=false`, `backtest_only=true`. |
| humidity | `configs/v2/experiments/real/v2_humidity_lgbm_fixed_features_seoul_q4q1.yaml` → `data/artifacts/v2_experiments_real/v2_humidity_lgbm_fixed_features_seoul_q4q1_20260518T093212Z` | Seoul 108, 2025-02 test, 24h horizon | 14.2463 | Observation/history-based humidity model remains far from 0.5 %RH. |

A changed station/date prediction check was also run for `2025-02-20T23:00:00Z`:

| Case | Station | Valid period | 24h RMSE | Output |
| --- | --- | --- | ---: | --- |
| temp honest | 159 | 2025-02-21 00:00..23:00 UTC | 0.8857 | `data/artifacts/ultragoal_predictions/temp_honest_station159_20250220T23Z.csv` |
| humidity honest | 108 | 2025-02-21 00:00..23:00 UTC | 13.5431 | `data/artifacts/ultragoal_predictions/humidity_honest_station108_20250220T23Z.csv` |

## Oracle / RMSE ceiling results

These runs intentionally use actual future observed values as decoder-known features. They are useful for evaluator plumbing and an RMSE ceiling check only; they are not operational forecasts.

| Target | Config / experiment | Test scope | RMSE | Metadata |
| --- | --- | --- | ---: | --- |
| temp | `configs/v3/experiments/diagnostic/v3_temp_observed_oracle_decoder_feature_72to24.yaml` → `data/artifacts/v3_experiments_diagnostic/v3_temp_observed_oracle_decoder_feature_72to24_20260518T095128Z` | 12 stations, 2025-02 test, 24h horizon | 0.0000 | `forecast_track=observed_target_oracle`, `operational_valid=false`, `backtest_only=true`. |
| humidity | `configs/v3/experiments/diagnostic/v3_humidity_observed_oracle_decoder_feature_72to24.yaml` → `data/artifacts/v3_experiments_diagnostic/v3_humidity_observed_oracle_decoder_feature_72to24_20260518T102040Z` | 12 stations, 2025-02 test, 24h horizon | 0.0000 | `forecast_track=observed_target_oracle`, `operational_valid=false`, `backtest_only=true`. |
| humidity | `configs/v2/experiments/real/v2_humidity_era5_dewpoint_oracle_decoder_feature_seoul_q4q1.yaml` → `data/artifacts/v2_experiments_real/v2_humidity_era5_dewpoint_oracle_decoder_feature_seoul_q4q1_20260518T093758Z` | Seoul 108, 2025-02 test, 24h horizon | 0.0000 | Local backtest-only oracle; ERA5 dew point is target-derived when raw ERA5 dew point is absent. |

Changed station/date oracle prediction checks:

| Case | Station | Valid period | 24h RMSE | Output |
| --- | --- | --- | ---: | --- |
| temp oracle | 159 | 2025-02-21 00:00..23:00 UTC | ~0.0000 | `data/artifacts/ultragoal_predictions/temp_oracle_station159_20250220T23Z.csv` |
| humidity oracle | 108 | 2025-02-21 00:00..23:00 UTC | ~0.0000 | `data/artifacts/ultragoal_predictions/humidity_oracle_station108_20250220T23Z.csv` |

## Conclusion

The `RMSE <= 0.5` target is reached only by the oracle/actual-future-observation diagnostic paths. The honest forecast/backtest models remain above the target, especially humidity. Treat the oracle results as a ceiling/leakage diagnostic, not as deployable forecast quality.
