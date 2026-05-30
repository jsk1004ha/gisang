# G028 Temperature Final Improvement

## Result

- Status: **FAIL**
- Best candidate: `g028_temp_final_ensemble`
- Best temp RMSE: **1.694°C**
- PASS threshold: `<= 1.500°C`
- NEAR_PASS threshold: `<= 1.600°C`
- Gap to PASS: `0.194°C`
- Benchmark reliability: `strong` (182 cycles)

G028 did not reach PASS or NEAR_PASS. Per the stop rule, temperature is frozen for G030 with an accuracy caveat after one bounded recovery rerun inside the allowlist.

## Candidate audit

| Candidate | Best model | RMSE °C | MAE °C | Bias °C | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `g028_temp_true_patch5_lgbm_calibrated` | `lgbm_true_patch5` | 1.740 | 1.378 | 0.004 | true_patch5 calibration `per_station_horizon_mean_bias` |
| `g028_temp_true_patch5_lgbm_station_region_calibrated` | `ensemble_constrained_least_squares` | 1.746 | 1.380 | 0.151 | full-grid calibration run worsened vs recovery |
| `g028_temp_catboost_true_patch5_optional` | `catboost_residual` | n/a | n/a | n/a | optional comparison skipped |
| `g028_temp_final_ensemble` | `ensemble_stationwise_inverse_rmse` | 1.694 | 1.338 | 0.148 | final selected run; bounded recovery |

## No-new-experiment check

- Allowlist: `g028_temp_true_patch5_lgbm_calibrated`, `g028_temp_true_patch5_lgbm_station_region_calibrated`, `g028_temp_catboost_true_patch5_optional`, `g028_temp_final_ensemble`
- Out-of-allowlist candidate ids: none.
- Recovery rerun: `g028_temp_final_ensemble` with `rerun_of=g028_temperature_final_improvement` metadata.
- CatBoost optional comparison was skipped because `catboost` is not installed; no dependency was added.

## Artifacts

- Final run: `data/artifacts/g028_temp_final_ensemble/`
- Full-grid calibration run: `data/artifacts/g028_temperature_final_improvement/`
- Evidence JSON: `.omx/reports/g028-temperature-final-evidence.json`
- Candidate audit CSV: `.omx/reports/g028-temperature-candidate-audit.csv`
