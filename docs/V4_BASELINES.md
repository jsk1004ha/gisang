# V4 Baselines

V4 starts from the locked V3/V3.5 station-level baselines below. Treat these as comparison anchors, not final V4 claims.

## Locked baseline rows

| Baseline | Config / run family | Target | Track | Current evidence | V4 interpretation |
| --- | --- | --- | --- | --- | --- |
| Temperature MOS ridge | `configs/v3/experiments/v3_temp_mos_residual_ridge_72to24.yaml` | temp | `nwp_assisted_mos` | RMSE about `1.065°C` on representative backtest evidence | Best station-level temp baseline before operational forecast CSV replacement. |
| Humidity honest NWP MOS | `configs/v3/experiments/v3_humidity_nwp_mos_lgbm_72to24.yaml` and `configs/v3/experiments/v3_humidity_openmeteo_ecmwf_mos_lgbm_72to24.yaml` | humidity | `nwp_assisted_mos` | Forecast-archive path exists; quality depends on issue-time aligned archive coverage | Honest humidity improvement track. |
| Humidity V3.5 ERA5 RH diagnostic | `configs/v3/experiments/v3_5_humidity_era5_rh_residual_lgbm_72to24.yaml` | humidity | `nwp_assisted_mos` | Diagnostic upper bound only; ERA5 RH is target-derived in current data | Excluded from operational claims and main V4 readiness proof. |
| Diagnostic oracle checks | `configs/v3/experiments/diagnostic/*oracle*72to24.yaml` | temp/humidity | `observed_target_oracle` | Can approach near-zero RMSE | Pipeline sanity only; never a production model. |

## Rules for V4 comparisons

- A V4 model must report `v4_stage`, `forecast_schema_valid`, and patch metadata when patch features are used.
- `operational_valid=true` requires a forecast source with issue/init time and schema validation, not ERA5 reanalysis.
- `backtest_only=true` rows may remain useful baselines, but they must not satisfy the operational gate.
- Main leaderboard and best-model outputs must continue excluding diagnostic/oracle and alias artifacts.
