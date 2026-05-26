# V4 Implementation Plan

V4 should start only after V3.5 proves that station-level MOS can run with real forecast inputs, not ERA5 reanalysis future values.  This plan defines the minimum operational, spatial, national, probabilistic, and physical-consistency work needed for V4.

## V4 entry gate

V4 work is ready when all of the following are true:

1. Temperature NWP-assisted/MOS is stable around `RMSE <= 1.0~1.1°C` on representative rolling-origin validation.
2. At least one temperature MOS experiment is `operational_valid=true` with an issue-time-aligned forecast source (`prepared_forecast_csv`, GFS, ECMWF, or KMA), not ERA5 reanalysis.
3. Humidity reaches `RMSE <= 10%p` with a non-oracle path.
4. Diagnostic/oracle and alias artifacts remain excluded from main leaderboard/KPI/best-model ranking.
5. Station metadata has no uncontrolled `unknown` region/terrain/coastal classifications for benchmark stations.
6. The benchmark covers at least 20 ASOS stations and at least one rolling-origin seasonal validation pass.

## Forecast NWP archive

V4 requires a forecast archive, not valid-time reanalysis.  The canonical station-level schema is:

```text
station_id
forecast_init_time
issue_time
valid_time
horizon_step
nwp_t2m
nwp_sp
nwp_u10
nwp_v10
nwp_tp
nwp_dew_point
nwp_relative_humidity
nwp_cloud_cover
source
```

Rules:

- `issue_time <= forecast_init_time` for every sample.
- `valid_time = forecast_init_time + horizon_step hours` for inference output.
- Temperature/dew point are Celsius after loading.
- Pressure is hPa after loading.
- Missing horizons are fatal in operational mode.
- ERA5 reanalysis remains `backtest_only=true` and cannot pass operational inference.

## NWP grid patch extraction

Station-level nearest-point features are not enough for V4.  Add grid patches around each station:

```text
station_id
forecast_init_time
valid_time
horizon_step
source
patch_size          # 3, 5, or 9
variable            # t2m, sp, u10, v10, tp, dew_point, rh, cloud_cover
row_offset          # centered at 0
col_offset          # centered at 0
value
lat
lon
```

Initial patch sizes:

- `3x3`: low-cost baseline for local gradients.
- `5x5`: preferred first V4 benchmark.
- `9x9`: optional for fronts/typhoons/coastal transitions.

Derived patch features for tree models:

- center value
- patch mean/std/min/max
- upwind/downwind gradient
- coast-normal gradient where coastal metadata exists
- precipitation coverage fraction

## PatchCNN baseline

The first neural V4 spatial baseline should be deliberately small:

```text
NWP patch tensor -> Conv2D blocks -> station/static embedding concat -> horizon residual head
```

Inputs:

- per-horizon NWP patch tensor `[variables, patch_y, patch_x]`
- station embedding or station metadata
- horizon embedding
- month/hour/doy cyclic features

Targets:

- temperature MOS residual: `observed_temp - nwp_t2m_center`
- humidity residual or dew-point/depression target after V3.5 humidity fixes

Evaluation must compare against V3.5 station-level MOS ridge/LGBM before adding complexity.

## Probabilistic forecast

V4 outputs should include uncertainty:

```text
p10
p50
p90
mean
spread
```

Candidate models:

- Quantile LightGBM/CatBoost
- TFT QuantileLoss
- ensemble spread calibration

Metrics:

- pinball loss
- empirical coverage for P10-P90
- prediction interval width
- CRPS approximation
- deterministic RMSE/MAE for P50/mean

## National ASOS/AWS expansion

V4 should expand beyond the current station set, but AWS requires quality weighting.

Station quality features:

```text
missing_rate
outlier_rate
valid_period_length
sensor_stability
metadata_completeness
station_quality_score
```

Training policy:

- ASOS remains the primary benchmark source.
- AWS can supplement gaps through priority merging.
- Low-quality AWS stations are downweighted or excluded from headline metrics.
- Split remains strictly time ordered.

## Physical consistency

V4 predictions must satisfy or report violations for:

```text
dew_point <= temperature
0 <= relative_humidity <= 100
daily_min <= daily_max
precipitation >= 0
```

Humidity should be restored from temperature/dew point/depression where possible instead of independently predicting unconstrained RH.

## Operational loop

V4 production loop:

```text
forecast NWP download/archive
-> station and patch feature extraction
-> prediction
-> calibration
-> physical consistency checks
-> report/API output
-> next-day observation ingest
-> automatic evaluation
-> leaderboard/report refresh
```

All artifacts should continue to feed the unified reporting system, with diagnostic/oracle rows excluded from main rankings.
