# G026 Accuracy Breakthrough Sprint

G026 moves the operational benchmark from the G025 90-cycle medium archive toward a strong benchmark and target-specific production model selection.

## Purpose

The sprint is not a new model-family expansion sprint. It focuses on breaking the remaining operational gates:

- temperature RMSE <= 1.5°C
- humidity RMSE <= 10%p
- real forecast archive reliability >= medium, with strong (>=180 cycles) preferred
- target-specific production model selection, because temperature and humidity have different best feature modes

## Data policy

Generated, smoke, fixture, diagnostic, oracle, and synthetic forecast sources remain disallowed for operational claims. Strong G026 archives should use real forecast issue cycles and real KMA ASOS observations.

The local G026 workflow attempts two data paths:

1. strong core GFS archive: real NOAA GFS public S3 forecast cycles, station-nearest `t2m`, 2 m RH, and dew point, merged with the G025 medium archive;
2. full-variable GFS enrichment: expanded GFS variables such as pressure, wind, gust, cloud, radiation, soil temperature, land/sea mask, specific humidity, precipitable water, and precipitation rate.

Full-variable public S3 extraction can be very large because each selected GRIB message is global. When the runtime cannot complete a full-variable strong download, reports must state the blocker and must not label the full-variable path as adequate.

## Target-specific production policy

Temperature may select true-grid patch and ensemble paths when validation holdout improves. Humidity keeps no-patch residual LightGBM as the official reference path; humidity patch features are limited to moisture/regime variables and humidity ensembles are rejected unless validation holdout improves without increasing bias magnitude.

The operational runner writes:

- `variable_coverage.csv`
- `production_model_manifest.json`
- target-specific ensemble acceptance/rejection reasons
- `operational_performance_report.html`

`variable_coverage.csv` includes overall coverage plus split-specific coverage
columns such as `train_coverage`, `val_coverage`, and `test_coverage` so
full-variable evidence is not overstated when a variable is present only outside
the training split. The production manifest records each target's artifact type:
single-model selections point to `models.pkl`, while accepted ensemble selections
also include their weights, metrics, and prediction artifact paths.

## Gate interpretation

V4-C and site operational beta remain blocked unless:

- best operational temperature RMSE <= 1.5°C
- best operational humidity RMSE <= 10%p
- benchmark reliability >= medium
- true-grid patch ablation/report artifacts exist
- no critical archive quality warnings exist


## Strong run result (local G026 evidence)

The local strong-core run used real NOAA GFS public S3 forecast cycles plus the existing G025 medium archive and real KMA ASOS observations:

- forecast cycles: 182
- stations: 30
- archive rows: 131,040
- horizon 1-24 coverage: 1.000
- archive adequate: `True`
- benchmark reliability: `strong`

Best operational models on this run:

| Target | Best model | RMSE | MAE | Bias | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Temperature | `ensemble_stationwise_inverse_rmse` | 1.694°C | 1.338°C | 0.148°C | FAIL (`<=1.5°C`) |
| Humidity | `operational_residual_lgbm_humidity` | 10.396%p | 8.087%p | 0.596%p | FAIL (`<=10%p`) |

The humidity ensemble was intentionally rejected because its bias magnitude worsened versus the target-specific no-patch baseline. V4-C and site operational beta therefore remain blocked.

Full-variable strong extraction was not promoted in this run: public GFS S3 full-variable extraction requires large global-message downloads, so the code path and variable schema are supported, but the local strong benchmark above uses the core real forecast variables only.
