# G030 Production Freeze + Final Report

G030 freezes model selection and turns the G028/G029 results into production-facing artifacts. No new model candidates are introduced after this point; G031 is site/API work only.

## Frozen model selection

| Target | Frozen model | Status | Metric |
| --- | --- | --- | --- |
| Temperature | `ensemble_stationwise_inverse_rmse` | `FAIL` / accuracy caveat | RMSE `1.694°C`, MAE `1.338°C`, Bias `0.148°C` |
| Humidity | `operational_residual_lgbm_humidity` no-patch LGBM | `NEAR_PASS` / beta | RMSE `10.396%p`, MAE `8.087%p`, Bias `0.596%p` |
| Weather code | `rule_based_beta` | beta | Rule-based from forecast variables |

Temperature did not reach the `1.5°C` PASS gate or `1.6°C` NEAR_PASS gate after the bounded G028 recovery. Humidity did not reach the `10%p` PASS gate, but it is inside the `10.5%p` NEAR_PASS beta threshold and passes the `|Bias| <= 2%p` rule.

## Generated artifacts

Generated under `data/artifacts/g030_production_freeze_final/`:

- `production_model_manifest.json`
- `production_model_freeze_record.json`
- `operational_performance_report.html`
- `operational_performance_summary.csv`
- `operational_performance_summary.json`
- `experiment_summary.json`
- `plots/*.png`

The freeze record stores a manifest checksum, model artifact checksums, the frozen forecast schema version (`forecast_points.v2-beta-sources`), and the source benchmark summary checksum. It deliberately hashes the G028 source benchmark summary rather than the final G030 summary to avoid a self-referential checksum.

## Dashboard requirements satisfied

The final HTML report keeps the human-readable dashboard structure restored in G027:

1. Overview with KPI cards, reliability, archive coverage, target RMSE, V4-C gate, and site readiness.
2. Official Baselines comparing Raw GFS and Operational Residual LGBM by target.
3. Temperature Analysis with forecast-vs-actual, scatter/residual, horizon, station/region, heatmap, daily max/min, patch ablation, calibration, and ensemble plots.
4. Humidity Analysis with forecast-vs-actual, scatter/residual, horizon, station/region, heatmap, dry/humid event, calibration, and model-comparison plots.
5. Gate & Readiness with V4-C/site condition tables, beta source readiness, missing conditions, and next actions.
6. Experiment Table with target metrics, model metrics, patch ablation, calibration candidates, and ensemble candidates.

Thumbnail image embedding is the default and every thumbnail links to its original PNG. Missing plot placeholders are supported by the report renderer.

## Freeze rules

- G030 is the final model-selection point.
- G031 may change only web/API/schema-consumption/docs behavior.
- No new training candidates, test-metric model selection, or diagnostic/smoke/oracle artifacts may enter the frozen benchmark/report.
- Site copy must show `temperature_accuracy_caveat`, `humidity_beta`, `weather_code_rule_based_beta`, and site readiness `WARN` honestly.
