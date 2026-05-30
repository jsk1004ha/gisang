# G027 Operational Report Dashboard

G027 keeps the operational benchmark gates from G024-G026, but presents the results as a human-readable dashboard instead of a table/JSON-heavy verification report.

## Goal

성능 개선 결과를 사람이 이해할 수 있도록, 이전 V1-V4 통합 리포트처럼 시각화 중심 HTML dashboard로 생성한다.

주의: 이번 G027 report는 표와 JSON 중심의 검증 리포트로 끝내지 말고, 이전 “기상 V1-V4 통합 실험 리포트”처럼 사람이 보기 좋은 HTML dashboard로 만든다. `forecast_vs_actual`, `horizon_error`, scatter, station/region bar, heatmap, calibration comparison, patch ablation, ensemble comparison 이미지를 포함한다.

## Outputs

The operational runner writes these report artifacts under the selected output directory:

- `operational_performance_report.html`
- `operational_performance_summary.csv`
- `operational_performance_summary.json`
- `plots/*.png`
- existing operational artifacts such as `experiment_summary.json`, `predictions_test.csv`, and `production_model_manifest.json`

## Image embedding

`weather_korea_forecast.v4.operational_performance` supports:

- `--embed-images thumbnail` (default): embed a resized PNG preview as base64 and link to the original PNG;
- `--embed-images full`: embed the full image directly;
- `--embed-images external-assets`: reference relative PNG assets.

Missing plots render as placeholders so report generation does not fail when an optional plot cannot be produced.

## Dashboard layout

1. Overview: benchmark reliability, archive cycles/stations/rows, best target RMSE, V4-C gate, and site readiness.
2. Official Baselines: Raw GFS vs Operational Residual LGBM for temperature and humidity, including RMSE improvement percentage.
3. Temperature Analysis: forecast-vs-actual, horizon error, station/region breakdown, patch ablation, calibration comparison, ensemble component comparison, and worst station/horizon text.
4. Humidity Analysis: forecast-vs-actual, scatter/residual views, dry/humid event error, calibration comparison, no-patch/patch/ensemble comparison, and humidity beta status.
5. Gate & Readiness: V4-C and site beta conditions with PASS/WARN/FAIL badges and next actions.
6. Experiment Table: target metrics, model metrics, patch ablation, calibration candidate, and ensemble candidate tables.

## Required plot names

Temperature:

- `temp_forecast_vs_actual.png`
- `temp_prediction_scatter.png`
- `temp_residual_scatter.png`
- `temp_horizon_error.png`
- `temp_station_rmse_bar.png`
- `temp_region_rmse_bar.png`
- `temp_horizon_station_heatmap.png`
- `temp_daily_max_min_error.png`
- `temp_patch_ablation_bar.png`
- `temp_calibration_raw_vs_corrected.png`
- `temp_ensemble_component_comparison.png`

Humidity:

- `humidity_forecast_vs_actual.png`
- `humidity_prediction_scatter.png`
- `humidity_residual_scatter.png`
- `humidity_horizon_error.png`
- `humidity_station_rmse_bar.png`
- `humidity_region_rmse_bar.png`
- `humidity_horizon_station_heatmap.png`
- `humidity_dry_humid_event_error.png`
- `humidity_calibration_raw_vs_corrected.png`
- `humidity_model_comparison.png`

## Visual rules

- PASS is green, WARN is amber, FAIL is red.
- Metric values use three decimals.
- Units are target-specific: temperature uses `°C`, humidity uses `%p`.
- Colors stay muted and report-oriented rather than decorative.
