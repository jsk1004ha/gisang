# Forecast Output Schema

Production forecast artifacts are separate from research artifacts.

## Research artifacts

- `predictions_test.csv`
- `metrics_summary.json`
- `experiment_report.html`
- `training_history.json`

## Production artifacts

- `data/forecasts/runs/{forecast_run_id}/forecast_run.json`
- `data/forecasts/runs/{forecast_run_id}/forecast_points.json`
- `data/forecasts/runs/{forecast_run_id}/forecast_points.csv`
- `data/forecasts/latest/*`

`forecast_run.json` records model version, forecast init time, source, operational flags, backtest flags, horizon, targets, notes, and warnings.

`forecast_points` contains station metadata, valid time, horizon, weather variables, weather code/label/icon, confidence, data source, and operational flags.
