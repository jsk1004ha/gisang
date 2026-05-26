# Unified Experiment Reporting

This repository now includes a dependency-free reporting package under
`weather_korea_forecast.reporting` that consolidates V1/V2/V3 experiment
artifacts into one CSV leaderboard and one standalone HTML dashboard.

## Command

```bash
PYTHONPATH=src .venv312/Scripts/python.exe -m weather_korea_forecast.reporting.generate_report \
  --experiments-root data/artifacts \
  --output-dir reports \
  --title "Gisang V1-V3 Experiment Report"
```

Outputs:

- `reports/experiment_report.html` — standalone dashboard; PNG plots are embedded as base64 by default.
- `reports/experiment_summary.csv` — one row per discovered experiment.
- `reports/experiment_summary.json` — JSON equivalent of the summary records.
- `reports/best_models.csv` — top 5 rows per `target_name`/`track` group.
- `reports/failed_or_incomplete_experiments.csv` — parse failures or incomplete artifact folders.

Use `--no-images` for a fast/light report and `--track-filter` or
`--target-filter` for smaller scoped reports. Image handling can be selected
with `--embed-images full|thumbnail|external-assets`; `full` embeds every PNG
as base64 for one-file sharing, `thumbnail` embeds clickable scaled images, and
`external-assets` references the artifact PNG paths without embedding them.

## Artifact discovery

The collector recursively scans the chosen `--experiments-root` for
`experiment_summary.json` files. It reads the files that are present and treats
missing optional files as warnings, not fatal errors:

- `experiment_summary.json`
- `metrics_summary.json` / `metrics_test.json`
- `experiment_config.yaml`
- `training_history.json`
- `bias_correction.json`
- `worst_case_summary.json`
- `predictions_test.csv`
- standard plot PNG files such as `forecast_vs_actual.png`, `horizon_error.png`,
  `prediction_scatter.png`, `raw_vs_corrected.png`, `station_rmse_bar.png`,
  `region_rmse_bar.png`, `daily_max_min_error.png`, `residual_scatter.png`, and
  `baseline_vs_final_scatter.png`.

A malformed experiment folder is included in the failed/incomplete CSV so one
bad run does not block dashboard generation.

## ExperimentRecord schema

`ExperimentRecord` in `src/weather_korea_forecast/reporting/schema.py` defines
the CSV/JSON shape. Important fields include:

- identity: `experiment_name`, `version`, `target_name`, `track`, `model_name`, `model_type`
- horizon/config: `encoder_length`, `prediction_length`, train/val/test ranges
- metrics: `rmse`, `mae`, `bias`, raw/validation metrics, worst horizon/station/region fields
- operational flags: `uses_future_weather_features`, `future_feature_source`,
  `operational_valid`, `backtest_only`, `leakage_risk_note`
- goal status: `rmse_goal`, `rmse_goal_met`, `rmse_gap_to_goal`
- interpretation/deduplication: `goal_eligible`, `is_diagnostic`,
  `is_alias_artifact`, `canonical_experiment_id`, `run_timestamp`,
  `is_representative_run`, `included_in_main_leaderboard`
- artifacts: `artifact_dir` plus plot path columns
- quality: `complete`, `warnings`, `error`

RMSE goals are assigned automatically:

- temp + `nwp_assisted_mos`: `1.0°C`
- temp + `observation_only`: `2.0°C`
- humidity: `10.0%p`

Oracle/diagnostic rows are excluded from main KPI, main leaderboard,
`best_models.csv`, and goal-hit counts. A row is diagnostic if it matches any of:

- `track` contains `oracle`
- `model_type == decoder_feature_baseline`
- `future_feature_source == observed_target_oracle`
- `rmse == 0` and `backtest_only == true`
- experiment metadata/path indicates a diagnostic run

Diagnostic rows still appear in the HTML under **Diagnostic / Oracle Checks**
with a warning that they are pipeline sanity checks, not forecast-model
performance. `best` and `latest` artifact aliases are also retained in detail
sections but excluded from the main representative-run leaderboard.
The HTML main leaderboard, KPI cards, summary charts, and `best_models.csv`
use only rows where `included_in_main_leaderboard == true`.

## HTML layout

The HTML dashboard contains:

1. header with generated time, root path, and completion counts;
2. KPI cards for target/track-specific best RMSEs, operational-valid best,
   backtest-only best, and diagnostic-free goal hits;
3. filterable/searchable leaderboard table;
4. inline SVG summary charts without external CDN dependencies;
5. best-model comparison cards;
6. Diagnostic / Oracle Checks section;
7. warnings/data-quality table;
8. per-experiment detail accordions with embedded plots or placeholders.

## Train/evaluate integration

V1/V2/V3 training and evaluation CLIs regenerate the unified report by default
after an experiment finishes:

```bash
PYTHONPATH=src .venv312/Scripts/python.exe -m weather_korea_forecast.v2.train \
  --config configs/v3/experiments/v3_temp_mos_residual_ridge_72to24.yaml
```

This regenerates `reports/experiment_report.html` and the CSV/JSON summaries
after the experiment finishes. Use `--no-update-report` only when you explicitly
want to skip that default report refresh.

## Known limitations

- The dashboard is static; filters are client-side JavaScript over the generated table.
- Very large numbers of embedded PNGs can make the HTML file large. Use `--no-images` for quick CI artifacts.
- The report compares available metrics only; if a legacy V1/V2 experiment did not save a metric, that field remains blank.
- Operational-valid status depends on each experiment summary/config writing accurate future-feature metadata.
- Main comparisons are intentionally target/track-specific because temperature
  RMSE (°C), humidity RMSE (%RH), and multi-target aggregate metrics should not
  be interpreted as one shared unit.
