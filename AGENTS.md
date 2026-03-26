# AGENTS.md

## Repository purpose

This repository builds station-level weather forecasting pipelines for Korea using observation data plus ERA5-derived features.
The current implementation is a V1 pipeline with data preparation, training, evaluation, and inference CLIs.

## Working conventions

- Work from the repository root: `C:\Users\js100\Desktop\coding\gisang`.
- Prefer `python -m ...` entrypoints over ad hoc scripts.
- If the package is not installed in the environment, use `PYTHONPATH=src`.
- Keep changes config-driven. Prefer updating `configs/` and loader logic over hardcoding paths, stations, or feature lists in code.
- V2 uses unified configs under `configs/v2/experiments/`. Keep V1 and V2 config flows separate.
- If you add a repo-local real-data bootstrap config for V2, keep it under `configs/v2/experiments/real/` and document that it is a local example, not the canonical multi-station benchmark config.
- Preserve both model backends:
  `fallback_torch` is the default lightweight path.
  `pytorch_forecasting` is the optional true-TFT path when extra dependencies are installed.
- Do not describe the fallback model as a true TFT. It is a simpler substitute model.
- V2 defaults to single-target experiments. Keep `temp` and `humidity` configs separate unless the task explicitly extends V2 multi-target support.

## Standard commands

- Build training table:
  `python -m weather_korea_forecast.data.build_training_table --config configs/data/dataset_v1.yaml`
- Prepare V2 training table:
  `python -m weather_korea_forecast.v2.prepare_data --config configs/v2/experiments/v2_temp_ridge.yaml`
- Train:
  `python -m weather_korea_forecast.training.train --data-config configs/data/dataset_v1.yaml --model-config configs/model/tft_v1.yaml --train-config configs/train/train_v1.yaml`
- Train V2:
  `python -m weather_korea_forecast.v2.train --config configs/v2/experiments/v2_temp_tft.yaml`
- Resume training from the current best checkpoint:
  `python -m weather_korea_forecast.training.train --data-config configs/data/dataset_v1.yaml --model-config configs/model/tft_v1.yaml --train-config configs/train/train_v1.yaml --resume-from data/artifacts/experiments/best/model.pt`
- Evaluate:
  `python -m weather_korea_forecast.evaluation.evaluate --experiment-dir data/artifacts/experiments/latest`
- Evaluate V2:
  `python -m weather_korea_forecast.v2.evaluate --experiment-dir data/artifacts/v2_experiments/latest`
- Predict:
  `python -m weather_korea_forecast.inference.predict --experiment-dir data/artifacts/experiments/latest --station-id SEOUL --forecast-init-time 2025-01-03T00:00:00Z`
- Predict V2:
  `python -m weather_korea_forecast.v2.predict --experiment-dir data/artifacts/v2_experiments/latest --station-id 108 --forecast-init-time 2025-01-03T00:00:00Z`
- Quick validation when dependencies are limited:
  `python -m compileall src`
- Tests if `pytest` is installed:
  `python -m pytest -q`

## Data and pipeline guardrails

- Observation inputs must normalize to the standard schema:
  `station_id, datetime, temp, humidity, pressure, wind_speed, precipitation, quality_flag`
- Time handling must remain explicit and UTC-normalized inside the pipeline.
- When extending observation ingestion, prefer the shared observation loader path over duplicating ASOS/AWS-specific parsing.
- `ASOS` remains the primary source unless config explicitly adds supplementary sources.
- Supplementary sources such as local `AWS` CSV files should fill gaps through config-based priority merging rather than replacing the base path implicitly.
- Keep train/val/test splitting strictly time-ordered.

## Modeling guardrails

- If you change training, inference, or dataset code, verify that train-time and predict-time feature handling still match.
- Multi-target support is not fully end-to-end yet. Do not assume `target_columns` works beyond the first target unless you have updated training, prediction, and evaluation paths together.
- Baseline comparisons matter. If model behavior changes, consider whether baseline configs or reports also need updates.
- Preserve the `latest/` alias behavior and the `best/` promotion behavior when touching experiment artifact management.
- V2 also maintains `latest/` and `best/` aliases under its own artifact root. Do not mix V1 and V2 artifact directories.

## Files to update together

- If you change config schema, update:
  `README.md`
  relevant files under `configs/`
- If you change V2 experiment config schema, update:
  `README.md`
  `docs/V2_plan.md`
  `CHANGELOG.md`
  files under `configs/v2/`
- If you change data loading or table-building behavior, update:
  `README.md`
  `tests/`
  example config files under `configs/data/`
- If you change training or inference behavior, update:
  `README.md`
  `configs/train/`
  `configs/model/`
  smoke or unit tests covering roundtrip behavior

## Repository hygiene

- Do not commit generated data under `data/raw/`, `data/processed/`, or experiment outputs under `data/artifacts/`.
- Temporary smoke directories should stay untracked.
- Prefer small, reviewable changes with explicit tests or a documented reason when tests cannot run.
