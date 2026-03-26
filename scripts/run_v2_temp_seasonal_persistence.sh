#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"
CONFIG="configs/v2/experiments/v2_temp_seasonal_persistence.yaml"

python -m weather_korea_forecast.v2.prepare_data --config "$CONFIG"
python -m weather_korea_forecast.v2.train --config "$CONFIG"
python -m weather_korea_forecast.v2.evaluate --experiment-dir data/artifacts/v2_experiments/latest

