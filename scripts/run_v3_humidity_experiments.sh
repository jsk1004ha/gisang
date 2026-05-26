#!/usr/bin/env bash
set -u
PYTHON_BIN="${PYTHON_BIN:-.venv312/Scripts/python.exe}"
LOG_DIR="${LOG_DIR:-logs/v3_runs}"
mkdir -p "$LOG_DIR"
CONFIGS=(
  configs/v3/experiments/v3_humidity_dewpoint_lgbm_72to24.yaml
  configs/v3/experiments/v3_humidity_dewpoint_depression_lgbm_72to24.yaml
  configs/v3/experiments/v3_humidity_nwp_mos_lgbm_72to24.yaml
)
for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  log="$LOG_DIR/${name}.log"
  echo "=== RUN $name ===" | tee "$log"
  if [[ "$name" == *nwp_mos* ]]; then
    PYTHONPATH=src "$PYTHON_BIN" -m weather_korea_forecast.v3.nwp_mos --config "$cfg" 2>&1 | tee -a "$log" || echo "FAILED $name; continuing" | tee -a "$log"
  else
    PYTHONPATH=src "$PYTHON_BIN" -m weather_korea_forecast.v2.train --config "$cfg" 2>&1 | tee -a "$log" || echo "FAILED $name; continuing" | tee -a "$log"
  fi
done
