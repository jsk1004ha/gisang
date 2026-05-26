#!/usr/bin/env bash
set -u
PYTHON_BIN="${PYTHON_BIN:-.venv312/Scripts/python.exe}"
LOG_DIR="${LOG_DIR:-logs/v3_runs}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-data/artifacts/v3_experiments}"
mkdir -p "$LOG_DIR" "$ARTIFACT_ROOT"
CONFIGS=(
  configs/v3/experiments/v3_temp_mos_residual_ridge_72to24.yaml
  configs/v3/experiments/v3_temp_mos_residual_ridge_168to24.yaml
  configs/v3/experiments/v3_temp_mos_horizonwise_residual_ridge_72to24.yaml
  configs/v3/experiments/v3_temp_mos_horizonwise_residual_ridge_168to24.yaml
  configs/v3/experiments/v3_temp_mos_residual_lgbm_72to24.yaml
  configs/v3/experiments/v3_temp_mos_residual_lgbm_168to24.yaml
)
for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  log="$LOG_DIR/${name}.log"
  echo "=== RUN $name ===" | tee "$log"
  if PYTHONPATH=src "$PYTHON_BIN" -m weather_korea_forecast.v2.train --config "$cfg" 2>&1 | tee -a "$log"; then
    exp_dir=$(tail -n 20 "$log" | grep -E 'data/artifacts/.+_[0-9]{8}T[0-9]{6}Z' | tail -n 1 | tr -d '\r')
    if [ -n "$exp_dir" ] && [ -f "$exp_dir/metrics_summary.json" ]; then
      PYTHONPATH=src "$PYTHON_BIN" - <<PY | tee -a "$log"
import json, pathlib
p=pathlib.Path(r"$exp_dir")
m=json.loads((p/'metrics_summary.json').read_text())
print(f"METRICS {p.name}: RMSE={m.get('rmse')} MAE={m.get('mae')} Bias={m.get('bias')}")
for req in ['experiment_summary.json','metrics_summary.json','predictions_test.csv']:
    print(req, 'OK' if (p/req).exists() else 'MISSING')
PY
    fi
  else
    echo "FAILED $name; continuing" | tee -a "$log"
  fi
done
# Ensemble after components exist.
PYTHONPATH=src "$PYTHON_BIN" -m weather_korea_forecast.v2.train --config configs/v3/experiments/v3_temp_mos_ensemble_72to24.yaml 2>&1 | tee "$LOG_DIR/v3_temp_mos_ensemble_72to24.log" || true
