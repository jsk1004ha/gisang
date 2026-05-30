from __future__ import annotations

TARGETS = {
    "temp": {"actual": "temp", "baseline": "nwp_t2m", "official_raw": "raw_gfs_t2m", "official_lgbm": "operational_residual_lgbm_temp"},
    "humidity": {"actual": "humidity", "baseline": "nwp_humidity", "official_raw": "raw_gfs_rh", "official_lgbm": "operational_residual_lgbm_humidity"},
}
TARGET_UNITS = {"temp": "°C", "humidity": "%p"}
TARGET_GATE_RMSE = {"temp": 1.5, "humidity": 10.0}
TARGET_NEAR_PASS_RMSE = {"temp": 1.6, "humidity": 10.5}
TARGET_BIAS_GATE = {"humidity": 2.0}
FORECAST_POINTS_SCHEMA_VERSION = "forecast_points.v2-beta-sources"
