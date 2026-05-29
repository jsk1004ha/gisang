from __future__ import annotations

import argparse
import html
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from weather_korea_forecast.v4.gfs_grid_patch import PATCH_FEATURE_MODE

try:  # pragma: no cover - dependency is present in normal dev env, optional in minimal envs.
    from lightgbm import LGBMRegressor
except Exception:  # noqa: BLE001
    LGBMRegressor = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency.
    from catboost import CatBoostRegressor
except Exception:  # noqa: BLE001
    CatBoostRegressor = None  # type: ignore[assignment]

TARGETS = {
    "temp": {"actual": "temp", "baseline": "nwp_t2m", "official_raw": "raw_gfs_t2m", "official_lgbm": "operational_residual_lgbm_temp"},
    "humidity": {"actual": "humidity", "baseline": "nwp_humidity", "official_raw": "raw_gfs_rh", "official_lgbm": "operational_residual_lgbm_humidity"},
}
BASE_FEATURES = [
    "horizon_step",
    "nwp_t2m",
    "nwp_dew_point",
    "nwp_humidity",
    "nwp_sp",
    "nwp_u10",
    "nwp_v10",
    "nwp_tp",
    "nwp_gust",
    "nwp_cloud_cover",
    "nwp_low_cloud_cover",
    "nwp_shortwave_radiation",
    "nwp_longwave_radiation",
    "nwp_soil_temperature",
    "nwp_land_sea_mask",
    "nwp_specific_humidity",
    "nwp_pwat",
    "nwp_precip_rate",
    "nwp_mslp",
    "nwp_wind_speed",
    "nwp_wind_direction",
    "nwp_dewpoint_depression",
    "nwp_humidity_logit",
    "nwp_temp_pressure_interaction",
    "lat",
    "lon",
    "elevation",
    "station_id",
    "issue_hour_sin",
    "issue_hour_cos",
    "issue_doy_sin",
    "issue_doy_cos",
    "valid_hour_sin",
    "valid_hour_cos",
    "valid_doy_sin",
    "valid_doy_cos",
]
FULL_VARIABLE_COLUMNS = [
    "nwp_t2m",
    "nwp_dew_point",
    "nwp_humidity",
    "nwp_sp",
    "nwp_u10",
    "nwp_v10",
    "nwp_tp",
    "nwp_gust",
    "nwp_cloud_cover",
    "nwp_low_cloud_cover",
    "nwp_shortwave_radiation",
    "nwp_longwave_radiation",
    "nwp_soil_temperature",
    "nwp_land_sea_mask",
    "nwp_specific_humidity",
    "nwp_pwat",
    "nwp_precip_rate",
    "nwp_mslp",
]
HUMIDITY_PATCH_VARIABLE_HINTS = (
    "humidity",
    "dew_point",
    "specific_humidity",
    "pwat",
    "cloud",
    "precip",
    "tp",
    "t2m",
)
DEFAULT_LGBM_GRID = [
    {"num_leaves": 15, "max_depth": 4, "learning_rate": 0.05, "n_estimators": 350, "min_child_samples": 20, "subsample": 0.9, "colsample_bytree": 0.9, "reg_alpha": 0.0, "reg_lambda": 0.0, "min_split_gain": 0.0},
    {"num_leaves": 31, "max_depth": -1, "learning_rate": 0.03, "n_estimators": 500, "min_child_samples": 20, "subsample": 0.9, "colsample_bytree": 0.9, "reg_alpha": 0.0, "reg_lambda": 0.0, "min_split_gain": 0.0},
    {"num_leaves": 31, "max_depth": 6, "learning_rate": 0.03, "n_estimators": 700, "min_child_samples": 10, "subsample": 0.95, "colsample_bytree": 0.95, "reg_alpha": 0.0, "reg_lambda": 0.1, "min_split_gain": 0.0},
    {"num_leaves": 63, "max_depth": -1, "learning_rate": 0.02, "n_estimators": 800, "min_child_samples": 15, "subsample": 0.9, "colsample_bytree": 0.9, "reg_alpha": 0.0, "reg_lambda": 0.1, "min_split_gain": 0.0},
    {"num_leaves": 63, "max_depth": 8, "learning_rate": 0.03, "n_estimators": 600, "min_child_samples": 8, "subsample": 0.85, "colsample_bytree": 0.9, "reg_alpha": 0.05, "reg_lambda": 0.2, "min_split_gain": 0.01},
    {"num_leaves": 127, "max_depth": 10, "learning_rate": 0.02, "n_estimators": 900, "min_child_samples": 8, "subsample": 0.9, "colsample_bytree": 0.85, "reg_alpha": 0.05, "reg_lambda": 0.5, "min_split_gain": 0.0},
    {"num_leaves": 127, "max_depth": -1, "learning_rate": 0.015, "n_estimators": 1200, "min_child_samples": 6, "subsample": 0.85, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.8, "min_split_gain": 0.02},
]
CALIBRATION_CANDIDATES = [
    "none",
    "global_mean_bias",
    "per_horizon_mean_bias",
    "per_station_horizon_mean_bias",
    "per_region_horizon_mean_bias",
    "global_affine",
    "per_station_horizon_affine",
    "per_region_horizon_affine",
    "per_station_month_hour_mean_bias",
    "quantile_mapping",
    "isotonic",
    "isotonic_calibration",
]
LGBM_GRID_RESULT_COLUMNS = {
    "grid_index",
    "num_leaves",
    "max_depth",
    "learning_rate",
    "n_estimators",
    "min_child_samples",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "min_split_gain",
    "val_rmse",
    "val_mae",
    "val_bias",
    "val_n",
}


@dataclass(frozen=True)
class CalibrationModel:
    name: str
    payload: dict[str, Any]
    actual_column: str = "actual"
    prediction_column: str = "prediction"

    def apply(self, frame: pd.DataFrame) -> np.ndarray:
        prediction = pd.to_numeric(frame[self.prediction_column], errors="coerce").astype(float).to_numpy()
        if self.name == "none":
            return prediction
        if self.name.endswith("mean_bias"):
            return prediction + self._lookup_adjustments(frame)
        if self.name.endswith("affine"):
            slopes, intercepts = self._lookup_affine(frame)
            return slopes * prediction + intercepts
        if self.name == "quantile_mapping":
            source = np.asarray(self.payload.get("prediction_quantiles", []), dtype=float)
            target = np.asarray(self.payload.get("actual_quantiles", []), dtype=float)
            if len(source) < 2 or len(target) < 2:
                return prediction
            return np.interp(prediction, source, target, left=target[0], right=target[-1])
        if self.name in {"isotonic", "isotonic_calibration"}:
            model = self.payload.get("model")
            if model is None:
                return prediction
            return np.asarray(model.predict(prediction), dtype=float)
        return prediction

    def _lookup_adjustments(self, frame: pd.DataFrame) -> np.ndarray:
        global_bias = float(self.payload.get("global_bias", 0.0))
        adjustments = np.full(len(frame), global_bias, dtype=float)
        groups = self.payload.get("groups", {}) or {}
        if self.name == "per_horizon_mean_bias":
            for i, horizon in enumerate(frame["horizon_step"].astype(int).astype(str)):
                adjustments[i] = float(groups.get(horizon, global_bias))
        elif self.name == "per_station_horizon_mean_bias":
            horizon_groups = self.payload.get("horizon_groups", {}) or {}
            for i, row in enumerate(frame[["station_id", "horizon_step"]].itertuples(index=False)):
                key = f"{row.station_id}|{int(row.horizon_step)}"
                adjustments[i] = float(groups.get(key, horizon_groups.get(str(int(row.horizon_step)), global_bias)))
        elif self.name == "per_region_horizon_mean_bias":
            horizon_groups = self.payload.get("horizon_groups", {}) or {}
            region_col = self.payload.get("region_column")
            for i, row in enumerate(frame.itertuples(index=False)):
                region = getattr(row, str(region_col), None) if region_col else None
                horizon = int(getattr(row, "horizon_step"))
                key = f"{region}|{horizon}"
                adjustments[i] = float(groups.get(key, horizon_groups.get(str(horizon), global_bias)))
        elif self.name == "per_station_month_hour_mean_bias":
            station_groups = self.payload.get("station_groups", {}) or {}
            for i, row in enumerate(frame.itertuples(index=False)):
                timestamp = pd.Timestamp(getattr(row, "valid_time"))
                key = f"{getattr(row, 'station_id')}|{int(timestamp.month)}|{int(timestamp.hour)}"
                station_key = str(getattr(row, "station_id"))
                adjustments[i] = float(groups.get(key, station_groups.get(station_key, global_bias)))
        return adjustments

    def _lookup_affine(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        default = self.payload.get("global", {"slope": 1.0, "intercept": 0.0})
        slopes = np.full(len(frame), float(default.get("slope", 1.0)), dtype=float)
        intercepts = np.full(len(frame), float(default.get("intercept", 0.0)), dtype=float)
        groups = self.payload.get("groups", {}) or {}
        if self.name == "per_station_horizon_affine":
            for i, row in enumerate(frame[["station_id", "horizon_step"]].itertuples(index=False)):
                key = f"{row.station_id}|{int(row.horizon_step)}"
                params = groups.get(key, default)
                slopes[i] = float(params.get("slope", slopes[i]))
                intercepts[i] = float(params.get("intercept", intercepts[i]))
        elif self.name == "per_region_horizon_affine":
            region_col = self.payload.get("region_column")
            if region_col and str(region_col) in frame.columns:
                for i, row in enumerate(frame.itertuples(index=False)):
                    key = f"{getattr(row, str(region_col))}|{int(getattr(row, 'horizon_step'))}"
                    params = groups.get(key, default)
                    slopes[i] = float(params.get("slope", slopes[i]))
                    intercepts[i] = float(params.get("intercept", intercepts[i]))
        return slopes, intercepts

    def to_jsonable(self) -> dict[str, Any]:
        payload = dict(self.payload)
        if "model" in payload:
            payload["model"] = "<isotonic-regression>"
        return {"name": self.name, "payload": payload, "actual_column": self.actual_column, "prediction_column": self.prediction_column}


@dataclass(frozen=True)
class CalibrationSelectionResult:
    selected_name: str
    selected: CalibrationModel
    candidate_results: list[dict[str, Any]]
    selected_holdout_metrics: dict[str, float]


def metric_dict(actual: Iterable[float], prediction: Iterable[float]) -> dict[str, float | int]:
    y = np.asarray(list(actual), dtype=float)
    p = np.asarray(list(prediction), dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "bias": float("nan"), "n": 0}
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
        "bias": float(np.mean(p - y)),
        "n": int(len(y)),
    }


def summarize_patch_ablation_results(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = [dict(row) for row in rows]
    if not records:
        return {
            "best_mode": None,
            "patch_improvement_rmse": None,
            "patch_improvement_worst_station": None,
            "patch_improvement_late_horizon": None,
        }
    best = min(records, key=lambda row: float(row.get("rmse", float("inf"))))
    baseline = next((row for row in records if row.get("mode") == "no_patch"), records[0])

    def improvement(metric: str) -> float | None:
        if metric not in baseline or metric not in best:
            return None
        return float(baseline[metric]) - float(best[metric])

    return {
        "best_mode": best.get("mode"),
        "patch_improvement_rmse": improvement("rmse"),
        "patch_improvement_worst_station": improvement("worst_station_rmse"),
        "patch_improvement_late_horizon": improvement("late_horizon_rmse"),
    }


def validate_lgbm_grid_results(rows: Iterable[dict[str, Any]]) -> None:
    for idx, row in enumerate(rows):
        missing = sorted(LGBM_GRID_RESULT_COLUMNS.difference(row))
        if missing:
            raise ValueError(f"LGBM grid result row {idx} missing required columns: {missing}")


def classify_benchmark_reliability(*, forecast_cycle_count: int, date_span_days: float, season_count: int) -> str:
    if int(season_count) >= 4 and float(date_span_days) >= 365 and int(forecast_cycle_count) >= 365:
        return "seasonal"
    if int(forecast_cycle_count) >= 180:
        return "strong"
    if int(forecast_cycle_count) >= 90:
        return "medium"
    if int(forecast_cycle_count) >= 30:
        return "short"
    return "smoke"


def summarize_variable_coverage(frame: pd.DataFrame, variables: Iterable[str] = FULL_VARIABLE_COLUMNS, *, split_column: str = "split") -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total = int(len(frame))
    split_values = []
    if split_column in frame.columns:
        split_values = sorted(str(value) for value in frame[split_column].dropna().unique())
    for column in variables:
        if column not in frame.columns:
            row = {
                "variable": column,
                "present": False,
                "non_null_count": 0,
                "missing_count": total,
                "coverage": 0.0,
                "min": None,
                "max": None,
                "split_coverage": {},
            }
            for split in split_values:
                row[f"{split}_coverage"] = 0.0
            rows.append(row)
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        non_null = int(values.notna().sum())
        split_coverage: dict[str, float] = {}
        for split in split_values:
            mask = frame[split_column].astype(str) == split
            denominator = int(mask.sum())
            split_coverage[split] = float(values[mask].notna().sum() / denominator) if denominator else 0.0
        rows.append(
            {
                "variable": column,
                "present": True,
                "non_null_count": non_null,
                "missing_count": int(total - non_null),
                "coverage": float(non_null / total) if total else 0.0,
                "min": float(values.min()) if non_null else None,
                "max": float(values.max()) if non_null else None,
                "split_coverage": split_coverage,
                **{f"{split}_coverage": split_coverage[split] for split in split_values},
            }
        )
    available = [row["variable"] for row in rows if bool(row["present"]) and float(row["coverage"]) > 0.0]
    missing = [row["variable"] for row in rows if not bool(row["present"]) or float(row["coverage"]) == 0.0]
    return {
        "row_count": total,
        "available_variables": available,
        "missing_variables": missing,
        "full_variable_count": len(available),
        "variables": rows,
    }


def fit_calibration_candidates(
    calibration_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    *,
    actual_column: str,
    prediction_column: str,
    candidates: Iterable[str] = CALIBRATION_CANDIDATES,
) -> CalibrationSelectionResult:
    candidate_results: list[dict[str, Any]] = []
    models: dict[str, CalibrationModel] = {}
    none_metrics: dict[str, float] | None = None
    for candidate in candidates:
        model = _fit_calibration_model(calibration_frame, actual_column=actual_column, prediction_column=prediction_column, name=str(candidate))
        corrected = model.apply(holdout_frame)
        metrics = metric_dict(holdout_frame[actual_column], corrected)
        row = {"candidate": str(candidate), **metrics}
        candidate_results.append(row)
        models[str(candidate)] = model
        if candidate == "none":
            none_metrics = metrics  # type: ignore[assignment]
    if none_metrics is None:
        none_model = _fit_calibration_model(calibration_frame, actual_column=actual_column, prediction_column=prediction_column, name="none")
        none_metrics = metric_dict(holdout_frame[actual_column], none_model.apply(holdout_frame))  # type: ignore[assignment]
        candidate_results.insert(0, {"candidate": "none", **none_metrics})
        models["none"] = none_model
    best = min(candidate_results, key=lambda row: float(row.get("rmse", float("inf"))))
    selected_name = str(best["candidate"])
    if float(best["rmse"]) >= float(none_metrics["rmse"]):
        selected_name = "none"
        best = next(row for row in candidate_results if row["candidate"] == "none")
    return CalibrationSelectionResult(
        selected_name=selected_name,
        selected=models[selected_name],
        candidate_results=candidate_results,
        selected_holdout_metrics={key: float(value) for key, value in best.items() if key != "candidate"},
    )


def residual_debug_summary(
    frame: pd.DataFrame,
    *,
    actual_column: str,
    baseline_column: str,
    predicted_residual_column: str,
    prediction_column: str,
) -> dict[str, Any]:
    actual = pd.to_numeric(frame[actual_column], errors="coerce").astype(float)
    baseline = pd.to_numeric(frame[baseline_column], errors="coerce").astype(float)
    predicted_residual = pd.to_numeric(frame[predicted_residual_column], errors="coerce").astype(float)
    prediction = pd.to_numeric(frame[prediction_column], errors="coerce").astype(float)
    residual = actual - baseline
    formula_error = prediction - (baseline + predicted_residual)
    return {
        "residual_target_formula": f"{actual_column} - {baseline_column}",
        "final_prediction_formula": f"{baseline_column} + {predicted_residual_column}",
        "residual_target_sign_ok": bool(np.allclose(residual, actual - baseline, equal_nan=True)),
        "final_prediction_formula_ok": bool(np.nanmax(np.abs(formula_error.to_numpy())) < 1e-8) if len(formula_error.dropna()) else False,
        "residual_target_mean": float(residual.mean()),
        "residual_target_std": float(residual.std(ddof=0)),
        "predicted_residual_mean": float(predicted_residual.mean()),
        "predicted_residual_std": float(predicted_residual.std(ddof=0)),
        "prediction_bias": float((prediction - actual).mean()),
        "max_formula_error_abs": float(np.nanmax(np.abs(formula_error.to_numpy()))) if len(formula_error.dropna()) else float("nan"),
    }


def evaluate_v4c_gate(summary: dict[str, Any]) -> dict[str, Any]:
    baselines = summary.get("official_baselines", {}) if isinstance(summary.get("official_baselines"), dict) else {}
    best_models = summary.get("best_operational_models", {}) if isinstance(summary.get("best_operational_models"), dict) else {}
    temp = baselines.get("temp", {}) if isinstance(baselines.get("temp"), dict) else {}
    humidity = baselines.get("humidity", {}) if isinstance(baselines.get("humidity"), dict) else {}
    temp_lgbm = best_models.get("temp") if isinstance(best_models.get("temp"), dict) else temp.get("operational_residual_lgbm_temp", {})
    humidity_lgbm = best_models.get("humidity") if isinstance(best_models.get("humidity"), dict) else humidity.get("operational_residual_lgbm_humidity", {})
    reliability = str(summary.get("benchmark_reliability") or "smoke")
    conditions = {
        "operational_valid=true temp model exists": temp_lgbm.get("rmse") is not None,
        "temp_rmse <= 1.5": temp_lgbm.get("rmse") is not None and float(temp_lgbm["rmse"]) <= 1.5,
        "humidity_rmse <= 10": humidity_lgbm.get("rmse") is not None and float(humidity_lgbm["rmse"]) <= 10.0,
        "benchmark_reliability >= medium": reliability in {"medium", "strong", "seasonal"},
        "patch_ablation_completed": bool(summary.get("patch_ablation_completed")),
        "report_generated": bool(summary.get("artifacts", {}).get("html_report") if isinstance(summary.get("artifacts"), dict) else False),
    }
    missing = [name for name, ok in conditions.items() if not ok]
    return {"status": "PASS" if not missing else "FAIL", "conditions": conditions, "missing_conditions": missing}


def evaluate_site_readiness_gate(summary: dict[str, Any]) -> dict[str, Any]:
    baselines = summary.get("official_baselines", {}) if isinstance(summary.get("official_baselines"), dict) else {}
    best_models = summary.get("best_operational_models", {}) if isinstance(summary.get("best_operational_models"), dict) else {}
    temp = baselines.get("temp", {}) if isinstance(baselines.get("temp"), dict) else {}
    humidity = baselines.get("humidity", {}) if isinstance(baselines.get("humidity"), dict) else {}
    temp_lgbm = best_models.get("temp") if isinstance(best_models.get("temp"), dict) else temp.get("operational_residual_lgbm_temp", {})
    humidity_lgbm = best_models.get("humidity") if isinstance(best_models.get("humidity"), dict) else humidity.get("operational_residual_lgbm_humidity", {})
    reliability = str(summary.get("benchmark_reliability") or "smoke")
    conditions = {
        "temp operational_valid=true": temp_lgbm.get("rmse") is not None,
        "temp RMSE <= 1.5": temp_lgbm.get("rmse") is not None and float(temp_lgbm["rmse"]) <= 1.5,
        "humidity RMSE <= 10 beta": humidity_lgbm.get("rmse") is not None and float(humidity_lgbm["rmse"]) <= 10.0,
        "benchmark_reliability >= medium": reliability in {"medium", "strong", "seasonal"},
        "patch_ablation_completed": bool(summary.get("patch_ablation_completed")),
        "report_generated": bool(summary.get("artifacts", {}).get("html_report")) if isinstance(summary.get("artifacts"), dict) else False,
        "weather_code rule-based allowed": True,
    }
    missing = [name for name, ok in conditions.items() if not ok]
    return {"status": "PASS" if not missing else "WARN", "conditions": conditions, "missing_conditions": missing}


def build_production_model_manifest(summary: dict[str, Any]) -> dict[str, Any]:
    best = summary.get("best_operational_models", {}) if isinstance(summary.get("best_operational_models"), dict) else {}
    temp = best.get("temp", {}) if isinstance(best.get("temp"), dict) else {}
    humidity = best.get("humidity", {}) if isinstance(best.get("humidity"), dict) else {}
    site = summary.get("site_readiness", {}) if isinstance(summary.get("site_readiness"), dict) else evaluate_site_readiness_gate(summary)
    v4c = summary.get("v4c_gate", {}) if isinstance(summary.get("v4c_gate"), dict) else evaluate_v4c_gate(summary)
    temp_artifact = _target_model_artifact_summary("temp", temp, summary)
    humidity_artifact = _target_model_artifact_summary("humidity", humidity, summary)
    return {
        "temp_model_artifact": temp_artifact["primary_artifact"],
        "humidity_model_artifact": humidity_artifact["primary_artifact"],
        "temp_model": temp.get("model"),
        "humidity_model": humidity.get("model"),
        "temp_artifact_type": temp_artifact["artifact_type"],
        "humidity_artifact_type": humidity_artifact["artifact_type"],
        "temp_selected_model_key": temp_artifact["selected_model_key"],
        "humidity_selected_model_key": humidity_artifact["selected_model_key"],
        "temp_artifacts": temp_artifact,
        "humidity_artifacts": humidity_artifact,
        "temp_rmse": temp.get("rmse"),
        "humidity_rmse": humidity.get("rmse"),
        "temp_mae": temp.get("mae"),
        "humidity_mae": humidity.get("mae"),
        "temp_bias": temp.get("bias"),
        "humidity_bias": humidity.get("bias"),
        "benchmark_reliability": summary.get("benchmark_reliability"),
        "site_readiness_status": site.get("status"),
        "v4c_gate_status": v4c.get("status"),
        "operational_beta_allowed": site.get("status") == "PASS",
        "target_specific_models": True,
    }


def _target_model_artifact_summary(target: str, metrics: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    artifacts = summary.get("artifacts", {}) if isinstance(summary.get("artifacts"), dict) else {}
    model_name = str(metrics.get("model") or "")
    output: dict[str, Any] = {
        "target": target,
        "selected_model_key": model_name,
        "component_model_artifact": artifacts.get("models"),
        "artifact_type": "model",
        "primary_artifact": artifacts.get("models"),
        "ensemble_artifacts": {},
    }
    if not model_name.startswith("ensemble_"):
        return output
    ensembles = summary.get("ensembles", {}) if isinstance(summary.get("ensembles"), dict) else {}
    ensemble = ensembles.get(target, {}) if isinstance(ensembles.get(target), dict) else {}
    ensemble_artifacts = ensemble.get("artifacts", {}) if isinstance(ensemble.get("artifacts"), dict) else {}
    output.update(
        {
            "artifact_type": "ensemble",
            "ensemble_method": model_name.removeprefix("ensemble_"),
            "ensemble_artifacts": ensemble_artifacts,
            "primary_artifact": ensemble_artifacts.get("weights") or artifacts.get("models"),
            "component_model_artifact": artifacts.get("models"),
        }
    )
    return output


def write_operational_performance_html(summary: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = output.parent / "summary_snapshot.json"
    snapshot.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    rows = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Operational Performance Report</title>",
        "<style>body{font-family:Inter,Segoe UI,Arial,sans-serif;margin:32px;background:#f8fafc;color:#0f172a}table{border-collapse:collapse;width:100%;background:white;margin:16px 0}th,td{border:1px solid #cbd5e1;padding:8px;text-align:left}th{background:#e2e8f0}.card{background:white;border:1px solid #cbd5e1;border-radius:12px;padding:16px;margin:16px 0}.pass{color:#15803d;font-weight:700}.fail{color:#b91c1c;font-weight:700}.warn{color:#b45309;font-weight:700}code{background:#e2e8f0;padding:1px 4px;border-radius:4px}</style>",
        "</head><body>",
        "<h1>Operational Performance Report</h1>",
        f"<div class='card'><b>benchmark_reliability:</b> <code>{html.escape(str(summary.get('benchmark_reliability')))}</code></div>",
        _html_metric_table(summary.get("official_baselines", {})),
        _html_gate("V4-C Gate", summary.get("v4c_gate", {})),
        _html_gate("Site Readiness", summary.get("site_readiness", {})),
        _html_section("Calibration", summary.get("calibration", {})),
        _html_section("Patch Ablation", summary.get("patch_ablation", {})),
        _html_section("Patch Improvement", summary.get("patch_improvement", {})),
        _html_section("Variable Coverage", summary.get("variable_coverage", {})),
        _html_section("Ensembles", summary.get("ensembles", {})),
        _html_section("CatBoost", summary.get("catboost", {})),
        _html_section("Production Model Manifest", summary.get("production_model_manifest", {})),
        _html_section("Ridge Residual Debug", summary.get("ridge_debug", {})),
        _html_section("Artifacts", summary.get("artifacts", {})),
        "</body></html>",
    ]
    output.write_text("\n".join(rows), encoding="utf-8")
    return output


def run_operational_benchmark(
    *,
    nwp_archive: str | Path,
    archive_quality_report: str | Path,
    observations: str | Path,
    station_metadata: str | Path,
    output_dir: str | Path,
    lgbm_grid: list[dict[str, Any]] | None = None,
    grid_patch_features: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    quality = json.loads(Path(archive_quality_report).read_text(encoding="utf-8"))
    if quality.get("forecast_archive_adequate") is not True:
        raise ValueError(f"forecast archive is inadequate: {quality.get('blocking_reasons')}")
    frame = load_joined_operational_frame(nwp_archive=nwp_archive, observations=observations, station_metadata=station_metadata)
    frame = add_time_ordered_splits(frame, train_cycles=20, val_cycles=5, test_cycles=5)
    frame, true_patch_features = _with_true_grid_patch_features(frame, grid_patch_features)
    benchmark = _benchmark_info(frame)
    variable_coverage = summarize_variable_coverage(frame)
    models: dict[str, Any] = {}
    predictions = frame.loc[frame["split"].eq("test"), ["station_id", "issue_time", "valid_time", "horizon_step", "temp", "humidity", "nwp_t2m", "nwp_humidity"]].copy()
    summary: dict[str, Any] = {
        "data": {
            "nwp_archive": str(nwp_archive),
            "archive_quality_report": str(archive_quality_report),
            "observations": str(observations),
            "joined_rows": int(len(frame)),
            "stations": int(frame["station_id"].nunique()),
            "forecast_cycles": int(frame["issue_time"].nunique()),
            "horizon_min": int(frame["horizon_step"].min()),
            "horizon_max": int(frame["horizon_step"].max()),
            "split_rows": {str(k): int(v) for k, v in frame["split"].value_counts().to_dict().items()},
            "split_cycles": {str(k): int(v) for k, v in frame.groupby("split")["issue_time"].nunique().to_dict().items()},
        },
        **benchmark,
        "official_baselines": {"temp": {}, "humidity": {}},
        "metrics": {},
        "calibration": {},
        "patch_ablation": {},
        "patch_improvement": {},
        "patch_ablation_completed": False,
        "patch_modes_present": [],
        "patch_ablation_required_modes": ["no_patch", "proxy_patch5", "true_patch3", "true_patch5"],
        "patch_ablation_missing_modes": ["no_patch", "proxy_patch5", "true_patch3", "true_patch5"],
        "variable_coverage": variable_coverage,
        "catboost": {},
        "ensembles": {},
        "best_operational_models": {},
        "ridge_debug": {},
        "artifacts": {},
    }
    grid = lgbm_grid or DEFAULT_LGBM_GRID
    pd.DataFrame(variable_coverage.get("variables", [])).to_csv(output / "variable_coverage.csv", index=False)
    patch_modes = [
        {"mode": "no_patch", "feature_columns": [], "patch_feature_mode": "none"},
        {"mode": "proxy_patch5", "feature_columns": _station_neighbor_features(frame, 5), "patch_feature_mode": "station_neighborhood_proxy"},
    ]
    for patch_size in sorted(true_patch_features):
        patch_modes.append(
            {
                "mode": f"true_patch{patch_size}",
                "feature_columns": true_patch_features[patch_size],
                "patch_feature_mode": PATCH_FEATURE_MODE,
            }
        )
    summary.update(patch_ablation_status([str(mode["mode"]) for mode in patch_modes]))
    for target_name, target_cfg in TARGETS.items():
        actual = target_cfg["actual"]
        baseline = target_cfg["baseline"]
        raw_name = target_cfg["official_raw"]
        lgbm_name = target_cfg["official_lgbm"]
        test = frame.loc[frame["split"].eq("test")].copy()
        raw_metrics = metric_dict(test[actual], test[baseline])
        summary["official_baselines"][target_name][raw_name] = raw_metrics
        summary["metrics"][raw_name] = {"test": raw_metrics}
        predictions[f"{raw_name}_prediction"] = test[baseline].to_numpy(dtype=float)

        ridge_payload = _fit_ridge_debug(frame, actual_column=actual, baseline_column=baseline, target_name=target_name, output_dir=output)
        summary["ridge_debug"][target_name] = ridge_payload["debug"]
        predictions[f"{target_name}_ridge_prediction"] = ridge_payload["test_prediction"]
        models[f"{target_name}_ridge"] = ridge_payload["model"]

        best_mode: str | None = None
        best_summary: dict[str, Any] | None = None
        best_model: Pipeline | None = None
        best_test_prediction: np.ndarray | None = None
        best_validation_rmse = float("inf")
        best_validation_metrics: dict[str, float | int] | None = None
        ablation_rows: list[dict[str, Any]] = []
        feature_importance_rows: list[pd.DataFrame] = []
        component_predictions: dict[str, dict[str, pd.DataFrame]] = {}
        for patch_mode in patch_modes:
            mode = str(patch_mode["mode"])
            extra_features = _target_patch_feature_columns(target_name, mode, list(patch_mode["feature_columns"]))
            patch_feature_mode = str(patch_mode["patch_feature_mode"])
            feature_cols = [column for column in [*BASE_FEATURES, *extra_features] if column in frame.columns]
            tuning = _tune_lgbm(frame, actual_column=actual, baseline_column=baseline, feature_columns=feature_cols, grid=grid)
            model = tuning["model"]
            val_pred = tuning["val_prediction_frame"]
            val_cycles = sorted(val_pred["issue_time"].drop_duplicates())
            split_at = max(1, len(val_cycles) // 2)
            calibration_frame = val_pred.loc[val_pred["issue_time"].isin(val_cycles[:split_at])].copy()
            holdout_frame = val_pred.loc[val_pred["issue_time"].isin(val_cycles[split_at:])].copy()
            if holdout_frame.empty:
                holdout_frame = calibration_frame.copy()
            calibration = fit_calibration_candidates(
                calibration_frame,
                holdout_frame,
                actual_column=actual,
                prediction_column="prediction",
                candidates=CALIBRATION_CANDIDATES,
            )
            corrected_holdout = calibration.selected.apply(holdout_frame)
            use_corrected = calibration.selected_name != "none"
            final_holdout = corrected_holdout if use_corrected else holdout_frame["prediction"].to_numpy(dtype=float)
            test_frame = tuning["test_prediction_frame"].copy()
            corrected_test = calibration.selected.apply(test_frame)
            test_metrics_raw = metric_dict(test_frame[actual], test_frame["prediction"])
            test_metrics_corrected = metric_dict(test_frame[actual], corrected_test)
            final_prediction = corrected_test if use_corrected else test_frame["prediction"].to_numpy(dtype=float)
            final_metrics = test_metrics_corrected if use_corrected else test_metrics_raw
            holdout_metrics = metric_dict(holdout_frame[actual], final_holdout)
            worst_station = _worst_group_rmse(test_frame.assign(final_prediction=final_prediction), actual, "final_prediction", "station_id")
            late_horizon = test_frame.loc[test_frame["horizon_step"].between(12, 24)].copy()
            late_metrics = metric_dict(late_horizon[actual], pd.Series(final_prediction, index=test_frame.index).loc[late_horizon.index])
            ablation = {
                "target": target_name,
                "mode": mode,
                "patch_feature_mode": patch_feature_mode,
                "rmse": final_metrics["rmse"],
                "mae": final_metrics["mae"],
                "bias": final_metrics["bias"],
                "worst_station_rmse": worst_station,
                "late_horizon_rmse": late_metrics["rmse"],
                "selected_calibration": calibration.selected_name if use_corrected else "none_holdout_not_improved",
                "selection_holdout_rmse": holdout_metrics["rmse"],
                "best_params": tuning["best_params"],
                "validation_rmse": tuning["best_validation_rmse"],
                "feature_count": len(feature_cols),
            }
            component_name = f"lgbm_{mode}"
            holdout_component = holdout_frame.copy()
            holdout_component["prediction"] = final_holdout
            test_component = test_frame.copy()
            test_component["prediction"] = final_prediction
            component_predictions[component_name] = {"val": holdout_component, "test": test_component}
            ablation_rows.append(ablation)
            _write_candidate_rows(calibration.candidate_results, output / f"{target_name}_{mode}_calibration_candidate_results.csv")
            _write_grid_rows(tuning["grid_results"], output / f"{target_name}_{mode}_lgbm_grid_search_results.csv")
            if tuning.get("feature_importance") is not None:
                fi = tuning["feature_importance"].copy()
                fi.insert(0, "target", target_name)
                fi.insert(1, "mode", mode)
                feature_importance_rows.append(fi)
            if _is_official_lgbm_candidate(target_name, mode) and float(holdout_metrics["rmse"]) < best_validation_rmse:
                best_mode = mode
                best_summary = ablation
                best_model = model
                best_test_prediction = final_prediction
                best_validation_rmse = float(holdout_metrics["rmse"])
                best_validation_metrics = holdout_metrics
                summary["calibration"][target_name] = {
                    "mode": mode,
                    "selected": calibration.selected.to_jsonable() if use_corrected else CalibrationModel("none", {}).to_jsonable(),
                    "selected_holdout_metrics": calibration.selected_holdout_metrics,
                    "raw_test_metrics": test_metrics_raw,
                    "corrected_test_metrics": test_metrics_corrected,
                    "corrected_applied_to_test": use_corrected,
                }
        if best_summary is None or best_model is None or best_test_prediction is None:
            raise RuntimeError(f"No LGBM model was trained for {target_name}.")
        catboost_payload = _fit_optional_catboost(frame, actual_column=actual, baseline_column=baseline, feature_columns=[column for column in BASE_FEATURES if column in frame.columns])
        summary["catboost"][target_name] = catboost_payload["summary"]
        if catboost_payload.get("model") is not None:
            component_predictions["catboost_residual"] = {
                "val": catboost_payload["val_prediction_frame"],
                "test": catboost_payload["test_prediction_frame"],
            }
            models[f"{target_name}_catboost"] = catboost_payload["model"]
        if best_validation_metrics is None:
            best_validation_metrics = {"rmse": best_validation_rmse, "mae": float("nan"), "bias": float("nan"), "n": 0}
        ensemble_payload = build_ensemble_results(
            component_predictions,
            actual_column=actual,
            output_dir=output,
            target_name=target_name,
            baseline_validation_metrics=best_validation_metrics,
            baseline_test_metrics=best_summary,
            reject_if_bias_worse=target_name == "humidity",
        )
        summary["ensembles"][target_name] = ensemble_payload["summary"]
        if ensemble_payload["best_test_prediction"] is not None:
            predictions[f"{target_name}_ensemble_prediction"] = ensemble_payload["best_test_prediction"]
        summary["official_baselines"][target_name][lgbm_name] = {
            "rmse": best_summary["rmse"],
            "mae": best_summary["mae"],
            "bias": best_summary["bias"],
            "n": int(test.shape[0]),
            "patch_mode": best_mode,
            "selected_calibration": best_summary["selected_calibration"],
        }
        best_operational = {
            "model": lgbm_name,
            "validation_rmse": best_validation_rmse,
            **summary["official_baselines"][target_name][lgbm_name],
        }
        ensemble_best = ensemble_payload["summary"].get("best") if isinstance(ensemble_payload["summary"], dict) else None
        ensemble_accepted = isinstance(ensemble_payload.get("summary"), dict) and ensemble_payload["summary"].get("selection_status") == "accepted"
        if ensemble_accepted and isinstance(ensemble_best, dict) and ensemble_best.get("validation_rmse") is not None and float(ensemble_best["validation_rmse"]) < best_operational["validation_rmse"]:
            best_operational = {"model": f"ensemble_{ensemble_best['method']}", **ensemble_best}
        summary["best_operational_models"][target_name] = best_operational
        summary["metrics"][lgbm_name] = {"test": summary["official_baselines"][target_name][lgbm_name]}
        summary["patch_ablation"][target_name] = ablation_rows
        summary["patch_improvement"][target_name] = summarize_patch_ablation_results(ablation_rows)
        predictions[f"{lgbm_name}_prediction"] = best_test_prediction
        models[lgbm_name] = best_model
        if feature_importance_rows:
            pd.concat(feature_importance_rows, ignore_index=True).to_csv(output / f"{target_name}_feature_importance.csv", index=False)
        pd.DataFrame(ablation_rows).to_csv(output / f"{target_name}_patch_ablation_results.csv", index=False)
        Path(output / f"{target_name}_best_lgbm_params.json").write_text(json.dumps(best_summary["best_params"], indent=2), encoding="utf-8")

    summary["v4c_gate"] = evaluate_v4c_gate(summary)
    summary["site_readiness"] = evaluate_site_readiness_gate(summary)
    predictions.to_csv(output / "predictions_test.csv", index=False)
    frame.to_csv(output / "joined_training_frame.csv", index=False)
    with (output / "models.pkl").open("wb") as handle:
        pickle.dump(models, handle)
    summary["artifacts"] = {
        "summary_json": str(output / "experiment_summary.json"),
        "html_report": str(output / "operational_performance_report.html"),
        "predictions_test": str(output / "predictions_test.csv"),
        "joined_training_frame": str(output / "joined_training_frame.csv"),
        "models": str(output / "models.pkl"),
        "variable_coverage": str(output / "variable_coverage.csv"),
    }
    summary["v4c_gate"] = evaluate_v4c_gate(summary)
    summary["site_readiness"] = evaluate_site_readiness_gate(summary)
    manifest_path = output / "production_model_manifest.json"
    manifest = build_production_model_manifest(summary)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    summary["production_model_manifest"] = manifest
    summary["artifacts"]["production_model_manifest"] = str(manifest_path)
    (output / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    write_operational_performance_html(summary, output / "operational_performance_report.html")
    _write_markdown_report(summary, output / "operational_performance_report.md")
    return summary


def load_joined_operational_frame(*, nwp_archive: str | Path, observations: str | Path, station_metadata: str | Path) -> pd.DataFrame:
    nwp = pd.read_csv(nwp_archive, dtype={"station_id": str}, low_memory=False)
    nwp["station_id"] = nwp["station_id"].astype(str)
    for column in ["forecast_init_time", "issue_time", "valid_time"]:
        if column in nwp.columns:
            nwp[column] = pd.to_datetime(nwp[column], utc=True, format="mixed")
    if "issue_time" not in nwp.columns:
        nwp["issue_time"] = nwp["forecast_init_time"]
    _normalize_nwp_columns_inplace(nwp)
    obs = pd.read_csv(observations, dtype={"station_id": str}, low_memory=False)
    obs["station_id"] = obs["station_id"].astype(str)
    obs["obs_time_kst"] = pd.to_datetime(obs["datetime"], errors="coerce")
    obs["valid_time"] = obs["obs_time_kst"].dt.tz_localize("Asia/Seoul").dt.tz_convert("UTC")
    for column in ["temp", "humidity", "pressure", "wind_speed", "precipitation"]:
        if column in obs.columns:
            obs[column] = pd.to_numeric(obs[column], errors="coerce")
    if "precipitation" in obs.columns:
        obs["precipitation"] = obs["precipitation"].fillna(0.0)
    stations = pd.read_csv(station_metadata, dtype={"station_id": str})
    keep_station_cols = [column for column in ["station_id", "lat", "lon", "elevation", "region", "region_class", "coastal_distance_km"] if column in stations.columns]
    frame = nwp.merge(obs[["station_id", "valid_time", "temp", "humidity", "pressure", "wind_speed", "precipitation"]], on=["station_id", "valid_time"], how="inner")
    frame = frame.merge(stations[keep_station_cols], on="station_id", how="left")
    frame = pd.concat([frame, _cyc_features(frame["issue_time"], "issue"), _cyc_features(frame["valid_time"], "valid")], axis=1)
    _add_operational_derived_features(frame)
    frame = frame.dropna(subset=["temp", "humidity", "nwp_t2m", "nwp_humidity"]).sort_values(["issue_time", "station_id", "horizon_step"]).reset_index(drop=True)
    return frame


def _normalize_nwp_columns_inplace(frame: pd.DataFrame) -> None:
    alias_pairs = {
        "nwp_humidity": ["nwp_relative_humidity", "nwp_relative_humidity_2m", "gfs_relative_humidity_2m"],
        "nwp_t2m": ["nwp_temp_2m_c", "gfs_temp_2m_c"],
        "nwp_dew_point": ["nwp_dew_point_2m_c", "gfs_dew_point_2m_c"],
        "nwp_sp": ["nwp_surface_pressure", "gfs_surface_pressure"],
        "nwp_tp": ["nwp_total_precipitation", "gfs_total_precipitation"],
    }
    for canonical, aliases in alias_pairs.items():
        if canonical not in frame.columns:
            for alias in aliases:
                if alias in frame.columns:
                    frame[canonical] = frame[alias]
                    break
    for column in FULL_VARIABLE_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ["nwp_t2m", "nwp_dew_point", "nwp_soil_temperature"]:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            frame[column] = np.where(values > 150.0, values - 273.15, values)
    for column in ["nwp_sp", "nwp_mslp"]:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            frame[column] = np.where(values > 2000.0, values / 100.0, values)


def _add_operational_derived_features(frame: pd.DataFrame) -> None:
    if {"nwp_t2m", "nwp_dew_point"}.issubset(frame.columns):
        frame["nwp_dewpoint_depression"] = pd.to_numeric(frame["nwp_t2m"], errors="coerce") - pd.to_numeric(frame["nwp_dew_point"], errors="coerce")
    if {"nwp_u10", "nwp_v10"}.issubset(frame.columns):
        u = pd.to_numeric(frame["nwp_u10"], errors="coerce")
        v = pd.to_numeric(frame["nwp_v10"], errors="coerce")
        if "nwp_wind_speed" not in frame.columns:
            frame["nwp_wind_speed"] = np.sqrt(u * u + v * v)
        if "nwp_wind_direction" not in frame.columns:
            frame["nwp_wind_direction"] = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    if "nwp_humidity" in frame.columns:
        rh = pd.to_numeric(frame["nwp_humidity"], errors="coerce").clip(lower=0.1, upper=99.9) / 100.0
        frame["nwp_humidity_logit"] = np.log(rh / (1.0 - rh))
    if {"nwp_t2m", "nwp_sp"}.issubset(frame.columns):
        frame["nwp_temp_pressure_interaction"] = pd.to_numeric(frame["nwp_t2m"], errors="coerce") * pd.to_numeric(frame["nwp_sp"], errors="coerce")


def add_time_ordered_splits(frame: pd.DataFrame, *, train_cycles: int, val_cycles: int, test_cycles: int) -> pd.DataFrame:
    output = frame.copy()
    cycles = sorted(output["issue_time"].drop_duplicates())
    needed = train_cycles + val_cycles + test_cycles
    if len(cycles) < needed:
        raise ValueError(f"Need at least {needed} forecast cycles; got {len(cycles)}")
    selected = cycles if len(cycles) > needed else cycles[-needed:]
    if len(selected) == needed:
        train_count, val_count, test_count = train_cycles, val_cycles, test_cycles
    else:
        total = len(selected)
        test_count = max(test_cycles, int(np.ceil(total * 0.15)))
        val_count = max(val_cycles, int(np.ceil(total * 0.15)))
        train_count = total - val_count - test_count
        if train_count < train_cycles:
            train_count = train_cycles
            remaining = total - train_count
            val_count = max(1, remaining // 2)
            test_count = max(1, remaining - val_count)
    split_map = {cycle: "train" for cycle in selected[:train_count]}
    split_map.update({cycle: "val" for cycle in selected[train_count : train_count + val_count]})
    split_map.update({cycle: "test" for cycle in selected[train_count + val_count : train_count + val_count + test_count]})
    output = output.loc[output["issue_time"].isin(selected)].copy()
    output["split"] = output["issue_time"].map(split_map)
    return output


def _with_true_grid_patch_features(frame: pd.DataFrame, feature_csv: str | Path | None) -> tuple[pd.DataFrame, dict[int, list[str]]]:
    if feature_csv is None:
        return frame, {}
    path = Path(feature_csv)
    if not path.exists():
        raise FileNotFoundError(path)
    features = pd.read_csv(path, dtype={"station_id": str}, low_memory=False)
    if features.empty:
        return frame, {}
    if "patch_feature_mode" not in features.columns:
        raise ValueError(f"{path} is missing patch_feature_mode; true GFS grid patch features must declare provenance.")
    modes = {str(value) for value in features["patch_feature_mode"].dropna().unique()}
    if modes != {PATCH_FEATURE_MODE}:
        raise ValueError(f"{path} contains non-true-grid patch_feature_mode values: {sorted(modes)}")
    if "patch_size" not in features.columns:
        raise ValueError(f"{path} is missing patch_size; true GFS grid patch features must include patch size.")
    for column in ["forecast_init_time", "valid_time"]:
        if column in features.columns:
            features[column] = pd.to_datetime(features[column], utc=True)
    features["station_id"] = features["station_id"].astype(str)
    features["horizon_step"] = pd.to_numeric(features["horizon_step"], errors="coerce").astype("Int64")
    id_columns = {"station_id", "forecast_init_time", "valid_time", "horizon_step", "source", "patch_size", "patch_feature_mode"}
    numeric_feature_columns = [
        column
        for column in features.columns
        if column not in id_columns and pd.api.types.is_numeric_dtype(features[column])
    ]
    if not numeric_feature_columns:
        raise ValueError(f"{path} contains no numeric true-grid patch feature columns.")
    output: dict[int, list[str]] = {}
    merged_frame = frame.copy()
    for patch_size, group in features.groupby("patch_size"):
        size = int(patch_size)
        rename = {column: f"true_patch{size}_{column}" for column in numeric_feature_columns}
        subset = (
            group[["station_id", "forecast_init_time", "valid_time", "horizon_step", *numeric_feature_columns]]
            .drop_duplicates(["station_id", "forecast_init_time", "valid_time", "horizon_step"], keep="last")
            .rename(columns=rename)
        )
        before_columns = set(merged_frame.columns)
        merged_frame = merged_frame.merge(
            subset,
            on=["station_id", "forecast_init_time", "valid_time", "horizon_step"],
            how="left",
        )
        output[size] = [column for column in merged_frame.columns if column not in before_columns and column.startswith(f"true_patch{size}_")]
        if not output[size]:
            raise ValueError(f"{path} patch_size={size} did not contribute numeric true-grid patch features.")
    return merged_frame.copy(), output


def patch_ablation_status(modes: Iterable[str]) -> dict[str, Any]:
    required = ["no_patch", "proxy_patch5", "true_patch3", "true_patch5"]
    present = sorted({str(mode) for mode in modes})
    missing = [mode for mode in required if mode not in present]
    return {
        "patch_ablation_completed": not missing,
        "patch_modes_present": present,
        "patch_ablation_required_modes": required,
        "patch_ablation_missing_modes": missing,
    }


def _target_patch_feature_columns(target_name: str, mode: str, columns: list[str]) -> list[str]:
    """Apply target-specific patch policy without forcing one shared pipeline.

    Temperature keeps the full true-grid/proxy patch feature set because G025 showed
    patch5 helped temperature. Humidity keeps `no_patch` as the reference path and
    only lets moisture-regime true-grid patch summaries compete, avoiding the broad
    patch noise that worsened the G025 medium humidity ensemble.
    """

    if target_name != "humidity" or mode == "no_patch":
        return columns
    if mode == "proxy_patch5":
        return []
    return [column for column in columns if any(hint in column for hint in HUMIDITY_PATCH_VARIABLE_HINTS)]


def _is_official_lgbm_candidate(target_name: str, mode: str) -> bool:
    """Return whether a patch mode may define the official residual LGBM path.

    G026 deliberately keeps humidity's official residual LightGBM baseline pinned
    to no-patch. Moisture patch variants still run as ablations and ensemble
    candidates, but they do not replace the target-specific official baseline.
    """

    return target_name != "humidity" or mode == "no_patch"


def _fit_calibration_model(frame: pd.DataFrame, *, actual_column: str, prediction_column: str, name: str) -> CalibrationModel:
    actual = pd.to_numeric(frame[actual_column], errors="coerce").astype(float)
    prediction = pd.to_numeric(frame[prediction_column], errors="coerce").astype(float)
    residual = actual - prediction
    global_bias = float(residual.mean()) if len(residual.dropna()) else 0.0
    if name == "none":
        return CalibrationModel(name="none", payload={}, actual_column=actual_column, prediction_column=prediction_column)
    if name == "global_mean_bias":
        return CalibrationModel(name=name, payload={"global_bias": global_bias}, actual_column=actual_column, prediction_column=prediction_column)
    if name == "per_horizon_mean_bias":
        groups = residual.groupby(frame["horizon_step"].astype(int).astype(str)).mean().to_dict()
        return CalibrationModel(name=name, payload={"global_bias": global_bias, "groups": {str(k): float(v) for k, v in groups.items()}}, actual_column=actual_column, prediction_column=prediction_column)
    if name == "per_station_horizon_mean_bias":
        key = frame["station_id"].astype(str) + "|" + frame["horizon_step"].astype(int).astype(str)
        groups = residual.groupby(key).mean().to_dict()
        horizon_groups = residual.groupby(frame["horizon_step"].astype(int).astype(str)).mean().to_dict()
        return CalibrationModel(name=name, payload={"global_bias": global_bias, "groups": {str(k): float(v) for k, v in groups.items()}, "horizon_groups": {str(k): float(v) for k, v in horizon_groups.items()}}, actual_column=actual_column, prediction_column=prediction_column)
    if name == "per_station_month_hour_mean_bias":
        valid_time = pd.to_datetime(frame["valid_time"], utc=True)
        key = frame["station_id"].astype(str) + "|" + valid_time.dt.month.astype(str) + "|" + valid_time.dt.hour.astype(str)
        groups = residual.groupby(key).mean().to_dict()
        station_groups = residual.groupby(frame["station_id"].astype(str)).mean().to_dict()
        return CalibrationModel(
            name=name,
            payload={
                "global_bias": global_bias,
                "groups": {str(k): float(v) for k, v in groups.items()},
                "station_groups": {str(k): float(v) for k, v in station_groups.items()},
            },
            actual_column=actual_column,
            prediction_column=prediction_column,
        )
    if name == "per_region_horizon_mean_bias":
        region_col = "region_class" if "region_class" in frame.columns else "region" if "region" in frame.columns else None
        if region_col is None:
            return CalibrationModel(name="global_mean_bias", payload={"global_bias": global_bias}, actual_column=actual_column, prediction_column=prediction_column)
        key = frame[region_col].astype(str) + "|" + frame["horizon_step"].astype(int).astype(str)
        groups = residual.groupby(key).mean().to_dict()
        horizon_groups = residual.groupby(frame["horizon_step"].astype(int).astype(str)).mean().to_dict()
        return CalibrationModel(name=name, payload={"global_bias": global_bias, "groups": {str(k): float(v) for k, v in groups.items()}, "horizon_groups": {str(k): float(v) for k, v in horizon_groups.items()}, "region_column": region_col}, actual_column=actual_column, prediction_column=prediction_column)
    if name == "global_affine":
        slope, intercept = _fit_affine(prediction.to_numpy(), actual.to_numpy())
        return CalibrationModel(name=name, payload={"global": {"slope": slope, "intercept": intercept}}, actual_column=actual_column, prediction_column=prediction_column)
    if name == "per_station_horizon_affine":
        groups: dict[str, dict[str, float]] = {}
        for key_value, group in frame.groupby(frame["station_id"].astype(str) + "|" + frame["horizon_step"].astype(int).astype(str)):
            slope, intercept = _fit_affine(group[prediction_column].to_numpy(dtype=float), group[actual_column].to_numpy(dtype=float))
            groups[str(key_value)] = {"slope": slope, "intercept": intercept}
        slope, intercept = _fit_affine(prediction.to_numpy(), actual.to_numpy())
        return CalibrationModel(name=name, payload={"global": {"slope": slope, "intercept": intercept}, "groups": groups}, actual_column=actual_column, prediction_column=prediction_column)
    if name == "per_region_horizon_affine":
        region_col = "region_class" if "region_class" in frame.columns else "region" if "region" in frame.columns else None
        if region_col is None:
            return _fit_calibration_model(frame, actual_column=actual_column, prediction_column=prediction_column, name="global_affine")
        groups = {}
        group_key = frame[region_col].astype(str) + "|" + frame["horizon_step"].astype(int).astype(str)
        for key_value, group in frame.groupby(group_key):
            slope, intercept = _fit_affine(group[prediction_column].to_numpy(dtype=float), group[actual_column].to_numpy(dtype=float))
            groups[str(key_value)] = {"slope": slope, "intercept": intercept}
        slope, intercept = _fit_affine(prediction.to_numpy(), actual.to_numpy())
        return CalibrationModel(
            name=name,
            payload={"global": {"slope": slope, "intercept": intercept}, "groups": groups, "region_column": region_col},
            actual_column=actual_column,
            prediction_column=prediction_column,
        )
    if name == "quantile_mapping":
        quantiles = np.linspace(0, 1, 21)
        return CalibrationModel(name=name, payload={"prediction_quantiles": np.quantile(prediction.dropna(), quantiles).tolist(), "actual_quantiles": np.quantile(actual.dropna(), quantiles).tolist()}, actual_column=actual_column, prediction_column=prediction_column)
    if name in {"isotonic", "isotonic_calibration"}:
        try:
            from sklearn.isotonic import IsotonicRegression
        except Exception:  # pragma: no cover
            return CalibrationModel(name="none", payload={}, actual_column=actual_column, prediction_column=prediction_column)
        model = IsotonicRegression(out_of_bounds="clip")
        clean = pd.DataFrame({"actual": actual, "prediction": prediction}).dropna()
        if len(clean) < 2:
            return CalibrationModel(name="none", payload={}, actual_column=actual_column, prediction_column=prediction_column)
        model.fit(clean["prediction"].to_numpy(), clean["actual"].to_numpy())
        return CalibrationModel(name=name, payload={"model": model}, actual_column=actual_column, prediction_column=prediction_column)
    raise ValueError(f"Unknown calibration candidate: {name}")


def _fit_affine(prediction: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(prediction) & np.isfinite(actual)
    if int(mask.sum()) < 2 or float(np.std(prediction[mask])) == 0.0:
        return 1.0, float(np.nanmean(actual[mask] - prediction[mask])) if int(mask.sum()) else 0.0
    slope, intercept = np.polyfit(prediction[mask], actual[mask], 1)
    return float(slope), float(intercept)


def _tune_lgbm(frame: pd.DataFrame, *, actual_column: str, baseline_column: str, feature_columns: list[str], grid: list[dict[str, Any]]) -> dict[str, Any]:
    if LGBMRegressor is None:
        raise RuntimeError("LightGBM is required for operational performance tuning.")
    train = frame.loc[frame["split"].eq("train")].copy()
    val = frame.loc[frame["split"].eq("val")].copy()
    test = frame.loc[frame["split"].eq("test")].copy()
    feature_columns = _drop_train_all_missing_features(train, feature_columns)
    categorical = [column for column in ["station_id", "region", "region_class"] if column in feature_columns]
    numeric = [column for column in feature_columns if column not in categorical]
    best: dict[str, Any] | None = None
    grid_rows = []
    for idx, params in enumerate(grid):
        model = Pipeline(
            [
                ("prep", _preprocessor(numeric, categorical)),
                ("model", LGBMRegressor(**_lgbm_params(params))),
            ]
        )
        y_train = train[actual_column].astype(float) - train[baseline_column].astype(float)
        model.fit(train[feature_columns], y_train)
        val_pred = val[baseline_column].astype(float).to_numpy() + model.predict(val[feature_columns])
        val_metrics = metric_dict(val[actual_column], val_pred)
        row = {"grid_index": idx, **params, **{f"val_{k}": v for k, v in val_metrics.items()}}
        grid_rows.append(row)
        if best is None or float(val_metrics["rmse"]) < float(best["metrics"]["rmse"]):
            best = {"model": model, "params": params, "metrics": val_metrics}
    if best is None:
        raise RuntimeError("LGBM grid search did not run.")
    model = best["model"]
    val_prediction = val[baseline_column].astype(float).to_numpy() + model.predict(val[feature_columns])
    test_prediction = test[baseline_column].astype(float).to_numpy() + model.predict(test[feature_columns])
    val_frame = val[["station_id", "issue_time", "valid_time", "horizon_step", actual_column, baseline_column, *[c for c in ["region", "region_class"] if c in val.columns]]].copy()
    test_frame = test[["station_id", "issue_time", "valid_time", "horizon_step", actual_column, baseline_column, *[c for c in ["region", "region_class"] if c in test.columns]]].copy()
    val_frame["prediction"] = val_prediction
    test_frame["prediction"] = test_prediction
    return {
        "model": model,
        "best_params": best["params"],
        "best_validation_rmse": best["metrics"]["rmse"],
        "grid_results": grid_rows,
        "val_prediction_frame": val_frame,
        "test_prediction_frame": test_frame,
        "feature_importance": _feature_importance(model, numeric, categorical),
    }


def _fit_optional_catboost(frame: pd.DataFrame, *, actual_column: str, baseline_column: str, feature_columns: list[str]) -> dict[str, Any]:
    if CatBoostRegressor is None:
        return {
            "summary": {"status": "skipped", "reason": "catboost is not installed"},
            "model": None,
            "val_prediction_frame": None,
            "test_prediction_frame": None,
        }
    train = frame.loc[frame["split"].eq("train")].copy()
    val = frame.loc[frame["split"].eq("val")].copy()
    test = frame.loc[frame["split"].eq("test")].copy()
    feature_columns = _drop_train_all_missing_features(train, feature_columns)
    cat_columns = [column for column in ["station_id", "region", "region_class", "horizon_step"] if column in feature_columns]
    cat_indices = [feature_columns.index(column) for column in cat_columns]
    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=700,
        depth=8,
        learning_rate=0.03,
        l2_leaf_reg=5.0,
        random_seed=20260528,
        verbose=False,
        allow_writing_files=False,
    )
    train_x = train[feature_columns].copy()
    val_x = val[feature_columns].copy()
    test_x = test[feature_columns].copy()
    for column in cat_columns:
        train_x[column] = train_x[column].astype(str)
        val_x[column] = val_x[column].astype(str)
        test_x[column] = test_x[column].astype(str)
    y_train = train[actual_column].astype(float) - train[baseline_column].astype(float)
    model.fit(train_x, y_train, cat_features=cat_indices)
    val_prediction = val[baseline_column].astype(float).to_numpy() + model.predict(val_x)
    test_prediction = test[baseline_column].astype(float).to_numpy() + model.predict(test_x)
    val_frame = val[["station_id", "issue_time", "valid_time", "horizon_step", actual_column, baseline_column, *[c for c in ["region", "region_class"] if c in val.columns]]].copy()
    test_frame = test[["station_id", "issue_time", "valid_time", "horizon_step", actual_column, baseline_column, *[c for c in ["region", "region_class"] if c in test.columns]]].copy()
    val_frame["prediction"] = val_prediction
    test_frame["prediction"] = test_prediction
    summary = {
        "status": "trained",
        "validation": metric_dict(val_frame[actual_column], val_frame["prediction"]),
        "test": metric_dict(test_frame[actual_column], test_frame["prediction"]),
        "categorical_features": cat_columns,
    }
    return {"summary": summary, "model": model, "val_prediction_frame": val_frame, "test_prediction_frame": test_frame}


def build_ensemble_results(
    components: dict[str, dict[str, pd.DataFrame]],
    *,
    actual_column: str,
    output_dir: str | Path,
    target_name: str,
    baseline_validation_metrics: dict[str, Any] | None = None,
    baseline_test_metrics: dict[str, Any] | None = None,
    reject_if_bias_worse: bool = False,
) -> dict[str, Any]:
    if len(components) < 2:
        return {"summary": {"status": "skipped", "reason": "fewer than two components"}, "best_test_prediction": None}
    val_matrix, val_actual, val_meta = _component_matrix(components, split="val", actual_column=actual_column)
    test_matrix, test_actual, test_meta = _component_matrix(components, split="test", actual_column=actual_column)
    if val_matrix.empty or test_matrix.empty:
        return {"summary": {"status": "skipped", "reason": "components do not align"}, "best_test_prediction": None}
    methods = {
        "simple_average": _simple_average_weights(list(val_matrix.columns)),
        "inverse_rmse_weight": _inverse_rmse_weights(val_matrix, val_actual),
    }
    constrained = _constrained_least_squares_weights(val_matrix, val_actual)
    if constrained is not None:
        methods["constrained_least_squares"] = constrained
    baseline_validation_rmse = float((baseline_validation_metrics or {}).get("rmse", float("inf")))
    baseline_validation_bias_abs = abs(float((baseline_validation_metrics or {}).get("bias", float("inf"))))
    rows: list[dict[str, Any]] = []
    predictions_by_method: dict[str, np.ndarray] = {}
    weight_payload: dict[str, Any] = {}
    for method, weights in methods.items():
        val_prediction = _weighted_prediction(val_matrix, weights)
        test_prediction = _weighted_prediction(test_matrix, weights)
        val_metrics = metric_dict(val_actual, val_prediction)
        test_metrics = metric_dict(test_actual, test_prediction)
        rows.append({"method": method, "validation_rmse": val_metrics["rmse"], "validation_bias": val_metrics["bias"], **{f"test_{k}": v for k, v in test_metrics.items()}})
        predictions_by_method[method] = test_prediction
        weight_payload[method] = weights
    for group_method, group_column in [("horizonwise_inverse_rmse", "horizon_step"), ("stationwise_inverse_rmse", "station_id")]:
        val_prediction, test_prediction, weights = _groupwise_inverse_rmse_ensemble(
            val_matrix,
            val_actual,
            val_meta,
            test_matrix,
            test_meta,
            group_column=group_column,
        )
        val_metrics = metric_dict(val_actual, val_prediction)
        test_metrics = metric_dict(test_actual, test_prediction)
        rows.append({"method": group_method, "validation_rmse": val_metrics["rmse"], "validation_bias": val_metrics["bias"], **{f"test_{k}": v for k, v in test_metrics.items()}})
        predictions_by_method[group_method] = test_prediction
        weight_payload[group_method] = weights
    best = min(rows, key=lambda row: float(row["validation_rmse"]))
    best_method = str(best["method"])
    accepted = True
    rejected_reasons: list[str] = []
    if np.isfinite(baseline_validation_rmse) and float(best["validation_rmse"]) >= baseline_validation_rmse:
        accepted = False
        rejected_reasons.append("validation holdout RMSE did not improve over target-specific baseline")
    if reject_if_bias_worse and np.isfinite(baseline_validation_bias_abs) and abs(float(best.get("validation_bias", 0.0))) > baseline_validation_bias_abs:
        accepted = False
        rejected_reasons.append("validation holdout bias magnitude worsened versus humidity baseline")
    output = Path(output_dir)
    pd.DataFrame(rows).to_csv(output / f"{target_name}_ensemble_component_metrics.csv", index=False)
    (output / f"{target_name}_ensemble_weights.json").write_text(json.dumps(weight_payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    prediction_frame = test_meta.copy()
    prediction_frame[actual_column] = test_actual.to_numpy(dtype=float)
    for method, prediction in predictions_by_method.items():
        prediction_frame[f"{method}_prediction"] = prediction
    prediction_frame.to_csv(output / f"{target_name}_ensemble_predictions_test.csv", index=False)
    _write_ensemble_scatter(test_actual, predictions_by_method[best_method], target_name=target_name, output_dir=output)
    summary = {
        "status": "accepted" if accepted else "rejected",
        "selection_status": "accepted" if accepted else "rejected",
        "rejected_reasons": rejected_reasons,
        "best": {
            "method": best_method,
            "validation_rmse": best["validation_rmse"],
            "validation_bias": best.get("validation_bias"),
            "rmse": best["test_rmse"],
            "mae": best["test_mae"],
            "bias": best["test_bias"],
            "n": best["test_n"],
        },
        "baseline_validation_metrics": baseline_validation_metrics or {},
        "baseline_test_metrics": baseline_test_metrics or {},
        "methods": rows,
        "artifacts": {
            "weights": str(output / f"{target_name}_ensemble_weights.json"),
            "metrics": str(output / f"{target_name}_ensemble_component_metrics.csv"),
            "predictions": str(output / f"{target_name}_ensemble_predictions_test.csv"),
            "scatter": str(output / f"{target_name}_baseline_vs_final_scatter.png"),
        },
    }
    return {"summary": summary, "best_test_prediction": predictions_by_method[best_method] if accepted else None}


def _component_matrix(
    components: dict[str, dict[str, pd.DataFrame]],
    *,
    split: str,
    actual_column: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    frames = {name: payload[split].copy() for name, payload in components.items() if payload.get(split) is not None}
    common_index: set[Any] | None = None
    for frame in frames.values():
        common_index = set(frame.index) if common_index is None else common_index.intersection(frame.index)
    if not common_index:
        return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame()
    ordered = sorted(common_index)
    first = next(iter(frames.values())).loc[ordered]
    matrix = pd.DataFrame({name: frame.loc[ordered, "prediction"].to_numpy(dtype=float) for name, frame in frames.items()}, index=ordered)
    actual = first[actual_column].astype(float)
    meta_columns = [column for column in ["station_id", "issue_time", "valid_time", "horizon_step", "region", "region_class"] if column in first.columns]
    return matrix, actual, first[meta_columns].copy()


def _simple_average_weights(component_names: list[str]) -> dict[str, float]:
    if not component_names:
        return {}
    weight = 1.0 / len(component_names)
    return {name: weight for name in component_names}


def _inverse_rmse_weights(matrix: pd.DataFrame, actual: pd.Series) -> dict[str, float]:
    raw = {}
    for column in matrix.columns:
        rmse = float(metric_dict(actual, matrix[column])["rmse"])
        raw[column] = 1.0 / max(rmse, 1e-6)
    total = sum(raw.values())
    return {name: float(value / total) for name, value in raw.items()} if total else _simple_average_weights(list(matrix.columns))


def _constrained_least_squares_weights(matrix: pd.DataFrame, actual: pd.Series) -> dict[str, float] | None:
    try:
        from scipy.optimize import minimize
    except Exception:  # pragma: no cover - scipy optional in minimal envs.
        return None
    columns = list(matrix.columns)
    x = matrix.to_numpy(dtype=float)
    y = actual.to_numpy(dtype=float)
    initial = np.full(len(columns), 1.0 / len(columns), dtype=float)

    def objective(weights: np.ndarray) -> float:
        return float(np.mean(np.square(x @ weights - y)))

    result = minimize(
        objective,
        initial,
        bounds=[(0.0, 1.0)] * len(columns),
        constraints=[{"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}],
        method="SLSQP",
    )
    if not result.success:
        return None
    weights = np.asarray(result.x, dtype=float)
    weights = weights / weights.sum() if weights.sum() else initial
    return {name: float(weight) for name, weight in zip(columns, weights, strict=True)}


def _weighted_prediction(matrix: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    prediction = np.zeros(len(matrix), dtype=float)
    for column, weight in weights.items():
        if column in matrix.columns:
            prediction += float(weight) * matrix[column].to_numpy(dtype=float)
    return prediction


def _groupwise_inverse_rmse_ensemble(
    val_matrix: pd.DataFrame,
    val_actual: pd.Series,
    val_meta: pd.DataFrame,
    test_matrix: pd.DataFrame,
    test_meta: pd.DataFrame,
    *,
    group_column: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float]]]:
    global_weights = _inverse_rmse_weights(val_matrix, val_actual)
    weights_by_group: dict[str, dict[str, float]] = {}
    val_prediction = np.zeros(len(val_matrix), dtype=float)
    test_prediction = np.zeros(len(test_matrix), dtype=float)
    for group, index in val_meta.groupby(group_column).groups.items():
        weights = _inverse_rmse_weights(val_matrix.loc[index], val_actual.loc[index]) if len(index) >= 2 else global_weights
        weights_by_group[str(group)] = weights
        val_prediction[val_matrix.index.get_indexer(index)] = _weighted_prediction(val_matrix.loc[index], weights)
    for group, index in test_meta.groupby(group_column).groups.items():
        weights = weights_by_group.get(str(group), global_weights)
        test_prediction[test_matrix.index.get_indexer(index)] = _weighted_prediction(test_matrix.loc[index], weights)
    return val_prediction, test_prediction, weights_by_group


def _write_ensemble_scatter(actual: pd.Series, prediction: np.ndarray, *, target_name: str, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return
    try:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(actual, prediction, s=8, alpha=0.35)
        lo = float(min(np.nanmin(actual), np.nanmin(prediction)))
        hi = float(max(np.nanmax(actual), np.nanmax(prediction)))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        ax.set_xlabel("actual")
        ax.set_ylabel("ensemble prediction")
        ax.set_title(f"{target_name} ensemble prediction")
        fig.tight_layout()
        fig.savefig(output_dir / f"{target_name}_baseline_vs_final_scatter.png", dpi=140)
        plt.close(fig)
    except Exception:  # pragma: no cover - headless matplotlib runtime.
        return


def _fit_ridge_debug(frame: pd.DataFrame, *, actual_column: str, baseline_column: str, target_name: str, output_dir: Path) -> dict[str, Any]:
    feature_cols = _drop_train_all_missing_features(frame.loc[frame["split"].eq("train")], [column for column in BASE_FEATURES if column in frame.columns])
    train = frame.loc[frame["split"].eq("train")].copy()
    test = frame.loc[frame["split"].eq("test")].copy()
    categorical = [column for column in ["station_id", "region", "region_class"] if column in feature_cols]
    numeric = [column for column in feature_cols if column not in categorical]
    model = Pipeline([("prep", _preprocessor(numeric, categorical)), ("model", Ridge(alpha=1.0))])
    y_train = train[actual_column].astype(float) - train[baseline_column].astype(float)
    model.fit(train[feature_cols], y_train)
    predicted_residual = model.predict(test[feature_cols])
    prediction = test[baseline_column].astype(float).to_numpy() + predicted_residual
    debug_frame = test[["station_id", "issue_time", "valid_time", "horizon_step", actual_column, baseline_column]].copy()
    debug_frame["residual_target"] = debug_frame[actual_column].astype(float) - debug_frame[baseline_column].astype(float)
    debug_frame["predicted_residual"] = predicted_residual
    debug_frame["prediction"] = prediction
    debug = residual_debug_summary(debug_frame, actual_column=actual_column, baseline_column=baseline_column, predicted_residual_column="predicted_residual", prediction_column="prediction")
    debug["test_metrics"] = metric_dict(debug_frame[actual_column], debug_frame["prediction"])
    debug["raw_baseline_metrics"] = metric_dict(debug_frame[actual_column], debug_frame[baseline_column])
    debug_frame.to_csv(output_dir / f"{target_name}_ridge_residual_debug_rows.csv", index=False)
    _write_ridge_debug_markdown(debug, output_dir / f"{target_name}_ridge_residual_debug_report.md")
    _write_ridge_debug_plots(debug_frame, target_name=target_name, actual_column=actual_column, output_dir=output_dir)
    return {"model": model, "debug": debug, "test_prediction": prediction}


def _drop_train_all_missing_features(train: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    categorical_candidates = {"station_id", "region", "region_class"}
    usable: list[str] = []
    for column in feature_columns:
        if column not in train.columns:
            continue
        if column in categorical_candidates:
            if train[column].notna().any():
                usable.append(column)
            continue
        if pd.to_numeric(train[column], errors="coerce").notna().any():
            usable.append(column)
    return usable


def _preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="drop",
    )


def _lgbm_params(params: dict[str, Any]) -> dict[str, Any]:
    output = {
        "random_state": 20260528,
        "n_jobs": -1,
        "verbosity": -1,
    }
    output.update(params)
    return output


def _station_neighbor_features(frame: pd.DataFrame, patch_size: int) -> list[str]:
    variables = [column for column in ["nwp_t2m", "nwp_dew_point", "nwp_humidity", "nwp_sp", "nwp_u10", "nwp_v10", "nwp_tp", "nwp_wind_speed"] if column in frame.columns]
    k = min(int(patch_size * patch_size), int(frame["station_id"].nunique()))
    stations = frame[["station_id", "lat", "lon"]].drop_duplicates("station_id").dropna()
    if stations.empty:
        return []
    neighbor_ids: dict[str, list[str]] = {}
    for row in stations.itertuples(index=False):
        distances = np.square(stations["lat"].astype(float) - float(row.lat)) + np.square(stations["lon"].astype(float) - float(row.lon))
        neighbor_ids[str(row.station_id)] = stations.loc[distances.nsmallest(k).index, "station_id"].astype(str).tolist()
    key_cols = ["issue_time", "horizon_step"]
    feature_names = [f"{var}_station_patch{patch_size}_{suffix}" for var in variables for suffix in ("mean", "std", "min", "max")]
    features = pd.DataFrame(index=frame.index, columns=feature_names, dtype=float)
    work = frame[["station_id", *key_cols, *variables]].copy()
    work["station_id"] = work["station_id"].astype(str)
    for station_id, neighbors in neighbor_ids.items():
        source = work.loc[work["station_id"].isin(neighbors), [*key_cols, *variables]]
        if source.empty:
            continue
        aggregated = source.groupby(key_cols, dropna=False)[variables].agg(["mean", "std", "min", "max"])
        aggregated.columns = [f"{var}_station_patch{patch_size}_{suffix}" for var, suffix in aggregated.columns]
        aggregated = aggregated.reset_index()
        target_index = frame.index[frame["station_id"].astype(str).eq(station_id)]
        target = frame.loc[target_index, key_cols].copy()
        target["__index"] = target_index
        merged = target.merge(aggregated, on=key_cols, how="left").set_index("__index")
        features.loc[target_index, feature_names] = merged[feature_names].to_numpy(dtype=float)
    frame[feature_names] = features
    return feature_names


def _cyc_features(series: pd.Series, prefix: str) -> pd.DataFrame:
    timestamps = pd.to_datetime(series, utc=True)
    hour = timestamps.dt.hour.astype(float)
    doy = timestamps.dt.dayofyear.astype(float)
    return pd.DataFrame(
        {
            f"{prefix}_hour_sin": np.sin(2.0 * np.pi * hour / 24.0),
            f"{prefix}_hour_cos": np.cos(2.0 * np.pi * hour / 24.0),
            f"{prefix}_doy_sin": np.sin(2.0 * np.pi * doy / 366.0),
            f"{prefix}_doy_cos": np.cos(2.0 * np.pi * doy / 366.0),
        },
        index=series.index,
    )


def _benchmark_info(frame: pd.DataFrame) -> dict[str, Any]:
    issue_times = pd.to_datetime(frame["issue_time"], utc=True).dropna()
    date_span_days = float((issue_times.max() - issue_times.min()) / pd.Timedelta(days=1)) if not issue_times.empty else 0.0
    season_count = len({_season(ts) for ts in issue_times})
    cycle_count = int(issue_times.nunique())
    return {
        "forecast_cycle_count": cycle_count,
        "station_count": int(frame["station_id"].nunique()),
        "date_span_days": date_span_days,
        "season_coverage": sorted({_season(ts) for ts in issue_times}),
        "issue_time_coverage": sorted(str(ts) for ts in issue_times.drop_duplicates()),
        "benchmark_reliability": classify_benchmark_reliability(forecast_cycle_count=cycle_count, date_span_days=date_span_days, season_count=season_count),
    }


def _season(timestamp: pd.Timestamp) -> str:
    month = pd.Timestamp(timestamp).month
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "autumn"


def _worst_group_rmse(frame: pd.DataFrame, actual_column: str, prediction_column: str, group_column: str) -> float:
    values = [metric_dict(group[actual_column], group[prediction_column])["rmse"] for _, group in frame.groupby(group_column)]
    return float(max(values)) if values else float("nan")


def _feature_importance(model: Pipeline, numeric: list[str], categorical: list[str]) -> pd.DataFrame | None:
    estimator = model.named_steps.get("model")
    if estimator is None or not hasattr(estimator, "feature_importances_"):
        return None
    prep = model.named_steps.get("prep")
    try:
        names = list(prep.get_feature_names_out()) if prep is not None else [*numeric, *categorical]
    except Exception:  # noqa: BLE001
        names = [*numeric, *categorical]
    importance = list(getattr(estimator, "feature_importances_"))
    if len(names) != len(importance):
        names = [f"feature_{idx}" for idx in range(len(importance))]
    return pd.DataFrame({"feature_name": names, "importance": [float(v) for v in importance]}).sort_values("importance", ascending=False)


def _write_candidate_rows(rows: list[dict[str, Any]], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_grid_rows(rows: list[dict[str, Any]], path: Path) -> None:
    validate_lgbm_grid_results(rows)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_ridge_debug_markdown(debug: dict[str, Any], path: Path) -> None:
    lines = ["# Ridge Residual Debug", "", f"residual_target_sign_ok: `{debug.get('residual_target_sign_ok')}`", f"final_prediction_formula_ok: `{debug.get('final_prediction_formula_ok')}`", ""]
    for key, value in debug.items():
        lines.append(f"- {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ridge_debug_plots(frame: pd.DataFrame, *, target_name: str, actual_column: str, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(frame["residual_target"], bins=30, alpha=0.6, label="actual residual")
    ax.hist(frame["predicted_residual"], bins=30, alpha=0.6, label="ridge predicted residual")
    ax.legend()
    ax.set_title(f"{target_name} residual distribution")
    fig.tight_layout()
    fig.savefig(output_dir / f"{target_name}_residual_distribution_plot.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(frame["residual_target"], frame["predicted_residual"], s=8, alpha=0.4)
    ax.set_xlabel("actual residual")
    ax.set_ylabel("ridge predicted residual")
    ax.set_title(f"{target_name} ridge residual scatter")
    fig.tight_layout()
    fig.savefig(output_dir / f"{target_name}_ridge_vs_lgbm_residual_scatter.png", dpi=140)
    plt.close(fig)


def _write_markdown_report(summary: dict[str, Any], path: Path) -> None:
    lines = ["# Operational Performance Report", "", f"benchmark_reliability: `{summary.get('benchmark_reliability')}`", "", "## Official Baselines", ""]
    for target, models in summary.get("official_baselines", {}).items():
        lines.append(f"### {target}")
        for model, metrics in models.items():
            lines.append(f"- `{model}`: RMSE={_fmt(metrics.get('rmse'))}, MAE={_fmt(metrics.get('mae'))}, Bias={_fmt(metrics.get('bias'))}")
        lines.append("")
    lines.append(f"## V4-C Gate\n\n`{summary.get('v4c_gate', {}).get('status')}` missing={summary.get('v4c_gate', {}).get('missing_conditions')}\n")
    lines.append(f"## Site Readiness\n\n`{summary.get('site_readiness', {}).get('status')}` missing={summary.get('site_readiness', {}).get('missing_conditions')}\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def _html_metric_table(baselines: Any) -> str:
    if not isinstance(baselines, dict):
        return "<div class='card'>No baselines.</div>"
    rows = ["<h2>Official Baselines</h2><table><tr><th>Target</th><th>Model</th><th>RMSE</th><th>MAE</th><th>Bias</th><th>Notes</th></tr>"]
    for target, models in baselines.items():
        if not isinstance(models, dict):
            continue
        for model, metrics in models.items():
            metrics = metrics if isinstance(metrics, dict) else {}
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(target))}</td><td><code>{html.escape(str(model))}</code></td>"
                f"<td>{_fmt(metrics.get('rmse'))}</td><td>{_fmt(metrics.get('mae'))}</td><td>{_fmt(metrics.get('bias'))}</td>"
                f"<td>{html.escape(str({k: v for k, v in metrics.items() if k not in {'rmse', 'mae', 'bias', 'n'}}))}</td>"
                "</tr>"
            )
    rows.append("</table>")
    return "\n".join(rows)


def _html_gate(title: str, gate: Any) -> str:
    gate = gate if isinstance(gate, dict) else {}
    status = str(gate.get("status", "n/a"))
    css = "pass" if status == "PASS" else "fail" if status == "FAIL" else "warn"
    missing = gate.get("missing_conditions", [])
    return f"<div class='card'><h2>{html.escape(title)}</h2><p>Status: <span class='{css}'>{html.escape(status)}</span></p><p>Missing: {html.escape(str(missing))}</p></div>"


def _html_section(title: str, payload: Any) -> str:
    return f"<div class='card'><h2>{html.escape(title)}</h2><pre>{html.escape(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))}</pre></div>"


def _fmt(value: Any) -> str:
    try:
        if value is None:
            return "n/a"
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run operational-valid MOS performance benchmarks and always write an HTML report.")
    parser.add_argument("--nwp-archive", required=True)
    parser.add_argument("--archive-quality-report", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--station-metadata", required=True)
    parser.add_argument("--grid-patch-features", help="Optional true GFS grid patch feature CSV from weather_korea_forecast.v4.gfs_grid_patch.")
    parser.add_argument("--output-dir", default="data/artifacts/g024_operational_performance")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = run_operational_benchmark(
        nwp_archive=args.nwp_archive,
        archive_quality_report=args.archive_quality_report,
        observations=args.observations,
        station_metadata=args.station_metadata,
        output_dir=args.output_dir,
        grid_patch_features=args.grid_patch_features,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
