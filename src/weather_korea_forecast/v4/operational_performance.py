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

try:  # pragma: no cover - dependency is present in normal dev env, optional in minimal envs.
    from lightgbm import LGBMRegressor
except Exception:  # noqa: BLE001
    LGBMRegressor = None  # type: ignore[assignment]

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
    "nwp_wind_speed",
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
DEFAULT_LGBM_GRID = [
    {"num_leaves": 15, "max_depth": 4, "learning_rate": 0.05, "n_estimators": 350, "min_child_samples": 20, "subsample": 0.9, "colsample_bytree": 0.9, "reg_alpha": 0.0, "reg_lambda": 0.0},
    {"num_leaves": 31, "max_depth": -1, "learning_rate": 0.03, "n_estimators": 500, "min_child_samples": 20, "subsample": 0.9, "colsample_bytree": 0.9, "reg_alpha": 0.0, "reg_lambda": 0.0},
    {"num_leaves": 31, "max_depth": 6, "learning_rate": 0.03, "n_estimators": 700, "min_child_samples": 10, "subsample": 0.95, "colsample_bytree": 0.95, "reg_alpha": 0.0, "reg_lambda": 0.1},
    {"num_leaves": 63, "max_depth": -1, "learning_rate": 0.02, "n_estimators": 800, "min_child_samples": 15, "subsample": 0.9, "colsample_bytree": 0.9, "reg_alpha": 0.0, "reg_lambda": 0.1},
    {"num_leaves": 63, "max_depth": 8, "learning_rate": 0.03, "n_estimators": 600, "min_child_samples": 8, "subsample": 0.85, "colsample_bytree": 0.9, "reg_alpha": 0.05, "reg_lambda": 0.2},
    {"num_leaves": 127, "max_depth": 10, "learning_rate": 0.02, "n_estimators": 900, "min_child_samples": 8, "subsample": 0.9, "colsample_bytree": 0.85, "reg_alpha": 0.05, "reg_lambda": 0.5},
]
CALIBRATION_CANDIDATES = [
    "none",
    "global_mean_bias",
    "per_horizon_mean_bias",
    "per_station_horizon_mean_bias",
    "per_region_horizon_mean_bias",
    "global_affine",
    "per_station_horizon_affine",
    "quantile_mapping",
    "isotonic",
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
        if self.name == "isotonic":
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
    temp = baselines.get("temp", {}) if isinstance(baselines.get("temp"), dict) else {}
    humidity = baselines.get("humidity", {}) if isinstance(baselines.get("humidity"), dict) else {}
    temp_lgbm = temp.get("operational_residual_lgbm_temp", {}) if isinstance(temp.get("operational_residual_lgbm_temp"), dict) else {}
    humidity_lgbm = humidity.get("operational_residual_lgbm_humidity", {}) if isinstance(humidity.get("operational_residual_lgbm_humidity"), dict) else {}
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
    temp = baselines.get("temp", {}) if isinstance(baselines.get("temp"), dict) else {}
    humidity = baselines.get("humidity", {}) if isinstance(baselines.get("humidity"), dict) else {}
    temp_lgbm = temp.get("operational_residual_lgbm_temp", {}) if isinstance(temp.get("operational_residual_lgbm_temp"), dict) else {}
    humidity_lgbm = humidity.get("operational_residual_lgbm_humidity", {}) if isinstance(humidity.get("operational_residual_lgbm_humidity"), dict) else {}
    reliability = str(summary.get("benchmark_reliability") or "smoke")
    conditions = {
        "temp operational_valid=true": temp_lgbm.get("rmse") is not None,
        "temp RMSE <= 1.5": temp_lgbm.get("rmse") is not None and float(temp_lgbm["rmse"]) <= 1.5,
        "humidity RMSE <= 10 beta": humidity_lgbm.get("rmse") is not None and float(humidity_lgbm["rmse"]) <= 10.0,
        "benchmark_reliability >= medium": reliability in {"medium", "strong", "seasonal"},
        "weather_code rule-based allowed": True,
    }
    missing = [name for name, ok in conditions.items() if not ok]
    return {"status": "PASS" if not missing else "WARN", "conditions": conditions, "missing_conditions": missing}


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
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    quality = json.loads(Path(archive_quality_report).read_text(encoding="utf-8"))
    if quality.get("forecast_archive_adequate") is not True:
        raise ValueError(f"forecast archive is inadequate: {quality.get('blocking_reasons')}")
    frame = load_joined_operational_frame(nwp_archive=nwp_archive, observations=observations, station_metadata=station_metadata)
    frame = add_time_ordered_splits(frame, train_cycles=20, val_cycles=5, test_cycles=5)
    benchmark = _benchmark_info(frame)
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
        },
        **benchmark,
        "official_baselines": {"temp": {}, "humidity": {}},
        "metrics": {},
        "calibration": {},
        "patch_ablation": {},
        "patch_improvement": {},
        "patch_ablation_completed": True,
        "ridge_debug": {},
        "artifacts": {},
    }
    grid = lgbm_grid or DEFAULT_LGBM_GRID
    patch_modes = {"no_patch": [], "patch3": _station_neighbor_features(frame, 3), "patch5": _station_neighbor_features(frame, 5)}
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
        ablation_rows: list[dict[str, Any]] = []
        feature_importance_rows: list[pd.DataFrame] = []
        for mode, extra_features in patch_modes.items():
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
            test_frame = tuning["test_prediction_frame"].copy()
            corrected_test = calibration.selected.apply(test_frame)
            test_metrics_raw = metric_dict(test_frame[actual], test_frame["prediction"])
            test_metrics_corrected = metric_dict(test_frame[actual], corrected_test)
            use_corrected = calibration.selected_name != "none"
            final_prediction = corrected_test if use_corrected else test_frame["prediction"].to_numpy(dtype=float)
            final_metrics = test_metrics_corrected if use_corrected else test_metrics_raw
            worst_station = _worst_group_rmse(test_frame.assign(final_prediction=final_prediction), actual, "final_prediction", "station_id")
            late_horizon = test_frame.loc[test_frame["horizon_step"].between(12, 24)].copy()
            late_metrics = metric_dict(late_horizon[actual], pd.Series(final_prediction, index=test_frame.index).loc[late_horizon.index])
            ablation = {
                "target": target_name,
                "mode": mode,
                "rmse": final_metrics["rmse"],
                "mae": final_metrics["mae"],
                "bias": final_metrics["bias"],
                "worst_station_rmse": worst_station,
                "late_horizon_rmse": late_metrics["rmse"],
                "selected_calibration": calibration.selected_name if use_corrected else "none_holdout_not_improved",
                "best_params": tuning["best_params"],
                "validation_rmse": tuning["best_validation_rmse"],
                "feature_count": len(feature_cols),
            }
            ablation_rows.append(ablation)
            _write_candidate_rows(calibration.candidate_results, output / f"{target_name}_{mode}_calibration_candidate_results.csv")
            _write_grid_rows(tuning["grid_results"], output / f"{target_name}_{mode}_lgbm_grid_search_results.csv")
            if tuning.get("feature_importance") is not None:
                fi = tuning["feature_importance"].copy()
                fi.insert(0, "target", target_name)
                fi.insert(1, "mode", mode)
                feature_importance_rows.append(fi)
            if best_summary is None or float(final_metrics["rmse"]) < float(best_summary["rmse"]):
                best_mode = mode
                best_summary = ablation
                best_model = model
                best_test_prediction = final_prediction
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
        summary["official_baselines"][target_name][lgbm_name] = {
            "rmse": best_summary["rmse"],
            "mae": best_summary["mae"],
            "bias": best_summary["bias"],
            "n": int(test.shape[0]),
            "patch_mode": best_mode,
            "selected_calibration": best_summary["selected_calibration"],
        }
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
    }
    summary["v4c_gate"] = evaluate_v4c_gate(summary)
    summary["site_readiness"] = evaluate_site_readiness_gate(summary)
    (output / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    write_operational_performance_html(summary, output / "operational_performance_report.html")
    _write_markdown_report(summary, output / "operational_performance_report.md")
    return summary


def load_joined_operational_frame(*, nwp_archive: str | Path, observations: str | Path, station_metadata: str | Path) -> pd.DataFrame:
    nwp = pd.read_csv(nwp_archive, dtype={"station_id": str})
    nwp["station_id"] = nwp["station_id"].astype(str)
    for column in ["forecast_init_time", "issue_time", "valid_time"]:
        if column in nwp.columns:
            nwp[column] = pd.to_datetime(nwp[column], utc=True)
    if "issue_time" not in nwp.columns:
        nwp["issue_time"] = nwp["forecast_init_time"]
    obs = pd.read_csv(observations, dtype={"station_id": str})
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
    frame = frame.dropna(subset=["temp", "humidity", "nwp_t2m", "nwp_humidity"]).sort_values(["issue_time", "station_id", "horizon_step"]).reset_index(drop=True)
    return frame


def add_time_ordered_splits(frame: pd.DataFrame, *, train_cycles: int, val_cycles: int, test_cycles: int) -> pd.DataFrame:
    output = frame.copy()
    cycles = sorted(output["issue_time"].drop_duplicates())
    needed = train_cycles + val_cycles + test_cycles
    if len(cycles) < needed:
        raise ValueError(f"Need at least {needed} forecast cycles; got {len(cycles)}")
    selected = cycles[-needed:]
    split_map = {cycle: "train" for cycle in selected[:train_cycles]}
    split_map.update({cycle: "val" for cycle in selected[train_cycles : train_cycles + val_cycles]})
    split_map.update({cycle: "test" for cycle in selected[train_cycles + val_cycles :]})
    output = output.loc[output["issue_time"].isin(selected)].copy()
    output["split"] = output["issue_time"].map(split_map)
    return output


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
    if name == "quantile_mapping":
        quantiles = np.linspace(0, 1, 21)
        return CalibrationModel(name=name, payload={"prediction_quantiles": np.quantile(prediction.dropna(), quantiles).tolist(), "actual_quantiles": np.quantile(actual.dropna(), quantiles).tolist()}, actual_column=actual_column, prediction_column=prediction_column)
    if name == "isotonic":
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


def _fit_ridge_debug(frame: pd.DataFrame, *, actual_column: str, baseline_column: str, target_name: str, output_dir: Path) -> dict[str, Any]:
    feature_cols = [column for column in BASE_FEATURES if column in frame.columns]
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
    lookup = frame.set_index(["station_id", *key_cols])[variables]
    feature_names = []
    rows: list[dict[str, Any]] = []
    for sample in frame[["station_id", *key_cols]].itertuples(index=False):
        values_by_var: dict[str, list[float]] = {var: [] for var in variables}
        for sid in neighbor_ids.get(str(sample.station_id), [str(sample.station_id)]):
            try:
                values = lookup.loc[(sid, sample.issue_time, sample.horizon_step)]
            except KeyError:
                continue
            for var in variables:
                value = values[var]
                if pd.notna(value):
                    values_by_var[var].append(float(value))
        out: dict[str, float] = {}
        for var, values in values_by_var.items():
            arr = np.asarray(values, dtype=float)
            for suffix, val in {
                "mean": float(np.nanmean(arr)) if arr.size else np.nan,
                "std": float(np.nanstd(arr)) if arr.size else np.nan,
                "min": float(np.nanmin(arr)) if arr.size else np.nan,
                "max": float(np.nanmax(arr)) if arr.size else np.nan,
            }.items():
                name = f"{var}_station_patch{patch_size}_{suffix}"
                out[name] = val
                if name not in feature_names:
                    feature_names.append(name)
        rows.append(out)
    features = pd.DataFrame(rows, index=frame.index)
    for column in features.columns:
        frame[column] = features[column]
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
    lines = ["# G024 Operational Performance Report", "", f"benchmark_reliability: `{summary.get('benchmark_reliability')}`", "", "## Official Baselines", ""]
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
    parser = argparse.ArgumentParser(description="Run G024 operational-valid MOS performance sprint and always write an HTML report.")
    parser.add_argument("--nwp-archive", required=True)
    parser.add_argument("--archive-quality-report", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--station-metadata", required=True)
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
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
