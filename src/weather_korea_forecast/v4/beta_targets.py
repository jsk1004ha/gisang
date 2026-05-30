from __future__ import annotations

from math import isfinite
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from weather_korea_forecast.service.beta_sources import normalize_probability_fraction

ITERATION_KEYS = ("1_raw_baseline", "2_calibration", "3_final_selection")


def circular_direction_mae(actual: Iterable[float], prediction: Iterable[float]) -> float | None:
    y = np.asarray(list(actual), dtype=float)
    p = np.asarray(list(prediction), dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if not mask.any():
        return None
    diff = np.abs((p[mask] - y[mask] + 180.0) % 360.0 - 180.0)
    return float(np.mean(diff))


def select_precip_probability_descriptor(
    *,
    raw_validation_metrics: dict[str, Any] | None,
    model_validation_metrics: dict[str, Any] | None,
    raw_test_metrics: dict[str, Any] | None = None,
    model_test_metrics: dict[str, Any] | None = None,
    direct_source: str = "nwp_direct",
    allow_ai_beta: bool = False,
) -> dict[str, Any]:
    raw_validation = raw_validation_metrics or {}
    model_validation = model_validation_metrics or {}
    raw_brier = _finite(raw_validation.get("brier"))
    model_brier = _finite(model_validation.get("brier"))
    if allow_ai_beta and model_brier is not None and (raw_brier is None or model_brier < raw_brier):
        return _descriptor(
            target="precip_probability",
            model_key="lgbm_classifier_calibrated_beta",
            source="ai_beta",
            status="beta",
            confidence="medium",
            unit="percent",
            selection_reason="validation_brier_improved",
            validation_metrics=model_validation,
            test_metrics=model_test_metrics or {},
        )
    if raw_brier is not None:
        fallback_reason = "ai_beta_unavailable"
        if model_brier is not None:
            fallback_reason = "validation_brier_not_improved" if model_brier >= raw_brier else "ai_beta_training_frozen"
        return _descriptor(
            target="precip_probability",
            model_key=direct_source,
            source=direct_source,
            status="direct",
            confidence="low",
            unit="percent",
            selection_reason="direct_baseline_selected",
            fallback_reason=fallback_reason,
            validation_metrics=raw_validation,
            test_metrics=raw_test_metrics or {},
        )
    return _descriptor(
        target="precip_probability",
        model_key="unavailable",
        source="unavailable",
        status="unavailable",
        confidence="unavailable",
        unit="percent",
        fallback_reason="nwp_precip_probability_missing",
    )


def build_beta_targets_summary(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        "precip_probability": _precip_probability_summary(frame),
        "wind": _wind_summary(frame),
        "cloud": _cloud_summary(frame),
        "weather_code": _weather_code_summary(),
    }


def binary_probability_metrics(actual_event: Iterable[Any], probability: Iterable[Any], *, input_unit: str = "auto") -> dict[str, Any]:
    y = pd.to_numeric(pd.Series(list(actual_event)), errors="coerce")
    p = pd.Series(list(probability)).map(lambda value: normalize_probability_fraction(value, input_unit=input_unit))
    mask = y.notna() & p.notna()
    if not mask.any():
        return {"brier": None, "roc_auc": None, "pr_auc": None, "false_alarm_rate": None, "hit_rate": None, "n": 0}
    y_arr = y[mask].astype(int).to_numpy()
    p_arr = p[mask].astype(float).to_numpy()
    pred = p_arr >= 0.5
    event = y_arr == 1
    false_alarms = int((pred & ~event).sum())
    hits = int((pred & event).sum())
    non_events = int((~event).sum())
    events = int(event.sum())
    return {
        "brier": float(np.mean((p_arr - y_arr) ** 2)),
        "roc_auc": _safe_auc(roc_auc_score, y_arr, p_arr),
        "pr_auc": _safe_auc(average_precision_score, y_arr, p_arr),
        "false_alarm_rate": float(false_alarms / non_events) if non_events else None,
        "hit_rate": float(hits / events) if events else None,
        "n": int(len(y_arr)),
        "reliability_bins": reliability_bins(y_arr, p_arr, input_unit="fraction"),
    }


def reliability_bins(actual_event: Iterable[Any], probability: Iterable[Any], bins: int = 5, *, input_unit: str = "auto") -> list[dict[str, Any]]:
    y = pd.to_numeric(pd.Series(list(actual_event)), errors="coerce")
    p = pd.Series(list(probability)).map(lambda value: normalize_probability_fraction(value, input_unit=input_unit))
    frame = pd.DataFrame({"actual": y, "probability": p}).dropna()
    if frame.empty:
        return []
    frame["bin"] = pd.cut(frame["probability"], bins=np.linspace(0.0, 1.0, bins + 1), include_lowest=True)
    rows = []
    for interval, group in frame.groupby("bin", observed=True):
        rows.append(
            {
                "bin": str(interval),
                "mean_probability": float(group["probability"].mean()),
                "observed_frequency": float(group["actual"].mean()),
                "n": int(len(group)),
            }
        )
    return rows


def wind_speed_metrics(actual_speed: Iterable[Any], predicted_speed: Iterable[Any]) -> dict[str, Any]:
    y = pd.to_numeric(pd.Series(list(actual_speed)), errors="coerce")
    p = pd.to_numeric(pd.Series(list(predicted_speed)), errors="coerce")
    mask = y.notna() & p.notna()
    if not mask.any():
        return {"wind_speed_rmse": None, "wind_speed_mae": None, "strong_wind_hit_rate": None, "n": 0}
    y_arr = y[mask].astype(float).to_numpy()
    p_arr = p[mask].astype(float).to_numpy()
    strong = y_arr >= 14.0
    predicted_strong = p_arr >= 14.0
    return {
        "wind_speed_rmse": float(np.sqrt(np.mean((p_arr - y_arr) ** 2))),
        "wind_speed_mae": float(np.mean(np.abs(p_arr - y_arr))),
        "strong_wind_hit_rate": float((predicted_strong & strong).sum() / strong.sum()) if strong.any() else None,
        "n": int(len(y_arr)),
    }


def _precip_probability_summary(frame: pd.DataFrame) -> dict[str, Any]:
    actual_event = (pd.to_numeric(frame.get("precipitation", pd.Series(dtype=float)), errors="coerce") > 0.1).astype(int)
    if "nwp_precip_probability" in frame.columns and frame["nwp_precip_probability"].notna().any():
        val = _split(frame, "val")
        test = _split(frame, "test")
        raw_validation = binary_probability_metrics(actual_event.loc[val.index], val["nwp_precip_probability"], input_unit="percent") if not val.empty else binary_probability_metrics(actual_event, frame["nwp_precip_probability"], input_unit="percent")
        raw_test = binary_probability_metrics(actual_event.loc[test.index], test["nwp_precip_probability"], input_unit="percent") if not test.empty else {}
        descriptor = select_precip_probability_descriptor(
            raw_validation_metrics=raw_validation,
            model_validation_metrics=None,
            raw_test_metrics=raw_test,
        )
        raw_stage = {"status": "available", "baseline": "nwp_precip_probability", "validation_metrics": raw_validation, "test_metrics": raw_test}
    else:
        descriptor = select_precip_probability_descriptor(raw_validation_metrics=None, model_validation_metrics=None)
        raw_stage = {"status": "unavailable", "reason": "nwp_precip_probability_missing", "forbidden_baseline": "nwp_tp"}
    return _summary(
        raw_stage=raw_stage,
        calibration_stage={"status": "not_applied", "reason": "model_improvement_frozen"},
        descriptor=descriptor,
    )


def _wind_summary(frame: pd.DataFrame) -> dict[str, Any]:
    raw_available = {"wind_speed", "nwp_wind_speed"}.issubset(frame.columns)
    raw_metrics = wind_speed_metrics(frame["wind_speed"], frame["nwp_wind_speed"]) if raw_available else {}
    has_direction = any(column in frame.columns and frame[column].notna().any() for column in ["wind_direction", "wind_direction_deg", "obs_wind_direction"])
    if raw_available:
        descriptor = _descriptor(
            target="wind",
            model_key="gfs_direct",
            source="gfs_direct",
            status="direct",
            confidence="low",
            unit="m/s,deg",
            selection_reason="direct_baseline_selected",
            fallback_reason=None if has_direction else "observed_wind_direction_missing",
            validation_metrics=raw_metrics,
        )
    else:
        descriptor = _descriptor(target="wind", model_key="unavailable", source="unavailable", status="unavailable", confidence="unavailable", unit="m/s,deg", fallback_reason="nwp_wind_speed_missing")
    return _summary(
        raw_stage={"status": "available" if raw_available else "unavailable", "baseline": "nwp_wind_speed", "metrics": raw_metrics},
        calibration_stage={"status": "not_applied", "reason": "observed_wind_direction_missing" if not has_direction else "wind_beta_training_pending"},
        descriptor=descriptor,
    )


def _cloud_summary(frame: pd.DataFrame) -> dict[str, Any]:
    label_column = next((column for column in ["cloud_class", "sky_code", "cloud_label"] if column in frame.columns and frame[column].notna().any()), None)
    direct_available = "nwp_cloud_cover" in frame.columns and frame["nwp_cloud_cover"].notna().any()
    if label_column is None:
        descriptor = _descriptor(
            target="cloud",
            model_key="nwp_direct_or_lgbm_beta",
            source="nwp_direct" if direct_available else "rule_based_beta",
            status="direct" if direct_available else "beta",
            confidence="low",
            unit="percent",
            selection_reason="direct_or_rule_based_selected",
            fallback_reason="observed_cloud_label_missing",
        )
        calibration_stage = {"status": "not_applicable", "reason": "observed_cloud_label_missing"}
    else:
        descriptor = _descriptor(
            target="cloud",
            model_key="nwp_direct_or_lgbm_beta",
            source="nwp_direct",
            status="direct",
            confidence="low",
            unit="class",
            selection_reason="direct_baseline_selected",
            fallback_reason="cloud_classifier_validation_not_run",
        )
        calibration_stage = {"status": "pending", "label_column": label_column}
    return _summary(
        raw_stage={"status": "available" if direct_available else "unavailable", "baseline": "nwp_cloud_cover"},
        calibration_stage=calibration_stage,
        descriptor=descriptor,
    )


def _weather_code_summary() -> dict[str, Any]:
    return _summary(
        raw_stage={"status": "available", "baseline": "rule_based_weather_code"},
        calibration_stage={"status": "not_applicable", "reason": "rule_based"},
        descriptor=_descriptor(
            target="weather_code",
            model_key="rule_based_beta",
            source="rule_based_beta",
            status="beta",
            confidence="low",
            selection_reason="rule_based_beta_selected",
        ),
    )


def _summary(*, raw_stage: dict[str, Any], calibration_stage: dict[str, Any], descriptor: dict[str, Any]) -> dict[str, Any]:
    return {
        "iterations": {
            "1_raw_baseline": raw_stage,
            "2_calibration": calibration_stage,
            "3_final_selection": {"descriptor": descriptor},
        },
        "descriptor": descriptor,
    }


def _descriptor(
    *,
    target: str,
    model_key: str,
    source: str,
    status: str,
    confidence: str,
    unit: str | None = None,
    selection_reason: str | None = None,
    fallback_reason: str | None = None,
    validation_metrics: dict[str, Any] | None = None,
    test_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "target": target,
        "model_key": model_key,
        "source": source,
        "status": status,
        "confidence": confidence,
        "unit": unit,
        "selection_reason": selection_reason,
        "fallback_reason": fallback_reason,
        "validation_metrics": validation_metrics or {},
        "test_metrics": test_metrics or {},
    }


def _split(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    if "split" not in frame.columns:
        return pd.DataFrame(columns=frame.columns)
    return frame.loc[frame["split"].astype(str).eq(split)].copy()


def _safe_auc(metric: Any, actual: np.ndarray, probability: np.ndarray) -> float | None:
    try:
        if len(np.unique(actual)) < 2:
            return None
        value = float(metric(actual, probability))
        return value if isfinite(value) else None
    except Exception:  # noqa: BLE001
        return None


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None
