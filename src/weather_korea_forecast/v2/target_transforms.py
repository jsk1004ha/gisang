from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


TARGET_CONTEXT_TEMP_C = "_target_context_temp_c"
TARGET_CONTEXT_BASELINE_VALUE = "_target_context_baseline_value"


def normalize_target_transform_config(config: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(config or {})
    transform_type = str(raw.get("type", raw.get("name", "none")) or "none").strip().lower().replace("-", "_")
    aliases = {
        "": "none",
        "off": "none",
        "disabled": "none",
        "identity": "none",
        "logit": "logit_rh",
        "rh_logit": "logit_rh",
        "relative_humidity_logit": "logit_rh",
        "dewpoint": "dew_point",
        "dewpoint_c": "dew_point",
        "dew_point_c": "dew_point",
        "dewpoint_depression": "dew_point_depression",
        "residual": "residual_from_feature",
        "feature_residual": "residual_from_feature",
        "forecast_residual": "residual_from_feature",
        "nwp_residual": "residual_from_feature",
        "era5_residual": "residual_from_feature",
        "residual_feature": "residual_from_feature",
        "future_feature_residual": "residual_from_feature",
    }
    raw["type"] = aliases.get(transform_type, transform_type)
    return raw


def target_transform_requires_temperature(transform_config: dict[str, Any] | None) -> bool:
    return normalize_target_transform_config(transform_config)["type"] in {"dew_point", "dew_point_depression"}


def target_transform_context_columns(transform_config: dict[str, Any] | None) -> list[str]:
    transform = normalize_target_transform_config(transform_config)
    if transform["type"] == "residual_from_feature":
        return [TARGET_CONTEXT_BASELINE_VALUE]
    if target_transform_requires_temperature(transform_config):
        return [TARGET_CONTEXT_TEMP_C]
    return []


def append_target_transform_context(frame: pd.DataFrame, transform_config: dict[str, Any] | None) -> pd.DataFrame:
    transform = normalize_target_transform_config(transform_config)
    enriched = frame.copy()
    if transform["type"] == "residual_from_feature":
        baseline_column = _residual_baseline_column(enriched, transform)
        enriched[TARGET_CONTEXT_BASELINE_VALUE] = enriched[baseline_column].astype(float)
        return enriched
    if not target_transform_requires_temperature(transform):
        return frame
    temperature_column = str(transform.get("temperature_column", "") or "")
    candidates = [temperature_column] if temperature_column else []
    candidates.extend(["obs_temp", "temp", "era5_t2m_c", "era5_t2m"])
    source_column = next((column for column in candidates if column and column in enriched.columns), None)
    if source_column is None:
        raise ValueError(
            "Humidity dew-point target transforms require a temperature context column. "
            "Configure data.target_transform.temperature_column or include obs_temp/temp."
        )
    enriched[TARGET_CONTEXT_TEMP_C] = temperature_series_to_celsius(enriched[source_column])
    return enriched


def apply_target_transform(frame: pd.DataFrame, target_name: str, transform_config: dict[str, Any] | None) -> pd.Series:
    transform = normalize_target_transform_config(transform_config)
    transform_type = transform["type"]
    if transform_type == "none":
        if target_name not in frame.columns:
            raise ValueError(f"Target column '{target_name}' is not present in the V2 training table.")
        return frame[target_name].astype(float)

    if transform_type == "logit_rh":
        source_column = str(transform.get("source_column", target_name if target_name in frame.columns else "humidity"))
        if source_column not in frame.columns:
            raise ValueError(f"logit_rh target transform requires column '{source_column}'.")
        return pd.Series(logit_relative_humidity(frame[source_column].astype(float)), index=frame.index)

    if transform_type == "dew_point":
        source_column = str(transform.get("source_column", "obs_dew_point_c"))
        if source_column not in frame.columns:
            raise ValueError(f"dew_point target transform requires column '{source_column}'.")
        return frame[source_column].astype(float)

    if transform_type == "dew_point_depression":
        source_column = str(transform.get("source_column", "obs_dew_point_depression"))
        if source_column not in frame.columns:
            raise ValueError(f"dew_point_depression target transform requires column '{source_column}'.")
        return frame[source_column].astype(float)

    if transform_type == "residual_from_feature":
        if target_name not in frame.columns:
            raise ValueError(f"Target column '{target_name}' is not present in the V2 training table.")
        if TARGET_CONTEXT_BASELINE_VALUE in frame.columns:
            baseline = frame[TARGET_CONTEXT_BASELINE_VALUE].astype(float)
        else:
            baseline_column = _residual_baseline_column(frame, transform)
            baseline = frame[baseline_column].astype(float)
        return frame[target_name].astype(float) - baseline

    raise ValueError(f"Unsupported target transform: {transform_type}")


def inverse_target_transform_value(
    value: float,
    transform_config: dict[str, Any] | None,
    context: dict[str, float] | None = None,
) -> float:
    transform = normalize_target_transform_config(transform_config)
    transform_type = transform["type"]
    if transform_type == "none":
        return float(value)
    if transform_type == "logit_rh":
        return float(inverse_logit_relative_humidity(np.asarray([value], dtype=float))[0])
    if transform_type == "dew_point":
        temp_c = _require_context_temperature(context, transform_type)
        return float(relative_humidity_from_dew_point(temp_c=np.asarray([temp_c]), dew_point_c=np.asarray([value]))[0])
    if transform_type == "dew_point_depression":
        temp_c = _require_context_temperature(context, transform_type)
        dew_point_c = temp_c - float(value)
        return float(relative_humidity_from_dew_point(temp_c=np.asarray([temp_c]), dew_point_c=np.asarray([dew_point_c]))[0])
    if transform_type == "residual_from_feature":
        baseline_value = _require_context_baseline_value(context, transform_type)
        return float(baseline_value + float(value))
    raise ValueError(f"Unsupported target transform: {transform_type}")


def logit_relative_humidity(relative_humidity: pd.Series | np.ndarray, eps: float = 1e-3) -> np.ndarray:
    rh01 = np.asarray(relative_humidity, dtype=float) / 100.0
    rh01 = np.clip(rh01, eps, 1.0 - eps)
    return np.log(rh01 / (1.0 - rh01))


def inverse_logit_relative_humidity(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return 100.0 / (1.0 + np.exp(-array))


def dew_point_from_relative_humidity(temp_c: pd.Series | np.ndarray, relative_humidity: pd.Series | np.ndarray) -> np.ndarray:
    temp = np.asarray(temp_c, dtype=float)
    humidity = np.clip(np.asarray(relative_humidity, dtype=float), 1e-3, 100.0)
    alpha = np.log(humidity / 100.0) + (17.625 * temp) / (243.04 + temp)
    return 243.04 * alpha / (17.625 - alpha)


def relative_humidity_from_dew_point(temp_c: pd.Series | np.ndarray, dew_point_c: pd.Series | np.ndarray) -> np.ndarray:
    temp = np.asarray(temp_c, dtype=float)
    dew_point = np.asarray(dew_point_c, dtype=float)
    exponent = (17.625 * dew_point) / (243.04 + dew_point) - (17.625 * temp) / (243.04 + temp)
    return np.clip(100.0 * np.exp(exponent), 0.0, 100.0)


def saturation_vapor_pressure_hpa(temp_c: pd.Series | np.ndarray) -> np.ndarray:
    temp = np.asarray(temp_c, dtype=float)
    return 6.1094 * np.exp((17.625 * temp) / (243.04 + temp))


def vapor_pressure_hpa(temp_c: pd.Series | np.ndarray, relative_humidity: pd.Series | np.ndarray) -> np.ndarray:
    humidity = np.clip(np.asarray(relative_humidity, dtype=float), 0.0, 100.0)
    return saturation_vapor_pressure_hpa(temp_c) * humidity / 100.0


def absolute_humidity_g_m3(temp_c: pd.Series | np.ndarray, relative_humidity: pd.Series | np.ndarray) -> np.ndarray:
    temp = np.asarray(temp_c, dtype=float)
    vapor_pressure = vapor_pressure_hpa(temp, relative_humidity)
    return 216.7 * vapor_pressure / (temp + 273.15)


def temperature_series_to_celsius(series: pd.Series | np.ndarray) -> pd.Series:
    values = pd.Series(series, copy=False).astype(float)
    finite = values.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return values
    median = float(finite.median())
    high_quantile = float(finite.quantile(0.95))
    if median > 170.0 or high_quantile > 170.0:
        return values - 273.15
    return values


def _require_context_temperature(context: dict[str, float] | None, transform_type: str) -> float:
    if context is None or TARGET_CONTEXT_TEMP_C not in context:
        raise ValueError(f"{transform_type} target transform requires '{TARGET_CONTEXT_TEMP_C}' context for RH restoration.")
    return float(context[TARGET_CONTEXT_TEMP_C])


def _require_context_baseline_value(context: dict[str, float] | None, transform_type: str) -> float:
    if context is None or TARGET_CONTEXT_BASELINE_VALUE not in context:
        raise ValueError(f"{transform_type} target transform requires '{TARGET_CONTEXT_BASELINE_VALUE}' context for restoration.")
    value = float(context[TARGET_CONTEXT_BASELINE_VALUE])
    if not np.isfinite(value):
        raise ValueError(f"{transform_type} target transform requires a finite baseline context value.")
    return value


def _residual_baseline_column(frame: pd.DataFrame, transform: dict[str, Any]) -> str:
    configured = str(
        transform.get("baseline_column")
        or transform.get("feature_column")
        or transform.get("forecast_feature")
        or ""
    )
    candidates = [configured] if configured else []
    candidates.extend(["era5_t2m_c", "era5_t2m", "predicted_temp"])
    baseline_column = next((column for column in candidates if column and column in frame.columns), None)
    if baseline_column is None:
        raise ValueError(
            "residual_from_feature target transform requires a baseline feature column. "
            "Configure data.target_transform.baseline_column, e.g. era5_t2m_c."
        )
    return baseline_column
