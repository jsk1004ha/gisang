from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_humidity_features(frame: pd.DataFrame, *, time_column: str = "valid_time") -> pd.DataFrame:
    output = frame.copy()
    if "era5_t2m_c" in output.columns:
        output["era5_t2m_c"] = _to_celsius_if_needed(output["era5_t2m_c"])
    elif "era5_t2m" in output.columns:
        output["era5_t2m_c"] = _to_celsius_if_needed(output["era5_t2m"])
    if "era5_dew_point_c" in output.columns:
        output["era5_dew_point_c"] = _to_celsius_if_needed(output["era5_dew_point_c"])
    elif "era5_d2m" in output.columns:
        output["era5_dew_point_c"] = _to_celsius_if_needed(output["era5_d2m"])
    elif "nwp_dew_point" in output.columns:
        output["era5_dew_point_c"] = _to_celsius_if_needed(output["nwp_dew_point"])
    if {"era5_t2m_c", "era5_dew_point_c"}.issubset(output.columns):
        output["era5_dew_point_depression"] = output["era5_t2m_c"].astype(float) - output["era5_dew_point_c"].astype(float)
    if time_column in output.columns:
        dt = pd.to_datetime(output[time_column], utc=True, errors="coerce")
        output["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24.0)
        output["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24.0)
        output["doy_sin"] = np.sin(2 * np.pi * dt.dt.dayofyear / 366.0)
        output["doy_cos"] = np.cos(2 * np.pi * dt.dt.dayofyear / 366.0)
    if "humidity" in output.columns:
        output["humidity"] = clip_relative_humidity(output["humidity"])
    if "humidity_percent" in output.columns:
        output["humidity_percent"] = clip_relative_humidity(output["humidity_percent"])
    return output


def restore_relative_humidity_from_dew_point(temp_c: pd.Series | np.ndarray | float, dew_point_c: pd.Series | np.ndarray | float) -> pd.Series:
    temp = pd.Series(temp_c, dtype="float64")
    dew = pd.Series(dew_point_c, dtype="float64")
    saturation_dew = np.exp((17.625 * dew) / (243.04 + dew))
    saturation_temp = np.exp((17.625 * temp) / (243.04 + temp))
    return clip_relative_humidity(100.0 * saturation_dew / saturation_temp)


def restore_relative_humidity_from_depression(temp_c: pd.Series | np.ndarray | float, dew_point_depression: pd.Series | np.ndarray | float) -> pd.Series:
    temp = pd.Series(temp_c, dtype="float64")
    depression = pd.Series(dew_point_depression, dtype="float64")
    return restore_relative_humidity_from_dew_point(temp, temp - depression)


def clip_relative_humidity(values: pd.Series | np.ndarray | float) -> pd.Series:
    return pd.Series(values, dtype="float64").clip(lower=0.0, upper=100.0)


def logit_relative_humidity(values: pd.Series | np.ndarray | float, *, eps: float = 1e-3) -> pd.Series:
    rh = clip_relative_humidity(values) / 100.0
    rh = rh.clip(lower=eps, upper=1.0 - eps)
    return np.log(rh / (1.0 - rh))


def inverse_logit_relative_humidity(values: pd.Series | np.ndarray | float) -> pd.Series:
    logits = pd.Series(values, dtype="float64")
    return clip_relative_humidity(100.0 / (1.0 + np.exp(-logits)))


def dry_humid_event_flags(values: pd.Series | np.ndarray | float, *, dry_threshold: float = 30.0, humid_threshold: float = 80.0) -> pd.DataFrame:
    rh = clip_relative_humidity(values)
    return pd.DataFrame({"is_dry_event": rh <= dry_threshold, "is_humid_event": rh >= humid_threshold})


def _to_celsius_if_needed(values: pd.Series | np.ndarray | float) -> pd.Series:
    series = pd.Series(values, dtype="float64")
    if series.median(skipna=True) > 150.0:
        return series - 273.15
    return series
