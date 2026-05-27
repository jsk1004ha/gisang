from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from weather_korea_forecast.data.extract_era5_at_station import extract_era5_at_stations
from weather_korea_forecast.data.load_kma_asos import load_kma_asos
from weather_korea_forecast.data.load_observations import load_observation_sources
from weather_korea_forecast.data.split_time_series import assign_time_splits
from weather_korea_forecast.data.station_metadata import load_station_metadata
from weather_korea_forecast.features.time_features import add_time_features
from weather_korea_forecast.utils.io import read_table, write_json, write_table
from weather_korea_forecast.utils.paths import resolve_path
from weather_korea_forecast.v2.target_transforms import (
    TARGET_CONTEXT_BASELINE_VALUE,
    absolute_humidity_g_m3,
    append_target_transform_context,
    apply_target_transform,
    dew_point_from_relative_humidity,
    normalize_target_transform_config,
    relative_humidity_from_dew_point,
    saturation_vapor_pressure_hpa,
    temperature_series_to_celsius,
    vapor_pressure_hpa,
)


def build_v2_training_table(config: dict) -> tuple[pd.DataFrame, dict[str, object]]:
    paths = config["paths"]
    data_config = config["data"]
    target_name = str(data_config["target_name"])

    observations = _load_observations_from_config(config)
    station_metadata = load_station_metadata(paths["station_metadata_csv"])
    era5_raw = read_table(paths["era5_csv"])
    era5_features = extract_era5_at_stations(
        era5_df=era5_raw,
        station_df=station_metadata,
        mode=data_config.get("era5", {}).get("extraction_mode", "nearest"),
    )

    observations["station_id"] = observations["station_id"].astype(str)
    station_metadata["station_id"] = station_metadata["station_id"].astype(str)
    era5_features["station_id"] = era5_features["station_id"].astype(str)

    merged = observations.merge(era5_features, on=["station_id", "datetime"], how="left")
    merged = merged.merge(station_metadata, on="station_id", how="left")
    merged = add_time_features(merged)
    merged = _ensure_region_columns(merged)
    merged = _add_observation_aliases(merged)
    merged = _add_physical_features(merged)
    merged = _merge_predicted_temperature_features(merged, config)
    merged = append_target_transform_context(merged, data_config.get("target_transform"))
    merged["target_value"] = apply_target_transform(merged, target_name, data_config.get("target_transform"))
    merged["target_name"] = target_name
    merged = _fill_raw_continuous_columns(merged, config)
    merged = assign_time_splits(merged, data_config["split"])
    merged = _apply_station_month_hour_anomaly_target(merged, target_name, data_config.get("target_transform"))
    merged = _apply_feature_engineering(merged, config)
    merged = merged.sort_values(["station_id", "datetime"]).reset_index(drop=True)
    merged["quality_flag"] = merged.get("quality_flag", "").fillna("")

    quality_report = summarize_time_index_quality(merged)
    quality_report["era5_feature_sanity"] = summarize_era5_feature_sanity(merged)
    if target_name == "humidity":
        quality_report["humidity_feature_sanity"] = summarize_humidity_feature_sanity(merged, config)
    return merged, quality_report


def prepare_v2_data(config: dict) -> tuple[Path, Path | None]:
    training_table, quality_report = build_v2_training_table(config)
    output_path = write_table(training_table, config["paths"]["output_training_table"])
    quality_path = None
    if "output_data_quality" in config["paths"]:
        quality_path = write_json(quality_report, config["paths"]["output_data_quality"])
    return output_path, quality_path


def summarize_time_index_quality(frame: pd.DataFrame) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for station_id, group in frame.sort_values(["station_id", "datetime"]).groupby("station_id"):
        datetimes = pd.to_datetime(group["datetime"], utc=True).drop_duplicates().sort_values()
        diffs = datetimes.diff().dropna()
        missing_hours = int(sum(max(int(diff / pd.Timedelta(hours=1)) - 1, 0) for diff in diffs))
        duplicate_count = int(group.duplicated(subset=["station_id", "datetime"]).sum())
        rows.append(
            {
                "station_id": str(station_id),
                "start": str(datetimes.min()) if not datetimes.empty else None,
                "end": str(datetimes.max()) if not datetimes.empty else None,
                "row_count": int(len(group)),
                "timestamp_count": int(datetimes.nunique()),
                "duplicate_count": duplicate_count,
                "missing_hour_count": missing_hours,
            }
        )
    return {
        "station_count": int(frame["station_id"].nunique()) if "station_id" in frame.columns else 0,
        "stations": rows,
    }


def summarize_era5_feature_sanity(frame: pd.DataFrame) -> dict[str, object]:
    checks: dict[str, object] = {}
    warnings: list[str] = []
    for column, bounds in {
        "era5_t2m": (-80.0, 60.0),
        "era5_t2m_c": (-80.0, 60.0),
        "era5_dew_point_c": (-100.0, 50.0),
        "era5_dew_point_depression": (-5.0, 80.0),
    }.items():
        if column not in frame.columns:
            continue
        series = frame[column].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            continue
        column_summary = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
        }
        checks[column] = column_summary
        lower, upper = bounds
        if column_summary["min"] < lower or column_summary["max"] > upper:
            warnings.append(f"{column} outside expected Celsius range [{lower}, {upper}]: {column_summary}")
    checks["warnings"] = warnings
    return checks


def summarize_humidity_feature_sanity(frame: pd.DataFrame, config: dict) -> dict[str, object]:
    """Sanity checks for humidity experiments where dew point/time features matter.

    The function reports warnings rather than failing preparation so existing
    experiments remain reproducible, but it makes Kelvin-like dew point values
    and missing diurnal/seasonal covariates visible in data-quality artifacts.
    """

    features = config.get("data", {}).get("features", {})
    configured = {
        str(column)
        for section in ("encoder_continuous", "decoder_known", "static_real", "static_categoricals")
        for column in features.get(section, [])
    }
    required_time_features = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    missing_configured = [column for column in required_time_features if column not in configured]
    missing_frame = [column for column in required_time_features if column not in frame.columns]
    warnings: list[str] = []
    if missing_configured:
        warnings.append("humidity time features missing from config: " + ", ".join(missing_configured))
    if missing_frame:
        warnings.append("humidity time features missing from table: " + ", ".join(missing_frame))

    checks: dict[str, object] = {
        "required_time_features": required_time_features,
        "missing_configured_time_features": missing_configured,
        "missing_frame_time_features": missing_frame,
    }
    for column in ("era5_dew_point_c", "era5_t2m_c", "era5_dew_point_depression"):
        if column not in frame.columns:
            continue
        series = frame[column].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            continue
        summary = {"min": float(series.min()), "max": float(series.max()), "mean": float(series.mean())}
        checks[column] = summary
        if column in {"era5_dew_point_c", "era5_t2m_c"} and (summary["mean"] > 100.0 or summary["max"] > 100.0):
            warnings.append(f"{column} appears to be Kelvin; convert to Celsius before humidity modeling: {summary}")
        if column == "era5_dew_point_depression" and (summary["min"] < -5.0 or summary["max"] > 80.0):
            warnings.append(f"{column} outside expected physical range: {summary}")

    humidity_column = "humidity" if "humidity" in frame.columns else "target_value" if config.get("data", {}).get("target_name") == "humidity" and "target_value" in frame.columns else None
    if humidity_column is not None:
        series = frame[humidity_column].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if not series.empty:
            summary = {"min": float(series.min()), "max": float(series.max()), "mean": float(series.mean())}
            checks["relative_humidity"] = summary
            if summary["min"] < 0.0 or summary["max"] > 100.0:
                warnings.append(f"relative humidity outside [0, 100]: {summary}")

    checks["warnings"] = warnings
    return checks


def _load_observations_from_config(config: dict) -> pd.DataFrame:
    data_config = config["data"]
    default_source_tz = data_config.get("timezone", {}).get("source", "Asia/Seoul")
    observations_config = data_config.get("observations", {})
    sources = observations_config.get("sources") or _legacy_observation_sources(config)
    if len(sources) == 1 and sources[0].get("kind", "asos") == "asos" and not sources[0].get("resample_rule"):
        return load_kma_asos(
            sources[0]["path"],
            column_mapping=sources[0].get("column_mapping"),
            source_tz=sources[0].get("source_tz", default_source_tz),
        )
    return load_observation_sources(
        sources=sources,
        default_source_tz=default_source_tz,
        merge_strategy=observations_config.get("merge_strategy", "priority"),
    )


def _legacy_observation_sources(config: dict) -> list[dict]:
    paths = config["paths"]
    data_config = config["data"]
    default_source_tz = data_config.get("timezone", {}).get("source", "Asia/Seoul")
    sources = [
        {
            "name": "asos",
            "kind": "asos",
            "path": paths["observation_csv"],
            "column_mapping": data_config.get("observation_columns"),
            "source_tz": default_source_tz,
            "priority": 0,
        }
    ]
    aws_path = paths.get("aws_observation_csv")
    if aws_path:
        aws_config = data_config.get("aws", {})
        sources.append(
            {
                "name": aws_config.get("name", "aws"),
                "kind": "aws",
                "path": aws_path,
                "column_mapping": data_config.get("aws_observation_columns", data_config.get("observation_columns")),
                "source_tz": aws_config.get("source_tz", default_source_tz),
                "priority": int(aws_config.get("priority", 1)),
                "prefer_columns": aws_config.get(
                    "prefer_columns",
                    ["humidity", "pressure", "wind_speed", "precipitation", "quality_flag"],
                ),
                "resample_rule": aws_config.get("resample_rule"),
                "aggregation": aws_config.get("aggregation"),
                "station_id": aws_config.get("station_id"),
            }
        )
    return sources


def _ensure_region_columns(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    if "region_class" not in enriched.columns:
        if "region" in enriched.columns:
            enriched["region_class"] = enriched["region"]
        else:
            enriched["region_class"] = "unknown"
    if "region" not in enriched.columns:
        enriched["region"] = enriched["region_class"]
    return enriched


def _add_observation_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    for source_column in ("temp", "humidity", "pressure", "wind_speed", "precipitation"):
        if source_column in enriched.columns:
            enriched[f"obs_{source_column}"] = enriched[source_column]
    return enriched


def _add_physical_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    if {"temp", "humidity"}.issubset(enriched.columns):
        temp = enriched["temp"].astype(float)
        humidity = enriched["humidity"].astype(float).clip(lower=1e-3, upper=100.0)
        dew_point = dew_point_from_relative_humidity(temp, humidity)
        enriched["obs_dew_point_c"] = dew_point
        enriched["obs_dew_point_depression"] = temp - dew_point
        enriched["obs_saturation_vapor_pressure_hpa"] = saturation_vapor_pressure_hpa(temp)
        enriched["obs_vapor_pressure_hpa"] = vapor_pressure_hpa(temp, humidity)
        enriched["obs_absolute_humidity_g_m3"] = absolute_humidity_g_m3(temp, humidity)
    if "era5_t2m" in enriched.columns:
        era5_temp_c = temperature_series_to_celsius(enriched["era5_t2m"])
        enriched["era5_t2m"] = era5_temp_c
        enriched["era5_t2m_c"] = era5_temp_c
    else:
        era5_temp_c = None
    dew_point_column = _era5_dew_point_column(enriched)
    if dew_point_column is not None:
        era5_dew_point = temperature_series_to_celsius(enriched[dew_point_column])
        enriched["era5_dew_point_c"] = era5_dew_point
    elif era5_temp_c is not None and "humidity" in enriched.columns:
        humidity = enriched["humidity"].astype(float).clip(lower=1e-3, upper=100.0)
        enriched["era5_dew_point_c"] = dew_point_from_relative_humidity(era5_temp_c, humidity)
    if era5_temp_c is not None and "era5_dew_point_c" in enriched.columns:
        enriched["era5_dew_point_depression"] = era5_temp_c.astype(float) - enriched["era5_dew_point_c"].astype(float)
        enriched["era5_relative_humidity"] = relative_humidity_from_dew_point(
            temp_c=era5_temp_c.astype(float),
            dew_point_c=enriched["era5_dew_point_c"].astype(float),
        )
    if era5_temp_c is not None:
        enriched["nwp_temp_c"] = era5_temp_c.astype(float)
        # Generic V3 MOS aliases let the same residual/bias features work with
        # ERA5 backtests today and forecast NWP adapters later.
        if "era5_relative_humidity" in enriched.columns and "nwp_relative_humidity_2m" not in enriched.columns:
            enriched["nwp_relative_humidity_2m"] = enriched["era5_relative_humidity"]
        if "era5_sp" in enriched.columns and "nwp_sp" not in enriched.columns:
            enriched["nwp_sp"] = enriched["era5_sp"]
        if "era5_u10" in enriched.columns and "nwp_u10" not in enriched.columns:
            enriched["nwp_u10"] = enriched["era5_u10"]
        if "era5_v10" in enriched.columns and "nwp_v10" not in enriched.columns:
            enriched["nwp_v10"] = enriched["era5_v10"]
        if "era5_tp" in enriched.columns and "nwp_tp" not in enriched.columns:
            enriched["nwp_tp"] = enriched["era5_tp"]
    if era5_temp_c is not None and "temp" in enriched.columns:
        enriched["obs_minus_era5_temp"] = enriched["temp"].astype(float) - era5_temp_c.astype(float)
        enriched["obs_minus_nwp_temp"] = enriched["temp"].astype(float) - enriched["nwp_temp_c"].astype(float)
    if "era5_relative_humidity" in enriched.columns and "humidity" in enriched.columns:
        enriched["obs_minus_era5_rh"] = enriched["humidity"].astype(float) - enriched["era5_relative_humidity"].astype(float)
        enriched["obs_minus_nwp_rh"] = enriched["humidity"].astype(float) - enriched["nwp_relative_humidity_2m"].astype(float)
    if {"era5_u10", "era5_v10"}.issubset(enriched.columns):
        u10 = enriched["era5_u10"].astype(float)
        v10 = enriched["era5_v10"].astype(float)
        speed = np.sqrt(np.square(u10) + np.square(v10))
        direction = np.arctan2(u10, v10)
        enriched["era5_wind_speed"] = speed
        enriched["era5_wind_dir_sin"] = np.sin(direction)
        enriched["era5_wind_dir_cos"] = np.cos(direction)
    if "precipitation" in enriched.columns:
        enriched["precipitation_flag"] = (enriched["precipitation"].astype(float) > 0.0).astype(int)
    return enriched


def _era5_dew_point_column(frame: pd.DataFrame) -> str | None:
    candidates = [
        "era5_d2m",
        "era5_dew_point",
        "era5_dewpoint",
        "era5_dew_point_temperature",
        "era5_2m_dewpoint_temperature",
    ]
    return next((column for column in candidates if column in frame.columns), None)


def _merge_predicted_temperature_features(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    path = config.get("paths", {}).get("predicted_temperature_csv")
    if not path:
        return frame
    prediction_path = resolve_path(path)
    if not prediction_path.exists():
        raise FileNotFoundError(f"Configured predicted temperature feature file does not exist: {prediction_path}")
    predicted = read_table(prediction_path).copy()
    if "station_id" not in predicted.columns:
        raise ValueError("predicted_temperature_csv requires a station_id column.")
    timestamp_column = next((column for column in ("datetime", "valid_time", "timestamp") if column in predicted.columns), None)
    if timestamp_column is None:
        raise ValueError("predicted_temperature_csv requires one of datetime, valid_time, or timestamp.")
    value_column = next(
        (column for column in ("predicted_temp_horizon", "predicted_temp", "prediction", "temp_prediction") if column in predicted.columns),
        None,
    )
    if value_column is None:
        raise ValueError("predicted_temperature_csv requires one of predicted_temp_horizon, predicted_temp, prediction, or temp_prediction.")

    predicted = predicted.rename(columns={timestamp_column: "datetime", value_column: "predicted_temp"})
    predicted["station_id"] = predicted["station_id"].astype(str)
    predicted["datetime"] = pd.to_datetime(predicted["datetime"], utc=True)
    merged = frame.merge(predicted[["station_id", "datetime", "predicted_temp"]], on=["station_id", "datetime"], how="left")
    merged["predicted_temp_delta"] = merged["predicted_temp"].astype(float) - merged.get("obs_temp", merged.get("temp")).astype(float)
    grouped = merged.sort_values(["station_id", "datetime"]).groupby("station_id", group_keys=False)
    merged["predicted_temp_vs_prev_day"] = grouped["predicted_temp"].transform(lambda series: series - series.shift(24))
    return merged


def _apply_station_month_hour_anomaly_target(
    frame: pd.DataFrame,
    target_name: str,
    transform_config: dict | None,
) -> pd.DataFrame:
    transform = normalize_target_transform_config(transform_config)
    if transform["type"] != "station_month_hour_anomaly":
        return frame
    if target_name not in frame.columns:
        raise ValueError(f"station_month_hour_anomaly target transform requires column '{target_name}'.")
    enriched = frame.copy()
    group_columns = [str(column) for column in transform.get("group_columns", ["station_id", "month", "hour"])]
    missing = [column for column in group_columns if column not in enriched.columns]
    if missing:
        raise ValueError(f"station_month_hour_anomaly target transform missing group columns: {missing}")
    train = enriched.loc[enriched["split"] == "train"].copy()
    if train.empty:
        raise ValueError("station_month_hour_anomaly target transform requires train split rows.")
    global_mean = float(train[target_name].astype(float).mean())
    grouped = train.groupby(group_columns, dropna=False)[target_name].mean().rename("station_month_hour_climatology").reset_index()
    enriched = enriched.merge(grouped, on=group_columns, how="left")
    station_mean = train.groupby("station_id")[target_name].mean().to_dict() if "station_id" in train.columns else {}
    if "station_id" in enriched.columns:
        enriched["station_month_hour_climatology"] = enriched["station_month_hour_climatology"].fillna(enriched["station_id"].map(station_mean))
    enriched["station_month_hour_climatology"] = enriched["station_month_hour_climatology"].fillna(global_mean).astype(float)
    enriched[TARGET_CONTEXT_BASELINE_VALUE] = enriched["station_month_hour_climatology"]
    enriched["target_value"] = enriched[target_name].astype(float) - enriched["station_month_hour_climatology"]
    return enriched


def _fill_raw_continuous_columns(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    enriched = frame.copy()
    data_config = config["data"]
    interpolate_limit = int(data_config.get("cleaning", {}).get("interpolate_limit_hours", 12))
    numeric_columns = [
        column
        for column in enriched.columns
        if column.startswith("obs_")
        or column.startswith("era5_")
        or column.startswith("_target_context")
        or column in {"target_value", "coastal_distance_km"}
        or column == "station_month_hour_climatology"
    ]
    static_numeric_columns = data_config.get("features", {}).get("static_real", [])
    numeric_columns.extend([column for column in static_numeric_columns if column in enriched.columns])
    numeric_columns = sorted(set(numeric_columns))

    for column in numeric_columns:
        enriched[column] = (
            enriched.groupby("station_id")[column]
            .transform(
                lambda series: series.astype(float)
                .interpolate(limit=interpolate_limit, limit_direction="both")
                .ffill()
                .bfill()
            )
        )
    return enriched


def _apply_feature_engineering(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    enriched = frame.sort_values(["station_id", "datetime"]).copy()
    feature_config = config["data"].get("feature_engineering", {})
    lag_features = feature_config.get(
        "lag_features",
        {
            "target_value": [1, 3, 6, 12, 24, 48, 72],
            "obs_temp": [1, 3, 6, 24],
            "obs_humidity": [1, 3, 6, 24],
            "obs_pressure": [1, 6, 24],
            "obs_wind_speed": [1, 6, 24],
            "obs_dew_point_c": [1, 3, 6, 24],
            "obs_dew_point_depression": [1, 3, 6, 24],
            "era5_t2m": [1, 3, 6, 12, 24],
            "era5_sp": [1, 6, 24],
            "obs_minus_nwp_temp": [1, 3, 6, 12, 24, 48, 72],
        },
    )
    rolling_features = feature_config.get(
        "rolling_features",
        {
            "target_value": [3, 6, 12, 24],
            "obs_humidity": [6, 24],
            "obs_temp": [6, 24],
            "obs_minus_nwp_temp": [6, 24, 72],
        },
    )
    delta_features = feature_config.get(
        "delta_features",
        {
            "target_value": [1, 6, 24],
            "obs_humidity": [1, 6, 24],
            "obs_temp": [1, 6, 24],
            "obs_minus_nwp_temp": [24],
        },
    )

    if "obs_minus_nwp_temp" in enriched.columns:
        lag_features = {**lag_features, "obs_minus_nwp_temp": sorted(set(lag_features.get("obs_minus_nwp_temp", []) + [1, 3, 6, 12, 24, 48, 72]))}
        rolling_features = {**rolling_features, "obs_minus_nwp_temp": sorted(set(rolling_features.get("obs_minus_nwp_temp", []) + [6, 24, 72]))}
        delta_features = {**delta_features, "obs_minus_nwp_temp": sorted(set(delta_features.get("obs_minus_nwp_temp", []) + [24]))}

    grouped = enriched.groupby("station_id", group_keys=False)
    for column, lags in lag_features.items():
        if column not in enriched.columns:
            continue
        for lag in sorted({int(value) for value in lags}):
            enriched[f"{column}_lag_{lag}"] = grouped[column].shift(lag)

    for column, windows in rolling_features.items():
        if column not in enriched.columns:
            continue
        shifted = grouped[column].shift(1)
        for window in sorted({int(value) for value in windows}):
            enriched[f"{column}_roll_mean_{window}"] = shifted.groupby(enriched["station_id"]).transform(lambda series: series.rolling(window).mean())
            enriched[f"{column}_roll_std_{window}"] = shifted.groupby(enriched["station_id"]).transform(
                lambda series: series.rolling(window).std(ddof=0)
            )

    for column, periods in delta_features.items():
        if column not in enriched.columns:
            continue
        for period in sorted({int(value) for value in periods}):
            enriched[f"{column}_delta_{period}"] = grouped[column].transform(lambda series: series - series.shift(period))

    if "target_value_lag_24" in enriched.columns:
        enriched["target_value_same_hour_prev_day"] = enriched["target_value_lag_24"]
    if "target_value_lag_24" in enriched.columns:
        enriched["target_value_diff_vs_prev_day"] = enriched["target_value"] - enriched["target_value_lag_24"]
    if {"obs_humidity", "obs_temp"}.issubset(enriched.columns):
        enriched["humidity_temp_interaction"] = enriched["obs_humidity"] * enriched["obs_temp"]
    return enriched


def load_or_prepare_v2_training_table(config: dict) -> pd.DataFrame:
    output_path = resolve_path(config["paths"]["output_training_table"])
    if output_path.exists():
        cached = read_table(output_path)
        missing_columns = _missing_configured_training_columns(cached, config)
        if not missing_columns:
            return cached
        print(
            "Cached V2/V3 training table is missing configured columns; rebuilding "
            f"{output_path}: {missing_columns[:12]}"
        )
    training_table, quality_report = build_v2_training_table(config)
    write_table(training_table, output_path)
    if "output_data_quality" in config["paths"]:
        write_json(quality_report, config["paths"]["output_data_quality"])
    return training_table


def _missing_configured_training_columns(frame: pd.DataFrame, config: dict) -> list[str]:
    data_config = config.get("data", {})
    features = data_config.get("features", {})
    required: set[str] = {"station_id", "datetime", "target_value", "split"}
    for section in ("encoder_continuous", "decoder_known", "static_real", "static_categoricals"):
        required.update(str(column) for column in features.get(section, []))
    required.update(str(column) for column in data_config.get("scaling", {}).get("columns", []))
    return sorted(column for column in required if column not in frame.columns)
