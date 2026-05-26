from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from weather_korea_forecast.utils.io import read_table
from weather_korea_forecast.utils.paths import resolve_path
from weather_korea_forecast.v2.target_transforms import (
    dew_point_from_relative_humidity,
    temperature_series_to_celsius,
)


BACKTEST_FUTURE_FEATURE_SOURCES = {"era5_reanalysis", "era5_reanalysis_backtest"}
OPERATIONAL_FUTURE_FEATURE_SOURCES = {
    "nwp_forecast",
    "gfs",
    "gfs_forecast",
    "ecmwf",
    "ecmwf_forecast",
    "kma_forecast",
    "gdps",
    "gdps_forecast",
    "um",
    "um_forecast",
    "custom_forecast",
    "prepared_forecast",
    "prepared_forecast_csv",
}
ALLOWED_FUTURE_FEATURE_SOURCES = BACKTEST_FUTURE_FEATURE_SOURCES | OPERATIONAL_FUTURE_FEATURE_SOURCES | {"none"}

WEATHER_FEATURE_PREFIXES = ("era5_", "nwp_", "gfs_", "ecmwf_", "kma_", "gdps_", "um_")


def build_future_feature_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Describe whether the experiment consumes future-valid weather covariates.

    Decoder weather covariates change the problem from pure observation-only
    forecasting into MOS/downscaling/backtest mode. Persisting this metadata in
    every artifact keeps leaderboards from mixing those tracks accidentally.
    """

    feature_config = config.get("data", {}).get("features", {})
    future_config = _future_feature_config(config)
    decoder_columns = [str(column) for column in feature_config.get("decoder_known", [])]
    configured_weather_columns = [str(column) for column in future_config.get("weather_columns", [])]
    weather_columns = sorted(
        {
            column
            for column in decoder_columns
            if _looks_like_future_weather_feature(column) or column in configured_weather_columns
        }
    )
    uses_future_weather = bool(weather_columns) or bool(future_config.get("uses_future_weather_features", False))
    source = _normalize_source(
        future_config.get("source")
        or config.get("data", {}).get("future_feature_source")
        or config.get("experiment", {}).get("future_feature_source")
        or ("era5_reanalysis" if uses_future_weather else "none")
    )
    if uses_future_weather and source not in ALLOWED_FUTURE_FEATURE_SOURCES:
        raise ValueError(
            "data.future_features.source must be one of "
            f"{sorted(ALLOWED_FUTURE_FEATURE_SOURCES)} when decoder_known contains future weather covariates; got {source!r}."
        )
    operational_valid = bool(
        future_config.get(
            "operational_valid",
            uses_future_weather and source in OPERATIONAL_FUTURE_FEATURE_SOURCES,
        )
    )
    backtest_only = bool(
        future_config.get(
            "backtest_only",
            uses_future_weather and source in BACKTEST_FUTURE_FEATURE_SOURCES and not operational_valid,
        )
    )
    forecast_track = str(
        future_config.get(
            "track",
            "nwp_assisted" if uses_future_weather else "observation_only",
        )
    )
    warnings: list[str] = []
    if uses_future_weather and source in BACKTEST_FUTURE_FEATURE_SOURCES:
        warnings.append(
            "Future ERA5/reanalysis decoder covariates are backtest/MOS upper-bound features; "
            "replace them with forecast NWP features for operational inference."
        )
    if uses_future_weather and source == "none":
        warnings.append("Future weather decoder covariates are configured but future_feature source is 'none'.")
    if backtest_only and not uses_future_weather:
        warnings.append(
            "This experiment is marked backtest-only without future weather covariates; "
            "check the config notes for oracle or otherwise non-operational decoder features."
        )
    leakage_risk_note = str(
        future_config.get(
            "leakage_risk_note",
            "Future ERA5/reanalysis decoder covariates are not available at real forecast issue time; this is backtest-only."
            if uses_future_weather and source in BACKTEST_FUTURE_FEATURE_SOURCES
            else "Future weather covariates must be issue-time-aligned forecasts, not valid-time observations."
            if uses_future_weather
            else "No future weather covariates configured.",
        )
    )
    note = str(
        future_config.get(
            "note",
            "Backtest/MOS only unless replaced by forecast NWP source"
            if uses_future_weather and not operational_valid
            else "",
        )
    )
    return {
        "uses_future_weather_features": uses_future_weather,
        "uses_future_nwp_features": uses_future_weather,
        "future_weather_feature_columns": weather_columns,
        "future_feature_source": source,
        "operational_valid": operational_valid,
        "backtest_only": backtest_only,
        "forecast_track": forecast_track,
        "track": forecast_track,
        "leakage_risk_note": leakage_risk_note,
        "note": note,
        "warnings": warnings,
    }


def future_weather_feature_columns(config: dict[str, Any]) -> list[str]:
    return list(build_future_feature_metadata(config)["future_weather_feature_columns"])


def load_future_weather_features(
    source: str,
    start_time: str | pd.Timestamp,
    horizon: int,
    stations: list[str] | tuple[str, ...],
    *,
    path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load future weather features through the V3 source adapter interface.

    Supported ``source`` values are ``era5_reanalysis_backtest``,
    ``gfs_forecast``, ``ecmwf_forecast`` and ``kma_forecast``.  Forecast sources
    currently use prepared CSV archives/tables; live API downloaders can be
    connected behind this function without changing MOS model code.
    """

    normalized = _normalize_source(source)
    if normalized == "era5_reanalysis":
        normalized = "era5_reanalysis_backtest"
    if normalized in {"prepared_forecast", "prepared_csv"}:
        normalized = "prepared_forecast_csv"
    if normalized not in {"era5_reanalysis_backtest", "gfs_forecast", "ecmwf_forecast", "kma_forecast", "prepared_forecast_csv"}:
        raise ValueError(
            "source must be one of: era5_reanalysis_backtest, gfs_forecast, ecmwf_forecast, kma_forecast, prepared_forecast_csv"
        )
    if path is None:
        raise NotImplementedError(
            f"{normalized} live adapter is not configured yet; provide a prepared CSV path for now."
        )

    cfg = dict(config or {})
    cfg.setdefault("data", {}).setdefault("future_features", {})["source"] = normalized
    start = pd.Timestamp(start_time)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")

    frames = []
    for station_id in stations:
        table = load_future_weather_table(path, cfg, str(station_id), start)
        valid_times = pd.date_range(start=start + pd.Timedelta(hours=1), periods=int(horizon), freq="1h", tz="UTC")
        table = table.loc[table["datetime"].isin(valid_times)].copy()
        table["horizon_step"] = ((table["datetime"] - start) / pd.Timedelta(hours=1)).astype(int)
        if "valid_time" in table.columns:
            table = table.drop(columns=["datetime"])
        else:
            table = table.rename(columns={"datetime": "valid_time"})
        frames.append(table)
    if not frames:
        return pd.DataFrame(columns=_canonical_future_weather_schema())
    output = pd.concat(frames, ignore_index=True, sort=False)
    output = _validate_prepared_forecast_features(output, start=start, horizon=int(horizon), stations=[str(s) for s in stations])
    for column in _canonical_future_weather_schema():
        if column not in output.columns:
            output[column] = pd.NA
    return output[_canonical_future_weather_schema() + [c for c in output.columns if c not in _canonical_future_weather_schema()]]


def _canonical_future_weather_schema() -> list[str]:
    return [
        "station_id",
        "forecast_init_time",
        "valid_time",
        "horizon_step",
        "nwp_t2m",
        "nwp_sp",
        "nwp_u10",
        "nwp_v10",
        "nwp_tp",
        "nwp_dew_point",
        "source",
        "issue_time",
    ]


def _validate_prepared_forecast_features(
    frame: pd.DataFrame,
    *,
    start: pd.Timestamp,
    horizon: int,
    stations: list[str],
) -> pd.DataFrame:
    normalized = frame.copy()
    if "valid_time" not in normalized.columns and "datetime" in normalized.columns:
        normalized["valid_time"] = normalized["datetime"]
    required = {"station_id", "valid_time", "horizon_step"}
    missing = sorted(required - set(normalized.columns))
    if missing:
        raise ValueError(f"Prepared forecast CSV is missing required columns: {missing}")
    normalized["station_id"] = normalized["station_id"].astype(str)
    normalized["valid_time"] = pd.to_datetime(normalized["valid_time"], utc=True)
    normalized["horizon_step"] = normalized["horizon_step"].astype(int)
    duplicate_count = int(normalized.duplicated(["station_id", "valid_time", "horizon_step"]).sum())
    if duplicate_count:
        raise ValueError(f"Prepared forecast CSV contains {duplicate_count} duplicate station/valid_time/horizon rows.")
    expected = {(station, step) for station in stations for step in range(1, horizon + 1)}
    observed = set(zip(normalized["station_id"], normalized["horizon_step"]))
    missing_horizons = sorted(expected - observed)
    if missing_horizons:
        preview = missing_horizons[:10]
        raise ValueError(f"Prepared forecast CSV is missing forecast horizons: {preview}")
    for temp_column in ["nwp_t2m", "era5_t2m", "era5_t2m_c", "nwp_temp_2m_c", "nwp_dew_point"]:
        if temp_column in normalized.columns:
            converted = temperature_series_to_celsius(normalized[temp_column])
            normalized[temp_column] = converted.to_numpy()
    if "nwp_sp" in normalized.columns and normalized["nwp_sp"].astype(float).median() > 2000:
        normalized["nwp_sp"] = normalized["nwp_sp"].astype(float) / 100.0
    if "source" not in normalized.columns:
        normalized["source"] = "prepared_forecast_csv"
    if "forecast_init_time" not in normalized.columns:
        normalized["forecast_init_time"] = start
    else:
        normalized["forecast_init_time"] = pd.to_datetime(normalized["forecast_init_time"], utc=True)
    if "issue_time" not in normalized.columns:
        normalized["issue_time"] = start
    else:
        normalized["issue_time"] = pd.to_datetime(normalized["issue_time"], utc=True)
    return normalized


def load_future_weather_table(
    path: str | Path,
    config: dict[str, Any],
    station_id: str,
    forecast_init_time: pd.Timestamp,
) -> pd.DataFrame:
    """Load prepared forecast NWP covariates for operational V2 inference.

    Expected canonical schema is:
    ``station_id, datetime|valid_time, <decoder weather columns>``.
    Optional issue/run time columns are supported. If present, the latest issue
    time not later than ``forecast_init_time`` is selected.
    """

    frame = read_table(resolve_path(path)).copy()
    if "station_id" not in frame.columns:
        raise ValueError("Future weather CSV requires a station_id column.")
    valid_time_column = _first_existing_column(frame, ["datetime", "valid_time", "timestamp"])
    if valid_time_column is None:
        raise ValueError("Future weather CSV requires one of datetime, valid_time, or timestamp.")

    frame = _apply_future_weather_column_mapping(frame, config)
    frame["station_id"] = frame["station_id"].astype(str)
    frame["datetime"] = pd.to_datetime(frame[valid_time_column], utc=True)
    frame = frame.loc[frame["station_id"] == str(station_id)].copy()
    if frame.empty:
        raise ValueError(f"Future weather CSV has no rows for station_id={station_id!r}.")

    issue_time_column = _first_existing_column(
        frame,
        ["issue_time", "forecast_init_time", "model_run_time", "run_time", "reference_time"],
    )
    if issue_time_column is not None:
        frame[issue_time_column] = pd.to_datetime(frame[issue_time_column], utc=True)
        eligible = frame.loc[frame[issue_time_column] <= forecast_init_time].copy()
        if eligible.empty:
            raise ValueError(
                "Future weather CSV has issue/run times, but none are at or before "
                f"forecast_init_time={forecast_init_time.isoformat()}."
            )
        selected_issue_time = eligible[issue_time_column].max()
        frame = eligible.loc[eligible[issue_time_column] == selected_issue_time].copy()

    frame = _add_future_weather_derived_columns(frame)
    return frame.drop_duplicates(["station_id", "datetime"]).sort_values(["station_id", "datetime"]).reset_index(drop=True)


def load_future_weather_archive(path: str | Path, config: dict[str, Any]) -> pd.DataFrame:
    """Load an issue-time-aligned forecast archive for honest MOS training.

    Canonical schema after loading:
    ``station_id, issue_time, valid_time, datetime, lead_hour, <forecast columns>``.

    ``datetime`` is kept as an alias for ``valid_time`` so the same prepared
    archive can also be used with predict-time future weather helpers.  Unlike
    valid-time-only ERA5/reanalysis tables, callers can train with
    ``issue_time`` and ``lead_hour`` to avoid selecting a forecast run that was
    issued after the forecast-init time.
    """

    frame = read_table(resolve_path(path)).copy()
    if "station_id" not in frame.columns:
        raise ValueError("Future weather archive requires a station_id column.")
    valid_time_column = _first_existing_column(frame, ["valid_time", "datetime", "timestamp"])
    issue_time_column = _first_existing_column(
        frame,
        ["issue_time", "forecast_init_time", "model_run_time", "run_time", "reference_time"],
    )
    if valid_time_column is None:
        raise ValueError("Future weather archive requires one of valid_time, datetime, or timestamp.")
    if issue_time_column is None and "lead_hour" not in frame.columns:
        raise ValueError("Future weather archive requires issue_time/run_time or lead_hour.")

    frame = _apply_future_weather_column_mapping(frame, config)
    frame["station_id"] = frame["station_id"].astype(str)
    frame["valid_time"] = pd.to_datetime(frame[valid_time_column], utc=True)
    frame["datetime"] = frame["valid_time"]
    if issue_time_column is not None:
        frame["issue_time"] = pd.to_datetime(frame[issue_time_column], utc=True)
    if "lead_hour" not in frame.columns:
        lead = (frame["valid_time"] - frame["issue_time"]) / pd.Timedelta(hours=1)
        frame["lead_hour"] = lead.astype(int)
    else:
        frame["lead_hour"] = frame["lead_hour"].astype(int)
        if "issue_time" not in frame.columns:
            frame["issue_time"] = frame["valid_time"] - pd.to_timedelta(frame["lead_hour"], unit="h")
    frame = _add_future_weather_derived_columns(frame)
    return (
        frame.drop_duplicates(["station_id", "issue_time", "valid_time", "lead_hour"])
        .sort_values(["station_id", "issue_time", "lead_hour", "valid_time"])
        .reset_index(drop=True)
    )


def _future_feature_config(config: dict[str, Any]) -> dict[str, Any]:
    data_config = config.get("data", {})
    raw = data_config.get("future_features", data_config.get("future_weather_features", {}))
    return dict(raw or {})


def _normalize_source(value: Any) -> str:
    source = str(value or "none").strip().lower().replace("-", "_")
    aliases = {
        "": "none",
        "off": "none",
        "disabled": "none",
        "era5": "era5_reanalysis",
        "era5_backtest": "era5_reanalysis_backtest",
        "reanalysis": "era5_reanalysis",
        "nwp": "nwp_forecast",
        "forecast": "nwp_forecast",
        "prepared": "prepared_forecast_csv",
        "prepared_csv": "prepared_forecast_csv",
    }
    return aliases.get(source, source)


def _looks_like_future_weather_feature(column: str) -> bool:
    return column.startswith(WEATHER_FEATURE_PREFIXES)


def _apply_future_weather_column_mapping(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    mapped = frame.copy()
    future_config = _future_feature_config(config)
    # Canonical form: expected_model_column: source_csv_column.
    for expected_column, source_column in dict(future_config.get("column_mapping", {})).items():
        expected = str(expected_column)
        source = str(source_column)
        if source in mapped.columns and expected not in mapped.columns:
            mapped[expected] = mapped[source]

    aliases = {
        "t2m": "era5_t2m",
        "temperature_2m": "era5_t2m",
        "temp_2m": "era5_t2m",
        "nwp_t2m": "era5_t2m",
        "gfs_t2m": "era5_t2m",
        "ecmwf_t2m": "era5_t2m",
        "sp": "era5_sp",
        "surface_pressure": "era5_sp",
        "nwp_sp": "era5_sp",
        "u10": "era5_u10",
        "u_wind_10m": "era5_u10",
        "nwp_u10": "era5_u10",
        "v10": "era5_v10",
        "v_wind_10m": "era5_v10",
        "nwp_v10": "era5_v10",
        "tp": "era5_tp",
        "precipitation": "era5_tp",
        "total_precipitation": "era5_tp",
        "nwp_tp": "era5_tp",
        "rh": "nwp_relative_humidity_2m",
        "r2": "nwp_relative_humidity_2m",
        "reh": "nwp_relative_humidity_2m",
        "relative_humidity": "nwp_relative_humidity_2m",
        "relative_humidity_2m": "nwp_relative_humidity_2m",
        "nwp_rh": "nwp_relative_humidity_2m",
        "nwp_rh2m": "nwp_relative_humidity_2m",
        "gfs_rh": "nwp_relative_humidity_2m",
        "gfs_rh2m": "nwp_relative_humidity_2m",
        "gfs_relative_humidity_2m": "nwp_relative_humidity_2m",
        "kma_reh": "nwp_relative_humidity_2m",
        "kma_relative_humidity": "nwp_relative_humidity_2m",
        "d2m": "nwp_dew_point_2m_c",
        "dew_point_2m": "nwp_dew_point_2m_c",
        "dewpoint_2m": "nwp_dew_point_2m_c",
        "nwp_d2m": "nwp_dew_point_2m_c",
        "gfs_d2m": "nwp_dew_point_2m_c",
        "gfs_dew_point_2m_c": "nwp_dew_point_2m_c",
        "nwp_temp_2m": "nwp_temp_2m_c",
        "nwp_t2m_c": "nwp_temp_2m_c",
        "gfs_temp_2m_c": "nwp_temp_2m_c",
        "nwp_surface_pressure": "nwp_surface_pressure",
        "gfs_surface_pressure": "nwp_surface_pressure",
        "nwp_u10": "nwp_u10",
        "gfs_u10": "nwp_u10",
        "nwp_v10": "nwp_v10",
        "gfs_v10": "nwp_v10",
        "nwp_total_precipitation": "nwp_total_precipitation",
        "gfs_total_precipitation": "nwp_total_precipitation",
    }
    for source, expected in aliases.items():
        if source in mapped.columns and expected not in mapped.columns:
            mapped[expected] = mapped[source]
    return mapped


def _add_future_weather_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    if "era5_t2m" in enriched.columns:
        enriched["era5_t2m"] = temperature_series_to_celsius(enriched["era5_t2m"])
        enriched["era5_t2m_c"] = enriched["era5_t2m"]
    if {"era5_u10", "era5_v10"}.issubset(enriched.columns):
        u10 = enriched["era5_u10"].astype(float)
        v10 = enriched["era5_v10"].astype(float)
        speed = np.sqrt(np.square(u10) + np.square(v10))
        direction = np.arctan2(u10, v10)
        enriched["era5_wind_speed"] = speed
        enriched["era5_wind_dir_sin"] = np.sin(direction)
        enriched["era5_wind_dir_cos"] = np.cos(direction)
    if "nwp_temp_2m_c" in enriched.columns:
        enriched["nwp_temp_2m_c"] = temperature_series_to_celsius(enriched["nwp_temp_2m_c"])
    if "nwp_dew_point_2m_c" in enriched.columns:
        enriched["nwp_dew_point_2m_c"] = temperature_series_to_celsius(enriched["nwp_dew_point_2m_c"])
    if "nwp_relative_humidity_2m" not in enriched.columns and {"nwp_temp_2m_c", "nwp_dew_point_2m_c"}.issubset(enriched.columns):
        temp = enriched["nwp_temp_2m_c"].astype(float)
        dew_point = enriched["nwp_dew_point_2m_c"].astype(float)
        # Invert the Magnus approximation used elsewhere in the pipeline.
        saturation = np.exp((17.625 * temp) / (243.04 + temp))
        vapor = np.exp((17.625 * dew_point) / (243.04 + dew_point))
        enriched["nwp_relative_humidity_2m"] = (100.0 * vapor / saturation).clip(lower=0.0, upper=100.0)
    if {"nwp_temp_2m_c", "nwp_relative_humidity_2m"}.issubset(enriched.columns) and "nwp_dew_point_2m_c" not in enriched.columns:
        enriched["nwp_dew_point_2m_c"] = dew_point_from_relative_humidity(
            enriched["nwp_temp_2m_c"].astype(float),
            enriched["nwp_relative_humidity_2m"].astype(float).clip(lower=1e-3, upper=100.0),
        )
    if {"nwp_temp_2m_c", "nwp_dew_point_2m_c"}.issubset(enriched.columns):
        enriched["nwp_dew_point_depression"] = enriched["nwp_temp_2m_c"].astype(float) - enriched["nwp_dew_point_2m_c"].astype(float)
    if {"nwp_u10", "nwp_v10"}.issubset(enriched.columns):
        u10 = enriched["nwp_u10"].astype(float)
        v10 = enriched["nwp_v10"].astype(float)
        speed = np.sqrt(np.square(u10) + np.square(v10))
        direction = np.arctan2(u10, v10)
        enriched["nwp_wind_speed"] = speed
        enriched["nwp_wind_dir_sin"] = np.sin(direction)
        enriched["nwp_wind_dir_cos"] = np.cos(direction)
    alias_pairs = {
        "era5_t2m_c": "nwp_t2m",
        "era5_sp": "nwp_sp",
        "era5_u10": "nwp_u10",
        "era5_v10": "nwp_v10",
        "era5_tp": "nwp_tp",
        "nwp_temp_2m_c": "nwp_t2m",
        "nwp_surface_pressure": "nwp_sp",
        "nwp_total_precipitation": "nwp_tp",
        "nwp_dew_point_2m_c": "nwp_dew_point",
    }
    for source_column, alias_column in alias_pairs.items():
        if source_column in enriched.columns and alias_column not in enriched.columns:
            enriched[alias_column] = enriched[source_column]
    return enriched


def _first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)
