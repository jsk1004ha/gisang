from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd
import torch

from weather_korea_forecast.features.time_features import add_time_features
from weather_korea_forecast.models.registry import build_model, resolve_model_config
from weather_korea_forecast.models.tft_model import TFTModelWrapper
from weather_korea_forecast.utils.config import load_yaml
from weather_korea_forecast.utils.io import write_json, write_table
from weather_korea_forecast.utils.paths import resolve_path
from weather_korea_forecast.v2.data import load_or_prepare_v2_training_table
from weather_korea_forecast.v2.dataset import V2DatasetBundle, build_v2_dataset_bundle
from weather_korea_forecast.v2.future_features import load_future_weather_table
from weather_korea_forecast.v2.target_transforms import (
    TARGET_CONTEXT_BASELINE_VALUE,
    TARGET_CONTEXT_TEMP_C,
    inverse_target_transform_value,
    normalize_target_transform_config,
)
from weather_korea_forecast.v2.train import apply_postprocessing


def generate_v2_forecast(
    experiment_dir: str | Path,
    station_id: str,
    forecast_init_time: str,
    output_timezone: str = "UTC",
    future_weather_csv: str | Path | None = None,
    operational_mode: bool = False,
) -> pd.DataFrame:
    experiment_path = resolve_path(experiment_dir)
    config = load_yaml(experiment_path / "experiment_config.yaml")
    resolved_model_config = resolve_model_config({"model": dict(config["model"])})
    training_table = load_or_prepare_v2_training_table(config)
    bundle = build_v2_dataset_bundle(training_table, config, backend=resolved_model_config["model"].get("backend", "fallback_torch"))
    _validate_future_feature_inference_mode(bundle, operational_mode=operational_mode)
    inference_device = str(config["training"].get("device", "cpu"))

    normalized_station_id = str(station_id)
    full_frame = bundle.full_frame.copy()
    station_frame = full_frame.loc[full_frame["station_id"].astype(str) == normalized_station_id].copy()
    init_time = _ensure_utc_timestamp(forecast_init_time)
    history_frame = station_frame.loc[station_frame["datetime"] <= init_time].copy().sort_values("datetime")
    encoder_frame = history_frame.tail(bundle.encoder_length).copy()
    if len(encoder_frame) < bundle.encoder_length:
        raise ValueError("Not enough history available for the requested forecast window.")
    if not _is_hourly_history(encoder_frame["datetime"]):
        raise ValueError("History window is not contiguous hourly data.")

    future_timestamps = pd.date_range(
        start=init_time + pd.Timedelta(hours=1),
        periods=bundle.prediction_length,
        freq="1h",
        tz="UTC",
    )
    future_weather_frame = None
    configured_future_weather_csv = future_weather_csv or config.get("paths", {}).get("future_weather_csv")
    _require_operational_forecast_csv(bundle, operational_mode, configured_future_weather_csv)
    if configured_future_weather_csv:
        future_weather_frame = load_future_weather_table(
            configured_future_weather_csv,
            config=config,
            station_id=normalized_station_id,
            forecast_init_time=init_time,
        )
    if operational_mode:
        _validate_operational_future_weather_frame(future_weather_frame, future_timestamps, bundle)
    decoder_frame = _build_forecast_decoder_frame(
        station_frame=station_frame,
        future_timestamps=future_timestamps,
        bundle=bundle,
        encoder_frame=encoder_frame,
        future_weather_frame=future_weather_frame,
    )

    batch = {
        "encoder_cont": torch.tensor(encoder_frame[bundle.encoder_columns].to_numpy(dtype="float32")).unsqueeze(0),
        "decoder_known": torch.tensor(decoder_frame[bundle.decoder_columns].to_numpy(dtype="float32")).unsqueeze(0),
        "static_real": (
            torch.tensor(encoder_frame.tail(1).iloc[0][bundle.static_baseline_columns].to_numpy(dtype="float32")).unsqueeze(0)
            if bundle.static_baseline_columns
            else torch.zeros((1, 0), dtype=torch.float32)
        ),
        "target": torch.zeros((1, bundle.prediction_length, len(bundle.target_columns)), dtype=torch.float32),
        "station_id": [normalized_station_id],
        "prediction_start": [future_timestamps[0]],
    }

    model_type = resolved_model_config["model"]["type"]
    if model_type in {
        "persistence",
        "seasonal_persistence",
        "decoder_feature",
        "decoder_feature_baseline",
        "future_feature_baseline",
        "ridge",
        "horizon_wise_ridge",
        "horizonwise_ridge",
        "lightgbm",
        "horizon_wise_lightgbm",
        "horizonwise_lightgbm",
        "catboost",
        "catboost_regressor",
        "horizon_wise_catboost",
        "horizonwise_catboost",
        "residual",
    }:
        predictor = build_model(resolved_model_config, bundle).load(experiment_path / "model.pt", bundle, resolved_model_config)
        prediction, _, _ = predictor.predict_loader([batch])
    elif resolved_model_config["model"].get("backend") == "pytorch_forecasting":
        prediction = _predict_with_v2_tft(
            experiment_path=experiment_path,
            bundle=bundle,
            encoder_frame=encoder_frame,
            decoder_frame=decoder_frame,
            future_timestamps=future_timestamps,
            station_id=normalized_station_id,
            device=inference_device,
        )
    else:
        wrapper = TFTModelWrapper.load(experiment_path / "model.pt", bundle)
        prediction, _, _ = wrapper.predict_loader([batch], device=inference_device)

    rows = []
    scaler_group = _scaler_group_for_forecast(normalized_station_id, encoder_frame, bundle)
    target_transform = normalize_target_transform_config(bundle.metadata.get("target_transform"))
    target_contexts = _forecast_target_contexts(bundle, encoder_frame, decoder_frame)
    for horizon_index in range(prediction.shape[1]):
        raw_prediction = float(prediction[0, horizon_index, 0].item())
        prediction_model_value = float(bundle.scaler.inverse_values("target_value", [raw_prediction], groups=[scaler_group])[0])
        prediction_value = inverse_target_transform_value(prediction_model_value, target_transform, target_contexts[horizon_index])
        rows.append(
            {
                "station_id": normalized_station_id,
                "timestamp": future_timestamps[horizon_index],
                "target_name": bundle.target_name,
                "prediction": prediction_value,
                "prediction_model_value": prediction_model_value,
            }
        )
    forecast = pd.DataFrame(rows)
    bias_payload_path = experiment_path / "bias_correction.json"
    if bias_payload_path.exists():
        bias_payload = json.loads(bias_payload_path.read_text(encoding="utf-8"))
        forecast = _add_forecast_postprocessing_context(forecast, encoder_frame)
        forecast = apply_postprocessing(forecast, config, bias_payload)
    if output_timezone != "UTC":
        forecast["timestamp"] = pd.to_datetime(forecast["timestamp"], utc=True).dt.tz_convert(output_timezone)
    return forecast[["station_id", "timestamp", "target_name", "prediction"]]


def _predict_with_v2_tft(
    experiment_path: Path,
    bundle: V2DatasetBundle,
    encoder_frame: pd.DataFrame,
    decoder_frame: pd.DataFrame,
    future_timestamps: pd.DatetimeIndex,
    station_id: str,
    device: str = "cpu",
) -> torch.Tensor:
    from pytorch_forecasting import TimeSeriesDataSet

    wrapper = TFTModelWrapper.load(experiment_path / "model.pt", bundle)
    base_row = encoder_frame.iloc[-1].copy()
    future_rows: list[dict[str, object]] = []
    last_time_idx = int(encoder_frame["time_idx"].iloc[-1])
    for offset, timestamp in enumerate(future_timestamps, start=1):
        row = base_row.to_dict()
        row["station_id"] = station_id
        row["datetime"] = timestamp
        row["time_idx"] = last_time_idx + offset
        row["target_value"] = base_row["target_value"]
        for column in bundle.unknown_columns:
            row[column] = base_row[column]
        for column in bundle.decoder_columns:
            row[column] = float(decoder_frame.iloc[offset - 1][column])
        future_rows.append(row)

    prediction_frame = pd.concat([encoder_frame.copy(), pd.DataFrame(future_rows)], ignore_index=True, sort=False)
    prediction_frame["time_idx"] = prediction_frame["time_idx"].astype(int)
    prediction_dataset = TimeSeriesDataSet.from_dataset(
        bundle.train_dataset,
        prediction_frame,
        predict=True,
        stop_randomization=True,
    )
    prediction_loader = prediction_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
    model = wrapper.model
    prediction_result = model.predict(
        prediction_loader,
        trainer_kwargs={
            "accelerator": "cpu" if device == "cpu" else "auto",
            "devices": 1,
            "logger": False,
            "enable_checkpointing": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        },
    )
    if prediction_result.ndim == 2:
        prediction_result = prediction_result.unsqueeze(-1)
    return prediction_result.cpu()


def _build_forecast_decoder_frame(
    station_frame: pd.DataFrame,
    future_timestamps: pd.DatetimeIndex,
    bundle: V2DatasetBundle,
    encoder_frame: pd.DataFrame,
    future_weather_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build predict-time decoder covariates from future rows when available.

    Training/evaluation windows read ``decoder_known`` columns from the decoder
    (future-valid) slice. For historical backtests, or operational runs where
    future NWP/ERA5-like covariates have been prepared in the training table,
    inference should use those same future-valid covariates instead of silently
    falling back to the last encoder row.
    """

    future_index = pd.DataFrame({"datetime": future_timestamps})
    available_columns = [
        column
        for column in set(bundle.decoder_columns + list(bundle.metadata.get("target_context_columns", [])))
        if column in station_frame.columns and column != "datetime"
    ]
    if available_columns:
        future_covariates = (
            station_frame[["datetime", *available_columns]]
            .drop_duplicates("datetime")
            .copy()
        )
        decoder_frame = future_index.merge(future_covariates, on="datetime", how="left")
    else:
        decoder_frame = future_index

    if future_weather_frame is not None and not future_weather_frame.empty:
        weather_columns = [
            column
            for column in set(bundle.decoder_columns + list(bundle.metadata.get("target_context_columns", [])))
            if column in future_weather_frame.columns and column != "datetime"
        ]
        if weather_columns:
            forecast_covariates = future_weather_frame[["datetime", *weather_columns]].drop_duplicates("datetime").copy()
            decoder_frame = decoder_frame.merge(forecast_covariates, on="datetime", how="left", suffixes=("", "_forecast"))
            for column in weather_columns:
                forecast_column = f"{column}_forecast"
                if forecast_column not in decoder_frame.columns:
                    continue
                if column in decoder_frame.columns:
                    decoder_frame[column] = decoder_frame[forecast_column].combine_first(decoder_frame[column])
                else:
                    decoder_frame[column] = decoder_frame[forecast_column]
                decoder_frame = decoder_frame.drop(columns=[forecast_column])
    decoder_frame = _fill_operational_decoder_lags(decoder_frame, station_frame, bundle, encoder_frame)
    _require_future_weather_covariates(decoder_frame, bundle)
    decoder_frame = add_time_features(decoder_frame)
    return _complete_decoder_columns(decoder_frame, bundle, encoder_frame)


def _complete_decoder_columns(decoder_frame: pd.DataFrame, bundle: V2DatasetBundle, encoder_frame: pd.DataFrame) -> pd.DataFrame:
    completed = decoder_frame.copy()
    base_row = encoder_frame.iloc[-1]
    for column in bundle.decoder_columns:
        if column not in completed.columns:
            completed[column] = float(base_row[column])
    return completed


def _fill_operational_decoder_lags(
    decoder_frame: pd.DataFrame,
    station_frame: pd.DataFrame,
    bundle: V2DatasetBundle,
    encoder_frame: pd.DataFrame,
) -> pd.DataFrame:
    completed = decoder_frame.copy()
    target_lag_columns = [
        column
        for column in bundle.decoder_columns
        if column.startswith("target_value_lag_") or column == "target_value_same_hour_prev_day"
    ]
    if not target_lag_columns:
        return completed

    lookup = station_frame.drop_duplicates("datetime").set_index("datetime")
    scaler_group = _scaler_group_for_forecast(str(encoder_frame.iloc[-1]["station_id"]), encoder_frame, bundle)
    for column in target_lag_columns:
        lag_hours = 24 if column == "target_value_same_hour_prev_day" else _parse_target_lag_hours(column)
        if lag_hours is None:
            continue
        values: list[float | None] = []
        for timestamp in pd.to_datetime(completed["datetime"], utc=True):
            source_time = timestamp - pd.Timedelta(hours=lag_hours)
            if source_time not in lookup.index or "target_value_raw" not in lookup.columns:
                values.append(None)
                continue
            raw_value = float(lookup.loc[source_time, "target_value_raw"])
            scaled = _scale_decoder_feature_value(raw_value, column, scaler_group, bundle)
            values.append(scaled)
        if column in completed.columns:
            completed[column] = pd.Series(values, index=completed.index).combine_first(completed[column])
        else:
            completed[column] = values
    return completed


def _parse_target_lag_hours(column: str) -> int | None:
    try:
        return int(column.rsplit("_lag_", 1)[1])
    except (IndexError, ValueError):
        return None


def _scale_decoder_feature_value(value: float, column: str, scaler_group: str, bundle: V2DatasetBundle) -> float:
    if column not in bundle.scaler.columns:
        return value
    if bundle.scaler.mode == "none":
        return value
    if bundle.scaler.mode == "global":
        return float((value - bundle.scaler.global_means[column]) / bundle.scaler.global_stds[column])
    mean = bundle.scaler.group_means.get(column, {}).get(scaler_group, bundle.scaler.global_means[column])
    std = bundle.scaler.group_stds.get(column, {}).get(scaler_group, bundle.scaler.global_stds[column])
    return float((value - mean) / std)


def _forecast_target_contexts(
    bundle: V2DatasetBundle,
    encoder_frame: pd.DataFrame,
    decoder_frame: pd.DataFrame,
) -> list[dict[str, float] | None]:
    context_columns = list(bundle.metadata.get("target_context_columns", []))
    if not context_columns:
        return [None for _ in range(bundle.prediction_length)]
    contexts: list[dict[str, float]] = []
    for horizon_index in range(bundle.prediction_length):
        context: dict[str, float] = {}
        for column in context_columns:
            if column == TARGET_CONTEXT_TEMP_C:
                context[column] = _forecast_temperature_context_c(bundle, encoder_frame, decoder_frame, horizon_index)
            elif column == TARGET_CONTEXT_BASELINE_VALUE:
                context[column] = _forecast_residual_baseline_context(bundle, encoder_frame, decoder_frame, horizon_index)
            elif column in decoder_frame.columns:
                value = float(decoder_frame.iloc[horizon_index][column])
                if pd.isna(value):
                    raise ValueError(f"Missing forecast target-transform context column {column!r} at horizon {horizon_index + 1}.")
                context[column] = value
            elif column in encoder_frame.columns:
                context[column] = float(encoder_frame.iloc[-1][column])
            else:
                raise ValueError(f"Cannot build forecast target-transform context column: {column}")
        contexts.append(context)
    return contexts


def _forecast_temperature_context_c(
    bundle: V2DatasetBundle,
    encoder_frame: pd.DataFrame,
    decoder_frame: pd.DataFrame,
    horizon_index: int,
) -> float:
    transform = normalize_target_transform_config(bundle.metadata.get("target_transform"))
    forecast_feature = str(transform.get("forecast_temperature_feature", "") or "")
    candidates = [forecast_feature] if forecast_feature else []
    candidates.extend(["predicted_temp", TARGET_CONTEXT_TEMP_C, "era5_t2m_c", "era5_t2m", "obs_temp", "temp"])
    for column in candidates:
        if not column:
            continue
        if column in decoder_frame.columns:
            return float(decoder_frame.iloc[horizon_index][column])
        if column in encoder_frame.columns:
            return float(encoder_frame.iloc[-1][column])
    raise ValueError(
        "Forecasting with dew-point humidity target transforms requires a future temperature feature "
        "or a historical temperature context fallback."
    )


def _forecast_residual_baseline_context(
    bundle: V2DatasetBundle,
    encoder_frame: pd.DataFrame,
    decoder_frame: pd.DataFrame,
    horizon_index: int,
) -> float:
    transform = normalize_target_transform_config(bundle.metadata.get("target_transform"))
    configured = str(
        transform.get("baseline_column")
        or transform.get("feature_column")
        or transform.get("forecast_feature")
        or ""
    )
    candidates = [TARGET_CONTEXT_BASELINE_VALUE]
    if configured:
        candidates.append(configured)
    candidates.extend(["era5_t2m_c", "era5_t2m", "predicted_temp"])
    for column in candidates:
        if column in decoder_frame.columns:
            value = float(decoder_frame.iloc[horizon_index][column])
            if pd.notna(value):
                return value
    raise ValueError(
        "Residual target inference requires a finite future baseline feature for every horizon. "
        "Prepare future ERA5/NWP covariates or choose an observation-only experiment."
    )


def _validate_future_feature_inference_mode(bundle: V2DatasetBundle, operational_mode: bool = False) -> None:
    future_features = dict(bundle.metadata.get("future_features", {}))
    if not future_features.get("uses_future_weather_features"):
        return
    source = str(future_features.get("future_feature_source", "unknown"))
    if bool(future_features.get("backtest_only", False)):
        message = (
            f"Experiment uses backtest-only future weather source={source!r}. "
            "ERA5/reanalysis future covariates are not operational forecast inputs."
        )
        if operational_mode:
            raise ValueError(message + " Provide GFS/ECMWF/KMA forecast features or disable operational mode.")
        warnings.warn(message, RuntimeWarning, stacklevel=2)


def _require_operational_forecast_csv(
    bundle: V2DatasetBundle,
    operational_mode: bool,
    future_weather_csv: str | Path | None,
) -> None:
    if not operational_mode:
        return
    future_features = dict(bundle.metadata.get("future_features", {}))
    if not future_features.get("uses_future_weather_features"):
        return
    if future_weather_csv:
        return
    source = str(future_features.get("future_feature_source", "unknown"))
    raise ValueError(
        "Operational NWP-assisted inference requires --future-weather-csv or paths.future_weather_csv "
        f"for source={source!r}; historical decoder rows cannot stand in for prepared forecast NWP inputs."
    )


def _validate_operational_future_weather_frame(
    future_weather_frame: pd.DataFrame | None,
    future_timestamps: pd.DatetimeIndex,
    bundle: V2DatasetBundle,
) -> None:
    future_features = dict(bundle.metadata.get("future_features", {}))
    required_columns = [str(column) for column in future_features.get("future_weather_feature_columns", [])]
    if not required_columns:
        return
    source = str(future_features.get("future_feature_source", "unknown"))
    if future_weather_frame is None or future_weather_frame.empty:
        raise ValueError(f"Operational forecast source={source!r} did not provide any future weather rows.")
    if "datetime" not in future_weather_frame.columns:
        raise ValueError(f"Operational forecast source={source!r} must include valid_time/datetime rows.")

    frame = future_weather_frame.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    expected_times = pd.DatetimeIndex(pd.to_datetime(future_timestamps, utc=True))
    horizon_frame = frame.loc[frame["datetime"].isin(expected_times)]
    observed_times = set(horizon_frame["datetime"])
    missing_times = [timestamp.isoformat() for timestamp in expected_times if timestamp not in observed_times]
    missing_columns = [column for column in required_columns if column not in horizon_frame.columns]
    incomplete_columns = [
        column
        for column in required_columns
        if column in horizon_frame.columns and horizon_frame[column].isna().any()
    ]
    if missing_times or missing_columns or incomplete_columns:
        raise ValueError(
            "Operational NWP-assisted inference requires prepared forecast CSV covariates for every decoder horizon. "
            f"source={source!r}, missing_times={missing_times[:5]}, missing={missing_columns}, incomplete={incomplete_columns}."
        )


def _require_future_weather_covariates(decoder_frame: pd.DataFrame, bundle: V2DatasetBundle) -> None:
    future_features = dict(bundle.metadata.get("future_features", {}))
    required_columns = [str(column) for column in future_features.get("future_weather_feature_columns", [])]
    if not required_columns:
        return
    missing = [column for column in required_columns if column not in decoder_frame.columns]
    incomplete = [
        column
        for column in required_columns
        if column in decoder_frame.columns and decoder_frame[column].isna().any()
    ]
    if missing or incomplete:
        source = future_features.get("future_feature_source", "unknown")
        raise ValueError(
            "NWP-assisted V2 inference requires future-valid weather covariates for all decoder horizons. "
            f"source={source!r}, missing={missing}, incomplete={incomplete}. "
            "For operational runs, provide forecast NWP features; ERA5 reanalysis configs are backtest/MOS only."
        )


def _scaler_group_for_forecast(station_id: str, encoder_frame: pd.DataFrame, bundle: V2DatasetBundle) -> str:
    group_column = getattr(bundle.scaler, "group_column", "station_id")
    if group_column == "station_id":
        return station_id
    if group_column in encoder_frame.columns:
        return str(encoder_frame.iloc[-1][group_column])
    if group_column == "region" and "region_class" in encoder_frame.columns:
        return str(encoder_frame.iloc[-1]["region_class"])
    return station_id


def _add_forecast_postprocessing_context(forecast: pd.DataFrame, encoder_frame: pd.DataFrame) -> pd.DataFrame:
    context = forecast.copy()
    context["horizon_step"] = range(1, len(context) + 1)
    base_row = encoder_frame.iloc[-1]
    region_class = str(base_row.get("region_class", "unknown"))
    context["region_class"] = region_class
    context["region"] = str(base_row.get("region", region_class))
    context["season"] = pd.to_datetime(context["timestamp"], utc=True).map(_season_from_timestamp)
    return context


def _ensure_utc_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _is_hourly_history(datetimes: pd.Series) -> bool:
    if len(datetimes) <= 1:
        return True
    diffs = pd.to_datetime(datetimes, utc=True).sort_values().diff().dropna()
    return bool((diffs == pd.Timedelta(hours=1)).all())


def _season_from_timestamp(timestamp: pd.Timestamp) -> str:
    month = timestamp.month
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "autumn"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V2 station-level inference for a saved experiment.")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--station-id", required=True)
    parser.add_argument("--forecast-init-time", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-timezone", default="UTC")
    parser.add_argument(
        "--future-weather-csv",
        default=None,
        help=(
            "Prepared forecast NWP covariates with station_id and valid_time/datetime columns. "
            "Required for operational NWP-assisted experiments."
        ),
    )
    parser.add_argument(
        "--operational",
        action="store_true",
        help="Fail fast if the saved experiment relies on backtest-only ERA5/reanalysis future features.",
    )
    args = parser.parse_args()

    forecast = generate_v2_forecast(
        args.experiment_dir,
        args.station_id,
        args.forecast_init_time,
        args.output_timezone,
        future_weather_csv=args.future_weather_csv,
        operational_mode=args.operational,
    )
    if args.output_csv:
        write_table(forecast, args.output_csv)
    if args.output_json:
        write_json(forecast.to_dict(orient="records"), args.output_json)
    print(forecast.to_string(index=False))


if __name__ == "__main__":
    main()
