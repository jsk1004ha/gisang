from __future__ import annotations

import argparse
import json
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
from weather_korea_forecast.v2.train import apply_postprocessing


def generate_v2_forecast(
    experiment_dir: str | Path,
    station_id: str,
    forecast_init_time: str,
    output_timezone: str = "UTC",
) -> pd.DataFrame:
    experiment_path = resolve_path(experiment_dir)
    config = load_yaml(experiment_path / "experiment_config.yaml")
    resolved_model_config = resolve_model_config({"model": dict(config["model"])})
    training_table = load_or_prepare_v2_training_table(config)
    bundle = build_v2_dataset_bundle(training_table, config, backend=resolved_model_config["model"].get("backend", "fallback_torch"))
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
    decoder_frame = add_time_features(pd.DataFrame({"datetime": future_timestamps}))
    decoder_frame = _complete_decoder_columns(decoder_frame, bundle, encoder_frame)

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
    if model_type in {"persistence", "seasonal_persistence", "ridge", "lightgbm"}:
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
    for horizon_index in range(prediction.shape[1]):
        raw_prediction = float(prediction[0, horizon_index, 0].item())
        prediction_value = float(bundle.scaler.inverse_values("target_value", [raw_prediction], groups=[normalized_station_id])[0])
        rows.append(
            {
                "station_id": normalized_station_id,
                "timestamp": future_timestamps[horizon_index],
                "target_name": bundle.target_name,
                "prediction": prediction_value,
            }
        )
    forecast = pd.DataFrame(rows)
    bias_payload_path = experiment_path / "bias_correction.json"
    if bias_payload_path.exists():
        bias_payload = json.loads(bias_payload_path.read_text(encoding="utf-8"))
        forecast["horizon_step"] = range(1, len(forecast) + 1)
        forecast["actual"] = forecast["prediction"]
        forecast = apply_postprocessing(forecast, config, bias_payload).drop(columns=["actual"])
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


def _complete_decoder_columns(decoder_frame: pd.DataFrame, bundle: V2DatasetBundle, encoder_frame: pd.DataFrame) -> pd.DataFrame:
    completed = decoder_frame.copy()
    base_row = encoder_frame.iloc[-1]
    for column in bundle.decoder_columns:
        if column not in completed.columns:
            completed[column] = float(base_row[column])
    return completed


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V2 station-level inference for a saved experiment.")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--station-id", required=True)
    parser.add_argument("--forecast-init-time", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-timezone", default="UTC")
    args = parser.parse_args()

    forecast = generate_v2_forecast(args.experiment_dir, args.station_id, args.forecast_init_time, args.output_timezone)
    if args.output_csv:
        write_table(forecast, args.output_csv)
    if args.output_json:
        write_json(forecast.to_dict(orient="records"), args.output_json)
    print(forecast.to_string(index=False))


if __name__ == "__main__":
    main()
