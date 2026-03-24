from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from weather_korea_forecast.data.dataset_tft import build_dataset_bundle
from weather_korea_forecast.features.time_features import add_time_features
from weather_korea_forecast.inference.schemas import ForecastPoint
from weather_korea_forecast.models.baselines import PersistenceBaseline
from weather_korea_forecast.models.tft_model import TFTModelWrapper
from weather_korea_forecast.training.train import _load_or_build_training_table
from weather_korea_forecast.utils.config import load_yaml
from weather_korea_forecast.utils.io import write_json, write_table
from weather_korea_forecast.utils.paths import resolve_path


def generate_forecast(
    experiment_dir: str | Path,
    station_id: str,
    forecast_init_time: str,
) -> pd.DataFrame:
    experiment_path = resolve_path(experiment_dir)
    data_config = load_yaml(experiment_path / "data_config.yaml")
    model_config = load_yaml(experiment_path / "model_config.yaml")
    _ = load_yaml(experiment_path / "train_config.yaml")

    training_table = _load_or_build_training_table(data_config)
    training_table = training_table.copy()
    training_table["datetime"] = pd.to_datetime(training_table["datetime"], utc=True)
    bundle = build_dataset_bundle(training_table, data_config, backend=model_config["model"].get("backend", "fallback_torch"))

    station_frame = training_table.loc[training_table["station_id"] == station_id].copy()
    init_time = _ensure_utc_timestamp(forecast_init_time)
    encoder_frame = station_frame.loc[station_frame["datetime"] <= init_time].tail(bundle.encoder_length).copy()
    if len(encoder_frame) < bundle.encoder_length:
        raise ValueError("Not enough history available for requested forecast window.")

    future_timestamps = pd.date_range(
        start=init_time + pd.Timedelta(hours=1),
        periods=bundle.prediction_length,
        freq="1h",
        tz="UTC",
    )
    decoder_frame = pd.DataFrame({"datetime": future_timestamps})
    decoder_frame = add_time_features(decoder_frame)
    scaling_columns = bundle.metadata.get("scaling_columns", [])
    encoder_frame = bundle.scaler.transform(encoder_frame, [column for column in scaling_columns if column in encoder_frame.columns])
    static_real = torch.tensor(encoder_frame.iloc[0][bundle.static_columns].to_numpy(dtype="float32")) if bundle.static_columns else torch.zeros(0)
    batch = {
        "encoder_cont": torch.tensor(encoder_frame[bundle.encoder_columns].to_numpy(dtype="float32")).unsqueeze(0),
        "decoder_known": torch.tensor(decoder_frame[bundle.decoder_columns].to_numpy(dtype="float32")).unsqueeze(0),
        "static_real": static_real.unsqueeze(0),
        "target": torch.zeros((1, bundle.prediction_length, len(bundle.target_columns)), dtype=torch.float32),
        "station_id": [station_id],
        "prediction_start": [future_timestamps[0]],
    }

    if model_config["model"]["type"] == "persistence":
        predictor = PersistenceBaseline(seasonal_period=model_config["model"].get("seasonal_period"))
        prediction = predictor.predict_batch(batch)
    else:
        wrapper = TFTModelWrapper.load(experiment_path / "model.pt", bundle)
        prediction, _, _ = wrapper.predict_loader([batch])

    points: list[ForecastPoint] = []
    target_column = bundle.target_columns[0]
    for horizon_index in range(prediction.shape[1]):
        prediction_value = float(prediction[0, horizon_index, 0].item())
        if target_column in bundle.scaler.means:
            prediction_value = float(bundle.scaler.inverse_values(target_column, prediction_value))
        points.append(
            ForecastPoint(
                station_id=station_id,
                timestamp=str(future_timestamps[horizon_index]),
                prediction=prediction_value,
            )
        )
    return pd.DataFrame([point.to_dict() for point in points])


def _ensure_utc_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run station-level inference for a saved experiment.")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--station-id", required=True)
    parser.add_argument("--forecast-init-time", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    forecast = generate_forecast(args.experiment_dir, args.station_id, args.forecast_init_time)
    if args.output_csv:
        write_table(forecast, args.output_csv)
    if args.output_json:
        write_json(forecast.to_dict(orient="records"), args.output_json)
    print(forecast.to_string(index=False))


if __name__ == "__main__":
    main()
