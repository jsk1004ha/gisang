from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from weather_korea_forecast.utils.io import read_table
from weather_korea_forecast.v2.data import build_v2_training_table
from weather_korea_forecast.v2.evaluate import evaluate_experiment
from weather_korea_forecast.v2.predict import generate_v2_forecast
from weather_korea_forecast.v2.train import train_v2_experiment


@pytest.fixture()
def synthetic_v2_project(tmp_path: Path) -> dict[str, object]:
    raw_dir = tmp_path / "data" / "raw"
    (raw_dir / "asos").mkdir(parents=True)
    (raw_dir / "era5").mkdir(parents=True)
    (raw_dir / "metadata").mkdir(parents=True)

    timestamps = pd.date_range("2024-01-01T00:00:00Z", periods=240, freq="1H")
    station_specs = [
        {"station_id": "108", "temp_offset": 0.0, "humidity_offset": 0.0, "lat": 37.57, "lon": 126.97, "elevation": 85.0, "region_class": "capital", "coastal_distance_km": 35.0},
        {"station_id": "159", "temp_offset": 2.5, "humidity_offset": 8.0, "lat": 35.10, "lon": 129.03, "elevation": 70.0, "region_class": "coastal", "coastal_distance_km": 2.0},
    ]
    observation_rows: list[dict[str, object]] = []
    era5_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []

    base = np.arange(len(timestamps))
    for spec in station_specs:
        temp = 10 + spec["temp_offset"] + 6 * np.sin(base / 12.0)
        humidity = 55 + spec["humidity_offset"] + 15 * np.cos(base / 18.0)
        pressure = 1005 + 2 * np.sin(base / 24.0)
        wind_speed = 2.5 + np.cos(base / 10.0)
        precipitation = np.where(base % 36 == 0, 1.0, 0.0)
        for index, timestamp in enumerate(timestamps):
            observation_rows.append(
                {
                    "station_id": spec["station_id"],
                    "datetime": str(timestamp),
                    "temp": float(temp[index]),
                    "humidity": float(np.clip(humidity[index], 0.0, 100.0)),
                    "pressure": float(pressure[index]),
                    "wind_speed": float(wind_speed[index]),
                    "precipitation": float(precipitation[index]),
                    "quality_flag": "",
                }
            )
            era5_rows.append(
                {
                    "station_id": spec["station_id"],
                    "datetime": str(timestamp),
                    "era5_t2m": float(temp[index] + 0.8),
                    "era5_sp": float(pressure[index] + 1.5),
                    "era5_u10": float(1.2 + np.sin(index / 8.0)),
                    "era5_v10": float(0.6 + np.cos(index / 8.0)),
                    "era5_tp": float(precipitation[index] * 0.9),
                }
            )
        metadata_rows.append(
            {
                "station_id": spec["station_id"],
                "lat": spec["lat"],
                "lon": spec["lon"],
                "elevation": spec["elevation"],
                "region_class": spec["region_class"],
                "coastal_distance_km": spec["coastal_distance_km"],
            }
        )

    obs_path = raw_dir / "asos" / "multistation.csv"
    era5_path = raw_dir / "era5" / "multistation_station.csv"
    metadata_path = raw_dir / "metadata" / "stations_v2.csv"
    pd.DataFrame(observation_rows).to_csv(obs_path, index=False)
    pd.DataFrame(era5_rows).to_csv(era5_path, index=False)
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)

    common_config = {
        "paths": {
            "observation_csv": str(obs_path),
            "era5_csv": str(era5_path),
            "station_metadata_csv": str(metadata_path),
            "output_training_table": str(tmp_path / "data" / "processed" / "training_table.csv"),
            "output_data_quality": str(tmp_path / "data" / "processed" / "data_quality.json"),
        },
        "data": {
            "timezone": {"source": "UTC"},
            "observation_columns": {
                "station_id": "station_id",
                "datetime": "datetime",
                "temp": "temp",
                "humidity": "humidity",
                "pressure": "pressure",
                "wind_speed": "wind_speed",
                "precipitation": "precipitation",
                "quality_flag": "quality_flag",
            },
            "era5": {"extraction_mode": "nearest"},
            "cleaning": {"interpolate_limit_hours": 6},
            "features": {
                "encoder_continuous": [
                    "target_value",
                    "target_value_lag_1",
                    "target_value_lag_3",
                    "target_value_lag_6",
                    "target_value_lag_12",
                    "target_value_lag_24",
                    "target_value_same_hour_prev_day",
                    "target_value_roll_mean_3",
                    "target_value_roll_std_3",
                    "target_value_roll_mean_6",
                    "target_value_roll_std_6",
                    "target_value_delta_1",
                    "target_value_delta_6",
                    "obs_pressure",
                    "obs_wind_speed",
                    "obs_precipitation",
                    "era5_t2m",
                    "era5_sp",
                    "era5_u10",
                    "era5_v10",
                    "era5_tp",
                    "hour_sin",
                    "hour_cos",
                    "doy_sin",
                    "doy_cos",
                ],
                "decoder_known": ["hour_sin", "hour_cos", "doy_sin", "doy_cos"],
                "static_real": ["lat", "lon", "elevation", "coastal_distance_km"],
                "static_categoricals": ["station_id", "region_class"],
            },
            "feature_engineering": {
                "lag_features": {
                    "target_value": [1, 3, 6, 12, 24],
                    "obs_pressure": [1, 6],
                    "obs_wind_speed": [1, 6],
                    "era5_t2m": [1, 3, 6],
                },
                "rolling_features": {"target_value": [3, 6]},
                "delta_features": {"target_value": [1, 6]},
            },
            "window": {"encoder_length": 24, "prediction_length": 6},
            "split": {
                "train_end": "2024-01-06T23:00:00Z",
                "val_end": "2024-01-08T23:00:00Z",
                "test_end": "2024-01-10T23:00:00Z",
            },
            "scaling": {
                "mode": "global",
                "group_column": "station_id",
                "columns": [
                    "target_value",
                    "target_value_lag_1",
                    "target_value_lag_3",
                    "target_value_lag_6",
                    "target_value_lag_12",
                    "target_value_lag_24",
                    "target_value_same_hour_prev_day",
                    "target_value_roll_mean_3",
                    "target_value_roll_std_3",
                    "target_value_roll_mean_6",
                    "target_value_roll_std_6",
                    "target_value_delta_1",
                    "target_value_delta_6",
                    "obs_pressure",
                    "obs_wind_speed",
                    "obs_precipitation",
                    "era5_t2m",
                    "era5_sp",
                    "era5_u10",
                    "era5_v10",
                    "era5_tp",
                    "lat",
                    "lon",
                    "elevation",
                    "coastal_distance_km",
                ],
            },
            "postprocess": {"clip_prediction": None},
        },
        "training": {
            "batch_size": 16,
            "max_epochs": 2,
            "early_stopping_patience": 1,
            "num_workers": 0,
            "device": "cpu",
            "gradient_clip_val": 0.1,
        },
        "evaluation": {"bias_correction": {"enabled": True, "mode": "per_horizon"}},
        "artifacts": {
            "root_dir": str(tmp_path / "artifacts" / "v2"),
            "leaderboard_path": str(tmp_path / "artifacts" / "v2" / "leaderboard.csv"),
        },
    }

    temp_ridge_config = {
        **common_config,
        "experiment": {"name": "synthetic_v2_temp_ridge", "version": "v2"},
        "data": {**common_config["data"], "target_name": "temp"},
        "model": {"name": "synthetic_v2_temp_ridge", "type": "ridge", "alpha": 1.0},
    }
    humidity_tft_config = {
        **common_config,
        "experiment": {"name": "synthetic_v2_humidity_tft", "version": "v2"},
        "data": {
            **common_config["data"],
            "target_name": "humidity",
            "postprocess": {"clip_prediction": [0, 100]},
        },
        "model": {
            "name": "synthetic_v2_humidity_tft",
            "type": "tft",
            "backend": "pytorch_forecasting",
            "allow_fallback_backend": False,
            "hidden_size": 8,
            "attention_head_size": 2,
            "hidden_continuous_size": 4,
            "dropout": 0.1,
            "learning_rate": 1e-2,
        },
        "training": {
            **common_config["training"],
            "max_epochs": 1,
            "gradient_clip_val": 0.0,
        },
    }
    return {"temp_ridge_config": temp_ridge_config, "humidity_tft_config": humidity_tft_config}


def test_build_v2_training_table_multistation_features(synthetic_v2_project: dict[str, object]) -> None:
    training_table, quality_report = build_v2_training_table(synthetic_v2_project["temp_ridge_config"])

    assert training_table["station_id"].nunique() == 2
    assert {"target_value", "target_value_lag_24", "target_value_roll_mean_6", "region_class", "coastal_distance_km"}.issubset(
        training_table.columns
    )
    assert quality_report["station_count"] == 2


def test_v2_ridge_roundtrip_updates_leaderboard(synthetic_v2_project: dict[str, object]) -> None:
    experiment_dir = train_v2_experiment(synthetic_v2_project["temp_ridge_config"])
    evaluation = evaluate_experiment(experiment_dir)
    forecast = generate_v2_forecast(
        experiment_dir=experiment_dir,
        station_id="108",
        forecast_init_time="2024-01-10T18:00:00Z",
    )

    leaderboard = read_table(Path(synthetic_v2_project["temp_ridge_config"]["artifacts"]["leaderboard_path"]))
    assert "rmse" in evaluation["metrics"]
    assert len(forecast) == synthetic_v2_project["temp_ridge_config"]["data"]["window"]["prediction_length"]
    assert "synthetic_v2_temp_ridge" in leaderboard["experiment_name"].tolist()
    assert (Path(experiment_dir) / "metrics_target_name_station_id.csv").exists()
    assert (Path(experiment_dir) / "experiment_summary.md").exists()


def test_v2_tft_roundtrip_for_humidity_with_clipping(synthetic_v2_project: dict[str, object]) -> None:
    pytest.importorskip("pytorch_forecasting")
    pytest.importorskip("lightning")

    experiment_dir = train_v2_experiment(synthetic_v2_project["humidity_tft_config"])
    evaluation = evaluate_experiment(experiment_dir)
    forecast = generate_v2_forecast(
        experiment_dir=experiment_dir,
        station_id="159",
        forecast_init_time="2024-01-10T18:00:00Z",
    )
    predictions = read_table(Path(experiment_dir) / "predictions_test.csv")

    assert "rmse" in evaluation["metrics"]
    assert predictions["target_name"].eq("humidity").all()
    assert forecast["prediction"].between(0.0, 100.0).all()
    assert (Path(experiment_dir) / "bias_correction.json").exists()
