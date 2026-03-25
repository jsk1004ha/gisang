from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def synthetic_project(tmp_path: Path) -> dict:
    raw_dir = tmp_path / "data" / "raw"
    (raw_dir / "asos").mkdir(parents=True)
    (raw_dir / "era5").mkdir(parents=True)
    (raw_dir / "metadata").mkdir(parents=True)

    timestamps = pd.date_range("2024-01-01T00:00:00Z", periods=120, freq="1H")
    temp = 10 + 5 * np.sin(np.arange(len(timestamps)) / 6.0)
    obs = pd.DataFrame(
        {
            "station_id": "SEOUL",
            "datetime": timestamps.astype(str),
            "temp": temp,
            "humidity": 55 + 10 * np.cos(np.arange(len(timestamps)) / 12.0),
            "pressure": 1005 + np.sin(np.arange(len(timestamps)) / 24.0),
            "wind_speed": 2.0 + np.cos(np.arange(len(timestamps)) / 10.0),
            "precipitation": np.zeros(len(timestamps)),
            "quality_flag": "",
        }
    )
    obs_path = raw_dir / "asos" / "seoul_observations.csv"
    obs.to_csv(obs_path, index=False)

    era5 = pd.DataFrame(
        {
            "station_id": "SEOUL",
            "datetime": timestamps.astype(str),
            "era5_t2m": temp + 0.4,
            "era5_sp": 1007 + np.sin(np.arange(len(timestamps)) / 18.0),
            "era5_u10": 1.0 + np.sin(np.arange(len(timestamps)) / 8.0),
            "era5_v10": 0.5 + np.cos(np.arange(len(timestamps)) / 8.0),
            "era5_tp": np.zeros(len(timestamps)),
        }
    )
    era5_path = raw_dir / "era5" / "seoul_station.csv"
    era5.to_csv(era5_path, index=False)

    metadata = pd.DataFrame(
        {
            "station_id": ["SEOUL"],
            "lat": [37.5665],
            "lon": [126.9780],
            "elevation": [38.0],
            "region": ["capital"],
            "coastal_distance_km": [35.0],
        }
    )
    metadata_path = raw_dir / "metadata" / "stations.csv"
    metadata.to_csv(metadata_path, index=False)

    data_config = {
        "paths": {
            "observation_csv": str(obs_path),
            "era5_csv": str(era5_path),
            "station_metadata_csv": str(metadata_path),
            "output_training_table": str(tmp_path / "data" / "processed" / "training_table.csv"),
        },
        "timezone": {"source": "UTC"},
        "targets": ["temp"],
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
        "features": {
            "encoder_continuous": [
                "obs_temp",
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
            "static_real": ["lat", "lon", "elevation"],
        },
        "window": {"encoder_length": 6, "prediction_length": 3},
        "split": {
            "train_end": "2024-01-02T23:00:00Z",
            "val_end": "2024-01-03T23:00:00Z",
            "test_end": "2024-01-05T23:00:00Z",
        },
        "scaling": {"mode": "global", "columns": ["obs_temp", "era5_t2m", "era5_sp", "era5_u10", "era5_v10", "era5_tp", "target_temp"]},
    }
    multi_target_data_config = {
        **data_config,
        "targets": ["temp", "humidity"],
        "features": {
            **data_config["features"],
            "encoder_continuous": [
                "obs_temp",
                "obs_humidity",
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
        },
        "scaling": {
            "mode": "global",
            "columns": [
                "obs_temp",
                "obs_humidity",
                "era5_t2m",
                "era5_sp",
                "era5_u10",
                "era5_v10",
                "era5_tp",
                "target_temp",
                "target_humidity",
            ],
        },
    }

    model_config = {
        "model": {
            "name": "tft_test",
            "type": "tft",
            "backend": "fallback_torch",
            "target_columns": ["target_temp"],
            "hidden_size": 8,
            "dropout": 0.0,
            "learning_rate": 1e-2,
        }
    }
    auto_tft_config = {
        "model": {
            **model_config["model"],
            "name": "tft_auto_test",
            "backend": "auto",
            "allow_fallback_backend": True,
            "target_columns": ["target_temp", "target_humidity"],
        }
    }
    baseline_config = {
        "model": {
            "name": "persistence_test",
            "type": "persistence",
            "seasonal_period": None,
            "target_columns": ["target_temp"],
        }
    }
    ridge_config = {
        "model": {
            "name": "ridge_test",
            "type": "ridge",
            "alpha": 1.0,
            "target_columns": ["target_temp", "target_humidity"],
        }
    }
    seasonal_baseline_config = {
        "model": {
            "name": "seasonal_persistence_test",
            "type": "seasonal_persistence",
            "seasonal_period": 24,
            "target_columns": ["target_temp", "target_humidity"],
        }
    }
    train_config = {
        "experiment": {"name": "synthetic_v1"},
        "training": {"batch_size": 8, "max_epochs": 2, "early_stopping_patience": 1, "num_workers": 0, "device": "cpu"},
        "artifacts": {"root_dir": str(tmp_path / "artifacts")},
        "seed": 7,
    }

    return {
        "root": tmp_path,
        "data_config": data_config,
        "multi_target_data_config": multi_target_data_config,
        "model_config": model_config,
        "auto_tft_config": auto_tft_config,
        "baseline_config": baseline_config,
        "ridge_config": ridge_config,
        "seasonal_baseline_config": seasonal_baseline_config,
        "train_config": train_config,
    }
