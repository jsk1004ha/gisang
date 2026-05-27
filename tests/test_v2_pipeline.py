from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from weather_korea_forecast.utils.io import read_table
from weather_korea_forecast.utils.config import load_yaml
from weather_korea_forecast.v2.data import build_v2_training_table
from weather_korea_forecast.v2.dataset import build_v2_dataset_bundle
from weather_korea_forecast.v2.evaluate import evaluate_experiment, evaluate_prediction_frame
from weather_korea_forecast.v2.future_features import build_future_feature_metadata, load_future_weather_table
from weather_korea_forecast.v2.predict import (
    _build_forecast_decoder_frame,
    _require_operational_forecast_csv,
    _validate_operational_future_weather_frame,
    generate_v2_forecast,
)
from weather_korea_forecast.v2.train import apply_postprocessing, compute_bias_correction, train_v2_experiment
from weather_korea_forecast.v2.target_transforms import logit_relative_humidity, relative_humidity_from_dew_point


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
    assert {
        "target_value",
        "target_value_lag_24",
        "target_value_roll_mean_6",
        "region_class",
        "coastal_distance_km",
        "obs_minus_era5_temp",
        "coastal_class",
        "terrain_class",
    }.issubset(training_table.columns)
    assert quality_report["station_count"] == 2


def test_v2_feature_engineering_uses_past_only_windows(synthetic_v2_project: dict[str, object]) -> None:
    training_table, _ = build_v2_training_table(synthetic_v2_project["temp_ridge_config"])
    station_frame = training_table.loc[training_table["station_id"].astype(str) == "108"].sort_values("datetime").reset_index(drop=True)
    row_index = 30

    assert station_frame.loc[row_index, "target_value_lag_1"] == pytest.approx(station_frame.loc[row_index - 1, "target_value"])
    assert station_frame.loc[row_index, "target_value_lag_24"] == pytest.approx(station_frame.loc[row_index - 24, "target_value"])
    assert station_frame.loc[row_index, "target_value_roll_mean_3"] == pytest.approx(
        station_frame.loc[row_index - 3 : row_index - 1, "target_value"].mean()
    )
    assert station_frame.loc[row_index, "target_value_delta_6"] == pytest.approx(
        station_frame.loc[row_index, "target_value"] - station_frame.loc[row_index - 6, "target_value"]
    )


def test_v2_era5_dew_point_kelvin_conversion_and_sanity(synthetic_v2_project: dict[str, object], tmp_path: Path) -> None:
    config = {
        **synthetic_v2_project["temp_ridge_config"],
        "paths": {
            **synthetic_v2_project["temp_ridge_config"]["paths"],
            "output_training_table": str(tmp_path / "kelvin" / "training_table.csv"),
            "output_data_quality": str(tmp_path / "kelvin" / "data_quality.json"),
        },
    }
    era5_path = Path(config["paths"]["era5_csv"])
    era5 = pd.read_csv(era5_path)
    era5["era5_d2m"] = era5["era5_t2m"] - 2.0 + 273.15
    era5["era5_t2m"] = era5["era5_t2m"] + 273.15
    era5.to_csv(era5_path, index=False)

    training_table, quality_report = build_v2_training_table(config)

    assert training_table["era5_t2m"].mean() < 40.0
    assert training_table["era5_t2m_c"].equals(training_table["era5_t2m"])
    assert training_table["era5_dew_point_c"].mean() < 40.0
    assert (
        training_table["era5_dew_point_depression"]
        - (training_table["era5_t2m_c"] - training_table["era5_dew_point_c"])
    ).abs().max() == pytest.approx(0.0)
    expected_era5_rh = relative_humidity_from_dew_point(
        training_table["era5_t2m_c"],
        training_table["era5_dew_point_c"],
    )
    assert np.nanmax(np.abs(training_table["era5_relative_humidity"] - expected_era5_rh)) == pytest.approx(0.0)
    assert np.nanmax(np.abs(training_table["nwp_relative_humidity_2m"] - expected_era5_rh)) == pytest.approx(0.0)
    assert quality_report["era5_feature_sanity"]["era5_dew_point_c"]["mean"] < 40.0


def test_v2_humidity_feature_sanity_reports_time_features_and_celsius_dewpoint(synthetic_v2_project: dict[str, object], tmp_path: Path) -> None:
    base_config = synthetic_v2_project["humidity_tft_config"]
    features = {
        **base_config["data"]["features"],
        "encoder_continuous": [
            column
            for column in base_config["data"]["features"]["encoder_continuous"]
            if column not in {"hour_sin", "hour_cos", "doy_sin", "doy_cos"}
        ],
        "decoder_known": [],
    }
    config = {
        **base_config,
        "paths": {
            **base_config["paths"],
            "output_training_table": str(tmp_path / "humidity_sanity" / "training_table.csv"),
            "output_data_quality": str(tmp_path / "humidity_sanity" / "data_quality.json"),
        },
        "data": {
            **base_config["data"],
            "features": features,
        },
    }
    era5_path = Path(config["paths"]["era5_csv"])
    era5 = pd.read_csv(era5_path)
    era5["era5_d2m"] = era5["era5_t2m"] - 2.0 + 273.15
    era5["era5_t2m"] = era5["era5_t2m"] + 273.15
    era5.to_csv(era5_path, index=False)

    training_table, quality_report = build_v2_training_table(config)
    sanity = quality_report["humidity_feature_sanity"]

    assert training_table["era5_dew_point_c"].mean() < 40.0
    assert set(sanity["missing_configured_time_features"]) == {"hour_sin", "hour_cos", "doy_sin", "doy_cos"}
    assert sanity["missing_frame_time_features"] == []
    assert any("humidity time features missing from config" in warning for warning in sanity["warnings"])
    assert not any("appears to be Kelvin" in warning for warning in sanity["warnings"])


def test_v2_humidity_can_merge_predicted_temp_horizon_feature(synthetic_v2_project: dict[str, object], tmp_path: Path) -> None:
    base_config = synthetic_v2_project["humidity_tft_config"]
    obs = pd.read_csv(base_config["paths"]["observation_csv"])
    predicted_path = tmp_path / "predicted_temp_horizon.csv"
    pd.DataFrame(
        {
            "station_id": obs["station_id"].astype(str),
            "valid_time": obs["datetime"],
            "predicted_temp_horizon": obs["temp"].astype(float) + 0.5,
        }
    ).to_csv(predicted_path, index=False)
    config = {
        **base_config,
        "paths": {
            **base_config["paths"],
            "predicted_temperature_csv": str(predicted_path),
            "output_training_table": str(tmp_path / "predicted_temp_horizon" / "training_table.csv"),
            "output_data_quality": str(tmp_path / "predicted_temp_horizon" / "data_quality.json"),
        },
        "data": {
            **base_config["data"],
            "features": {
                **base_config["data"]["features"],
                "encoder_continuous": [
                    *base_config["data"]["features"]["encoder_continuous"],
                    "predicted_temp",
                    "predicted_temp_delta",
                    "predicted_temp_vs_prev_day",
                ],
            },
            "scaling": {
                **base_config["data"]["scaling"],
                "columns": [
                    *base_config["data"]["scaling"]["columns"],
                    "predicted_temp",
                    "predicted_temp_delta",
                    "predicted_temp_vs_prev_day",
                ],
            },
        },
    }

    training_table, _ = build_v2_training_table(config)

    assert "predicted_temp" in training_table.columns
    assert "predicted_temp_delta" in training_table.columns
    assert training_table["predicted_temp_delta"].dropna().median() == pytest.approx(0.5)


def test_v2_station_month_hour_anomaly_target_uses_train_climatology(
    synthetic_v2_project: dict[str, object],
    tmp_path: Path,
) -> None:
    base_config = synthetic_v2_project["temp_ridge_config"]
    config = {
        **base_config,
        "paths": {
            **base_config["paths"],
            "output_training_table": str(tmp_path / "anomaly" / "training_table.csv"),
            "output_data_quality": str(tmp_path / "anomaly" / "data_quality.json"),
        },
        "data": {
            **base_config["data"],
            "target_transform": {"type": "station_month_hour_anomaly", "group_columns": ["station_id", "month", "hour"]},
        },
    }

    training_table, _ = build_v2_training_table(config)
    row = training_table.loc[training_table["split"] == "train"].iloc[30]
    train_group = training_table.loc[
        (training_table["split"] == "train")
        & (training_table["station_id"].astype(str) == str(row["station_id"]))
        & (training_table["month"] == row["month"])
        & (training_table["hour"] == row["hour"])
    ]
    expected_climatology = train_group["temp"].astype(float).mean()

    assert row["station_month_hour_climatology"] == pytest.approx(expected_climatology)
    assert row["target_value"] == pytest.approx(row["temp"] - expected_climatology)
    lagged = (
        training_table.loc[
            (training_table["station_id"].astype(str) == str(row["station_id"]))
            & (pd.to_datetime(training_table["datetime"], utc=True) == pd.to_datetime(row["datetime"], utc=True) - pd.Timedelta(hours=24))
        ]
        .iloc[0]
    )
    assert row["target_value_lag_24"] == pytest.approx(lagged["target_value"])


def test_v2_stationwise_and_regionwise_scaling_fit_train_groups_only(synthetic_v2_project: dict[str, object]) -> None:
    training_table, _ = build_v2_training_table(synthetic_v2_project["temp_ridge_config"])
    config = synthetic_v2_project["temp_ridge_config"]

    station_config = {
        **config,
        "data": {
            **config["data"],
            "scaling": {**config["data"]["scaling"], "mode": "stationwise", "group_column": "station_id"},
        },
    }
    station_bundle = build_v2_dataset_bundle(training_table, station_config)
    train_raw = training_table.loc[training_table["split"] == "train"].copy()
    expected_station_mean = train_raw.loc[train_raw["station_id"].astype(str) == "108", "target_value"].astype(float).mean()

    assert station_bundle.scaler.mode == "station_wise"
    assert station_bundle.scaler.group_means["target_value"]["108"] == pytest.approx(expected_station_mean)
    assert station_bundle.scaler.group_means["target_value"]["108"] != pytest.approx(training_table["target_value"].astype(float).mean())

    region_config = {
        **config,
        "data": {
            **config["data"],
            "scaling": {**config["data"]["scaling"], "mode": "regionwise", "group_column": "region_class"},
        },
    }
    region_bundle = build_v2_dataset_bundle(training_table, region_config)
    assert region_bundle.scaler.mode == "region_wise"
    assert set(region_bundle.scaler.group_means["target_value"]) == {"capital", "coastal"}


def test_v2_logit_humidity_target_restores_rh_predictions(synthetic_v2_project: dict[str, object], tmp_path: Path) -> None:
    base_config = synthetic_v2_project["temp_ridge_config"]
    config = {
        **base_config,
        "experiment": {"name": "synthetic_v2_humidity_logit_ridge", "version": "v2"},
        "paths": {
            **base_config["paths"],
            "output_training_table": str(tmp_path / "logit" / "training_table.csv"),
            "output_data_quality": str(tmp_path / "logit" / "data_quality.json"),
        },
        "data": {
            **base_config["data"],
            "target_name": "humidity",
            "target_transform": {"type": "logit_rh", "source_column": "humidity"},
            "postprocess": {"clip_prediction": [0, 100]},
        },
        "model": {"name": "synthetic_v2_humidity_logit_ridge", "type": "ridge", "alpha": 1.0},
        "artifacts": {
            "root_dir": str(tmp_path / "artifacts" / "logit"),
            "leaderboard_path": str(tmp_path / "artifacts" / "logit" / "leaderboard.csv"),
        },
    }
    training_table, _ = build_v2_training_table(config)
    first = training_table.iloc[0]
    assert first["target_value"] == pytest.approx(logit_relative_humidity(np.asarray([first["humidity"]]))[0])

    experiment_dir = train_v2_experiment(config)
    predictions = read_table(Path(experiment_dir) / "predictions_test.csv")

    assert predictions["target_name"].eq("humidity").all()
    assert predictions["prediction"].between(0.0, 100.0).all()
    assert predictions["actual"].between(0.0, 100.0).all()
    assert "prediction_model_value" in predictions.columns


def test_v2_humidity_extreme_sample_weighting_marks_sequences(synthetic_v2_project: dict[str, object], tmp_path: Path) -> None:
    base_config = synthetic_v2_project["temp_ridge_config"]
    config = {
        **base_config,
        "paths": {
            **base_config["paths"],
            "output_training_table": str(tmp_path / "weighted" / "training_table.csv"),
            "output_data_quality": str(tmp_path / "weighted" / "data_quality.json"),
        },
        "data": {
            **base_config["data"],
            "target_name": "humidity",
            "sample_weighting": {
                "enabled": True,
                "mode": "humidity_extremes",
                "source_column": "humidity",
                "low_threshold": 50.0,
                "high_threshold": 70.0,
                "extreme_weight": 3.0,
            },
        },
    }
    training_table, _ = build_v2_training_table(config)
    bundle = build_v2_dataset_bundle(training_table, config)
    weights = [float(sample["sample_weight"].item()) for sample in bundle.train_dataset.samples]

    assert max(weights) == pytest.approx(3.0)
    assert min(weights) == pytest.approx(1.0)


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
    assert {
        "scaling_mode",
        "num_stations",
        "best_horizon",
        "worst_horizon",
        "raw_rmse",
        "corrected_rmse",
        "forecast_track",
        "model_family",
        "uses_future_nwp_features",
        "future_feature_source",
        "operational_valid",
        "backtest_only",
    }.issubset(leaderboard.columns)


def test_v2_forecast_supports_region_horizon_bias_correction(synthetic_v2_project: dict[str, object]) -> None:
    base_config = synthetic_v2_project["temp_ridge_config"]
    config = {
        **base_config,
        "experiment": {"name": "synthetic_v2_temp_region_bias", "version": "v2"},
        "evaluation": {
            "bias_correction": {
                **base_config["evaluation"]["bias_correction"],
                "mode": "per_region_horizon",
            }
        },
    }
    experiment_dir = train_v2_experiment(config)

    forecast = generate_v2_forecast(
        experiment_dir=experiment_dir,
        station_id="108",
        forecast_init_time="2024-01-10T18:00:00Z",
    )
    bias_payload = json.loads((Path(experiment_dir) / "bias_correction.json").read_text(encoding="utf-8"))

    assert bias_payload["mode"] == "per_region_horizon"
    assert list(forecast.columns) == ["station_id", "timestamp", "target_name", "prediction"]
    assert len(forecast) == config["data"]["window"]["prediction_length"]
    assert forecast["prediction"].notna().all()


def test_v2_forecast_decoder_uses_future_known_covariates(synthetic_v2_project: dict[str, object]) -> None:
    base_config = synthetic_v2_project["temp_ridge_config"]
    config = {
        **base_config,
        "data": {
            **base_config["data"],
            "features": {
                **base_config["data"]["features"],
                "decoder_known": [
                    *base_config["data"]["features"]["decoder_known"],
                    "era5_t2m",
                ],
            },
            "feature_engineering": {
                **base_config["data"]["feature_engineering"],
                "lag_features": {
                    **base_config["data"]["feature_engineering"]["lag_features"],
                    "target_value": [1, 3, 6, 12, 24, 48, 72],
                },
            },
            "scaling": {
                **base_config["data"]["scaling"],
                "columns": [
                    *base_config["data"]["scaling"]["columns"],
                    "target_value_lag_48",
                    "target_value_lag_72",
                ],
            },
        },
    }
    training_table, _ = build_v2_training_table(config)
    bundle = build_v2_dataset_bundle(training_table, config)
    station_frame = bundle.full_frame.loc[bundle.full_frame["station_id"].astype(str) == "108"].copy()
    init_time = pd.Timestamp("2024-01-09T12:00:00Z")
    encoder_frame = station_frame.loc[station_frame["datetime"] <= init_time].tail(bundle.encoder_length).copy()
    future_timestamps = pd.date_range(
        start=init_time + pd.Timedelta(hours=1),
        periods=bundle.prediction_length,
        freq="1h",
        tz="UTC",
    )

    decoder_frame = _build_forecast_decoder_frame(station_frame, future_timestamps, bundle, encoder_frame)
    expected = (
        station_frame.set_index("datetime")
        .loc[future_timestamps, "era5_t2m"]
        .astype(float)
        .to_numpy()
    )

    assert np.allclose(decoder_frame["era5_t2m"].astype(float).to_numpy(), expected)
    assert not np.allclose(
        decoder_frame["era5_t2m"].astype(float).to_numpy(),
        np.repeat(float(encoder_frame.iloc[-1]["era5_t2m"]), len(decoder_frame)),
    )




def test_v2_operational_predict_requires_explicit_forecast_csv() -> None:
    class OperationalBundle:
        metadata = {
            "future_features": {
                "uses_future_weather_features": True,
                "future_feature_source": "prepared_forecast_csv",
                "future_weather_feature_columns": ["era5_t2m"],
            }
        }

    with pytest.raises(ValueError, match="--future-weather-csv"):
        _require_operational_forecast_csv(OperationalBundle(), operational_mode=True, future_weather_csv=None)

    _require_operational_forecast_csv(OperationalBundle(), operational_mode=True, future_weather_csv="forecast.csv")


def test_v2_operational_predict_validates_prepared_forecast_horizon_coverage() -> None:
    class OperationalBundle:
        metadata = {
            "future_features": {
                "uses_future_weather_features": True,
                "future_feature_source": "prepared_forecast_csv",
                "future_weather_feature_columns": ["era5_t2m", "era5_sp"],
            }
        }

    future_timestamps = pd.date_range("2024-01-01T01:00:00Z", periods=2, freq="1h", tz="UTC")
    incomplete_frame = pd.DataFrame(
        [
            {"datetime": "2024-01-01T01:00:00Z", "era5_t2m": 8.0, "era5_sp": 1013.0},
            {"datetime": "2024-01-01T02:00:00Z", "era5_t2m": 9.0, "era5_sp": np.nan},
        ]
    )
    with pytest.raises(ValueError, match="prepared forecast CSV covariates"):
        _validate_operational_future_weather_frame(incomplete_frame, future_timestamps, OperationalBundle())

    complete_frame = incomplete_frame.fillna({"era5_sp": 1014.0})
    _validate_operational_future_weather_frame(complete_frame, future_timestamps, OperationalBundle())


def test_v2_future_weather_decoder_requires_future_valid_covariates(synthetic_v2_project: dict[str, object]) -> None:
    base_config = synthetic_v2_project["temp_ridge_config"]
    config = {
        **base_config,
        "data": {
            **base_config["data"],
            "future_features": {"track": "nwp_assisted", "source": "era5_reanalysis", "operational_valid": False},
            "features": {
                **base_config["data"]["features"],
                "decoder_known": [
                    *base_config["data"]["features"]["decoder_known"],
                    "era5_t2m",
                ],
            },
        },
    }
    training_table, _ = build_v2_training_table(config)
    bundle = build_v2_dataset_bundle(training_table, config)
    station_frame = bundle.full_frame.loc[bundle.full_frame["station_id"].astype(str) == "108"].copy()
    init_time = pd.to_datetime(station_frame["datetime"], utc=True).max() - pd.Timedelta(hours=2)
    encoder_frame = station_frame.loc[station_frame["datetime"] <= init_time].tail(bundle.encoder_length).copy()
    future_timestamps = pd.date_range(
        start=init_time + pd.Timedelta(hours=1),
        periods=bundle.prediction_length,
        freq="1h",
        tz="UTC",
    )

    with pytest.raises(ValueError, match="requires future-valid weather covariates"):
        _build_forecast_decoder_frame(station_frame, future_timestamps, bundle, encoder_frame)


def test_v2_operational_decoder_uses_forecast_csv_and_history_lags(
    synthetic_v2_project: dict[str, object],
    tmp_path: Path,
) -> None:
    base_config = synthetic_v2_project["temp_ridge_config"]
    config = {
        **base_config,
        "data": {
            **base_config["data"],
            "future_features": {
                "track": "nwp_assisted",
                "source": "gfs_forecast",
                "operational_valid": True,
                "column_mapping": {
                    "era5_t2m": "gfs_t2m",
                    "era5_sp": "gfs_sp",
                    "era5_u10": "gfs_u10",
                    "era5_v10": "gfs_v10",
                    "era5_tp": "gfs_tp",
                },
            },
            "features": {
                **base_config["data"]["features"],
                "decoder_known": [
                    *base_config["data"]["features"]["decoder_known"],
                    "era5_t2m",
                    "era5_sp",
                    "era5_u10",
                    "era5_v10",
                    "era5_tp",
                    "era5_wind_speed",
                    "era5_wind_dir_sin",
                    "era5_wind_dir_cos",
                    "target_value_lag_24",
                    "target_value_lag_48",
                    "target_value_lag_72",
                    "target_value_same_hour_prev_day",
                ],
            },
            "feature_engineering": {
                **base_config["data"]["feature_engineering"],
                "lag_features": {
                    **base_config["data"]["feature_engineering"]["lag_features"],
                    "target_value": [1, 3, 6, 12, 24, 48, 72],
                },
            },
            "scaling": {
                **base_config["data"]["scaling"],
                "columns": [
                    *base_config["data"]["scaling"]["columns"],
                    "target_value_lag_48",
                    "target_value_lag_72",
                ],
            },
        },
    }
    training_table, _ = build_v2_training_table(config)
    bundle = build_v2_dataset_bundle(training_table, config)
    station_frame = bundle.full_frame.loc[bundle.full_frame["station_id"].astype(str) == "108"].copy()
    init_time = pd.Timestamp("2024-01-09T12:00:00Z")
    encoder_frame = station_frame.loc[station_frame["datetime"] <= init_time].tail(bundle.encoder_length).copy()
    future_timestamps = pd.date_range(
        start=init_time + pd.Timedelta(hours=1),
        periods=bundle.prediction_length,
        freq="1h",
        tz="UTC",
    )
    forecast_rows = []
    for index, timestamp in enumerate(future_timestamps):
        forecast_rows.append(
            {
                "station_id": "108",
                "issue_time": str(init_time),
                "valid_time": str(timestamp),
                "gfs_t2m": 280.15 + index,
                "gfs_sp": 1008.0 + index,
                "gfs_u10": 1.0,
                "gfs_v10": 2.0,
                "gfs_tp": 0.1 * index,
            }
        )
    forecast_path = tmp_path / "future_gfs.csv"
    pd.DataFrame(forecast_rows).to_csv(forecast_path, index=False)
    future_weather = load_future_weather_table(forecast_path, config, station_id="108", forecast_init_time=init_time)

    decoder_frame = _build_forecast_decoder_frame(station_frame, future_timestamps, bundle, encoder_frame, future_weather)
    expected_lag24_time = future_timestamps[0] - pd.Timedelta(hours=24)
    expected_raw_lag24 = float(
        station_frame.loc[station_frame["datetime"] == expected_lag24_time, "target_value_raw"].iloc[0]
    )
    expected_scaled_lag24 = (
        expected_raw_lag24 - bundle.scaler.global_means["target_value_lag_24"]
    ) / bundle.scaler.global_stds["target_value_lag_24"]

    assert decoder_frame["era5_t2m"].iloc[0] == pytest.approx(7.0)
    assert decoder_frame["era5_wind_speed"].iloc[0] == pytest.approx(np.sqrt(5.0))
    assert decoder_frame["target_value_lag_24"].iloc[0] == pytest.approx(expected_scaled_lag24)
    assert decoder_frame["target_value_same_hour_prev_day"].iloc[0] == pytest.approx(expected_scaled_lag24)


def test_v2_future_era5_residual_target_restores_absolute_temperature(
    synthetic_v2_project: dict[str, object],
    tmp_path: Path,
) -> None:
    base_config = synthetic_v2_project["temp_ridge_config"]
    residual_features = [
        "obs_minus_era5_temp_lag_1",
        "obs_minus_era5_temp_lag_6",
        "obs_minus_era5_temp_lag_24",
        "obs_minus_era5_temp_lag_48",
        "obs_minus_era5_temp_roll_mean_24",
        "obs_minus_era5_temp_roll_mean_72",
    ]
    config = {
        **base_config,
        "experiment": {"name": "synthetic_v2_temp_future_era5_residual", "version": "v2"},
        "paths": {
            **base_config["paths"],
            "output_training_table": str(tmp_path / "residual" / "training_table.csv"),
            "output_data_quality": str(tmp_path / "residual" / "data_quality.json"),
        },
        "data": {
            **base_config["data"],
            "target_transform": {"type": "residual_from_feature", "baseline_column": "era5_t2m_c"},
            "future_features": {
                "track": "nwp_assisted",
                "source": "era5_reanalysis",
                "operational_valid": False,
            },
            "features": {
                **base_config["data"]["features"],
                "encoder_continuous": [
                    *base_config["data"]["features"]["encoder_continuous"],
                    *residual_features,
                ],
                "decoder_known": [
                    *base_config["data"]["features"]["decoder_known"],
                    "era5_t2m",
                ],
            },
            "feature_engineering": {
                **base_config["data"]["feature_engineering"],
                "lag_features": {
                    **base_config["data"]["feature_engineering"]["lag_features"],
                    "obs_minus_era5_temp": [1, 6, 24, 48],
                },
                "rolling_features": {
                    **base_config["data"]["feature_engineering"]["rolling_features"],
                    "obs_minus_era5_temp": [24, 72],
                },
            },
            "scaling": {
                **base_config["data"]["scaling"],
                "columns": [
                    *base_config["data"]["scaling"]["columns"],
                    *residual_features,
                ],
            },
        },
        "artifacts": {
            "root_dir": str(tmp_path / "artifacts" / "residual"),
            "leaderboard_path": str(tmp_path / "artifacts" / "residual" / "leaderboard.csv"),
        },
    }
    training_table, _ = build_v2_training_table(config)
    row = training_table.iloc[24]
    station_frame = training_table.loc[training_table["station_id"].astype(str) == str(row["station_id"])].sort_values("datetime").reset_index(drop=True)

    assert row["target_value"] == pytest.approx(row["temp"] - row["era5_t2m_c"])
    assert "obs_minus_era5_temp_lag_24" in training_table.columns
    assert station_frame.loc[48, "obs_minus_era5_temp_lag_24"] == pytest.approx(station_frame.loc[24, "obs_minus_era5_temp"])

    experiment_dir = train_v2_experiment(config)
    predictions = read_table(Path(experiment_dir) / "predictions_test.csv")
    metadata = json.loads((Path(experiment_dir) / "future_feature_metadata.json").read_text(encoding="utf-8"))
    leaderboard = read_table(Path(config["artifacts"]["leaderboard_path"]))

    assert predictions["actual"].between(training_table["temp"].min() - 1.0, training_table["temp"].max() + 1.0).all()
    assert not predictions["actual"].round(6).equals(predictions["actual_model_value"].round(6))
    assert metadata["uses_future_nwp_features"] is True
    assert metadata["future_feature_source"] == "era5_reanalysis"
    assert metadata["operational_valid"] is False
    assert metadata["backtest_only"] is True
    assert leaderboard.loc[0, "forecast_track"] == "nwp_assisted"
    assert leaderboard.loc[0, "model_family"] == "ridge"
    assert bool(leaderboard.loc[0, "backtest_only"]) is True
    assert (Path(config["artifacts"]["leaderboard_path"]).with_name("leaderboard_nwp_assisted.csv")).exists()


def test_v3_temp_mos_residual_config_declares_backtest_track() -> None:
    config = load_yaml("configs/v3/experiments/v3_temp_mos_residual_ridge_72to24.yaml")
    metadata = build_future_feature_metadata(config)

    assert config["experiment"]["version"] == "v3"
    assert config["model"]["name"] == "v3_temp_mos_residual_ridge_72to24"
    assert config["data"]["target_transform"] == {"type": "residual_from_feature", "baseline_column": "era5_t2m_c"}
    assert "obs_minus_era5_temp_roll_std_24" in config["data"]["features"]["encoder_continuous"]
    assert metadata["forecast_track"] == "nwp_assisted_mos"
    assert metadata["future_feature_source"] == "era5_reanalysis"
    assert metadata["operational_valid"] is False
    assert metadata["backtest_only"] is True


def test_v3_observation_only_anomaly_config_declares_stationwise_transform() -> None:
    config = load_yaml("configs/v3/experiments/v3_temp_observation_only_anomaly_stationwise_horizonwise_ridge_168to24.yaml")
    metadata = build_future_feature_metadata(config)

    assert config["experiment"]["version"] == "v3"
    assert config["data"]["target_transform"]["type"] == "station_month_hour_anomaly"
    assert config["data"]["scaling"]["mode"] == "stationwise"
    assert config["model"]["type"] == "horizon_wise_ridge"
    assert metadata["forecast_track"] == "observation_only"
    assert metadata["uses_future_weather_features"] is False


def test_v3_temp_observed_oracle_config_is_marked_diagnostic_only() -> None:
    config = load_yaml("configs/v3/experiments/diagnostic/v3_temp_observed_oracle_decoder_feature_72to24.yaml")
    metadata = build_future_feature_metadata(config)

    assert config["experiment"]["version"] == "v3"
    assert config["model"]["type"] == "decoder_feature_baseline"
    assert config["model"]["target_source_features"] == ["target_value"]
    assert config["data"]["features"]["decoder_known"][0] == "target_value"
    assert metadata["forecast_track"] == "observed_target_oracle"
    assert metadata["future_feature_source"] == "observed_target_oracle"
    assert metadata["operational_valid"] is False
    assert metadata["backtest_only"] is True
    assert any("oracle" in warning or "non-operational" in warning for warning in metadata["warnings"])


def test_v3_humidity_observed_oracle_config_is_marked_diagnostic_only() -> None:
    config = load_yaml("configs/v3/experiments/diagnostic/v3_humidity_observed_oracle_decoder_feature_72to24.yaml")
    metadata = build_future_feature_metadata(config)

    assert config["experiment"]["version"] == "v3"
    assert config["data"]["target_name"] == "humidity"
    assert config["model"]["type"] == "decoder_feature_baseline"
    assert config["model"]["target_source_features"] == ["target_value"]
    assert config["data"]["features"]["decoder_known"][0] == "target_value"
    assert config["data"]["postprocess"]["clip_prediction"] == [0, 100]
    assert metadata["forecast_track"] == "observed_target_oracle"
    assert metadata["future_feature_source"] == "observed_target_oracle"
    assert metadata["operational_valid"] is False
    assert metadata["backtest_only"] is True
    assert any("oracle" in warning or "non-operational" in warning for warning in metadata["warnings"])


@pytest.mark.parametrize(
    ("config_path", "experiment_name", "target_transform"),
    [
        (
            "configs/v3/experiments/v3_humidity_direct_lgbm_72to24.yaml",
            "v3_humidity_direct_lgbm_72to24",
            None,
        ),
        (
            "configs/v3/experiments/v3_humidity_dewpoint_lgbm_72to24.yaml",
            "v3_humidity_dewpoint_lgbm_72to24",
            {
                "type": "dew_point",
                "source_column": "obs_dew_point_c",
                "temperature_column": "obs_temp",
            },
        ),
        (
            "configs/v3/experiments/v3_humidity_dewpoint_depression_lgbm_72to24.yaml",
            "v3_humidity_dewpoint_depression_lgbm_72to24",
            {
                "type": "dew_point_depression",
                "source_column": "obs_dew_point_depression",
                "temperature_column": "obs_temp",
            },
        ),
    ],
)
def test_v3_humidity_physical_target_configs_restore_rh(
    config_path: str, experiment_name: str, target_transform: dict[str, str] | None
) -> None:
    config = load_yaml(config_path)
    metadata = build_future_feature_metadata(config)

    assert config["experiment"]["version"] == "v3"
    assert config["experiment"]["name"] == experiment_name
    assert config["model"]["name"] == experiment_name
    assert config["data"]["target_name"] == "humidity"
    assert config["data"].get("target_transform") == target_transform
    assert config["data"]["postprocess"]["clip_prediction"] == [0, 100]
    assert config["artifacts"]["root_dir"] == "data/artifacts/v3_experiments"
    assert metadata["forecast_track"] == "observation_only"
    assert metadata["uses_future_weather_features"] is False
    assert metadata["operational_valid"] is False
    assert metadata["backtest_only"] is False


def test_v2_horizonwise_ridge_writes_horizon_metrics_and_plots(synthetic_v2_project: dict[str, object]) -> None:
    config = {
        **synthetic_v2_project["temp_ridge_config"],
        "experiment": {"name": "synthetic_v2_temp_horizonwise_ridge", "version": "v2"},
        "model": {
            "name": "synthetic_v2_temp_horizonwise_ridge",
            "type": "horizon_wise_ridge",
            "alpha": 1.0,
            "alpha_grid": [0.01, 0.1, 1.0],
            "alpha_selection": {"metric": "val_mse"},
        },
    }
    experiment_dir = train_v2_experiment(config)
    horizon_metrics = read_table(Path(experiment_dir) / "horizon_model_metrics.csv")

    assert set(horizon_metrics["horizon_step"]) == set(range(1, config["data"]["window"]["prediction_length"] + 1))
    assert {"alpha", "val_rmse", "val_mae", "val_bias"}.issubset(horizon_metrics.columns)
    assert (Path(experiment_dir) / "horizon_station_heatmap.png").exists()
    assert (Path(experiment_dir) / "station_rmse_bar.png").exists()
    assert (Path(experiment_dir) / "region_rmse_bar.png").exists()
    assert (Path(experiment_dir) / "metrics_daily_target.csv").exists()
    assert (Path(experiment_dir) / "worst_case_summary.json").exists()


def test_v2_evaluation_uses_target_named_daily_reports_and_humidity_metrics(tmp_path: Path) -> None:
    predictions = pd.DataFrame(
        [
            {
                "station_id": "108",
                "prediction_start": "2024-01-01T00:00:00Z",
                "valid_time": "2024-01-01T01:00:00Z",
                "horizon_step": 1,
                "target_name": "humidity",
                "prediction": 45.0,
                "actual": 35.0,
                "region": "capital",
                "season": "winter",
            },
            {
                "station_id": "108",
                "prediction_start": "2024-01-01T00:00:00Z",
                "valid_time": "2024-01-01T02:00:00Z",
                "horizon_step": 2,
                "target_name": "humidity",
                "prediction": 75.0,
                "actual": 85.0,
                "region": "capital",
                "season": "winter",
            },
            {
                "station_id": "159",
                "prediction_start": "2024-01-02T00:00:00Z",
                "valid_time": "2024-01-02T01:00:00Z",
                "horizon_step": 1,
                "target_name": "humidity",
                "prediction": 38.0,
                "actual": 37.0,
                "region": "coastal",
                "season": "winter",
            },
            {
                "station_id": "159",
                "prediction_start": "2024-01-02T00:00:00Z",
                "valid_time": "2024-01-02T02:00:00Z",
                "horizon_step": 2,
                "target_name": "humidity",
                "prediction": 88.0,
                "actual": 86.0,
                "region": "coastal",
                "season": "winter",
            },
        ]
    )

    evaluate_prediction_frame(predictions, tmp_path)
    daily_metrics = read_table(tmp_path / "metrics_daily_target.csv")
    humidity_metrics = read_table(tmp_path / "metrics_humidity_extremes.csv")

    assert (tmp_path / "daily_target_errors.csv").exists()
    assert (tmp_path / "extreme_target_scatter.png").exists()
    assert not (tmp_path / "daily_temperature_errors.csv").exists()
    assert set(daily_metrics["metric"]) == {"daily_max_rh_error", "daily_min_rh_error", "daily_rh_range_error"}
    assert humidity_metrics.loc[0, "dry_event_hit_rate"] == pytest.approx(0.5)
    assert humidity_metrics.loc[0, "humid_event_hit_rate"] == pytest.approx(0.5)


def test_v2_minimal_artifact_profile_keeps_report_csvs_without_plots(tmp_path: Path) -> None:
    predictions = pd.DataFrame(
        [
            {
                "station_id": "108",
                "prediction_start": "2024-01-01T00:00:00Z",
                "valid_time": "2024-01-01T01:00:00Z",
                "horizon_step": 1,
                "target_name": "humidity",
                "prediction": 45.0,
                "actual": 43.0,
                "region": "capital",
                "season": "winter",
            },
            {
                "station_id": "108",
                "prediction_start": "2024-01-01T00:00:00Z",
                "valid_time": "2024-01-01T02:00:00Z",
                "horizon_step": 2,
                "target_name": "humidity",
                "prediction": 50.0,
                "actual": 52.0,
                "region": "capital",
                "season": "winter",
            },
        ]
    )

    evaluate_prediction_frame(predictions, tmp_path, artifact_config={"profile": "minimal"})

    assert (tmp_path / "metrics_summary.json").exists()
    assert (tmp_path / "metrics_target_name.csv").exists()
    assert (tmp_path / "metrics_target_name_horizon_step.csv").exists()
    assert (tmp_path / "metrics_target_name_station_id.csv").exists()
    assert (tmp_path / "daily_target_errors.csv").exists()
    assert not (tmp_path / "forecast_vs_actual.png").exists()
    assert not (tmp_path / "worst_case_samples.csv").exists()


def test_v2_residual_framework_saves_component_predictions(synthetic_v2_project: dict[str, object]) -> None:
    config = {
        **synthetic_v2_project["temp_ridge_config"],
        "experiment": {"name": "synthetic_v2_temp_residual_ridge", "version": "v2"},
        "model": {
            "name": "synthetic_v2_temp_residual_ridge",
            "type": "residual",
            "baseline": {"name": "baseline_ridge", "type": "ridge", "alpha": 1.0},
            "residual": {
                "name": "residual_horizonwise_ridge",
                "type": "horizon_wise_ridge",
                "alpha": 0.1,
                "alpha_grid": [0.1],
            },
        },
    }
    experiment_dir = train_v2_experiment(config)
    components = read_table(Path(experiment_dir) / "predictions_test_components.csv")

    assert {"baseline", "residual", "final"}.issubset(set(components["component"]))
    assert (Path(experiment_dir) / "horizon_model_metrics.csv").exists()


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


def test_v2_lightgbm_roundtrip_writes_importance_and_raw_metrics(synthetic_v2_project: dict[str, object]) -> None:
    pytest.importorskip("lightgbm")

    config = {
        **synthetic_v2_project["temp_ridge_config"],
        "experiment": {"name": "synthetic_v2_temp_lgbm", "version": "v2"},
        "model": {
            "name": "synthetic_v2_temp_lgbm",
            "type": "lightgbm",
            "params": {
                "n_estimators": 50,
                "learning_rate": 0.05,
                "num_leaves": 15,
            },
        },
    }
    experiment_dir = train_v2_experiment(config)

    metrics_summary = read_table(Path(experiment_dir) / "metrics_target_name.csv")
    assert not metrics_summary.empty
    assert (Path(experiment_dir) / "feature_importance.csv").exists()
    assert (Path(experiment_dir) / "metrics_raw_target_name_horizon_step.csv").exists()


def test_bias_correction_holdout_guard_disables_harmful_correction() -> None:
    config = {
        "data": {"postprocess": {"clip_prediction": None}},
        "evaluation": {
            "bias_correction": {
                "enabled": True,
                "mode": "per_horizon",
                "calibration_fraction": 0.5,
                "apply_when": "improves_on_holdout",
            }
        },
    }
    frame = pd.DataFrame(
        [
            {
                "station_id": "108",
                "prediction_start": "2024-01-01T00:00:00Z",
                "valid_time": "2024-01-01T01:00:00Z",
                "horizon_step": 1,
                "target_name": "temp",
                "prediction": 12.0,
                "actual": 10.0,
            },
            {
                "station_id": "108",
                "prediction_start": "2024-01-02T00:00:00Z",
                "valid_time": "2024-01-02T01:00:00Z",
                "horizon_step": 1,
                "target_name": "temp",
                "prediction": 10.0,
                "actual": 10.0,
            },
        ]
    )

    payload = compute_bias_correction(frame, config)
    corrected = apply_postprocessing(frame, config, payload)

    assert payload["enabled"] is False
    assert payload["selection"]["accepted"] is False
    assert corrected["prediction"].tolist() == [12.0, 10.0]


def test_bias_correction_holdout_guard_keeps_helpful_correction() -> None:
    config = {
        "data": {"postprocess": {"clip_prediction": None}},
        "evaluation": {
            "bias_correction": {
                "enabled": True,
                "mode": "per_horizon",
                "calibration_fraction": 0.5,
                "apply_when": "improves_on_holdout",
            }
        },
    }
    frame = pd.DataFrame(
        [
            {
                "station_id": "108",
                "prediction_start": "2024-01-01T00:00:00Z",
                "valid_time": "2024-01-01T01:00:00Z",
                "horizon_step": 1,
                "target_name": "temp",
                "prediction": 12.0,
                "actual": 10.0,
            },
            {
                "station_id": "108",
                "prediction_start": "2024-01-02T00:00:00Z",
                "valid_time": "2024-01-02T01:00:00Z",
                "horizon_step": 1,
                "target_name": "temp",
                "prediction": 12.0,
                "actual": 10.0,
            },
        ]
    )

    payload = compute_bias_correction(frame, config)
    corrected = apply_postprocessing(frame, config, payload)

    assert payload["enabled"] is True
    assert payload["selection"]["accepted"] is True
    assert corrected["prediction"].tolist() == [10.0, 10.0]


def test_affine_calibration_learns_horizon_slope_and_intercept() -> None:
    config = {
        "data": {"postprocess": {"clip_prediction": None}},
        "evaluation": {
            "bias_correction": {
                "enabled": True,
                "mode": "per_horizon",
                "method": "affine",
                "calibration_fraction": 0.5,
                "apply_when": "improves_on_holdout",
            }
        },
    }
    rows = []
    for day in range(1, 5):
        for prediction in (1.0, 3.0):
            rows.append(
                {
                    "station_id": "108",
                    "prediction_start": f"2024-01-0{day}T00:00:00Z",
                    "valid_time": f"2024-01-0{day}T01:00:00Z",
                    "horizon_step": 1,
                    "target_name": "temp",
                    "prediction": prediction,
                    "actual": 2.0 * prediction + 1.0,
                }
            )
    frame = pd.DataFrame(rows)

    payload = compute_bias_correction(frame, config)
    corrected = apply_postprocessing(frame, config, payload)

    assert payload["enabled"] is True
    assert payload["method"] == "affine"
    assert payload["selection"]["accepted"] is True
    assert corrected["prediction"].round(6).tolist() == corrected["actual"].round(6).tolist()


def test_auto_calibration_selects_best_holdout_candidate() -> None:
    config = {
        "data": {"postprocess": {"clip_prediction": None}},
        "evaluation": {
            "bias_correction": {
                "enabled": True,
                "mode": "auto",
                "method": "auto",
                "candidate_modes": ["global", "per_station_horizon"],
                "candidate_methods": ["mean_bias"],
                "calibration_fraction": 0.5,
                "apply_when": "improves_on_holdout",
            }
        },
    }
    rows = []
    for day in range(1, 5):
        for station_id, prediction, actual in (("108", 12.0, 10.0), ("159", 8.0, 10.0)):
            rows.append(
                {
                    "station_id": station_id,
                    "prediction_start": f"2024-01-0{day}T00:00:00Z",
                    "valid_time": f"2024-01-0{day}T01:00:00Z",
                    "horizon_step": 1,
                    "target_name": "temp",
                    "prediction": prediction,
                    "actual": actual,
                }
            )
    frame = pd.DataFrame(rows)

    payload = compute_bias_correction(frame, config)
    corrected = apply_postprocessing(frame, config, payload)

    assert payload["enabled"] is True
    assert payload["mode"] == "per_station_horizon"
    assert payload["method"] == "mean_bias"
    assert payload["selection"]["selected_mode"] == "per_station_horizon"
    assert corrected["prediction"].round(6).tolist() == corrected["actual"].round(6).tolist()


def test_v3_generic_nwp_bias_features_are_past_only(synthetic_v2_project: dict[str, object], tmp_path: Path) -> None:
    base_config = synthetic_v2_project["temp_ridge_config"]
    config = {
        **base_config,
        "paths": {
            **base_config["paths"],
            "output_training_table": str(tmp_path / "nwp_bias" / "training_table.csv"),
            "output_data_quality": str(tmp_path / "nwp_bias" / "data_quality.json"),
        },
        "data": {
            **base_config["data"],
            "features": {
                **base_config["data"]["features"],
                "encoder_continuous": [
                    *base_config["data"]["features"]["encoder_continuous"],
                    "nwp_temp_c",
                    "obs_minus_nwp_temp_lag_24",
                    "obs_minus_nwp_temp_roll_mean_24",
                    "obs_minus_nwp_temp_delta_24",
                ],
            },
        },
    }
    training_table, _ = build_v2_training_table(config)
    station_frame = training_table.loc[training_table["station_id"].astype(str) == "108"].sort_values("datetime").reset_index(drop=True)

    assert "nwp_temp_c" in training_table.columns
    assert "obs_minus_nwp_temp" in training_table.columns
    assert station_frame.loc[48, "obs_minus_nwp_temp_lag_24"] == pytest.approx(station_frame.loc[24, "obs_minus_nwp_temp"])
    expected_roll = station_frame.loc[24:47, "obs_minus_nwp_temp"].mean()
    assert station_frame.loc[48, "obs_minus_nwp_temp_roll_mean_24"] == pytest.approx(expected_roll)
    assert station_frame.loc[48, "obs_minus_nwp_temp_delta_24"] == pytest.approx(
        station_frame.loc[48, "obs_minus_nwp_temp"] - station_frame.loc[24, "obs_minus_nwp_temp"]
    )


def test_v3_configs_declare_required_mos_tracks_and_alpha_grid() -> None:
    expected_configs = [
        "v3_temp_mos_residual_ridge_72to24",
        "v3_temp_mos_residual_ridge_168to24",
        "v3_temp_mos_horizonwise_residual_ridge_72to24",
        "v3_temp_mos_horizonwise_residual_ridge_168to24",
        "v3_temp_mos_residual_lgbm_72to24",
        "v3_temp_mos_residual_lgbm_168to24",
        "v3_temp_mos_residual_catboost_72to24",
    ]
    for name in expected_configs:
        config = load_yaml(f"configs/v3/experiments/{name}.yaml")
        metadata = build_future_feature_metadata(config)
        assert config["experiment"]["name"] == name
        assert config["data"]["target_transform"]["type"] == "residual_from_feature"
        assert metadata["track"] == "nwp_assisted_mos"
        assert metadata["backtest_only"] is True
        assert "obs_minus_nwp_temp_lag_72" in config["data"]["features"]["encoder_continuous"]
        if config["model"]["type"] in {"ridge", "horizon_wise_ridge"}:
            assert len(config["model"]["alpha_grid"]) == 33
            assert config["model"]["alpha_grid"][0] == pytest.approx(1e-4)
            assert config["model"]["alpha_grid"][-1] == pytest.approx(1e4)


def test_v3_operational_mode_rejects_backtest_only_future_features() -> None:
    from weather_korea_forecast.v2.predict import _validate_future_feature_inference_mode

    class DummyBundle:
        metadata = {
            "future_features": {
                "uses_future_weather_features": True,
                "future_feature_source": "era5_reanalysis",
                "backtest_only": True,
            }
        }

    with pytest.raises(ValueError, match="backtest-only"):
        _validate_future_feature_inference_mode(DummyBundle(), operational_mode=True)
