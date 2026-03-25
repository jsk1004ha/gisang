from __future__ import annotations

from weather_korea_forecast.evaluation.evaluate import evaluate_experiment
from weather_korea_forecast.inference.predict import generate_forecast
from weather_korea_forecast.training.train import train_experiment


def test_train_evaluate_and_predict_roundtrip(synthetic_project: dict) -> None:
    experiment_dir = train_experiment(
        synthetic_project["data_config"],
        synthetic_project["model_config"],
        synthetic_project["train_config"],
    )
    evaluation = evaluate_experiment(experiment_dir)
    forecast = generate_forecast(
        experiment_dir=experiment_dir,
        station_id="SEOUL",
        forecast_init_time="2024-01-05T20:00:00Z",
    )

    assert "rmse" in evaluation["metrics"]
    assert len(forecast) == synthetic_project["data_config"]["window"]["prediction_length"]
    assert forecast["station_id"].nunique() == 1


def test_multitarget_roundtrip_and_auto_backend_resolution(synthetic_project: dict) -> None:
    experiment_dir = train_experiment(
        synthetic_project["multi_target_data_config"],
        synthetic_project["auto_tft_config"],
        synthetic_project["train_config"],
    )
    evaluation = evaluate_experiment(experiment_dir)
    forecast = generate_forecast(
        experiment_dir=experiment_dir,
        station_id="SEOUL",
        forecast_init_time="2024-01-05T20:00:00Z",
    )

    assert evaluation["metrics"]["aggregation"] in {"global", "macro_by_target"}
    assert sorted(forecast["target_name"].unique().tolist()) == ["humidity", "temp"]
    assert len(forecast) == synthetic_project["multi_target_data_config"]["window"]["prediction_length"] * 2


def test_ridge_baseline_roundtrip(synthetic_project: dict) -> None:
    experiment_dir = train_experiment(
        synthetic_project["multi_target_data_config"],
        synthetic_project["ridge_config"],
        synthetic_project["train_config"],
    )
    forecast = generate_forecast(
        experiment_dir=experiment_dir,
        station_id="SEOUL",
        forecast_init_time="2024-01-05T20:00:00Z",
    )

    assert sorted(forecast["target_name"].unique().tolist()) == ["humidity", "temp"]
    assert len(forecast) == synthetic_project["multi_target_data_config"]["window"]["prediction_length"] * 2
