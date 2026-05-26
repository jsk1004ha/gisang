from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from weather_korea_forecast.dashboard.status import load_dashboard_state, render_dashboard_html
from weather_korea_forecast.v2.ensemble import build_prediction_ensemble


def _write_component_experiment(path: Path, name: str, predictions: list[float]) -> None:
    path.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "station_id": ["108", "108", "159", "159"],
            "prediction_start": ["2025-02-01T00:00:00Z"] * 4,
            "valid_time": [
                "2025-02-01T01:00:00Z",
                "2025-02-01T02:00:00Z",
                "2025-02-01T01:00:00Z",
                "2025-02-01T02:00:00Z",
            ],
            "horizon_step": [1, 2, 1, 2],
            "target_name": ["humidity"] * 4,
            "prediction": predictions,
            "actual": [40.0, 60.0, 50.0, 70.0],
            "region": ["capital", "capital", "coastal", "coastal"],
            "season": ["winter"] * 4,
        }
    )
    frame.to_csv(path / "predictions_test.csv", index=False)
    (path / "experiment_summary.json").write_text(
        json.dumps(
            {
                "experiment_name": name,
                "target_name": "humidity",
                "encoder_length": 72,
                "prediction_length": 24,
                "train_end": "2024-12-31T23:00:00Z",
                "val_end": "2025-01-31T23:00:00Z",
                "test_end": "2025-02-28T23:00:00Z",
            }
        ),
        encoding="utf-8",
    )


def test_prediction_ensemble_writes_metrics_and_leaderboard(tmp_path: Path) -> None:
    component_a = tmp_path / "component_a"
    component_b = tmp_path / "component_b"
    _write_component_experiment(component_a, "component_a", [42.0, 58.0, 55.0, 75.0])
    _write_component_experiment(component_b, "component_b", [38.0, 62.0, 45.0, 65.0])

    output_root = tmp_path / "artifacts"
    ensemble_dir = build_prediction_ensemble(
        experiment_dirs=[component_a, component_b],
        output_root=output_root,
        name="v3_humidity_mean_ensemble_smoke",
        method="mean",
        clip_min=0.0,
        clip_max=100.0,
        leaderboard_path=output_root / "leaderboard.csv",
    )

    predictions = pd.read_csv(ensemble_dir / "predictions_test.csv")
    assert predictions["prediction"].tolist() == pytest.approx([40.0, 60.0, 50.0, 70.0])
    metrics = json.loads((ensemble_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    assert metrics["rmse"] == pytest.approx(0.0)
    summary = json.loads((ensemble_dir / "experiment_summary.json").read_text(encoding="utf-8"))
    assert summary["model_family"] == "ensemble"
    leaderboard = pd.read_csv(output_root / "leaderboard.csv")
    assert leaderboard.loc[0, "model_family"] == "ensemble"
    assert (output_root / "leaderboard_humidity.csv").exists()
    assert (ensemble_dir / "predictions_test_components.csv").exists()


def test_dashboard_state_and_html_render_artifact_status(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    pd.DataFrame(
        [
            {
                "experiment_name": "v3_temp_mos_residual_ridge",
                "target_name": "temp",
                "model_type": "ridge",
                "model_family": "ridge",
                "forecast_track": "nwp_assisted_mos",
                "uses_future_weather_features": True,
                "operational_valid": False,
                "rmse": 1.06,
                "mae": 0.8,
                "bias": 0.06,
                "experiment_dir": str(artifact_root / "temp_run"),
            },
            {
                "experiment_name": "v3_humidity_mean_ensemble",
                "target_name": "humidity",
                "model_type": "ensemble_mean",
                "model_family": "ensemble",
                "forecast_track": "observation_only",
                "uses_future_weather_features": False,
                "operational_valid": False,
                "rmse": 19.9,
                "mae": 17.0,
                "bias": 9.2,
                "experiment_dir": str(artifact_root / "humidity_run"),
            },
        ]
    ).to_csv(artifact_root / "leaderboard.csv", index=False)
    (artifact_root / "v3_humidity_comparison.md").write_text("# comparison\n", encoding="utf-8")

    state = load_dashboard_state(artifact_root)
    assert state["counts"] == {"experiments": 2, "targets": 2, "tracks": 2}
    assert {row["group"] for row in state["best_by_target"]} == {"humidity", "temp"}
    assert state["comparisons"][0]["name"] == "v3_humidity_comparison.md"

    html = render_dashboard_html(state)
    assert "Weather Korea Forecast Dashboard" in html
    assert "v3_humidity_mean_ensemble" in html
    assert "/api/status" in html
