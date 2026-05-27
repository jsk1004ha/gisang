from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd

from weather_korea_forecast.reporting.collect_experiments import (
    collect_all,
    main_leaderboard_records,
    write_best_models_csv,
    write_failed_csv,
    write_summary_csv,
)


def _write_experiment(root: Path, name: str, *, target: str, track: str, rmse: float, backtest: bool = False, complete: bool = True) -> Path:
    exp = root / name
    exp.mkdir(parents=True)
    (exp / "experiment_summary.json").write_text(
        json.dumps(
            {
                "experiment_name": name,
                "version": "v3" if name.startswith("v3") else "v2",
                "target_name": target,
                "track": track,
                "model_name": name,
                "model_type": "ridge",
                "encoder_length": 72,
                "prediction_length": 24,
                "metrics": {"rmse": rmse, "mae": rmse * 0.8, "bias": 0.1},
                "uses_future_weather_features": backtest,
                "future_feature_source": "era5_reanalysis" if backtest else "none",
                "operational_valid": not backtest,
                "backtest_only": backtest,
                "leakage_risk_note": "ERA5 future features are backtest-only" if backtest else None,
            }
        ),
        encoding="utf-8",
    )
    (exp / "metrics_summary.json").write_text(json.dumps({"rmse": rmse, "mae": rmse * 0.8, "bias": 0.1}), encoding="utf-8")
    if complete:
        pd.DataFrame({"station_id": ["108"], "prediction": [1.0], "actual": [1.1]}).to_csv(exp / "predictions_test.csv", index=False)
    pd.DataFrame({"horizon_step": [1, 24], "rmse": [rmse, rmse + 0.2]}).to_csv(exp / "metrics_target_name_horizon_step.csv", index=False)
    pd.DataFrame({"station_id": ["108"], "rmse": [rmse + 0.1]}).to_csv(exp / "metrics_target_name_station_id.csv", index=False)
    return exp


def _write_experiment_with_summary(root: Path, name: str, summary: dict) -> Path:
    exp = root / name
    exp.mkdir(parents=True)
    payload = {
        "experiment_name": name,
        "version": "v3",
        "target_name": "temp",
        "track": "nwp_assisted_mos",
        "model_name": name,
        "model_type": "ridge",
        "metrics": {"rmse": 1.2, "mae": 0.9, "bias": 0.1},
        "operational_valid": False,
        "backtest_only": True,
    }
    payload.update(summary)
    (exp / "experiment_summary.json").write_text(json.dumps(payload), encoding="utf-8")
    metrics = payload.get("metrics", {})
    (exp / "metrics_summary.json").write_text(json.dumps(metrics), encoding="utf-8")
    pd.DataFrame({"station_id": ["108"], "prediction": [1.0], "actual": [1.1]}).to_csv(exp / "predictions_test.csv", index=False)
    return exp


def test_collect_experiments_calculates_goals_and_warnings(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    _write_experiment(root, "v3_temp_mos", target="temp", track="nwp_assisted_mos", rmse=0.98, backtest=True)
    _write_experiment(root, "v3_humidity", target="humidity", track="humidity", rmse=11.0, complete=False)
    broken = root / "broken"
    broken.mkdir(parents=True)
    (broken / "experiment_summary.json").write_text("{bad json", encoding="utf-8")

    records = collect_all(root)
    by_name = {record.experiment_name: record for record in records}

    assert by_name["v3_temp_mos"].rmse_goal == 1.0
    assert by_name["v3_temp_mos"].rmse_goal_met is True
    assert by_name["v3_temp_mos"].backtest_only is True
    assert any("backtest" in warning for warning in by_name["v3_temp_mos"].warnings)
    assert by_name["v3_humidity"].rmse_goal == 10.0
    assert by_name["v3_humidity"].rmse_goal_met is False
    assert by_name["broken"].complete is False

    summary_path = write_summary_csv(records, tmp_path / "reports" / "experiment_summary.csv")
    best_path = write_best_models_csv(records, tmp_path / "reports" / "best_models.csv")
    failed_path = write_failed_csv(records, tmp_path / "reports" / "failed_or_incomplete_experiments.csv")
    assert summary_path.exists()
    assert best_path.exists()
    failed = pd.read_csv(failed_path)
    assert set(failed["experiment_name"]) >= {"v3_humidity", "broken"}


def test_diagnostic_oracle_rows_excluded_from_main_and_best(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    _write_experiment(root, "v3_temp_real", target="temp", track="nwp_assisted_mos", rmse=1.1, backtest=True)
    _write_experiment_with_summary(
        root,
        "v3_temp_oracle",
        {
            "track": "oracle",
            "model_type": "decoder_feature_baseline",
            "future_feature_source": "observed_target_oracle",
            "metrics": {"rmse": 0.0, "mae": 0.0, "bias": 0.0},
            "backtest_only": True,
            "operational_valid": False,
        },
    )

    records = collect_all(root)
    by_name = {record.experiment_name: record for record in records}
    assert by_name["v3_temp_oracle"].is_diagnostic is True
    assert by_name["v3_temp_oracle"].rmse_goal_met is None
    assert by_name["v3_temp_oracle"].included_in_main_leaderboard is False
    assert by_name["v3_temp_real"].included_in_main_leaderboard is True
    assert [record.experiment_name for record in main_leaderboard_records(records)] == ["v3_temp_real"]

    best_path = write_best_models_csv(records, tmp_path / "reports" / "best_models.csv")
    best = pd.read_csv(best_path)
    assert "v3_temp_oracle" not in set(best["experiment_name"])
    assert best["included_in_main_leaderboard"].all()


def test_alias_artifacts_are_not_main_representatives(tmp_path: Path) -> None:
    root = tmp_path / "artifacts" / "v3_experiments"
    _write_experiment(root, "v3_temp_mos_20260101T000000Z", target="temp", track="nwp_assisted_mos", rmse=1.2, backtest=True)
    _write_experiment(root, "latest", target="temp", track="nwp_assisted_mos", rmse=1.2, backtest=True)

    records = collect_all(root.parent)
    by_dir = {Path(record.artifact_dir).name: record for record in records}
    assert by_dir["latest"].is_alias_artifact is True
    assert by_dir["latest"].is_representative_run is False
    assert by_dir["latest"].included_in_main_leaderboard is False
    main = main_leaderboard_records(records)
    assert [Path(record.artifact_dir).name for record in main] == ["v3_temp_mos_20260101T000000Z"]


def test_minimal_artifact_profile_does_not_warn_for_missing_plots(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    exp = _write_experiment(root, "v3_5_humidity_minimal", target="humidity", track="nwp_assisted_mos", rmse=4.9, backtest=True)
    summary = json.loads((exp / "experiment_summary.json").read_text())
    summary["version"] = "v3.5"
    summary["artifact_profile"] = "minimal"
    (exp / "experiment_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    record = collect_all(root)[0]

    assert record.artifact_profile == "minimal"
    assert not any("png missing" in warning for warning in record.warnings)


def test_collect_experiment_preserves_v4_schema_and_patch_fields(tmp_path: Path) -> None:
    root = tmp_path / "artifacts" / "v4_experiments"
    _write_experiment_with_summary(
        root,
        "v4_temp_patch_lgbm_20260527T000000Z",
        {
            "version": "v4",
            "target_name": "temp",
            "track": "nwp_assisted_mos",
            "metrics": {"rmse": 0.9, "mae": 0.7, "bias": 0.0},
            "future_feature_source": "prepared_forecast_csv",
            "operational_valid": True,
            "backtest_only": False,
            "forecast_schema": {"version": "v4-prepared-forecast-v1", "valid": True},
            "patch_features": {"enabled": True, "patch_size": 5, "feature_set": "summary_v1"},
        },
    )

    record = collect_all(root.parent)[0]

    assert record.v4_stage == "v4_operational_candidate"
    assert record.forecast_schema_version == "v4-prepared-forecast-v1"
    assert record.forecast_schema_valid is True
    assert record.patch_features_enabled is True
    assert record.patch_size == 5
    assert record.patch_feature_set == "summary_v1"

    summary_path = write_summary_csv([record], tmp_path / "reports" / "experiment_summary.csv")
    summary = pd.read_csv(summary_path)
    assert summary.loc[0, "v4_stage"] == "v4_operational_candidate"
    assert bool(summary.loc[0, "forecast_schema_valid"]) is True


def test_collect_experiment_warns_when_v4_schema_invalid(tmp_path: Path) -> None:
    root = tmp_path / "artifacts" / "v4_experiments"
    _write_experiment_with_summary(
        root,
        "v4_temp_invalid_schema",
        {
            "version": "v4",
            "future_feature_source": "prepared_forecast_csv",
            "operational_valid": False,
            "forecast_schema_valid": False,
            "patch_features_enabled": False,
        },
    )

    record = collect_all(root.parent)[0]

    assert record.forecast_schema_valid is False
    assert record.patch_features_enabled is False
    assert "forecast_schema_valid=false" in record.warnings
