from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from weather_korea_forecast.v2.train import apply_forecast_archive_overrides


def _config() -> dict:
    return {
        "experiment": {"name": "g022_temp_operational_residual_ridge_72to24", "version": "g022"},
        "paths": {"prepared_forecast_archive": "old.csv"},
        "data": {
            "features": {"decoder_known": ["nwp_t2m"]},
            "future_features": {
                "source": "prepared_forecast_csv",
                "operational_valid": True,
                "schema": {"valid": True},
                "archive": {"adequate": True},
            },
        },
    }


def test_operational_training_gate_blocks_inadequate_archive_quality_report(tmp_path: Path) -> None:
    report = tmp_path / "archive_quality_report.json"
    report.write_text(json.dumps({"forecast_archive_adequate": False, "blocking_reasons": ["station_count >= 20"]}), encoding="utf-8")

    with pytest.raises(ValueError, match="station_count >= 20"):
        apply_forecast_archive_overrides(_config(), future_weather_csv=tmp_path / "prepared.csv", archive_quality_report=report)


def test_operational_training_gate_applies_future_weather_csv_when_quality_report_passes(tmp_path: Path) -> None:
    archive, digest = _write_bound_archive(tmp_path)
    report = tmp_path / "archive_quality_report.json"
    report.write_text(
        json.dumps({"forecast_archive_adequate": True, "archive_content_sha256": digest, "expected_columns": ["nwp_t2m"], "expected_column_missing_rates": {"nwp_t2m": 0.0}, "station_count": 20, "forecast_cycle_count": 30, "row_count": 14400}),
        encoding="utf-8",
    )

    config = apply_forecast_archive_overrides(_config(), future_weather_csv=archive, archive_quality_report=report)

    assert config["paths"]["prepared_forecast_archive"] == str(archive)
    assert config["paths"]["future_weather_csv"] == str(archive)
    future = config["data"]["future_features"]
    assert future["forecast_archive_adequate"] is True
    assert future["forecast_source_path"] == str(archive)
    assert future["archive"]["station_count"] == 20


def _write_bound_archive(tmp_path: Path, value: float = 1.0) -> tuple[Path, str]:
    archive = tmp_path / "prepared.csv"
    pd.DataFrame(
        {
            "station_id": ["108"],
            "forecast_init_time": ["2026-01-01T00:00:00Z"],
            "valid_time": ["2026-01-01T01:00:00Z"],
            "horizon_step": [1],
            "nwp_t2m": [value],
        }
    ).to_csv(archive, index=False)
    return archive, hashlib.sha256(archive.read_bytes()).hexdigest()


def test_operational_training_gate_rejects_string_false_and_stale_report(tmp_path: Path) -> None:
    archive, digest = _write_bound_archive(tmp_path)
    report = tmp_path / "archive_quality_report.json"
    report.write_text(json.dumps({"forecast_archive_adequate": "false", "archive_content_sha256": digest}), encoding="utf-8")

    with pytest.raises(ValueError, match="strict JSON boolean true"):
        apply_forecast_archive_overrides(_config(), future_weather_csv=archive, archive_quality_report=report)

    stale = tmp_path / "stale_report.json"
    stale.write_text(json.dumps({"forecast_archive_adequate": True, "archive_content_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        apply_forecast_archive_overrides(_config(), future_weather_csv=archive, archive_quality_report=stale)


def test_operational_training_gate_rejects_missing_configured_weather_columns(tmp_path: Path) -> None:
    archive, digest = _write_bound_archive(tmp_path)
    report = tmp_path / "archive_quality_report.json"
    report.write_text(json.dumps({"forecast_archive_adequate": True, "archive_content_sha256": digest, "expected_columns": ["nwp_t2m", "nwp_dew_point"], "expected_column_missing_rates": {"nwp_t2m": 0.0, "nwp_dew_point": 0.0}}), encoding="utf-8")
    config = _config()
    config["data"]["features"]["decoder_known"] = ["nwp_t2m", "nwp_dew_point"]

    with pytest.raises(ValueError, match="missing configured forecast columns"):
        apply_forecast_archive_overrides(config, future_weather_csv=archive, archive_quality_report=report)


def test_operational_training_gate_requires_report_expected_columns_cover_config(tmp_path: Path) -> None:
    archive, digest = _write_bound_archive(tmp_path)
    report = tmp_path / "archive_quality_report.json"
    report.write_text(
        json.dumps({"forecast_archive_adequate": True, "archive_content_sha256": digest, "expected_columns": [], "expected_column_missing_rates": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected_columns must include configured forecast columns"):
        apply_forecast_archive_overrides(_config(), future_weather_csv=archive, archive_quality_report=report)


def test_operational_training_gate_rejects_report_expected_column_missing_rate(tmp_path: Path) -> None:
    archive, digest = _write_bound_archive(tmp_path)
    report = tmp_path / "archive_quality_report.json"
    report.write_text(
        json.dumps(
            {
                "forecast_archive_adequate": True,
                "archive_content_sha256": digest,
                "expected_columns": ["nwp_t2m"],
                "expected_column_missing_rates": {"nwp_t2m": 1.0},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected_column_missing_rates"):
        apply_forecast_archive_overrides(_config(), future_weather_csv=archive, archive_quality_report=report)


def test_train_v2_experiment_enforces_g022_operational_gate_for_library_call() -> None:
    with pytest.raises(ValueError, match="archive quality report"):
        from weather_korea_forecast.v2.train import train_v2_experiment

        train_v2_experiment(_config())


def test_operational_training_gate_binds_report_to_configured_archive_path(tmp_path: Path) -> None:
    archive, _digest = _write_bound_archive(tmp_path)
    stale = tmp_path / "stale_report.json"
    stale.write_text(
        json.dumps(
            {
                "forecast_archive_adequate": True,
                "archive_content_sha256": "0" * 64,
                "expected_columns": ["nwp_t2m"],
                "expected_column_missing_rates": {"nwp_t2m": 0.0},
            }
        ),
        encoding="utf-8",
    )
    config = _config()
    config["paths"]["prepared_forecast_archive"] = str(archive)

    with pytest.raises(ValueError, match="does not match"):
        apply_forecast_archive_overrides(config, archive_quality_report=stale)


def test_operational_training_gate_report_only_path_validates_feature_contract(tmp_path: Path) -> None:
    archive, digest = _write_bound_archive(tmp_path)
    report = tmp_path / "archive_quality_report.json"
    report.write_text(json.dumps({"forecast_archive_adequate": True, "archive_content_sha256": digest}), encoding="utf-8")
    config = _config()
    config["paths"]["prepared_forecast_archive"] = str(archive)

    with pytest.raises(ValueError, match="expected_columns must include configured forecast columns"):
        apply_forecast_archive_overrides(config, archive_quality_report=report)
