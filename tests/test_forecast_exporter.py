import json
from pathlib import Path

import pandas as pd
import pytest

from weather_korea_forecast.service.export_forecast import (
    ForecastExportError,
    export_forecast,
    is_blocked_provenance,
)


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    pred = tmp_path / "predictions_inference.csv"
    pd.DataFrame(
        {
            "station_id": ["108", "108"],
            "valid_time": ["2026-01-01T01:00:00Z", "2026-01-01T02:00:00Z"],
            "horizon_step": [1, 2],
            "temperature_c": [1.0, 2.0],
            "humidity_percent": [60, 65],
            "cloud_cover_percent": [10, 90],
            "nwp_precip_probability": [30, 80],
            "nwp_wind_speed": [2.4, 4.0],
            "nwp_wind_direction": [310, 45],
        }
    ).to_csv(pred, index=False)
    meta = tmp_path / "stations.csv"
    pd.DataFrame({"station_id": ["108"], "station_name": ["서울"], "lat": [37.5], "lon": [127.0], "region_class": ["metro"]}).to_csv(meta, index=False)
    return pred, meta


def test_export_forecast_writes_run_and_points_with_warnings(tmp_path: Path):
    pred, meta = _inputs(tmp_path)
    result = export_forecast(
        predictions_path=pred,
        station_metadata_path=meta,
        output_dir=tmp_path / "forecasts",
        forecast_run_id="run1",
        model_version="model-a",
        source="era5_reanalysis_backtest",
        operational_valid=False,
        backtest_only=True,
        temp_rmse=1.064,
    )

    latest = Path(result.latest_dir)
    run = json.loads((latest / "forecast_run.json").read_text(encoding="utf-8"))
    points = json.loads((latest / "forecast_points.json").read_text(encoding="utf-8"))
    assert run["operational_valid"] is False
    assert run["backtest_only"] is True
    assert run["warnings"]
    assert "humidity_beta: humidity output must be displayed as AI MOS beta" in run["warnings"]
    assert "weather_code_rule_based_beta: weather_code is rule-based beta" in run["warnings"]
    assert len(points) == 2
    assert points[0]["station_name"] == "서울"
    assert points[0]["weather_code"] == "clear"
    assert points[0]["confidence"] == "research"
    assert points[0]["temperature_source"] == "ai_mos_model"
    assert points[0]["temperature_status"] == "model"
    assert points[0]["humidity_source"] == "ai_mos_model_beta"
    assert points[0]["humidity_status"] == "beta"
    assert points[0]["precip_probability"] == 30
    assert points[0]["precip_probability_source"] == "nwp_direct"
    assert points[0]["precip_probability_status"] == "direct"
    assert points[0]["wind_direction_deg"] == 310
    assert points[0]["wind_source"] == "nwp_direct"
    assert points[0]["cloud_source"] == "nwp_direct"
    assert points[0]["weather_code_source"] == "rule_based_beta"
    assert "precip_probability" in run["targets"]
    assert "wind" in run["targets"]
    assert "cloud" in run["targets"]
    assert run["beta_targets"]["precip_probability"]["sources"] == ["nwp_direct"]
    assert run["beta_targets"]["precip_probability"]["statuses"] == ["direct"]


def test_export_forecast_preserves_explicit_gfs_direct_temperature_humidity(tmp_path: Path):
    pred = tmp_path / "gfs_direct_predictions.csv"
    pd.DataFrame(
        {
            "station_id": ["108"],
            "valid_time": ["2026-05-31T00:00:00Z"],
            "valid_date": ["2026-05-31"],
            "forecast_day": ["D+1"],
            "horizon_step": [24],
            "temperature_c": [24.1],
            "temp_max_c": [29.5],
            "temp_min_c": [21.2],
            "temperature_source": ["gfs_direct"],
            "temperature_status": ["direct"],
            "temperature_confidence": ["low"],
            "humidity_percent": [58],
            "humidity_source": ["gfs_direct"],
            "humidity_status": ["direct"],
            "humidity_confidence": ["low"],
            "wind_speed_ms": [2.4],
            "wind_source": ["gfs_direct"],
            "cloud_cover_percent": [62],
            "cloud_source": ["gfs_direct"],
            "precip_probability_source": ["unavailable"],
        }
    ).to_csv(pred, index=False)
    meta = tmp_path / "stations.csv"
    pd.DataFrame({"station_id": ["108"], "station_name": ["서울"], "lat": [37.5], "lon": [127.0], "region_class": ["수도권"]}).to_csv(meta, index=False)

    result = export_forecast(
        predictions_path=pred,
        station_metadata_path=meta,
        output_dir=tmp_path / "forecasts",
        forecast_run_id="gfs_direct_run",
        model_version="gfs_direct_site_beta",
        source="gfs_nomads_live",
        operational_valid=False,
        backtest_only=True,
    )

    latest = Path(result.latest_dir)
    run = json.loads((latest / "forecast_run.json").read_text(encoding="utf-8"))
    points = json.loads((latest / "forecast_points.json").read_text(encoding="utf-8"))
    assert run["humidity_beta"] is False
    assert "humidity_beta: humidity output must be displayed as AI MOS beta" not in run["warnings"]
    assert points[0]["temperature_source"] == "gfs_direct"
    assert points[0]["temperature_status"] == "direct"
    assert points[0]["valid_date"] == "2026-05-31"
    assert points[0]["forecast_day"] == "D+1"
    assert points[0]["temp_max_c"] == 29.5
    assert points[0]["temp_min_c"] == 21.2
    assert points[0]["humidity_source"] == "gfs_direct"
    assert points[0]["humidity_status"] == "direct"
    assert points[0]["precip_probability"] is None
    assert points[0]["precip_probability_source"] == "unavailable"


def test_export_forecast_blocks_diagnostic_artifacts(tmp_path: Path):
    pred, meta = _inputs(tmp_path)
    with pytest.raises(ForecastExportError):
        export_forecast(
            predictions_path=pred,
            station_metadata_path=meta,
            output_dir=tmp_path / "forecasts",
            forecast_run_id="synthetic_smoke_run",
            source="prepared_forecast_csv",
            operational_valid=True,
            backtest_only=False,
        )


def test_export_forecast_rejects_untrusted_operational_flag(tmp_path: Path):
    pred, meta = _inputs(tmp_path)
    with pytest.raises(ForecastExportError, match="trusted"):
        export_forecast(
            predictions_path=pred,
            station_metadata_path=meta,
            output_dir=tmp_path / "forecasts",
            forecast_run_id="run2",
            source="prepared_forecast_csv",
            operational_valid=True,
            backtest_only=False,
        )


def test_operational_export_requires_trusted_summary_source(tmp_path: Path):
    pred, meta = _inputs(tmp_path)
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "operational_valid": True,
                "backtest_only": False,
                "forecast_source_schema_valid": True,
                "forecast_archive_adequate": True,
                "forecast_source_path": "data/raw/nwp/archive/prepared_forecast_archive.csv",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ForecastExportError, match="future_feature_source"):
        export_forecast(
            predictions_path=pred,
            station_metadata_path=meta,
            output_dir=tmp_path / "forecasts",
            forecast_run_id="run3",
            source="prepared_forecast_csv",
            operational_valid=True,
            backtest_only=False,
            trusted_experiment_summary_path=summary,
        )


def test_operational_export_requires_trusted_forecast_source_path(tmp_path: Path):
    pred, meta = _inputs(tmp_path)
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "future_feature_source": "prepared_forecast_csv",
                "operational_valid": True,
                "backtest_only": False,
                "forecast_source_schema_valid": True,
                "forecast_archive_adequate": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ForecastExportError, match="forecast_source_path"):
        export_forecast(
            predictions_path=pred,
            station_metadata_path=meta,
            output_dir=tmp_path / "forecasts",
            forecast_run_id="run-path",
            source="prepared_forecast_csv",
            operational_valid=True,
            backtest_only=False,
            trusted_experiment_summary_path=summary,
        )


def test_operational_export_requires_explicit_backtest_false(tmp_path: Path):
    pred, meta = _inputs(tmp_path)
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "future_feature_source": "prepared_forecast_csv",
                "operational_valid": True,
                "backtest_only": None,
                "forecast_source_schema_valid": True,
                "forecast_archive_adequate": True,
                "forecast_source_path": "data/raw/nwp/archive/prepared_forecast_archive.csv",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ForecastExportError, match="backtest_only"):
        export_forecast(
            predictions_path=pred,
            station_metadata_path=meta,
            output_dir=tmp_path / "forecasts",
            forecast_run_id="run4",
            source="prepared_forecast_csv",
            operational_valid=True,
            backtest_only=False,
            trusted_experiment_summary_path=summary,
        )


def test_operational_export_rejects_nested_smoke_provenance(tmp_path: Path):
    pred, meta = _inputs(tmp_path)
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "future_feature_source": "prepared_forecast_csv",
                "operational_valid": True,
                "backtest_only": False,
                "forecast_source_schema_valid": True,
                "forecast_archive_adequate": True,
                "provenance": {"notes": ["real archive", "synthetic smoke bootstrap"]},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ForecastExportError, match="diagnostic/smoke"):
        export_forecast(
            predictions_path=pred,
            station_metadata_path=meta,
            output_dir=tmp_path / "forecasts",
            forecast_run_id="run5",
            source="prepared_forecast_csv",
            operational_valid=True,
            backtest_only=False,
            trusted_experiment_summary_path=summary,
        )


def test_operational_export_allows_report_generated_metadata_key() -> None:
    assert not is_blocked_provenance(
        {
            "future_feature_source": "prepared_forecast_csv",
            "operational_valid": True,
            "backtest_only": False,
            "report_generated": True,
            "forecast_source_path": "data/raw/nwp/archive/prepared_forecast_archive.csv",
        }
    )


def test_operational_export_allows_freeze_policy_blocked_token_declarations() -> None:
    assert not is_blocked_provenance(
        {
            "future_feature_source": "prepared_forecast_csv",
            "operational_valid": True,
            "backtest_only": False,
            "forecast_source_path": "data/raw/nwp/archive/prepared_forecast_archive.csv",
            "freeze_policy": {
                "forbidden_after_freeze": [
                    "new_training_candidates",
                    "test_metric_model_selection",
                    "diagnostic_smoke_oracle_leaderboard_mixing",
                ]
            },
            "benchmark_filter_policy": {
                "excluded_tokens": ["diagnostic", "smoke", "oracle", "synthetic", "generated", "fixture"]
            },
        }
    )


def test_operational_export_still_rejects_blocked_source_or_path_values() -> None:
    assert is_blocked_provenance(
        {
            "future_feature_source": "prepared_forecast_csv",
            "forecast_source_path": "data/generated/smoke/prepared_forecast_archive.csv",
        }
    )


def test_operational_export_rejects_caller_source_conflict(tmp_path: Path):
    pred, meta = _inputs(tmp_path)
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "future_feature_source": "prepared_forecast_csv",
                "operational_valid": True,
                "backtest_only": False,
                "forecast_source_schema_valid": True,
                "forecast_archive_adequate": True,
                "forecast_source_path": "data/raw/nwp/archive/prepared_forecast_archive.csv",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ForecastExportError, match="conflicts"):
        export_forecast(
            predictions_path=pred,
            station_metadata_path=meta,
            output_dir=tmp_path / "forecasts",
            forecast_run_id="run6",
            source="era5_reanalysis_backtest",
            operational_valid=True,
            backtest_only=False,
            trusted_experiment_summary_path=summary,
        )


def test_operational_export_writes_trusted_source_fields(tmp_path: Path):
    pred, meta = _inputs(tmp_path)
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "future_feature_source": "prepared_forecast_csv",
                "operational_valid": True,
                "backtest_only": False,
                "forecast_source_schema_valid": True,
                "forecast_archive_adequate": True,
                "forecast_source_path": "data/raw/nwp/archive/prepared_forecast_archive.csv",
            }
        ),
        encoding="utf-8",
    )

    result = export_forecast(
        predictions_path=pred,
        station_metadata_path=meta,
        output_dir=tmp_path / "forecasts",
        forecast_run_id="run7",
        source="unknown",
        operational_valid=True,
        backtest_only=False,
        trusted_experiment_summary_path=summary,
    )

    run = json.loads((Path(result.latest_dir) / "forecast_run.json").read_text(encoding="utf-8"))
    points = json.loads((Path(result.latest_dir) / "forecast_points.json").read_text(encoding="utf-8"))
    assert run["source"] == "prepared_forecast_csv"
    assert run["operational_valid"] is True
    assert run["backtest_only"] is False
    assert {point["data_source"] for point in points} == {"prepared_forecast_csv"}


def test_export_forecast_does_not_turn_precip_amount_into_probability(tmp_path: Path) -> None:
    pred = tmp_path / "predictions_inference.csv"
    pd.DataFrame(
        {
            "station_id": ["108"],
            "valid_time": ["2026-01-01T01:00:00Z"],
            "horizon_step": [1],
            "temperature_c": [1.0],
            "precip_mm": [12.0],
        }
    ).to_csv(pred, index=False)
    meta = tmp_path / "stations.csv"
    pd.DataFrame({"station_id": ["108"], "station_name": ["서울"]}).to_csv(meta, index=False)

    result = export_forecast(
        predictions_path=pred,
        station_metadata_path=meta,
        output_dir=tmp_path / "forecasts",
        forecast_run_id="run-no-pop",
        model_version="model-a",
        source="gfs_forecast",
        operational_valid=False,
        backtest_only=True,
    )

    points = json.loads((Path(result.latest_dir) / "forecast_points.json").read_text(encoding="utf-8"))
    assert points[0]["precip_mm"] == 12.0
    assert points[0]["precip_probability"] is None
    assert points[0]["precip_probability_source"] == "unavailable"
    assert points[0]["precip_probability_status"] == "unavailable"


def test_export_forecast_treats_nwp_pop_as_percent_not_fraction(tmp_path: Path) -> None:
    pred = tmp_path / "predictions_inference.csv"
    pd.DataFrame(
        {
            "station_id": ["108"],
            "valid_time": ["2026-01-01T01:00:00Z"],
            "horizon_step": [1],
            "temperature_c": [1.0],
            "nwp_precip_probability": [1],
        }
    ).to_csv(pred, index=False)
    meta = tmp_path / "stations.csv"
    pd.DataFrame({"station_id": ["108"], "station_name": ["서울"]}).to_csv(meta, index=False)

    result = export_forecast(
        predictions_path=pred,
        station_metadata_path=meta,
        output_dir=tmp_path / "forecasts",
        forecast_run_id="run-pop-unit",
        model_version="model-a",
        source="kma_forecast",
        operational_valid=False,
        backtest_only=True,
    )

    points = json.loads((Path(result.latest_dir) / "forecast_points.json").read_text(encoding="utf-8"))
    assert points[0]["precip_probability"] == 1
    assert points[0]["precip_probability_source"] == "kma_direct"
    assert points[0]["weather_code"] not in {"rain", "heavy_rain", "snow", "sleet"}


def test_export_forecast_downgrades_untrusted_ai_beta_source_from_input(tmp_path: Path) -> None:
    pred = tmp_path / "predictions_inference.csv"
    pd.DataFrame(
        {
            "station_id": ["108"],
            "valid_time": ["2026-01-01T01:00:00Z"],
            "horizon_step": [1],
            "temperature_c": [1.0],
            "nwp_precip_probability": [1],
            "precip_probability_source": ["ai_beta"],
            "precip_probability_status": ["beta"],
            "precip_probability_confidence": ["medium"],
        }
    ).to_csv(pred, index=False)
    meta = tmp_path / "stations.csv"
    pd.DataFrame({"station_id": ["108"], "station_name": ["서울"]}).to_csv(meta, index=False)

    result = export_forecast(
        predictions_path=pred,
        station_metadata_path=meta,
        output_dir=tmp_path / "forecasts",
        forecast_run_id="run-untrusted-ai",
        model_version="model-a",
        source="kma_forecast",
        operational_valid=False,
        backtest_only=True,
    )

    run = json.loads((Path(result.latest_dir) / "forecast_run.json").read_text(encoding="utf-8"))
    points = json.loads((Path(result.latest_dir) / "forecast_points.json").read_text(encoding="utf-8"))
    assert points[0]["precip_probability"] == 1
    assert points[0]["precip_probability_source"] == "kma_direct"
    assert points[0]["precip_probability_status"] == "direct"
    assert points[0]["precip_probability_confidence"] == "low"
    assert run["beta_targets"]["precip_probability"]["source"] == "kma_direct"


def test_export_forecast_allows_ai_beta_only_with_trusted_validation_descriptor(tmp_path: Path) -> None:
    pred = tmp_path / "predictions_inference.csv"
    pd.DataFrame(
        {
            "station_id": ["108"],
            "valid_time": ["2026-01-01T01:00:00Z"],
            "horizon_step": [1],
            "temperature_c": [1.0],
            "precip_probability": [0.32],
            "precip_probability_source": ["ai_beta"],
            "precip_probability_status": ["beta"],
            "precip_probability_confidence": ["medium"],
        }
    ).to_csv(pred, index=False)
    meta = tmp_path / "stations.csv"
    pd.DataFrame({"station_id": ["108"], "station_name": ["서울"]}).to_csv(meta, index=False)
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "beta_targets": {
                    "precip_probability": {
                        "descriptor": {
                            "source": "ai_beta",
                            "status": "beta",
                            "selection_reason": "validation_brier_improved",
                            "validation_metrics": {"brier": 0.1},
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = export_forecast(
        predictions_path=pred,
        station_metadata_path=meta,
        output_dir=tmp_path / "forecasts",
        forecast_run_id="run-trusted-ai",
        model_version="model-a",
        source="prepared_forecast_csv",
        operational_valid=False,
        backtest_only=True,
        trusted_experiment_summary_path=summary,
    )

    points = json.loads((Path(result.latest_dir) / "forecast_points.json").read_text(encoding="utf-8"))
    assert points[0]["precip_probability"] == 32
    assert points[0]["precip_probability_source"] == "ai_beta"
    assert points[0]["precip_probability_status"] == "beta"


def test_export_forecast_aggregates_mixed_beta_target_sources(tmp_path: Path) -> None:
    pred = tmp_path / "predictions_inference.csv"
    pd.DataFrame(
        {
            "station_id": ["108", "108"],
            "valid_time": ["2026-01-01T01:00:00Z", "2026-01-01T02:00:00Z"],
            "horizon_step": [1, 2],
            "temperature_c": [1.0, 2.0],
            "precip_probability": [40.0, None],
            "precip_probability_source": ["nwp_direct", "unavailable"],
            "precip_probability_status": ["direct", "unavailable"],
        }
    ).to_csv(pred, index=False)
    meta = tmp_path / "stations.csv"
    pd.DataFrame({"station_id": ["108"], "station_name": ["서울"]}).to_csv(meta, index=False)

    result = export_forecast(
        predictions_path=pred,
        station_metadata_path=meta,
        output_dir=tmp_path / "forecasts",
        forecast_run_id="run-mixed",
        model_version="model-a",
        source="prepared_forecast_csv",
        operational_valid=False,
        backtest_only=True,
    )

    run = json.loads((Path(result.latest_dir) / "forecast_run.json").read_text(encoding="utf-8"))
    precip = run["beta_targets"]["precip_probability"]
    assert precip["source"] == "mixed"
    assert precip["status"] == "mixed"
    assert precip["sources"] == ["nwp_direct", "unavailable"]
    assert precip["statuses"] == ["direct", "unavailable"]
    assert precip["counts_by_source"] == {"nwp_direct": 1, "unavailable": 1}
