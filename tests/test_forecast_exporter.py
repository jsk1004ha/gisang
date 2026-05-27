import json
from pathlib import Path

import pandas as pd
import pytest

from weather_korea_forecast.service.export_forecast import ForecastExportError, export_forecast


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
    assert len(points) == 2
    assert points[0]["station_name"] == "서울"
    assert points[0]["weather_code"] == "clear"
    assert points[0]["confidence"] == "research"


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
