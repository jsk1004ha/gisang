import json
from pathlib import Path

import pandas as pd
import pytest

from weather_korea_forecast.service.export_forecast import ForecastExportError
from weather_korea_forecast.service.run_forecast_pipeline import run_forecast_pipeline


def _files(tmp_path: Path, *, operational_valid: bool = False, backtest_only: bool = True) -> tuple[Path, Path, Path, Path]:
    forecast = tmp_path / "forecast.csv"
    pd.DataFrame(
        {
            "station_id": ["108"],
            "valid_time": ["2026-01-01T01:00:00Z"],
            "horizon_step": [1],
            "nwp_t2m": [1.0],
            "nwp_precip_probability": [40],
            "nwp_wind_speed": [3.2],
            "nwp_wind_direction": [280],
            "nwp_cloud_cover": [65],
        }
    ).to_csv(forecast, index=False)
    exp = tmp_path / "exp"
    exp.mkdir()
    summary = {
        "experiment_name": "model",
        "future_feature_source": "prepared_forecast_csv",
        "operational_valid": operational_valid,
        "backtest_only": backtest_only,
        "forecast_source_schema_valid": operational_valid,
        "forecast_archive_adequate": operational_valid,
        "forecast_source_path": str(forecast) if operational_valid else None,
        "metrics": {"rmse": 1.2},
    }
    (exp / "experiment_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    meta = tmp_path / "stations.csv"
    pd.DataFrame({"station_id": ["108"], "station_name": ["서울"]}).to_csv(meta, index=False)
    pred = tmp_path / "pred.csv"
    pd.DataFrame({"station_id": ["108"], "valid_time": ["2026-01-01T01:00:00Z"], "horizon_step": [1], "temperature_c": [1.0]}).to_csv(pred, index=False)
    return forecast, exp, meta, pred


def test_pipeline_research_mode_exports_with_warning(tmp_path: Path):
    forecast, exp, meta, pred = _files(tmp_path)
    result = run_forecast_pipeline(forecast_csv=forecast, experiment_dir=exp, station_metadata=meta, output_dir=tmp_path / "out", predictions=pred, research=True)
    assert result["status"] == "ok"
    assert result["warnings"]


def test_pipeline_operational_mode_blocks_backtest_model(tmp_path: Path):
    forecast, exp, meta, pred = _files(tmp_path, operational_valid=False, backtest_only=True)
    with pytest.raises(ForecastExportError):
        run_forecast_pipeline(forecast_csv=forecast, experiment_dir=exp, station_metadata=meta, output_dir=tmp_path / "out", predictions=pred, operational=True)


def test_pipeline_operational_mode_allows_operational_model(tmp_path: Path):
    forecast, exp, meta, pred = _files(tmp_path, operational_valid=True, backtest_only=False)
    result = run_forecast_pipeline(forecast_csv=forecast, experiment_dir=exp, station_metadata=meta, output_dir=tmp_path / "out", predictions=pred, operational=True)
    assert result["status"] == "ok"


def test_pipeline_forecast_csv_maps_beta_direct_fields(tmp_path: Path):
    forecast, exp, meta, _pred = _files(tmp_path)

    result = run_forecast_pipeline(
        forecast_csv=forecast,
        experiment_dir=exp,
        station_metadata=meta,
        output_dir=tmp_path / "out",
        predictions=None,
        research=True,
    )

    points = json.loads((tmp_path / "out" / "latest" / "forecast_points.json").read_text(encoding="utf-8"))
    assert result["status"] == "ok"
    assert points[0]["precip_probability"] == 40
    assert points[0]["precip_probability_source"] == "nwp_direct"
    assert points[0]["wind_direction_deg"] == 280
    assert points[0]["wind_source"] == "nwp_direct"
    assert points[0]["cloud_source"] == "nwp_direct"


def test_pipeline_merges_beta_direct_fields_into_supplied_predictions(tmp_path: Path):
    forecast, exp, meta, pred = _files(tmp_path)

    result = run_forecast_pipeline(
        forecast_csv=forecast,
        experiment_dir=exp,
        station_metadata=meta,
        output_dir=tmp_path / "out",
        predictions=pred,
        research=True,
    )

    points = json.loads((tmp_path / "out" / "latest" / "forecast_points.json").read_text(encoding="utf-8"))
    assert result["status"] == "ok"
    assert points[0]["temperature_c"] == 1.0
    assert points[0]["precip_probability"] == 40
    assert points[0]["wind_direction_deg"] == 280
    assert points[0]["cloud_cover_percent"] == 65


def test_pipeline_merge_requires_stable_point_keys(tmp_path: Path):
    forecast, exp, meta, pred = _files(tmp_path)
    pd.DataFrame({"station_id": ["108"], "temperature_c": [1.0]}).to_csv(pred, index=False)

    with pytest.raises(ForecastExportError, match="stable keys"):
        run_forecast_pipeline(
            forecast_csv=forecast,
            experiment_dir=exp,
            station_metadata=meta,
            output_dir=tmp_path / "out",
            predictions=pred,
            research=True,
        )


def test_pipeline_merge_rejects_duplicate_direct_point_keys(tmp_path: Path):
    forecast, exp, meta, pred = _files(tmp_path)
    frame = pd.read_csv(forecast)
    pd.concat([frame, frame], ignore_index=True).to_csv(forecast, index=False)

    with pytest.raises(ForecastExportError, match="duplicate station_id/valid_time/horizon_step"):
        run_forecast_pipeline(
            forecast_csv=forecast,
            experiment_dir=exp,
            station_metadata=meta,
            output_dir=tmp_path / "out",
            predictions=pred,
            research=True,
        )


def test_pipeline_operational_mode_blocks_nested_smoke_provenance(tmp_path: Path):
    forecast, exp, meta, pred = _files(tmp_path, operational_valid=True, backtest_only=False)
    summary = json.loads((exp / "experiment_summary.json").read_text(encoding="utf-8"))
    summary["metadata"] = {"provenance": {"note": "synthetic smoke archive"}}
    (exp / "experiment_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ForecastExportError, match="synthetic/smoke"):
        run_forecast_pipeline(forecast_csv=forecast, experiment_dir=exp, station_metadata=meta, output_dir=tmp_path / "out", predictions=pred, operational=True)
