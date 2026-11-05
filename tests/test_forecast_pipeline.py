import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

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


def test_pipeline_operational_mode_requires_inference_predictions(tmp_path: Path):
    forecast, exp, meta, _pred = _files(tmp_path, operational_valid=True, backtest_only=False)
    output = tmp_path / "out"
    with pytest.raises(ForecastExportError, match="requires --predictions"):
        run_forecast_pipeline(forecast_csv=forecast, experiment_dir=exp, station_metadata=meta, output_dir=output, operational=True)
    assert not output.exists()


@pytest.mark.parametrize("explicit_source, expected_source", [(None, "nwp_direct"), ("gfs_direct", "gfs_direct"), ("kma_direct", "kma_direct"), ("ai_mos_model", "nwp_direct"), ("unavailable", "unavailable")])
def test_pipeline_raw_research_export_cannot_inherit_model_claims(tmp_path: Path, explicit_source, expected_source):
    forecast, exp, meta, _pred = _files(tmp_path, operational_valid=True, backtest_only=False)
    summary_path = exp / "experiment_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["notes"] = "provider coverage is limited"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    frame = pd.read_csv(forecast)
    frame["nwp_t2m"] = 17.25
    frame["humidity_percent"] = 62
    if explicit_source:
        frame["temperature_source"] = explicit_source
        frame["humidity_source"] = explicit_source
    frame.to_csv(forecast, index=False)

    result = run_forecast_pipeline(forecast_csv=forecast, experiment_dir=exp, station_metadata=meta, output_dir=tmp_path / "out", research=True)
    run = json.loads((Path(result["run_dir"]) / "forecast_run.json").read_text(encoding="utf-8"))
    point = json.loads((Path(result["run_dir"]) / "forecast_points.json").read_text(encoding="utf-8"))[0]

    assert run["model_version"] == "direct_forecast_csv"
    assert run["operational_valid"] is False
    assert run["backtest_only"] is True
    assert run["humidity_beta"] is False
    assert "provider coverage is limited" in run["notes"]
    assert point["temperature_c"] == 17.25
    assert point["humidity_percent"] == 62
    assert point["temperature_source"] == expected_source
    assert point["humidity_source"] == expected_source
    expected_status = "unavailable" if expected_source == "unavailable" else "direct"
    assert point["temperature_status"] == point["humidity_status"] == expected_status
    assert point["confidence"] == "research"
    assert point["temperature_confidence"] == ("unavailable" if expected_source == "unavailable" else "low")


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
    predicted = pd.read_csv(pred)
    predicted["temperature_c"] = 3.5
    predicted["temperature_source"] = "gfs_direct"
    predicted["temperature_status"] = "direct"
    predicted.to_csv(pred, index=False)

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
    assert points[0]["temperature_c"] == 3.5
    assert points[0]["temperature_source"] == "gfs_direct"
    assert points[0]["temperature_status"] == "direct"
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


@pytest.mark.parametrize("with_predictions", [False, True])
def test_concurrent_pipelines_keep_their_own_input_values(tmp_path: Path, monkeypatch, with_predictions):
    pipeline = importlib.import_module("weather_korea_forecast.service.run_forecast_pipeline")
    jobs = []
    for index in (1, 2):
        inputs = tmp_path / f"input_{index}"
        inputs.mkdir()
        forecast, exp, meta, pred = _files(inputs, operational_valid=True, backtest_only=False)
        frame = pd.read_csv(forecast)
        frame["nwp_t2m"] = 10 * index
        frame["nwp_wind_speed"] = index
        frame.to_csv(forecast, index=False)
        frame = pd.read_csv(pred)
        frame["temperature_c"] = 10 * index + 1
        frame.to_csv(pred, index=False)
        jobs.append(dict(forecast_csv=forecast, experiment_dir=exp, station_metadata=meta,
                         output_dir=tmp_path / "out", predictions=pred if with_predictions else None,
                         operational=with_predictions, research=not with_predictions))

    # Both invocations finish each intermediate write before either is allowed
    # to read it. Shared filenames would deterministically mix the two inputs.
    direct_ready, predictions_ready = Barrier(2), Barrier(2)
    original_direct = pipeline._predictions_from_forecast_csv
    original_predictions = pipeline._prediction_path_for_export

    def synchronize_direct(*args, **kwargs):
        path = original_direct(*args, **kwargs)
        direct_ready.wait(timeout=5)
        return path

    def synchronize_predictions(*args, **kwargs):
        path = original_predictions(*args, **kwargs)
        predictions_ready.wait(timeout=5)
        return path

    monkeypatch.setattr(pipeline, "_predictions_from_forecast_csv", synchronize_direct)
    monkeypatch.setattr(pipeline, "_prediction_path_for_export", synchronize_predictions)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_forecast_pipeline, **job) for job in jobs]
        results = [future.result(timeout=10) for future in futures]

    assert results[0]["forecast_run_id"] != results[1]["forecast_run_id"]
    for index, result in enumerate(results, 1):
        point = json.loads((Path(result["run_dir"]) / "forecast_points.json").read_text())[0]
        assert point["forecast_run_id"] == result["forecast_run_id"]
        assert point["temperature_c"] == 10 * index + int(with_predictions)
        assert point["wind_speed_ms"] == index
    assert not list((tmp_path / "out" / "pipeline_work").iterdir())
