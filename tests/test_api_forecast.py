import json
from pathlib import Path

from weather_korea_forecast.api.main import health, latest_forecast, model_status, station_daily, station_forecast, station_hourly, stations


def _forecast_dir(tmp_path: Path) -> Path:
    latest = tmp_path / "latest"
    latest.mkdir()
    (latest / "forecast_run.json").write_text(
        json.dumps({"forecast_run_id": "run1", "model_version": "m", "operational_valid": False, "backtest_only": True, "created_at": "2026-01-01T00:00:00Z", "warnings": []}),
        encoding="utf-8",
    )
    (latest / "forecast_points.json").write_text(
        json.dumps(
            [
                {"station_id": "108", "station_name": "서울", "lat": 37.5, "lon": 127, "region_class": "metro", "valid_time": "2026-01-01T01:00:00Z", "temperature_c": 1.0, "humidity_percent": 60},
                {"station_id": "108", "station_name": "서울", "lat": 37.5, "lon": 127, "region_class": "metro", "valid_time": "2026-01-01T02:00:00Z", "temperature_c": 3.0, "humidity_percent": 70},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return latest


def test_api_helpers_return_forecast_and_warnings(tmp_path: Path):
    forecast_dir = _forecast_dir(tmp_path)
    assert health()["status"] == "ok"
    assert stations(forecast_dir)[0]["station_id"] == "108"
    latest = latest_forecast(forecast_dir)
    assert latest["warnings"]
    assert latest["humidity_beta"] is True
    assert len(station_forecast("108", forecast_dir)["points"]) == 2
    assert len(station_hourly("108", forecast_dir)) == 2
    daily = station_daily("108", forecast_dir)
    assert daily["daily"][0]["temp_max_c"] == 3.0


def test_model_status_reads_evaluation_summary(tmp_path: Path):
    forecast_dir = _forecast_dir(tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "experiment_summary.csv").write_text("experiment_name,target_name,rmse,included_in_main_leaderboard\na,temp,1.2,True\nb,humidity,14.0,True\n", encoding="utf-8")
    status = model_status(forecast_dir, reports)
    assert status["operational_valid"] is False
    assert status["temp_rmse"] == 1.2
    assert status["humidity_rmse"] == 14.0
