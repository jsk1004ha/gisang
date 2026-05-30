import json
from pathlib import Path

from weather_korea_forecast.api import main as api_main
from weather_korea_forecast.api.main import health, latest_forecast, model_status, station_daily, station_forecast, station_hourly, stations


def _forecast_dir(tmp_path: Path) -> Path:
    latest = tmp_path / "latest"
    latest.mkdir()
    (latest / "forecast_run.json").write_text(
        json.dumps(
            {
                "forecast_run_id": "run1",
                "model_version": "m",
                "operational_valid": False,
                "backtest_only": True,
                "created_at": "2026-01-01T00:00:00Z",
                "warnings": [],
                "beta_targets": {
                    "precip_probability": {"source": "nwp_direct", "status": "direct", "confidence": "low", "sources": ["nwp_direct"], "statuses": ["direct"]},
                    "wind": {"source": "nwp_direct", "status": "direct", "confidence": "low", "sources": ["nwp_direct"], "statuses": ["direct"]},
                },
            }
        ),
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
    assert latest["beta_targets"]["precip_probability"]["source"] == "nwp_direct"
    assert len(station_forecast("108", forecast_dir)["points"]) == 2
    assert len(station_hourly("108", forecast_dir)) == 2
    daily = station_daily("108", forecast_dir)
    assert daily["daily"][0]["temp_max_c"] == 3.0


def test_api_humidity_beta_respects_explicit_direct_source(tmp_path: Path):
    latest = tmp_path / "latest"
    latest.mkdir()
    (latest / "forecast_run.json").write_text(
        json.dumps(
            {
                "forecast_run_id": "run-direct",
                "model_version": "gfs_direct_site_beta",
                "operational_valid": False,
                "backtest_only": True,
                "humidity_beta": False,
            }
        ),
        encoding="utf-8",
    )
    (latest / "forecast_points.json").write_text(
        json.dumps(
            [
                {
                    "station_id": "108",
                    "station_name": "서울",
                    "valid_time": "2026-05-31T00:00:00Z",
                    "temperature_c": 24.1,
                    "temperature_source": "gfs_direct",
                    "humidity_percent": 58,
                    "humidity_source": "gfs_direct",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    latest_payload = latest_forecast(latest)
    station_payload = station_forecast("108", latest)

    assert latest_payload["humidity_beta"] is False
    assert station_payload["humidity_beta"] is False


def test_model_status_reads_evaluation_summary(tmp_path: Path):
    forecast_dir = _forecast_dir(tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "experiment_summary.csv").write_text("experiment_name,target_name,rmse,included_in_main_leaderboard\na,temp,1.2,True\nb,humidity,14.0,True\n", encoding="utf-8")
    status = model_status(forecast_dir, reports)
    assert status["operational_valid"] is False
    assert status["temp_rmse"] == 1.2
    assert status["humidity_rmse"] == 14.0
    assert status["beta_targets"]["wind"]["status"] == "direct"


def test_model_status_does_not_fallback_to_global_manifest_for_local_forecast(tmp_path: Path, monkeypatch):
    forecast_dir = _forecast_dir(tmp_path)
    global_manifest = tmp_path / "global_production_model_manifest.json"
    global_manifest.write_text(
        json.dumps(
            {
                "operational_valid": True,
                "temperature_status": "PASS",
                "humidity_status": "PASS",
                "humidity_bias_status": "PASS",
                "site_readiness": "PASS",
                "v4c_gate_status": "PASS",
                "benchmark_reliability": "strong",
                "temperature_rmse": 0.1,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api_main, "DEFAULT_PRODUCTION_MANIFEST", global_manifest)

    status = api_main.model_status(forecast_dir=forecast_dir, reports_dir=api_main.DEFAULT_REPORTS_DIR)

    assert status["operational_valid"] is False
    assert status["temperature_rmse"] != 0.1


def test_model_status_prefers_frozen_production_manifest_metrics_but_rechecks_gates(tmp_path: Path):
    forecast_dir = _forecast_dir(tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    manifest = reports / "production_model_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "freeze_iteration": "G030",
                "temperature_model": "ensemble_stationwise_inverse_rmse",
                "temperature_rmse": 1.694,
                "temperature_status": "FAIL",
                "humidity_model": "operational_residual_lgbm_humidity",
                "humidity_rmse": 10.396,
                "humidity_bias": 0.596,
                "humidity_status": "NEAR_PASS",
                "humidity_bias_status": "PASS",
                "weather_code_model": "rule_based_beta",
                "benchmark_reliability": "strong",
                "site_readiness": "WARN",
                "v4c_gate_status": "FAIL",
                "operational_valid": True,
                "forecast_schema_version": "forecast_points.v2-beta-sources",
                "model_improvement_frozen": True,
                "site_caveats": [
                    "temperature_accuracy_caveat",
                    "humidity_beta",
                    "weather_code_rule_based_beta",
                    "site_readiness_not_pass",
                ],
                "beta_label_required": {"temperature": True, "humidity": True, "weather_code": True},
            }
        ),
        encoding="utf-8",
    )

    status = model_status(forecast_dir, reports)

    assert status["temp_rmse"] == 1.694
    assert status["humidity_rmse"] == 10.396
    assert status["benchmark_reliability"] == "strong"
    assert status["site_readiness"] == "WARN"
    assert status["operational_valid"] is False
    assert status["forecast_schema_version"] == "forecast_points.v2-beta-sources"
    assert status["humidity_beta"] is True
    assert status["weather_code_rule_based_beta"] is True
    assert status["model_improvement_frozen"] is True
    assert "temperature_accuracy_caveat" in status["site_caveats"]
    assert any("temperature_accuracy_caveat" in warning for warning in status["warnings"])
    assert any("humidity_beta" in warning for warning in status["warnings"])
    assert any("operational_valid=false" in warning for warning in status["warnings"])


def test_api_infers_mixed_beta_targets_from_points(tmp_path: Path):
    latest = tmp_path / "latest"
    latest.mkdir()
    (latest / "forecast_run.json").write_text(
        json.dumps({"forecast_run_id": "run1", "model_version": "m", "operational_valid": False, "backtest_only": True}),
        encoding="utf-8",
    )
    (latest / "forecast_points.json").write_text(
        json.dumps(
            [
                {"station_id": "108", "valid_time": "2026-01-01T01:00:00Z", "precip_probability_source": "nwp_direct", "precip_probability_status": "direct", "precip_probability_confidence": "low"},
                {"station_id": "108", "valid_time": "2026-01-01T02:00:00Z", "precip_probability_source": "unavailable", "precip_probability_status": "unavailable", "precip_probability_confidence": "unavailable"},
            ]
        ),
        encoding="utf-8",
    )

    latest_payload = latest_forecast(latest)

    precip = latest_payload["beta_targets"]["precip_probability"]
    assert precip["source"] == "mixed"
    assert precip["sources"] == ["nwp_direct", "unavailable"]


def test_api_prefers_point_level_beta_targets_over_stale_run_summary(tmp_path: Path):
    latest = tmp_path / "latest"
    latest.mkdir()
    (latest / "forecast_run.json").write_text(
        json.dumps(
            {
                "forecast_run_id": "run1",
                "model_version": "m",
                "operational_valid": False,
                "backtest_only": True,
                "beta_targets": {
                    "precip_probability": {"source": "ai_beta", "status": "beta", "confidence": "medium"},
                },
            }
        ),
        encoding="utf-8",
    )
    (latest / "forecast_points.json").write_text(
        json.dumps(
            [
                {
                    "station_id": "108",
                    "valid_time": "2026-01-01T01:00:00Z",
                    "precip_probability_source": "nwp_direct",
                    "precip_probability_status": "direct",
                    "precip_probability_confidence": "low",
                },
                {
                    "station_id": "109",
                    "valid_time": "2026-01-01T01:00:00Z",
                    "precip_probability_source": "unavailable",
                    "precip_probability_status": "unavailable",
                    "precip_probability_confidence": "unavailable",
                },
            ]
        ),
        encoding="utf-8",
    )

    latest_payload = latest_forecast(latest)
    station_payload = station_forecast("108", latest)

    assert latest_payload["beta_targets"]["precip_probability"]["source"] == "mixed"
    assert latest_payload["beta_targets"]["precip_probability"]["counts_by_source"] == {"nwp_direct": 1, "unavailable": 1}
    assert station_payload["beta_targets"]["precip_probability"]["source"] == "nwp_direct"
    assert station_payload["beta_targets"]["precip_probability"]["counts_by_source"] == {"nwp_direct": 1}


def test_model_status_derives_rule_based_weather_flag_from_point_beta_targets(tmp_path: Path):
    latest = tmp_path / "latest"
    latest.mkdir()
    (latest / "forecast_run.json").write_text(
        json.dumps({"forecast_run_id": "run1", "model_version": "m", "operational_valid": False, "backtest_only": True}),
        encoding="utf-8",
    )
    (latest / "forecast_points.json").write_text(
        json.dumps(
            [
                {
                    "station_id": "108",
                    "valid_time": "2026-01-01T01:00:00Z",
                    "weather_code_source": "rule_based_beta",
                    "weather_code_status": "beta",
                    "weather_code_confidence": "low",
                }
            ]
        ),
        encoding="utf-8",
    )

    status = model_status(latest, tmp_path / "reports")

    assert status["weather_code_rule_based_beta"] is True
    assert status["beta_targets"]["weather_code"]["source"] == "rule_based_beta"
