from __future__ import annotations

from weather_korea_forecast.reporting.html_template import render_html_report
from weather_korea_forecast.reporting.schema import ExperimentRecord


def test_html_report_includes_nwp_archive_status_section() -> None:
    record = ExperimentRecord(
        experiment_name="g022_temp_operational_residual_ridge_72to24",
        version="g022",
        track="nwp_assisted_mos",
        target_name="temp",
        rmse=1.4,
        future_feature_source="prepared_forecast_csv",
        operational_valid=True,
        backtest_only=False,
        forecast_source_schema_valid=True,
        forecast_archive_adequate=True,
        forecast_archive_station_count=20,
        forecast_archive_issue_time_count=30,
        forecast_archive_horizon_coverage=0.96,
        forecast_archive_blocking_reasons=[],
        forecast_source_path="data/raw/nwp/archive/prepared_forecast_archive.csv",
    )

    html = render_html_report([record], title="test")

    assert "NWP Archive Status" in html
    assert "data/raw/nwp/archive/prepared_forecast_archive.csv" in html
    assert "0.960" in html or "0.96" in html
    assert "operational training enabled" in html
