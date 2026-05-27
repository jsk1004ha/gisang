from weather_korea_forecast.reporting.schema import ExperimentRecord
from weather_korea_forecast.reporting.site_readiness import evaluate_site_readiness


def test_site_readiness_requires_operational_temp_under_threshold() -> None:
    diagnostic = ExperimentRecord(experiment_name="smoke", is_diagnostic=True, included_in_main_leaderboard=False)
    temp = ExperimentRecord(
        experiment_name="op-temp",
        target_name="temp",
        track="nwp_assisted_mos",
        rmse=1.4,
        operational_valid=True,
        backtest_only=False,
        future_feature_source="prepared_forecast_csv",
        forecast_source_schema_valid=True,
        forecast_archive_adequate=True,
        forecast_source_path="data/raw/nwp/archive/prepared_forecast_archive.csv",
        is_diagnostic=False,
    )
    humidity = ExperimentRecord(experiment_name="humidity", target_name="humidity", rmse=14.0, is_diagnostic=False)

    readiness = evaluate_site_readiness([diagnostic, temp, humidity])

    assert readiness.status == "PASS"
    assert readiness.humidity_beta_allowed is True
    assert readiness.best_operational_temp_rmse == 1.4


def test_site_readiness_warns_without_operational_temp() -> None:
    readiness = evaluate_site_readiness([])

    assert readiness.status == "WARN"
    assert "trusted_operational_temp_model" in readiness.missing_conditions


def test_g022_section_appears_in_html_report(tmp_path):
    from weather_korea_forecast.reporting.html_template import render_report

    record = ExperimentRecord(
        experiment_name="g022_temp_operational_residual_ridge_72to24",
        version="g022",
        target_name="temp",
        track="nwp_assisted_mos",
        rmse=1.4,
        operational_valid=True,
        backtest_only=False,
        future_feature_source="prepared_forecast_csv",
        forecast_source_schema_valid=True,
        forecast_archive_adequate=True,
        forecast_source_path="data/raw/nwp/archive/prepared_forecast_archive.csv",
        included_in_main_leaderboard=True,
        artifact_dir=str(tmp_path / "exp"),
    )

    html = render_report([record], title="Report", experiments_root=tmp_path, include_images=False)

    assert "G022 Model Performance Sprint" in html
    assert "site readiness status" in html


def test_site_readiness_rejects_inadequate_archive_operational_claim() -> None:
    record = ExperimentRecord(
        experiment_name="unsafe-op-temp",
        target_name="temp",
        rmse=1.4,
        operational_valid=True,
        backtest_only=False,
        future_feature_source="prepared_forecast_csv",
        forecast_source_schema_valid=True,
        forecast_archive_adequate=False,
        is_diagnostic=False,
    )

    readiness = evaluate_site_readiness([record])

    assert readiness.status == "WARN"
    assert readiness.best_operational_temp_rmse is None
    assert "trusted_operational_temp_model" in readiness.missing_conditions


def test_site_readiness_rejects_missing_forecast_source_path() -> None:
    record = ExperimentRecord(
        experiment_name="op-temp-no-path",
        target_name="temp",
        rmse=1.4,
        operational_valid=True,
        backtest_only=False,
        future_feature_source="prepared_forecast_csv",
        forecast_source_schema_valid=True,
        forecast_archive_adequate=True,
        is_diagnostic=False,
    )

    readiness = evaluate_site_readiness([record])

    assert readiness.status == "WARN"
    assert readiness.best_operational_temp_rmse is None
    assert "trusted_operational_temp_model" in readiness.missing_conditions


def test_g022_section_ignores_non_g022_operational_records(tmp_path):
    from weather_korea_forecast.reporting.html_template import render_report

    old_operational = ExperimentRecord(
        experiment_name="v4_old_operational",
        version="v4",
        target_name="temp",
        track="nwp_assisted_mos",
        rmse=1.1,
        operational_valid=True,
        backtest_only=False,
        future_feature_source="prepared_forecast_csv",
        forecast_source_schema_valid=True,
        forecast_archive_adequate=True,
        forecast_source_path="data/raw/nwp/archive/prepared_forecast_archive.csv",
        included_in_main_leaderboard=True,
        artifact_dir=str(tmp_path / "old"),
    )
    g022_placeholder = ExperimentRecord(
        experiment_name="g022_temp_operational_residual_ridge_72to24",
        version="g022",
        target_name="temp",
        track="nwp_assisted_mos",
        rmse=1.4,
        operational_valid=True,
        backtest_only=False,
        future_feature_source="prepared_forecast_csv",
        forecast_source_schema_valid=True,
        forecast_archive_adequate=True,
        forecast_source_path="data/raw/nwp/archive/prepared_forecast_archive.csv",
        included_in_main_leaderboard=True,
        artifact_dir=str(tmp_path / "g022"),
    )

    html = render_report([old_operational, g022_placeholder], title="Report", experiments_root=tmp_path, include_images=False)

    section = html.split('<section class="card g022-performance">', 1)[1].split("</section>", 1)[0]
    assert "g022_temp_operational_residual_ridge_72to24" in section
    assert "v4_old_operational" not in section


def test_g022_section_rejects_generated_forecast_source_path(tmp_path):
    from weather_korea_forecast.reporting.html_template import render_report

    record = ExperimentRecord(
        experiment_name="g022_temp_operational_residual_ridge_72to24",
        version="g022",
        target_name="temp",
        track="nwp_assisted_mos",
        rmse=1.0,
        operational_valid=True,
        backtest_only=False,
        future_feature_source="prepared_forecast_csv",
        forecast_source_schema_valid=True,
        forecast_archive_adequate=True,
        forecast_source_path="data/raw/nwp/archive/generated_prepared_forecast.csv",
        included_in_main_leaderboard=True,
        artifact_dir=str(tmp_path / "g022"),
    )

    html = render_report([record], title="Report", experiments_root=tmp_path, include_images=False)

    section = html.split('<section class="card g022-performance">', 1)[1].split("</section>", 1)[0]
    assert "operational temp best model</dt><dd>n/a" in section
    assert "trusted_operational_temp_model" in section
