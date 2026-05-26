from __future__ import annotations

import base64
from pathlib import Path

from weather_korea_forecast.reporting.html_template import render_report
from weather_korea_forecast.reporting.plot_embed import embed_image_tag
from weather_korea_forecast.reporting.schema import ExperimentRecord

PNG_1X1 = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=")


def test_embed_image_tag_handles_missing_and_existing_png(tmp_path: Path) -> None:
    missing = embed_image_tag(tmp_path / "missing.png", "missing")
    assert "not available" in missing
    image = tmp_path / "forecast_vs_actual.png"
    image.write_bytes(PNG_1X1)
    embedded = embed_image_tag(image, "plot")
    assert "data:image/png;base64" in embedded


def test_render_report_contains_filters_badges_and_embedded_plot(tmp_path: Path) -> None:
    exp = tmp_path / "exp"
    exp.mkdir()
    (exp / "forecast_vs_actual.png").write_bytes(PNG_1X1)
    record = ExperimentRecord(
        experiment_name="v3_temp_mos_residual_ridge_72to24",
        version="v3",
        target_name="temp",
        track="nwp_assisted_mos",
        model_type="residual",
        rmse=0.99,
        mae=0.75,
        bias=0.02,
        rmse_goal=1.0,
        rmse_goal_met=True,
        future_feature_source="era5_reanalysis",
        operational_valid=False,
        backtest_only=True,
        artifact_dir=str(exp),
    )
    html = render_report([record], title="Report", experiments_root=tmp_path, include_images=True)
    assert "리더보드" in html
    assert "백테스트" in html
    assert "data:image/png;base64" in html
    assert "versionFilter" in html


def test_render_report_separates_diagnostic_and_warns_without_operational_model(tmp_path: Path) -> None:
    main_exp = tmp_path / "main"
    oracle_exp = tmp_path / "oracle"
    main_exp.mkdir()
    oracle_exp.mkdir()
    main = ExperimentRecord(
        experiment_name="v3_temp_mos_residual_ridge_72to24",
        version="v3",
        target_name="temp",
        track="nwp_assisted_mos",
        model_type="ridge",
        rmse=1.05,
        mae=0.8,
        bias=0.1,
        rmse_goal=1.0,
        rmse_goal_met=False,
        future_feature_source="era5_reanalysis",
        operational_valid=False,
        backtest_only=True,
        included_in_main_leaderboard=True,
        artifact_dir=str(main_exp),
    )
    oracle = ExperimentRecord(
        experiment_name="observed_target_oracle",
        version="v3",
        target_name="humidity",
        track="oracle",
        model_type="decoder_feature_baseline",
        rmse=0.0,
        mae=0.0,
        bias=0.0,
        rmse_goal=10.0,
        rmse_goal_met=None,
        goal_eligible=False,
        is_diagnostic=True,
        included_in_main_leaderboard=False,
        future_feature_source="observed_target_oracle",
        operational_valid=False,
        backtest_only=True,
        artifact_dir=str(oracle_exp),
    )

    html = render_report([main, oracle], title="Report", experiments_root=tmp_path, include_images=False)

    assert "Diagnostic / Oracle Checks" in html
    assert "실제 예측 모델 성능이나 best RMSE로 해석하지 마십시오" in html
    assert "운영 가능 모델 없음" in html
    assert "전체 Best RMSE" not in html
    assert "0.000" in html  # diagnostic section still displays oracle metrics.

    main_tbody = html.split('<table id="leaderboard">', 1)[1].split("</tbody>", 1)[0]
    assert "observed_target_oracle" not in main_tbody
    assert "v3_temp_mos_residual_ridge_72to24" in main_tbody
    diagnostic_section = html.split("Diagnostic / Oracle Checks", 1)[1].split("경고 / 데이터 품질", 1)[0]
    assert 'badge-ok">목표 달성' not in diagnostic_section
    assert "목표 집계 제외" in html


def test_render_report_charts_use_main_records_only(tmp_path: Path) -> None:
    main_exp = tmp_path / "main"
    diagnostic_exp = tmp_path / "diagnostic"
    main_exp.mkdir()
    diagnostic_exp.mkdir()
    main = ExperimentRecord(
        experiment_name="real_model",
        version="v3",
        target_name="temp",
        track="nwp_assisted_mos",
        model_type="ridge",
        rmse=1.2,
        mae=0.9,
        bias=0.1,
        rmse_goal=1.0,
        rmse_goal_met=False,
        goal_eligible=True,
        backtest_only=True,
        operational_valid=False,
        included_in_main_leaderboard=True,
        artifact_dir=str(main_exp),
    )
    diagnostic = ExperimentRecord(
        experiment_name="oracle_model",
        version="v3",
        target_name="temp",
        track="oracle",
        model_type="decoder_feature_baseline",
        rmse=0.0,
        mae=0.0,
        bias=0.0,
        rmse_goal=1.0,
        rmse_goal_met=None,
        goal_eligible=False,
        is_diagnostic=True,
        backtest_only=True,
        operational_valid=False,
        included_in_main_leaderboard=False,
        artifact_dir=str(diagnostic_exp),
    )
    html = render_report([main, diagnostic], title="Report", experiments_root=tmp_path, include_images=False)
    charts = html.split('<section class="grid two">', 1)[1].split("</section>", 1)[0]

    assert "real_model" in charts
    assert "oracle_model" not in charts
    assert "decoder_feature_baseline" not in charts
