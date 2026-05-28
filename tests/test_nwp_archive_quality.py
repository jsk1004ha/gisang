from __future__ import annotations

import json

import pandas as pd

from weather_korea_forecast.data.nwp_archive import build_prepared_forecast_archive, evaluate_archive_quality, normalize_prepared_forecast_archive
from tests.test_nwp_archive_builder import _forecast_rows


def test_archive_quality_report_contains_required_g023_fields(tmp_path) -> None:
    csv = tmp_path / "kma_prepared_forecast.csv"
    _forecast_rows(stations=20, cycles=30, source="kma_forecast").to_csv(csv, index=False)

    _, report = build_prepared_forecast_archive([csv], quality_json=tmp_path / "quality.json", quality_markdown=tmp_path / "quality.md")
    payload = json.loads((tmp_path / "quality.json").read_text(encoding="utf-8"))

    assert report.forecast_archive_adequate is True
    for key in [
        "source",
        "station_count",
        "forecast_cycle_count",
        "min_forecast_init_time",
        "max_forecast_init_time",
        "horizon_min",
        "horizon_max",
        "horizon_1_24_coverage",
        "missing_rate",
        "duplicate_count",
        "station_coverage_table",
        "horizon_coverage_table",
        "issue_time_coverage_table",
        "has_train_val_test_split",
        "forecast_archive_adequate",
        "blocking_reasons",
    ]:
        assert key in payload
    assert payload["source"] == ["kma_forecast"]
    assert payload["forecast_cycle_count"] == 30
    assert payload["missing_rate"] == 0.0
    assert payload["blocking_reasons"] == []
    assert "NWP Forecast Archive Quality" in (tmp_path / "quality.md").read_text(encoding="utf-8")


def test_archive_quality_records_humidity_sanity_warnings() -> None:
    frame = _forecast_rows(stations=20, cycles=30, source="kma_forecast")
    frame["nwp_humidity"] = 120.0
    frame["nwp_dew_point"] = 30.0
    frame["nwp_t2m"] = 20.0
    frame["nwp_precip_probability"] = -1.0
    frame["nwp_sky_code"] = 99
    normalized = normalize_prepared_forecast_archive(frame)

    report = evaluate_archive_quality(normalized)

    assert report.forecast_archive_adequate is False
    assert any("nwp_humidity" in warning for warning in report.warnings)
    assert any("dew_point" in warning for warning in report.warnings)
    assert any("precip_probability" in warning for warning in report.warnings)
    assert any("sky_code" in warning for warning in report.warnings)
    assert any("sanity" in reason for reason in report.blocking_reasons)


def test_archive_quality_tolerates_small_grib_numeric_sanity_epsilon() -> None:
    frame = normalize_prepared_forecast_archive(_forecast_rows(stations=20, cycles=30, source="gfs_forecast"))
    frame["nwp_humidity"] = 100.000006
    frame["nwp_relative_humidity"] = 100.000006
    frame["nwp_dew_point"] = frame["nwp_t2m"] + 0.061

    report = evaluate_archive_quality(frame)

    assert report.forecast_archive_adequate is True
    assert not any("sanity:" in warning for warning in report.warnings)


def test_archive_builder_coalesces_overlapping_sources_without_losing_fields(tmp_path) -> None:
    base = _forecast_rows(stations=20, cycles=30, source="kma_forecast")
    base["nwp_humidity"] = 55.0
    gfs = _forecast_rows(stations=20, cycles=30, source="gfs_forecast")
    gfs["nwp_dew_point"] = 1.0
    gfs = gfs.drop(columns=["nwp_t2m"])
    kma_csv = tmp_path / "kma_prepared_forecast.csv"
    gfs_csv = tmp_path / "gfs_prepared_forecast.csv"
    base.to_csv(kma_csv, index=False)
    gfs.to_csv(gfs_csv, index=False)

    archive, report = build_prepared_forecast_archive([kma_csv, gfs_csv], expected_columns=["nwp_t2m", "nwp_dew_point", "nwp_humidity"])

    assert archive.duplicated(["station_id", "forecast_init_time", "horizon_step"]).sum() == 0
    assert archive["nwp_t2m"].notna().all()
    assert archive["nwp_dew_point"].notna().all()
    assert archive["nwp_humidity"].notna().all()
    assert report.forecast_archive_adequate is True
    assert report.duplicate_count > 0
    assert any("kma_forecast" in source for source in archive["source"].unique())
    assert any("gfs_forecast" in source for source in archive["source"].unique())


def test_archive_quality_blocks_missing_expected_columns_and_uneven_station_coverage() -> None:
    frame = _forecast_rows(stations=20, cycles=30, source="gfs_forecast")
    frame = frame.loc[~((frame["station_id"] == "100") & (pd.to_datetime(frame["forecast_init_time"]) > pd.Timestamp("2026-01-02T00:00:00Z")))].copy()
    normalized = normalize_prepared_forecast_archive(frame.drop(columns=["nwp_t2m"]))

    report = evaluate_archive_quality(normalized, expected_columns=["nwp_t2m"])

    assert report.forecast_archive_adequate is False
    assert report.expected_column_missing_rates["nwp_t2m"] == 1.0
    assert any("expected column nwp_t2m" in reason for reason in report.blocking_reasons)
    assert any("station_level" in reason for reason in report.blocking_reasons)
