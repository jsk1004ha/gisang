from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from weather_korea_forecast.data.nwp_archive import build_prepared_forecast_archive, evaluate_archive_quality, normalize_prepared_forecast_archive


def _forecast_rows(stations: int = 2, cycles: int = 2, horizons: int = 24, *, source: str = "prepared_forecast_csv") -> pd.DataFrame:
    rows = []
    init_times = pd.date_range("2026-01-01T00:00:00Z", periods=cycles, freq="6h")
    for station_idx in range(stations):
        station = str(100 + station_idx)
        for init in init_times:
            for horizon in range(1, horizons + 1):
                rows.append(
                    {
                        "station_id": station,
                        "forecast_init_time": init.isoformat(),
                        "issue_time": init.isoformat(),
                        "valid_time": (init + pd.Timedelta(hours=horizon)).isoformat(),
                        "horizon_step": horizon,
                        "nwp_t2m": 273.15 + 10.0,
                        "nwp_sp": 101325,
                        "nwp_u10": 1.0,
                        "nwp_v10": 2.0,
                        "nwp_tp": 0.0,
                        "source": source,
                    }
                )
    return pd.DataFrame(rows)


def test_archive_builder_marks_small_archive_inadequate_and_converts_units(tmp_path: Path) -> None:
    csv = tmp_path / "prepared.csv"
    _forecast_rows().to_csv(csv, index=False)

    archive, report = build_prepared_forecast_archive(
        [csv],
        output_csv=tmp_path / "archive.csv",
        quality_json=tmp_path / "quality.json",
        quality_markdown=tmp_path / "quality.md",
    )

    assert archive["nwp_t2m"].median() == 10.0
    assert archive["nwp_sp"].median() == 1013.25
    assert report.forecast_source_schema_valid is True
    assert report.forecast_archive_adequate is False
    assert "station_count >= 20" in report.adequacy_reasons
    assert json.loads((tmp_path / "quality.json").read_text(encoding="utf-8"))["forecast_archive_adequate"] is False


def test_archive_builder_accepts_sufficient_archive(tmp_path: Path) -> None:
    csv = tmp_path / "prepared.csv"
    _forecast_rows(stations=20, cycles=30).to_csv(csv, index=False)

    _, report = build_prepared_forecast_archive([csv])

    assert report.station_count == 20
    assert report.forecast_init_time_count == 30
    assert report.horizon_1_24_coverage == 1.0
    assert report.forecast_archive_adequate is True


def test_archive_builder_preserves_mixed_iso_and_space_timestamp_formats(tmp_path: Path) -> None:
    first = _forecast_rows(stations=20, cycles=30)
    second = _forecast_rows(stations=20, cycles=30)
    for column in ["forecast_init_time", "issue_time", "valid_time"]:
        second[column] = pd.to_datetime(second[column], utc=True) + pd.Timedelta(days=10)
        second[column] = second[column].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    csv_a = tmp_path / "iso.csv"
    csv_b = tmp_path / "space.csv"
    first.to_csv(csv_a, index=False)
    second.to_csv(csv_b, index=False)

    archive, report = build_prepared_forecast_archive([csv_a, csv_b], min_stations=20, min_issue_cycles=60)

    assert archive["forecast_init_time"].isna().sum() == 0
    assert report.forecast_init_time_count == 60
    assert report.duplicate_count == 0
    assert report.forecast_source_schema_valid is True


def test_archive_builder_rejects_smoke_as_adequate(tmp_path: Path) -> None:
    frame = normalize_prepared_forecast_archive(_forecast_rows(stations=20, cycles=30, source="synthetic_smoke"))
    report = evaluate_archive_quality(frame, source_paths=["synthetic_smoke.csv"])

    assert report.forecast_archive_adequate is False
    assert any("synthetic" in reason for reason in report.adequacy_reasons)


def test_archive_builder_detects_missing_horizon_coverage() -> None:
    frame = normalize_prepared_forecast_archive(_forecast_rows(stations=20, cycles=30, horizons=22))
    report = evaluate_archive_quality(frame)

    assert report.horizon_1_24_coverage < 0.95
    assert report.forecast_archive_adequate is False
    assert report.missing_horizon_count > 0
