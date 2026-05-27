from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from weather_korea_forecast.data.nwp_archive import build_prepared_forecast_archive
from weather_korea_forecast.service.export_forecast import ForecastExportError, export_forecast


def test_operational_export_requires_trusted_archive_adequacy(tmp_path: Path) -> None:
    pred = tmp_path / "pred.csv"
    pd.DataFrame({"station_id": ["108"], "valid_time": ["2026-01-01T01:00:00Z"], "horizon_step": [1], "temperature_c": [1.0]}).to_csv(pred, index=False)
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "future_feature_source": "prepared_forecast_csv",
                "operational_valid": True,
                "backtest_only": False,
                "forecast_source_schema_valid": True,
                "forecast_archive_adequate": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ForecastExportError, match="forecast_archive_adequate"):
        export_forecast(
            predictions_path=pred,
            station_metadata_path=None,
            output_dir=tmp_path / "out",
            operational_valid=True,
            backtest_only=False,
            trusted_experiment_summary_path=summary,
        )


def test_archive_quality_blocks_operational_training_when_coverage_short(tmp_path: Path) -> None:
    csv = tmp_path / "short.csv"
    pd.DataFrame(
        {
            "station_id": ["108"],
            "forecast_init_time": ["2026-01-01T00:00:00Z"],
            "issue_time": ["2026-01-01T00:00:00Z"],
            "valid_time": ["2026-01-01T01:00:00Z"],
            "horizon_step": [1],
            "nwp_t2m": [280.0],
            "nwp_sp": [101325.0],
            "nwp_u10": [1.0],
            "nwp_v10": [0.0],
            "nwp_tp": [0.0],
            "source": ["prepared_forecast_csv"],
        }
    ).to_csv(csv, index=False)

    _, report = build_prepared_forecast_archive([csv])

    assert report.forecast_archive_adequate is False
    assert report.horizon_1_24_coverage < 0.95
