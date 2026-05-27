from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from weather_korea_forecast.service.export_forecast import ForecastExportError, export_forecast, is_blocked_artifact, is_blocked_provenance


def _load_summary(experiment_dir: Path) -> dict[str, Any]:
    path = experiment_dir / "experiment_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _predictions_from_forecast_csv(forecast_csv: Path, output_dir: Path) -> Path:
    frame = pd.read_csv(forecast_csv)
    if "station_id" not in frame.columns:
        raise ForecastExportError("forecast CSV must include station_id")
    out = output_dir / "pipeline_work" / "forecast_csv_predictions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    mapped = pd.DataFrame()
    mapped["station_id"] = frame["station_id"]
    mapped["valid_time"] = frame.get("valid_time")
    mapped["horizon_step"] = frame.get("horizon_step", frame.get("lead_hour", 1))
    mapped["temperature_c"] = frame.get("nwp_t2m", frame.get("nwp_t2m_c", frame.get("temperature_c")))
    mapped["humidity_percent"] = frame.get("humidity_percent")
    mapped["dew_point_c"] = frame.get("nwp_dew_point", frame.get("nwp_dew_point_c"))
    mapped["wind_speed_ms"] = frame.get("nwp_wind_speed")
    mapped["precip_mm"] = frame.get("nwp_tp")
    mapped["cloud_cover_percent"] = frame.get("nwp_cloud_cover")
    mapped.to_csv(out, index=False)
    return out


def run_forecast_pipeline(
    *,
    forecast_csv: Path,
    experiment_dir: Path,
    station_metadata: Path | None,
    output_dir: Path,
    predictions: Path | None = None,
    operational: bool = False,
    research: bool = False,
) -> dict[str, Any]:
    summary = _load_summary(experiment_dir)
    operational_valid = _bool(summary.get("operational_valid"))
    backtest_only = _bool(summary.get("backtest_only", True))
    source = str(summary.get("future_feature_source") or summary.get("source") or "unknown")
    if operational:
        if not operational_valid or backtest_only:
            raise ForecastExportError("operational mode requires operational_valid=true and backtest_only=false model")
        if _bool(summary.get("forecast_source_schema_valid", summary.get("forecast_schema_valid"))) is not True:
            raise ForecastExportError("operational mode requires forecast_source_schema_valid=true")
        if _bool(summary.get("forecast_archive_adequate", summary.get("real_forecast_archive_adequate"))) is not True:
            raise ForecastExportError("operational mode requires forecast_archive_adequate=true")
        if is_blocked_artifact(forecast_csv, source) or is_blocked_provenance(summary):
            raise ForecastExportError("operational mode blocks synthetic/smoke/diagnostic forecast sources")
    elif not research:
        raise ForecastExportError("pipeline requires either --operational or --research mode")
    prediction_path = predictions or _predictions_from_forecast_csv(forecast_csv, output_dir)
    result = export_forecast(
        predictions_path=prediction_path,
        station_metadata_path=station_metadata,
        output_dir=output_dir,
        model_version=str(summary.get("experiment_name") or experiment_dir.name),
        source=source,
        operational_valid=operational_valid,
        backtest_only=backtest_only,
        notes=str(summary.get("notes") or ("research export" if research else "operational export")),
        artifact_profile=str(summary.get("artifact_profile") or "full"),
        temp_rmse=(summary.get("metrics") or {}).get("rmse"),
        trusted_experiment_summary_path=experiment_dir / "experiment_summary.json" if (operational_valid or not backtest_only) else None,
    )
    return {"status": "ok", **result.__dict__}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run forecast pipeline skeleton.")
    parser.add_argument("--forecast-csv", required=True, type=Path)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--station-metadata", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--operational", action="store_true")
    parser.add_argument("--research", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(json.dumps(run_forecast_pipeline(**vars(args)), ensure_ascii=False))


if __name__ == "__main__":
    main()
