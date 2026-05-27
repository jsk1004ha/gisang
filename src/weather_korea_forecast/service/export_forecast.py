from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from weather_korea_forecast.service.confidence import estimate_confidence
from weather_korea_forecast.service.weather_code import weather_code_for_row

BLOCKED_TOKENS = ("diagnostic", "oracle", "synthetic", "smoke", "fixture", "generated")


class ForecastExportError(ValueError):
    pass


@dataclass(frozen=True)
class ForecastExportResult:
    forecast_run_id: str
    run_dir: str
    latest_dir: str
    points_count: int
    warnings: list[str]


def _blocked_text(*values: Any) -> str:
    return " ".join(str(value or "").lower() for value in values)


def is_blocked_artifact(*values: Any) -> bool:
    text = _blocked_text(*values)
    return any(token in text for token in BLOCKED_TOKENS)


def _walk_values(value: Any) -> list[Any]:
    if isinstance(value, dict):
        walked: list[Any] = []
        for key, nested in value.items():
            walked.append(key)
            walked.extend(_walk_values(nested))
        return walked
    if isinstance(value, (list, tuple, set)):
        walked = []
        for nested in value:
            walked.extend(_walk_values(nested))
        return walked
    return [value]


def is_blocked_provenance(metadata: dict[str, Any]) -> bool:
    return is_blocked_artifact(*_walk_values(metadata))


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _utc_now_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ForecastExportError(f"metadata file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ForecastExportError(f"metadata file must contain a JSON object: {path}")
    return payload


def _explicit_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    return None


def _metadata_bool(metadata: dict[str, Any], key: str) -> bool | None:
    if key in metadata:
        return _explicit_bool(metadata.get(key))
    future = metadata.get("future_features") if isinstance(metadata.get("future_features"), dict) else {}
    if key in future:
        return _explicit_bool(future.get(key))
    forecast_archive = metadata.get("forecast_archive") if isinstance(metadata.get("forecast_archive"), dict) else {}
    if key == "forecast_archive_adequate" and "adequate" in forecast_archive:
        return _explicit_bool(forecast_archive.get("adequate"))
    return None


def _trusted_metadata_value(metadata: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metadata and metadata.get(key) not in (None, ""):
            return metadata.get(key)
    future = metadata.get("future_features") if isinstance(metadata.get("future_features"), dict) else {}
    for key in keys:
        if key in future and future.get(key) not in (None, ""):
            return future.get(key)
    return None


def _validate_operational_metadata(metadata: dict[str, Any]) -> str:
    source_from_metadata = str(_trusted_metadata_value(metadata, "future_feature_source", "source") or "")
    checks = {
        "trusted future_feature_source=prepared_forecast_csv": source_from_metadata == "prepared_forecast_csv",
        "trusted operational_valid=true": _metadata_bool(metadata, "operational_valid") is True,
        "trusted backtest_only=false": _metadata_bool(metadata, "backtest_only") is False,
        "trusted forecast_source_schema_valid=true": _metadata_bool(metadata, "forecast_source_schema_valid") is True
        or _metadata_bool(metadata, "forecast_schema_valid") is True,
        "trusted forecast_archive_adequate=true": _metadata_bool(metadata, "forecast_archive_adequate") is True
        or _metadata_bool(metadata, "real_forecast_archive_adequate") is True,
    }
    checks["not diagnostic/smoke/synthetic"] = not is_blocked_provenance(metadata)
    missing = [name for name, ok in checks.items() if not ok]
    if missing:
        raise ForecastExportError("operational export requires trusted real forecast metadata: " + ", ".join(missing))
    return source_from_metadata


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ForecastExportError(f"input file not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return pd.DataFrame(payload if isinstance(payload, list) else payload.get("points", []))
    return pd.read_csv(path)


def _load_station_metadata(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["station_id", "station_name", "lat", "lon", "region_class"])
    frame = pd.read_csv(path)
    frame["station_id"] = frame["station_id"].astype(str)
    return frame


def _first_existing(row: pd.Series, names: list[str], default: Any = None) -> Any:
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return default


def _normalize_points(predictions: pd.DataFrame, station_meta: pd.DataFrame, *, forecast_run_id: str, model_version: str, source: str, operational_valid: bool, backtest_only: bool, temp_rmse: float | None, humidity_rmse: float | None) -> pd.DataFrame:
    if "station_id" not in predictions.columns:
        raise ForecastExportError("predictions must include station_id")
    frame = predictions.copy()
    frame["station_id"] = frame["station_id"].astype(str)
    if not station_meta.empty:
        frame = frame.merge(station_meta, on="station_id", how="left", suffixes=("", "_meta"))
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        horizon = _first_existing(row, ["horizon_step", "lead_hour", "horizon"], 1)
        valid_time = _first_existing(row, ["valid_time", "timestamp", "datetime"])
        temp = _first_existing(row, ["temperature_c", "temp", "prediction", "pred_temp", "nwp_t2m"])
        humidity = _first_existing(row, ["humidity_percent", "humidity", "pred_humidity"])
        point = {
            "forecast_run_id": forecast_run_id,
            "station_id": str(row["station_id"]),
            "station_name": _first_existing(row, ["station_name", "name", "station_name_meta"], str(row["station_id"])),
            "lat": _first_existing(row, ["lat", "latitude", "lat_meta"]),
            "lon": _first_existing(row, ["lon", "longitude", "lon_meta"]),
            "region_class": _first_existing(row, ["region_class", "region", "region_class_meta"], "unknown"),
            "valid_time": str(valid_time) if valid_time is not None else None,
            "horizon_step": int(float(horizon)) if horizon is not None else None,
            "temperature_c": temp,
            "humidity_percent": humidity,
            "dew_point_c": _first_existing(row, ["dew_point_c", "nwp_dew_point"]),
            "feels_like_c": _first_existing(row, ["feels_like_c"], temp),
            "wind_speed_ms": _first_existing(row, ["wind_speed_ms", "wind_speed", "nwp_wind_speed"]),
            "wind_direction_deg": _first_existing(row, ["wind_direction_deg", "wind_dir"]),
            "precip_probability": _first_existing(row, ["precip_probability", "pop"]),
            "precip_mm": _first_existing(row, ["precip_mm", "precipitation", "nwp_tp"]),
            "cloud_cover_percent": _first_existing(row, ["cloud_cover_percent", "cloud_cover", "nwp_cloud_cover"]),
            "model_version": model_version,
            "data_source": source,
            "operational_valid": operational_valid,
            "backtest_only": backtest_only,
        }
        weather = weather_code_for_row(point)
        metric = humidity_rmse if humidity is not None and temp is None else temp_rmse
        conf = estimate_confidence(
            target_name="humidity" if humidity is not None and temp is None else "temp",
            horizon_step=point["horizon_step"],
            rmse=metric,
            operational_valid=operational_valid,
            backtest_only=backtest_only,
            humidity_beta=humidity is not None,
        )
        point.update(asdict(weather))
        point.update(asdict(conf))
        rows.append(point)
    return pd.DataFrame(rows)


def export_forecast(
    *,
    predictions_path: Path,
    station_metadata_path: Path | None,
    output_dir: Path,
    forecast_run_id: str | None = None,
    model_version: str = "unknown",
    source: str = "unknown",
    operational_valid: bool = False,
    backtest_only: bool = True,
    notes: str = "",
    artifact_profile: str = "full",
    temp_rmse: float | None = None,
    humidity_rmse: float | None = None,
    trusted_experiment_summary_path: Path | None = None,
) -> ForecastExportResult:
    if is_blocked_artifact(predictions_path, source, notes, artifact_profile, forecast_run_id):
        raise ForecastExportError("diagnostic/oracle/synthetic/smoke/generated/fixture artifacts cannot be exported")
    if operational_valid is True or backtest_only is False:
        if trusted_experiment_summary_path is None:
            raise ForecastExportError("operational export requires --trusted-experiment-summary metadata")
        trusted_source = _validate_operational_metadata(_load_json(trusted_experiment_summary_path))
        if source not in {"", "unknown", trusted_source}:
            raise ForecastExportError(f"operational export source {source!r} conflicts with trusted source {trusted_source!r}")
        source = trusted_source
        operational_valid = True
        backtest_only = False
    run_id = forecast_run_id or f"forecast_{_utc_now_id()}"
    predictions = _read_frame(predictions_path)
    station_meta = _load_station_metadata(station_metadata_path)
    warnings: list[str] = []
    if not operational_valid:
        warnings.append("operational_valid=false: research/backtest warning required")
    if backtest_only:
        warnings.append("backtest_only=true: do not present as live operational forecast")
    points = _normalize_points(
        predictions,
        station_meta,
        forecast_run_id=run_id,
        model_version=model_version,
        source=source,
        operational_valid=operational_valid,
        backtest_only=backtest_only,
        temp_rmse=temp_rmse,
        humidity_rmse=humidity_rmse,
    )
    if points.empty:
        raise ForecastExportError("no forecast points produced")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / "runs" / run_id
    latest_dir = output_dir / "latest"
    archive_dir = output_dir / "archive"
    run_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    run = {
        "forecast_run_id": run_id,
        "model_version": model_version,
        "forecast_init_time": str(predictions.get("forecast_init_time", predictions.get("issue_time", pd.Series([None]))).iloc[0]),
        "created_at": datetime.now(UTC).isoformat(),
        "source": source,
        "operational_valid": operational_valid,
        "backtest_only": backtest_only,
        "horizon_hours": int(points["horizon_step"].max()),
        "targets": ["temp", "humidity", "weather_code"],
        "notes": notes,
        "warnings": warnings,
    }
    for directory in (run_dir, latest_dir):
        if directory == latest_dir and directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "forecast_run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
        points.to_json(directory / "forecast_points.json", orient="records", force_ascii=False, indent=2)
        points.to_csv(directory / "forecast_points.csv", index=False)
    shutil.copy2(run_dir / "forecast_run.json", archive_dir / f"{run_id}_forecast_run.json")
    return ForecastExportResult(run_id, str(run_dir), str(latest_dir), len(points), warnings)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export model predictions as site forecast artifacts.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--station-metadata", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--forecast-run-id", default=None)
    parser.add_argument("--model-version", default="unknown")
    parser.add_argument("--source", default="unknown")
    parser.add_argument("--operational-valid", default="false")
    parser.add_argument("--backtest-only", default="true")
    parser.add_argument("--notes", default="")
    parser.add_argument("--artifact-profile", default="full")
    parser.add_argument("--temp-rmse", type=float)
    parser.add_argument("--humidity-rmse", type=float)
    parser.add_argument("--trusted-experiment-summary", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = export_forecast(
        predictions_path=args.predictions,
        station_metadata_path=args.station_metadata,
        output_dir=args.output_dir,
        forecast_run_id=None if args.forecast_run_id == "auto" else args.forecast_run_id,
        model_version=args.model_version,
        source=args.source,
        operational_valid=_bool(args.operational_valid),
        backtest_only=_bool(args.backtest_only),
        notes=args.notes,
        artifact_profile=args.artifact_profile,
        temp_rmse=args.temp_rmse,
        humidity_rmse=args.humidity_rmse,
        trusted_experiment_summary_path=args.trusted_experiment_summary,
    )
    print(json.dumps(asdict(result), ensure_ascii=False))


if __name__ == "__main__":
    main()
