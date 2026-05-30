from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from weather_korea_forecast.service.beta_sources import (
    BETA_FORECAST_POINTS_SCHEMA_VERSION,
    DIRECT_SOURCES,
    descriptor_fields,
    direct_source_for_provider,
    normalize_probability_percent,
    summarize_beta_targets_from_points,
)
from weather_korea_forecast.service.confidence import estimate_confidence
from weather_korea_forecast.service.weather_code import weather_code_for_row

BLOCKED_TOKENS = ("diagnostic", "oracle", "synthetic", "smoke", "fixture", "generated")
PROVENANCE_SCAN_KEY_FRAGMENTS = ("source", "path", "uri", "url", "provenance", "artifact", "profile", "note", "run_id")
PROVENANCE_POLICY_KEY_FRAGMENTS = ("policy", "gate", "threshold", "metric", "candidate")


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


def _is_provenance_key(key: Any) -> bool:
    text = str(key).lower()
    return any(fragment in text for fragment in PROVENANCE_SCAN_KEY_FRAGMENTS)


def _is_policy_key(key: Any) -> bool:
    text = str(key).lower()
    return any(fragment in text for fragment in PROVENANCE_POLICY_KEY_FRAGMENTS)


def _walk_values(value: Any, *, key: str = "", in_provenance_branch: bool = False) -> list[Any]:
    if isinstance(value, dict):
        walked: list[Any] = []
        for nested_key, nested in value.items():
            if _is_policy_key(nested_key):
                continue
            next_in_provenance_branch = in_provenance_branch or _is_provenance_key(nested_key)
            walked.extend(_walk_values(nested, key=str(nested_key), in_provenance_branch=next_in_provenance_branch))
        return walked
    if isinstance(value, (list, tuple, set)):
        walked = []
        for nested in value:
            if isinstance(nested, dict):
                walked.extend(_walk_values(nested, key=key, in_provenance_branch=in_provenance_branch))
            elif in_provenance_branch:
                walked.append(nested)
        return walked
    if in_provenance_branch or _is_provenance_key(key):
        return [value]
    return []


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
    forecast_source_path = _trusted_metadata_value(metadata, "forecast_source_path", "prepared_forecast_archive", "prepared_forecast_csv")
    checks = {
        "trusted future_feature_source=prepared_forecast_csv": source_from_metadata == "prepared_forecast_csv",
        "trusted operational_valid=true": _metadata_bool(metadata, "operational_valid") is True,
        "trusted backtest_only=false": _metadata_bool(metadata, "backtest_only") is False,
        "trusted forecast_source_schema_valid=true": _metadata_bool(metadata, "forecast_source_schema_valid") is True
        or _metadata_bool(metadata, "forecast_schema_valid") is True,
        "trusted forecast_archive_adequate=true": _metadata_bool(metadata, "forecast_archive_adequate") is True
        or _metadata_bool(metadata, "real_forecast_archive_adequate") is True,
        "trusted forecast_source_path present": bool(forecast_source_path),
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


def _first_existing_with_name(row: pd.Series, names: list[str], default: Any = None) -> tuple[str | None, Any]:
    for name in names:
        if name in row and pd.notna(row[name]):
            return name, row[name]
    return None, default


def _trusted_ai_beta_targets(metadata: dict[str, Any] | None) -> set[str]:
    if not isinstance(metadata, dict):
        return set()
    beta_targets = metadata.get("beta_targets") if isinstance(metadata.get("beta_targets"), dict) else {}
    trusted: set[str] = set()
    for target, payload in beta_targets.items():
        descriptor = payload.get("descriptor", payload) if isinstance(payload, dict) else {}
        if not isinstance(descriptor, dict):
            continue
        selection_reason = str(descriptor.get("selection_reason") or "")
        validation_metrics = descriptor.get("validation_metrics")
        if (
            descriptor.get("source") == "ai_beta"
            and descriptor.get("status") == "beta"
            and ("validation" in selection_reason or bool(validation_metrics))
        ):
            trusted.add(str(target))
    return trusted


def _explicit_or_default_source(
    row: pd.Series,
    field: str,
    default_source: str,
    *,
    trusted_ai_beta_targets: set[str] | None = None,
) -> tuple[str, bool]:
    explicit = _first_existing(row, [field])
    if explicit in (None, ""):
        return default_source, False
    explicit_source = str(explicit)
    target = field.removesuffix("_source")
    if explicit_source == "ai_beta" and target not in (trusted_ai_beta_targets or set()):
        return default_source, False
    return explicit_source, True


def _expected_status_for_source(source: str) -> str:
    if source in DIRECT_SOURCES:
        return "direct"
    if source in {"ai_mos_model", "ai_mos"}:
        return "model"
    if source in {"ai_beta", "rule_based_beta", "ai_mos_model_beta", "ai_mos_beta"}:
        return "beta"
    return "unavailable"


def _explicit_or_default_status(row: pd.Series, field: str, source: str, *, allow_explicit: bool = True) -> str | None:
    explicit = _first_existing(row, [field])
    expected = _expected_status_for_source(source)
    if allow_explicit and explicit not in (None, "") and str(explicit) == expected:
        return str(explicit)
    return expected


def _explicit_or_default_confidence(row: pd.Series, field: str) -> str | None:
    explicit = _first_existing(row, [field])
    return str(explicit) if explicit not in (None, "") else None


def _source_fields(row: pd.Series, *, prefix: str, source: str, allow_explicit_metadata: bool = True) -> dict[str, str]:
    status = _explicit_or_default_status(row, f"{prefix}_status", source, allow_explicit=allow_explicit_metadata)
    confidence = _explicit_or_default_confidence(row, f"{prefix}_confidence") if allow_explicit_metadata else None
    return descriptor_fields(prefix=prefix, source=source, status=status, confidence=confidence)


def _beta_targets_from_points(points: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if points.empty:
        return {}
    return summarize_beta_targets_from_points(points.to_dict(orient="records"))


def _has_humidity_beta(points: pd.DataFrame) -> bool:
    if points.empty or "humidity_source" not in points:
        return False
    beta_sources = {"ai_mos_model_beta", "ai_mos_beta", "ai_beta"}
    return bool(points["humidity_source"].astype(str).isin(beta_sources).any())


def _probability_input_unit(column_name: str | None) -> str:
    normalized = str(column_name or "").lower()
    if normalized in {"pop", "nwp_precip_probability", "kma_pop"}:
        return "percent"
    return "auto"


def _append_warning_once(warnings: list[str], warning: str) -> None:
    if warning not in warnings:
        warnings.append(warning)


def _normalize_points(predictions: pd.DataFrame, station_meta: pd.DataFrame, *, forecast_run_id: str, model_version: str, source: str, operational_valid: bool, backtest_only: bool, temp_rmse: float | None, humidity_rmse: float | None, trusted_ai_beta_targets: set[str] | None = None) -> pd.DataFrame:
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
        precip_probability_name, precip_probability_raw = _first_existing_with_name(row, ["precip_probability", "pop", "nwp_precip_probability"])
        precip_probability = normalize_probability_percent(
            precip_probability_raw,
            input_unit=_probability_input_unit(precip_probability_name),
        )
        precip_source, precip_source_trusted = _explicit_or_default_source(
            row,
            "precip_probability_source",
            direct_source_for_provider(source) if precip_probability is not None else "unavailable",
            trusted_ai_beta_targets=trusted_ai_beta_targets,
        )
        wind_speed_name, wind_speed = _first_existing_with_name(row, ["wind_speed_ms", "wind_speed", "nwp_wind_speed"])
        wind_direction = _first_existing(row, ["wind_direction_deg", "wind_dir", "nwp_wind_direction"])
        wind_source, wind_source_trusted = _explicit_or_default_source(
            row,
            "wind_source",
            direct_source_for_provider(source) if wind_speed is not None or wind_direction is not None or wind_speed_name is not None else "unavailable",
            trusted_ai_beta_targets=trusted_ai_beta_targets,
        )
        cloud_name, cloud_cover = _first_existing_with_name(row, ["cloud_cover_percent", "cloud_cover", "nwp_cloud_cover"])
        cloud_source, cloud_source_trusted = _explicit_or_default_source(
            row,
            "cloud_source",
            direct_source_for_provider(source) if cloud_cover is not None or cloud_name is not None else "unavailable",
            trusted_ai_beta_targets=trusted_ai_beta_targets,
        )
        temperature_source, temperature_source_trusted = _explicit_or_default_source(
            row,
            "temperature_source",
            "ai_mos_model" if temp is not None else "unavailable",
            trusted_ai_beta_targets=trusted_ai_beta_targets,
        )
        humidity_source, humidity_source_trusted = _explicit_or_default_source(
            row,
            "humidity_source",
            "ai_mos_model_beta" if humidity is not None else "unavailable",
            trusted_ai_beta_targets=trusted_ai_beta_targets,
        )
        point = {
            "forecast_run_id": forecast_run_id,
            "station_id": str(row["station_id"]),
            "station_name": _first_existing(row, ["station_name", "name", "station_name_meta"], str(row["station_id"])),
            "lat": _first_existing(row, ["lat", "latitude", "lat_meta"]),
            "lon": _first_existing(row, ["lon", "longitude", "lon_meta"]),
            "region_class": _first_existing(row, ["region_class", "region", "region_class_meta"], "unknown"),
            "valid_time": str(valid_time) if valid_time is not None else None,
            "valid_date": _first_existing(row, ["valid_date", "forecast_valid_date", "date"]),
            "forecast_day": _first_existing(row, ["forecast_day", "lead_day", "day_ahead"]),
            "horizon_step": int(float(horizon)) if horizon is not None else None,
            "temperature_c": temp,
            "temp_max_c": _first_existing(row, ["temp_max_c", "temperature_max_c", "high_temperature_c"]),
            "temp_min_c": _first_existing(row, ["temp_min_c", "temperature_min_c", "low_temperature_c"]),
            "humidity_percent": humidity,
            "dew_point_c": _first_existing(row, ["dew_point_c", "nwp_dew_point"]),
            "feels_like_c": _first_existing(row, ["feels_like_c"], temp),
            "wind_speed_ms": wind_speed,
            "wind_direction_deg": wind_direction,
            "precip_probability": precip_probability,
            "precip_mm": _first_existing(row, ["precip_mm", "precipitation", "nwp_tp"]),
            "cloud_cover_percent": cloud_cover,
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
        point.update(_source_fields(row, prefix="temperature", source=temperature_source, allow_explicit_metadata=temperature_source_trusted))
        point.update(_source_fields(row, prefix="humidity", source=humidity_source, allow_explicit_metadata=humidity_source_trusted))
        point.update(_source_fields(row, prefix="precip_probability", source=precip_source, allow_explicit_metadata=precip_source_trusted))
        point.update(_source_fields(row, prefix="wind", source=wind_source, allow_explicit_metadata=wind_source_trusted))
        point.update(_source_fields(row, prefix="cloud", source=cloud_source, allow_explicit_metadata=cloud_source_trusted))
        weather_source, weather_source_trusted = _explicit_or_default_source(
            row,
            "weather_code_source",
            "rule_based_beta",
            trusted_ai_beta_targets=trusted_ai_beta_targets,
        )
        point.update(_source_fields(row, prefix="weather_code", source=weather_source, allow_explicit_metadata=weather_source_trusted))
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
    trusted_metadata = _load_json(trusted_experiment_summary_path) if trusted_experiment_summary_path is not None else {}
    trusted_ai_beta_targets = _trusted_ai_beta_targets(trusted_metadata)
    if operational_valid is True or backtest_only is False:
        if trusted_experiment_summary_path is None:
            raise ForecastExportError("operational export requires --trusted-experiment-summary metadata")
        trusted_source = _validate_operational_metadata(trusted_metadata)
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
        trusted_ai_beta_targets=trusted_ai_beta_targets,
    )
    if points.empty:
        raise ForecastExportError("no forecast points produced")
    humidity_beta = _has_humidity_beta(points)
    if humidity_beta:
        _append_warning_once(warnings, "humidity_beta: humidity output must be displayed as AI MOS beta")
    if "weather_code_source" in points and points["weather_code_source"].astype(str).eq("rule_based_beta").any():
        _append_warning_once(warnings, "weather_code_rule_based_beta: weather_code is rule-based beta")
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
        "forecast_schema_version": BETA_FORECAST_POINTS_SCHEMA_VERSION,
        "operational_valid": operational_valid,
        "backtest_only": backtest_only,
        "humidity_beta": humidity_beta,
        "horizon_hours": int(points["horizon_step"].max()),
        "targets": ["temp", "humidity", "precip_probability", "wind", "cloud", "weather_code"],
        "beta_targets": _beta_targets_from_points(points),
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
