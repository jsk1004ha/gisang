from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

REQUIRED_PREPARED_FORECAST_COLUMNS = {
    "station_id",
    "forecast_init_time",
    "valid_time",
    "horizon_step",
    "source",
    "issue_time",
}
BLOCKED_SOURCE_TOKENS = ("synthetic", "smoke", "fixture", "generated", "oracle", "diagnostic")
TEMP_COLUMNS = ("nwp_t2m", "nwp_dew_point", "nwp_d2m", "era5_t2m", "era5_d2m", "era5_dew_point_c")
HUMIDITY_BOUNDS_EPSILON = 1e-3
DEW_POINT_ORDER_EPSILON_C = 0.1


@dataclass(frozen=True)
class ArchiveQualityReport:
    row_count: int
    source: list[str]
    station_count: int
    forecast_cycle_count: int
    forecast_init_time_count: int
    issue_time_count: int
    min_forecast_init_time: str | None
    max_forecast_init_time: str | None
    horizon_min: int | None
    horizon_max: int | None
    horizon_1_24_coverage: float
    missing_rate: float
    duplicate_count: int
    missing_required_columns: list[str]
    invalid_valid_time_count: int
    missing_horizon_count: int
    station_coverage_table: list[dict[str, Any]]
    horizon_coverage_table: list[dict[str, Any]]
    issue_time_coverage_table: list[dict[str, Any]]
    has_train_val_test_split: bool
    train_val_test_split_possible: bool
    encoder_prediction_window_possible: bool
    forecast_source_schema_valid: bool
    forecast_archive_adequate: bool
    archive_path: str | None = None
    archive_content_sha256: str | None = None
    input_paths: list[str] = field(default_factory=list)
    expected_columns: list[str] = field(default_factory=list)
    expected_column_missing_rates: dict[str, float] = field(default_factory=dict)
    blocking_reasons: list[str] = field(default_factory=list)
    adequacy_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_prepared_forecast_archive(
    input_paths: Iterable[str | Path],
    *,
    output_csv: str | Path | None = None,
    quality_json: str | Path | None = None,
    quality_markdown: str | Path | None = None,
    min_stations: int = 20,
    min_issue_cycles: int = 30,
    required_horizon: int = 24,
    min_horizon_coverage: float = 0.95,
    encoder_length: int = 72,
    prediction_length: int = 24,
    expected_columns: Iterable[str] = (),
    max_expected_column_missing_rate: float = 0.05,
) -> tuple[pd.DataFrame, ArchiveQualityReport]:
    paths = [Path(path) for path in input_paths]
    if not paths:
        raise ValueError("At least one prepared forecast CSV path is required.")
    frames = [_read_prepared_forecast_csv(path) for path in paths]
    archive = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    normalized = normalize_prepared_forecast_archive(archive)
    duplicate_count = _duplicate_count(normalized)
    deduped = _coalesce_duplicate_forecasts(normalized)
    deduped = deduped.sort_values(["station_id", "forecast_init_time", "horizon_step", "valid_time"]).reset_index(drop=True)
    report = evaluate_archive_quality(
        deduped,
        source_paths=paths,
        min_stations=min_stations,
        min_issue_cycles=min_issue_cycles,
        required_horizon=required_horizon,
        min_horizon_coverage=min_horizon_coverage,
        encoder_length=encoder_length,
        prediction_length=prediction_length,
        expected_columns=expected_columns,
        max_expected_column_missing_rate=max_expected_column_missing_rate,
        duplicate_count_override=duplicate_count,
    )
    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        deduped.to_csv(output_path, index=False)
        report = replace(
            report,
            archive_path=str(output_path),
            archive_content_sha256=_sha256_file(output_path),
        )
    if quality_json is not None:
        Path(quality_json).parent.mkdir(parents=True, exist_ok=True)
        Path(quality_json).write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    if quality_markdown is not None:
        Path(quality_markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(quality_markdown).write_text(render_archive_quality_markdown(report), encoding="utf-8")
    return deduped, report


def normalize_prepared_forecast_archive(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "datetime" in normalized.columns and "valid_time" not in normalized.columns:
        normalized["valid_time"] = normalized["datetime"]
    if "lead_hour" in normalized.columns and "horizon_step" not in normalized.columns:
        normalized["horizon_step"] = normalized["lead_hour"]
    if "forecast_init_time" not in normalized.columns and "issue_time" in normalized.columns:
        normalized["forecast_init_time"] = normalized["issue_time"]
    if "issue_time" not in normalized.columns and "forecast_init_time" in normalized.columns:
        normalized["issue_time"] = normalized["forecast_init_time"]
    if "source" not in normalized.columns:
        normalized["source"] = "prepared_forecast_csv"
    if "station_id" in normalized.columns:
        normalized["station_id"] = normalized["station_id"].astype(str)
    for column in ("forecast_init_time", "issue_time", "valid_time"):
        if column in normalized.columns:
            normalized[column] = pd.to_datetime(normalized[column], utc=True, errors="coerce")
    if "horizon_step" in normalized.columns:
        normalized["horizon_step"] = pd.to_numeric(normalized["horizon_step"], errors="coerce").astype("Int64")
    for column in TEMP_COLUMNS:
        if column in normalized.columns:
            normalized[column] = _temperature_to_celsius_if_needed(normalized[column])
    if "nwp_sp" in normalized.columns:
        pressure = pd.to_numeric(normalized["nwp_sp"], errors="coerce")
        if pressure.median(skipna=True) > 2000:
            normalized["nwp_sp"] = pressure / 100.0
    return normalized


def evaluate_archive_quality(
    frame: pd.DataFrame,
    *,
    source_paths: Iterable[str | Path] = (),
    min_stations: int = 20,
    min_issue_cycles: int = 30,
    required_horizon: int = 24,
    min_horizon_coverage: float = 0.95,
    encoder_length: int = 72,
    prediction_length: int = 24,
    expected_columns: Iterable[str] = (),
    max_expected_column_missing_rate: float = 0.05,
    duplicate_count_override: int | None = None,
) -> ArchiveQualityReport:
    missing_required = sorted(REQUIRED_PREPARED_FORECAST_COLUMNS - set(frame.columns))
    duplicate_count = _duplicate_count(frame) if duplicate_count_override is None else int(duplicate_count_override)
    invalid_valid_time_count = _invalid_valid_time_count(frame)
    station_count = int(frame["station_id"].nunique()) if "station_id" in frame.columns else 0
    forecast_init_time_count = int(frame["forecast_init_time"].nunique()) if "forecast_init_time" in frame.columns else 0
    issue_time_count = int(frame["issue_time"].nunique()) if "issue_time" in frame.columns else 0
    horizon_min = _nullable_int(frame["horizon_step"].min()) if "horizon_step" in frame.columns and not frame.empty else None
    horizon_max = _nullable_int(frame["horizon_step"].max()) if "horizon_step" in frame.columns and not frame.empty else None
    coverage, missing_horizon_count = _horizon_coverage(frame, required_horizon=required_horizon)
    split_possible = forecast_init_time_count >= min_issue_cycles
    window_possible = _time_span_hours(frame) >= encoder_length + prediction_length
    schema_valid = not missing_required and invalid_valid_time_count == 0
    warnings: list[str] = []
    warnings.extend(_sanity_warnings(frame))
    if duplicate_count:
        warnings.append(f"{duplicate_count} duplicate station/forecast_init_time/horizon rows will be de-duplicated")
    blocked_source = _has_blocked_source(frame, source_paths)
    if blocked_source:
        warnings.append("archive provenance contains synthetic/smoke/diagnostic/generated token")
    expected_horizon_count = _expected_horizon_count(frame, required_horizon=required_horizon)
    missing_rate = float(missing_horizon_count / expected_horizon_count) if expected_horizon_count else 1.0
    sanity_ok = not any("sanity:" in warning for warning in warnings)
    station_coverage_table = _station_coverage_table(frame, required_horizon=required_horizon)
    station_level_ok = bool(station_coverage_table) and all(
        int(row.get("forecast_cycle_count") or 0) >= min_issue_cycles
        and float(row.get("horizon_1_24_coverage") or 0.0) >= min_horizon_coverage
        for row in station_coverage_table
    )
    expected_columns_list = [str(column) for column in expected_columns]
    expected_missing_rates = _expected_column_missing_rates(frame, expected_columns_list)
    expected_columns_ok = all(rate <= max_expected_column_missing_rate for rate in expected_missing_rates.values())
    adequacy_checks = {
        f"station_count >= {min_stations}": station_count >= min_stations,
        f"forecast_init_time_count >= {min_issue_cycles}": forecast_init_time_count >= min_issue_cycles,
        f"horizon_1_{required_horizon}_coverage >= {min_horizon_coverage:.2f}": coverage >= min_horizon_coverage,
        "missing_rate <= 0.05": missing_rate <= 0.05,
        "station_level_cycle_and_horizon_coverage": station_level_ok,
        "expected forecast feature columns present": expected_columns_ok,
        "train_val_test_split_possible": split_possible,
        f"encoder_{encoder_length}_prediction_{prediction_length}_window_possible": window_possible,
        "forecast_source_schema_valid": schema_valid,
        "not synthetic/smoke/diagnostic": not blocked_source,
        "humidity/dew-point sanity checks passed": sanity_ok,
    }
    reasons = [name for name, ok in adequacy_checks.items() if not ok]
    for column, rate in expected_missing_rates.items():
        if rate > max_expected_column_missing_rate:
            reasons.append(f"expected column {column} missing_rate {rate:.3f} > {max_expected_column_missing_rate:.3f}")
    source_values = sorted(frame["source"].dropna().astype(str).unique().tolist()) if "source" in frame.columns else []
    min_init = _iso_or_none(frame["forecast_init_time"].min()) if "forecast_init_time" in frame.columns and not frame.empty else None
    max_init = _iso_or_none(frame["forecast_init_time"].max()) if "forecast_init_time" in frame.columns and not frame.empty else None
    return ArchiveQualityReport(
        row_count=int(len(frame)),
        source=source_values,
        station_count=station_count,
        forecast_cycle_count=forecast_init_time_count,
        forecast_init_time_count=forecast_init_time_count,
        issue_time_count=issue_time_count,
        min_forecast_init_time=min_init,
        max_forecast_init_time=max_init,
        horizon_min=horizon_min,
        horizon_max=horizon_max,
        horizon_1_24_coverage=float(coverage),
        missing_rate=missing_rate,
        duplicate_count=duplicate_count,
        missing_required_columns=missing_required,
        invalid_valid_time_count=invalid_valid_time_count,
        missing_horizon_count=missing_horizon_count,
        station_coverage_table=station_coverage_table,
        horizon_coverage_table=_horizon_coverage_table(frame),
        issue_time_coverage_table=_issue_time_coverage_table(frame),
        has_train_val_test_split=split_possible,
        train_val_test_split_possible=split_possible,
        encoder_prediction_window_possible=window_possible,
        forecast_source_schema_valid=schema_valid,
        forecast_archive_adequate=not reasons,
        input_paths=[str(path) for path in source_paths],
        expected_columns=expected_columns_list,
        expected_column_missing_rates=expected_missing_rates,
        blocking_reasons=reasons,
        adequacy_reasons=reasons,
        warnings=warnings,
    )


def render_archive_quality_markdown(report: ArchiveQualityReport) -> str:
    rows = ["# NWP Forecast Archive Quality", "", f"adequate: `{report.forecast_archive_adequate}`", ""]
    for key, value in report.to_dict().items():
        rows.append(f"- {key}: {value}")
    rows.append("")
    if not report.forecast_archive_adequate:
        rows.append("This archive is suitable for schema/smoke validation only, not honest operational model training.")
    return "\n".join(rows) + "\n"


def _read_prepared_forecast_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _temperature_to_celsius_if_needed(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.median(skipna=True) > 150:
        return values - 273.15
    return values


def _duplicate_count(frame: pd.DataFrame) -> int:
    keys = ["station_id", "forecast_init_time", "horizon_step"]
    if not set(keys).issubset(frame.columns):
        return 0
    return int(frame.duplicated(keys).sum())


def _coalesce_duplicate_forecasts(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["station_id", "forecast_init_time", "horizon_step"]
    if not set(keys).issubset(frame.columns) or frame.empty:
        return frame.copy()
    if not frame.duplicated(keys).any():
        return frame.copy()
    rows: list[dict[str, Any]] = []
    for _, group in frame.groupby(keys, sort=False, dropna=False):
        row: dict[str, Any] = {}
        for column in frame.columns:
            if column in keys:
                row[column] = group[column].iloc[0]
                continue
            if column == "source":
                sources = [str(value) for value in group[column].dropna().astype(str).unique()]
                row[column] = "|".join(sources)
                continue
            values = group[column].dropna()
            row[column] = values.iloc[-1] if not values.empty else pd.NA
        rows.append(row)
    return pd.DataFrame(rows, columns=frame.columns)


def _invalid_valid_time_count(frame: pd.DataFrame) -> int:
    required = {"forecast_init_time", "valid_time", "horizon_step"}
    if not required.issubset(frame.columns):
        return 0
    expected = frame["forecast_init_time"] + pd.to_timedelta(frame["horizon_step"].astype("float"), unit="h")
    invalid = frame["valid_time"].isna() | frame["forecast_init_time"].isna() | frame["horizon_step"].isna() | (frame["valid_time"] != expected)
    return int(invalid.sum())


def _horizon_coverage(frame: pd.DataFrame, *, required_horizon: int) -> tuple[float, int]:
    keys = {"station_id", "forecast_init_time", "horizon_step"}
    if not keys.issubset(frame.columns) or frame.empty:
        return 0.0, 0
    pairs = frame[["station_id", "forecast_init_time"]].drop_duplicates()
    if pairs.empty:
        return 0.0, 0
    expected_count = int(len(pairs) * required_horizon)
    observed = frame.loc[frame["horizon_step"].between(1, required_horizon), ["station_id", "forecast_init_time", "horizon_step"]].drop_duplicates()
    observed_count = int(len(observed))
    missing = max(expected_count - observed_count, 0)
    return (observed_count / expected_count if expected_count else 0.0), missing


def _expected_horizon_count(frame: pd.DataFrame, *, required_horizon: int) -> int:
    keys = {"station_id", "forecast_init_time"}
    if not keys.issubset(frame.columns) or frame.empty:
        return 0
    return int(len(frame[["station_id", "forecast_init_time"]].drop_duplicates()) * required_horizon)


def _station_coverage_table(frame: pd.DataFrame, *, required_horizon: int) -> list[dict[str, Any]]:
    keys = {"station_id", "forecast_init_time", "horizon_step"}
    if not keys.issubset(frame.columns) or frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for station_id, group in frame.groupby("station_id"):
        cycles = int(group["forecast_init_time"].nunique())
        expected = cycles * required_horizon
        observed = int(len(group.loc[group["horizon_step"].between(1, required_horizon), ["forecast_init_time", "horizon_step"]].drop_duplicates()))
        rows.append(
            {
                "station_id": str(station_id),
                "forecast_cycle_count": cycles,
                "horizon_1_24_coverage": float(observed / expected) if expected else 0.0,
            }
        )
    return rows


def _horizon_coverage_table(frame: pd.DataFrame) -> list[dict[str, Any]]:
    keys = {"horizon_step", "station_id", "forecast_init_time"}
    if not keys.issubset(frame.columns) or frame.empty:
        return []
    denominator = int(len(frame[["station_id", "forecast_init_time"]].drop_duplicates()))
    rows: list[dict[str, Any]] = []
    for horizon, group in frame.groupby("horizon_step"):
        observed = int(len(group[["station_id", "forecast_init_time"]].drop_duplicates()))
        rows.append({"horizon_step": int(horizon), "coverage": float(observed / denominator) if denominator else 0.0})
    return sorted(rows, key=lambda row: int(row["horizon_step"]))


def _issue_time_coverage_table(frame: pd.DataFrame) -> list[dict[str, Any]]:
    keys = {"issue_time", "station_id", "horizon_step"}
    if not keys.issubset(frame.columns) or frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for issue_time, group in frame.groupby("issue_time"):
        rows.append(
            {
                "issue_time": _iso_or_none(issue_time),
                "station_count": int(group["station_id"].nunique()),
                "row_count": int(len(group)),
                "horizon_min": _nullable_int(group["horizon_step"].min()),
                "horizon_max": _nullable_int(group["horizon_step"].max()),
            }
        )
    return rows


def _sanity_warnings(frame: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    if "nwp_humidity" in frame.columns:
        humidity = pd.to_numeric(frame["nwp_humidity"], errors="coerce")
        bad = humidity.notna() & ~humidity.between(-HUMIDITY_BOUNDS_EPSILON, 100.0 + HUMIDITY_BOUNDS_EPSILON)
        if bad.any():
            warnings.append(f"sanity: nwp_humidity outside 0..100 in {int(bad.sum())} rows")
    if "nwp_relative_humidity" in frame.columns:
        humidity = pd.to_numeric(frame["nwp_relative_humidity"], errors="coerce")
        bad = humidity.notna() & ~humidity.between(-HUMIDITY_BOUNDS_EPSILON, 100.0 + HUMIDITY_BOUNDS_EPSILON)
        if bad.any():
            warnings.append(f"sanity: nwp_relative_humidity outside 0..100 in {int(bad.sum())} rows")
    dew_column = next((column for column in ("nwp_dew_point", "nwp_d2m", "era5_dew_point_c") if column in frame.columns), None)
    temp_column = next((column for column in ("nwp_t2m", "era5_t2m", "nwp_temp_2m_c") if column in frame.columns), None)
    if dew_column:
        dew = pd.to_numeric(frame[dew_column], errors="coerce")
        bad_range = dew.notna() & ~dew.between(-90.0, 60.0)
        if bad_range.any():
            warnings.append(f"sanity: dew_point Celsius range invalid in {int(bad_range.sum())} rows")
        if temp_column:
            temp = pd.to_numeric(frame[temp_column], errors="coerce")
            bad_order = dew.notna() & temp.notna() & (dew > temp + DEW_POINT_ORDER_EPSILON_C)
            if bad_order.any():
                warnings.append(f"sanity: dew_point exceeds temperature in {int(bad_order.sum())} rows")
    if "nwp_precip_probability" in frame.columns:
        pop = pd.to_numeric(frame["nwp_precip_probability"], errors="coerce")
        bad = pop.notna() & ~pop.between(0.0, 100.0)
        if bad.any():
            warnings.append(f"sanity: nwp_precip_probability outside 0..100 in {int(bad.sum())} rows")
    if "nwp_sky_code" in frame.columns:
        sky = pd.to_numeric(frame["nwp_sky_code"], errors="coerce")
        bad = sky.notna() & ~sky.isin([1, 3, 4])
        if bad.any():
            warnings.append(f"sanity: sky_code invalid in {int(bad.sum())} rows")
    if "nwp_precip_type" in frame.columns:
        pty = pd.to_numeric(frame["nwp_precip_type"], errors="coerce")
        bad = pty.notna() & ~pty.between(0, 7)
        if bad.any():
            warnings.append(f"sanity: precip_type invalid in {int(bad.sum())} rows")
    return warnings


def _expected_column_missing_rates(frame: pd.DataFrame, expected_columns: list[str]) -> dict[str, float]:
    if not expected_columns:
        return {}
    row_count = int(len(frame))
    rates: dict[str, float] = {}
    for column in expected_columns:
        if column not in frame.columns:
            rates[column] = 1.0
        elif row_count == 0:
            rates[column] = 1.0
        else:
            rates[column] = float(frame[column].isna().mean())
    return rates


def _time_span_hours(frame: pd.DataFrame) -> float:
    if "forecast_init_time" not in frame.columns or frame["forecast_init_time"].dropna().empty:
        return 0.0
    times = frame["forecast_init_time"].dropna()
    return float((times.max() - times.min()) / pd.Timedelta(hours=1))


def _has_blocked_source(frame: pd.DataFrame, paths: Iterable[str | Path]) -> bool:
    text = " ".join(str(path).lower() for path in paths)
    if "source" in frame.columns:
        text += " " + " ".join(frame["source"].astype(str).str.lower().unique())
    return any(token in text for token in BLOCKED_SOURCE_TOKENS)


def _nullable_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def _iso_or_none(value: Any) -> str | None:
    if pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and validate a prepared forecast NWP archive.")
    parser.add_argument("--input", action="append", help="Prepared forecast CSV path. Repeat for multiple files.")
    parser.add_argument("--inputs", nargs="+", help="Prepared forecast CSV paths. Alternative to repeated --input.")
    parser.add_argument("--output-csv", default="data/raw/nwp/archive/prepared_forecast_archive.csv")
    parser.add_argument("--output", dest="output_csv_alias", help="Alias for --output-csv.")
    parser.add_argument("--quality-json", default="data/raw/nwp/archive/archive_quality_report.json")
    parser.add_argument("--quality-report", dest="quality_json_alias", help="Alias for --quality-json.")
    parser.add_argument("--quality-md", default="data/raw/nwp/archive/archive_quality_report.md")
    parser.add_argument("--min-stations", type=int, default=20)
    parser.add_argument("--min-issue-cycles", type=int, default=30)
    parser.add_argument("--required-horizon", type=int, default=24)
    parser.add_argument("--min-horizon-coverage", type=float, default=0.95)
    parser.add_argument("--expected-columns", default="", help="Comma-separated forecast feature columns required for operational training.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    inputs = list(args.input or []) + list(args.inputs or [])
    if not inputs:
        raise SystemExit("At least one --input or --inputs path is required.")
    _, report = build_prepared_forecast_archive(
        inputs,
        output_csv=args.output_csv_alias or args.output_csv,
        quality_json=args.quality_json_alias or args.quality_json,
        quality_markdown=args.quality_md,
        min_stations=args.min_stations,
        min_issue_cycles=args.min_issue_cycles,
        required_horizon=args.required_horizon,
        min_horizon_coverage=args.min_horizon_coverage,
        expected_columns=[column.strip() for column in args.expected_columns.split(",") if column.strip()],
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
