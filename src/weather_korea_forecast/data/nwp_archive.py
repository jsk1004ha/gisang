from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

REQUIRED_PREPARED_FORECAST_COLUMNS = {
    "station_id",
    "forecast_init_time",
    "valid_time",
    "horizon_step",
    "nwp_t2m",
    "nwp_sp",
    "nwp_u10",
    "nwp_v10",
    "nwp_tp",
    "source",
    "issue_time",
}
BLOCKED_SOURCE_TOKENS = ("synthetic", "smoke", "fixture", "generated", "oracle", "diagnostic")
TEMP_COLUMNS = ("nwp_t2m", "nwp_dew_point", "nwp_d2m", "era5_t2m", "era5_d2m", "era5_dew_point_c")


@dataclass(frozen=True)
class ArchiveQualityReport:
    row_count: int
    station_count: int
    forecast_init_time_count: int
    issue_time_count: int
    horizon_min: int | None
    horizon_max: int | None
    horizon_1_24_coverage: float
    duplicate_count: int
    missing_required_columns: list[str]
    invalid_valid_time_count: int
    missing_horizon_count: int
    train_val_test_split_possible: bool
    encoder_prediction_window_possible: bool
    forecast_source_schema_valid: bool
    forecast_archive_adequate: bool
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
) -> tuple[pd.DataFrame, ArchiveQualityReport]:
    paths = [Path(path) for path in input_paths]
    if not paths:
        raise ValueError("At least one prepared forecast CSV path is required.")
    frames = [_read_prepared_forecast_csv(path) for path in paths]
    archive = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    normalized = normalize_prepared_forecast_archive(archive)
    report = evaluate_archive_quality(
        normalized,
        source_paths=paths,
        min_stations=min_stations,
        min_issue_cycles=min_issue_cycles,
        required_horizon=required_horizon,
        min_horizon_coverage=min_horizon_coverage,
        encoder_length=encoder_length,
        prediction_length=prediction_length,
    )
    deduped = normalized.drop_duplicates(["station_id", "forecast_init_time", "horizon_step"], keep="last")
    deduped = deduped.sort_values(["station_id", "forecast_init_time", "horizon_step", "valid_time"]).reset_index(drop=True)
    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        deduped.to_csv(output_csv, index=False)
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
) -> ArchiveQualityReport:
    missing_required = sorted(REQUIRED_PREPARED_FORECAST_COLUMNS - set(frame.columns))
    duplicate_count = _duplicate_count(frame)
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
    if duplicate_count:
        warnings.append(f"{duplicate_count} duplicate station/forecast_init_time/horizon rows will be de-duplicated")
    blocked_source = _has_blocked_source(frame, source_paths)
    if blocked_source:
        warnings.append("archive provenance contains synthetic/smoke/diagnostic/generated token")
    adequacy_checks = {
        f"station_count >= {min_stations}": station_count >= min_stations,
        f"forecast_init_time_count >= {min_issue_cycles}": forecast_init_time_count >= min_issue_cycles,
        f"horizon_1_{required_horizon}_coverage >= {min_horizon_coverage:.2f}": coverage >= min_horizon_coverage,
        "train_val_test_split_possible": split_possible,
        f"encoder_{encoder_length}_prediction_{prediction_length}_window_possible": window_possible,
        "forecast_source_schema_valid": schema_valid,
        "not synthetic/smoke/diagnostic": not blocked_source,
    }
    reasons = [name for name, ok in adequacy_checks.items() if not ok]
    return ArchiveQualityReport(
        row_count=int(len(frame)),
        station_count=station_count,
        forecast_init_time_count=forecast_init_time_count,
        issue_time_count=issue_time_count,
        horizon_min=horizon_min,
        horizon_max=horizon_max,
        horizon_1_24_coverage=float(coverage),
        duplicate_count=duplicate_count,
        missing_required_columns=missing_required,
        invalid_valid_time_count=invalid_valid_time_count,
        missing_horizon_count=missing_horizon_count,
        train_val_test_split_possible=split_possible,
        encoder_prediction_window_possible=window_possible,
        forecast_source_schema_valid=schema_valid,
        forecast_archive_adequate=not reasons,
        adequacy_reasons=reasons,
        warnings=warnings,
    )


def render_archive_quality_markdown(report: ArchiveQualityReport) -> str:
    rows = ["# Prepared Forecast Archive Quality", "", f"adequate: `{report.forecast_archive_adequate}`", ""]
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and validate a prepared forecast NWP archive.")
    parser.add_argument("--input", action="append", required=True, help="Prepared forecast CSV path. Repeat for multiple files.")
    parser.add_argument("--output-csv", default="data/raw/nwp/archive/prepared_forecast_archive.csv")
    parser.add_argument("--quality-json", default="data/raw/nwp/archive/archive_quality_report.json")
    parser.add_argument("--quality-md", default="data/raw/nwp/archive/archive_quality_report.md")
    parser.add_argument("--min-stations", type=int, default=20)
    parser.add_argument("--min-issue-cycles", type=int, default=30)
    parser.add_argument("--required-horizon", type=int, default=24)
    parser.add_argument("--min-horizon-coverage", type=float, default=0.95)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    _, report = build_prepared_forecast_archive(
        args.input,
        output_csv=args.output_csv,
        quality_json=args.quality_json,
        quality_markdown=args.quality_md,
        min_stations=args.min_stations,
        min_issue_cycles=args.min_issue_cycles,
        required_horizon=args.required_horizon,
        min_horizon_coverage=args.min_horizon_coverage,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
