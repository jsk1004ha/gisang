from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from weather_korea_forecast.reporting.schema import CSV_COLUMNS, PLOT_FILES, ExperimentRecord


def find_experiment_dirs(root: Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted({path.parent for path in root.rglob("experiment_summary.json")})


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {"value": payload}


def load_yaml_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {"value": payload}


def collect_experiment(exp_dir: Path) -> ExperimentRecord:
    exp_dir = Path(exp_dir)
    warnings: list[str] = []
    try:
        summary = load_json_if_exists(exp_dir / "experiment_summary.json")
    except Exception as exc:
        return ExperimentRecord(experiment_name=exp_dir.name, artifact_dir=str(exp_dir), complete=False, error=f"summary_json_error: {exc}")
    metrics = _metrics_from_files(exp_dir, summary, warnings)
    raw_metrics = _raw_metrics_from_files(exp_dir, summary, warnings)
    config = _load_experiment_config(exp_dir)
    bias = load_json_if_exists(exp_dir / "bias_correction.json")
    worst = load_json_if_exists(exp_dir / "worst_case_summary.json")
    future = dict(summary.get("future_features") or config.get("data", {}).get("future_features") or config.get("future_features") or {})
    future_source_hint = str(summary.get("future_feature_source") or future.get("future_feature_source") or future.get("source") or "none")
    target_name = _target_name(summary, config)
    track = _normalize_track(
        str(
            summary.get("track")
            or summary.get("forecast_track")
            or future.get("track")
            or future.get("forecast_track")
            or "observation_only"
        ),
        target_name=target_name,
        future_source=future_source_hint,
    )
    if track == "observation_only" and target_name == "temp":
        lower_name = f"{exp_dir.name} {summary.get('experiment_name', '')} {config.get('experiment', {}).get('name', '')}".lower()
        if future_source_hint not in {"", "none"} or "future_era5" in lower_name or "mos" in lower_name:
            track = "nwp_assisted_mos"
    rmse_goal = _rmse_goal(target_name, track)
    rmse = _float(metrics.get("rmse"))
    pred_path = exp_dir / "predictions_test.csv"
    if not pred_path.exists():
        warnings.append("predictions_test.csv missing")
    for plot in PLOT_FILES:
        if not (exp_dir / plot).exists():
            warnings.append(f"{plot} missing")
    future_source_for_warning = str(summary.get("future_feature_source") or future.get("future_feature_source") or future.get("source") or "")
    if future_source_for_warning in {"era5_reanalysis", "era5_reanalysis_backtest"}:
        warnings.append("ERA5 reanalysis future features are backtest-only")
    if summary.get("operational_valid") is False or future.get("operational_valid") is False:
        warnings.append("operational_valid=false")
    if _unknown_region(exp_dir):
        warnings.append("region_class contains unknown")
    if bias and not bool(bias.get("enabled", False)) and bias.get("disabled_reason"):
        warnings.append(f"bias correction disabled: {bias.get('disabled_reason')}")
    sample_count = _sample_count(exp_dir, metrics)
    if sample_count is not None and sample_count < 24:
        warnings.append("test sample count is very small")
    if target_name == "temp" and (metrics.get("mape") is not None):
        warnings.append("temperature MAPE unreliable near zero")
    if not metrics:
        warnings.append("metrics file missing or empty")
    operational_valid = _bool(summary.get("operational_valid", future.get("operational_valid")))
    backtest_only = _bool(summary.get("backtest_only", future.get("backtest_only")))
    if backtest_only is None and future_source_for_warning in {"era5_reanalysis", "era5_reanalysis_backtest"}:
        backtest_only = True
    if operational_valid is None and backtest_only is True:
        operational_valid = False
    is_alias_artifact = _is_alias_artifact(exp_dir)
    run_timestamp = _run_timestamp(exp_dir)
    experiment_name = str(summary.get("experiment_name") or config.get("experiment", {}).get("name") or exp_dir.name)
    model_type = str(summary.get("model_type") or config.get("model", {}).get("type") or "unknown")
    is_diagnostic = _is_diagnostic(
        track=track,
        model_type=model_type,
        future_feature_source=future_source_hint,
        rmse=rmse,
        backtest_only=backtest_only,
        exp_dir=exp_dir,
        summary=summary,
    )
    goal_eligible = bool(not is_diagnostic and rmse_goal is not None and rmse is not None)
    record = ExperimentRecord(
        experiment_name=experiment_name,
        version=_infer_version(exp_dir, summary, config),
        track=track,
        target_name=target_name,
        model_name=str(summary.get("model_name") or config.get("model", {}).get("name") or "unknown"),
        model_type=model_type,
        encoder_length=_int(summary.get("encoder_length") or config.get("data", {}).get("window", {}).get("encoder_length") or config.get("window", {}).get("encoder_length")),
        prediction_length=_int(summary.get("prediction_length") or config.get("data", {}).get("window", {}).get("prediction_length") or config.get("window", {}).get("prediction_length")),
        train_start=_str(summary.get("train_start") or config.get("split", {}).get("train_start") or config.get("data", {}).get("split", {}).get("train_start")),
        train_end=_str(summary.get("train_end") or config.get("split", {}).get("train_end") or config.get("data", {}).get("split", {}).get("train_end")),
        val_start=_str(summary.get("val_start") or config.get("split", {}).get("val_start") or config.get("data", {}).get("split", {}).get("val_start")),
        val_end=_str(summary.get("val_end") or config.get("split", {}).get("val_end") or config.get("data", {}).get("split", {}).get("val_end")),
        test_start=_str(summary.get("test_start") or config.get("split", {}).get("test_start") or config.get("data", {}).get("split", {}).get("test_start")),
        test_end=_str(summary.get("test_end") or config.get("split", {}).get("test_end") or config.get("data", {}).get("split", {}).get("test_end")),
        rmse=rmse, mae=_float(metrics.get("mae")), bias=_float(metrics.get("bias")), mape=_float(metrics.get("mape")),
        raw_rmse=_float(raw_metrics.get("rmse")), raw_mae=_float(raw_metrics.get("mae")), raw_bias=_float(raw_metrics.get("bias")),
        val_rmse=_float((summary.get("val_metrics") or {}).get("rmse")), best_val_loss=_float(summary.get("best_val_loss")), best_epoch=_int(summary.get("best_epoch")),
        sample_count=sample_count,
        worst_horizon_step=_int(summary.get("worst_horizon") or (worst.get("worst_horizon_step") or worst.get("worst_horizon") or {}).get("horizon_step")),
        worst_horizon_rmse=_breakdown_rmse(exp_dir, "metrics_target_name_horizon_step.csv", "horizon_step"),
        worst_horizon_mae=_breakdown_metric(exp_dir, "metrics_target_name_horizon_step.csv", "mae"),
        worst_horizon_bias=_breakdown_metric(exp_dir, "metrics_target_name_horizon_step.csv", "bias"),
        worst_station_id=_breakdown_key(exp_dir, "metrics_target_name_station_id.csv", "station_id"),
        worst_station_rmse=_breakdown_rmse(exp_dir, "metrics_target_name_station_id.csv", "station_id"),
        worst_region=_breakdown_key(exp_dir, "metrics_target_name_region.csv", "region"),
        worst_region_rmse=_breakdown_rmse(exp_dir, "metrics_target_name_region.csv", "region"),
        worst_season=_breakdown_key(exp_dir, "metrics_target_name_season.csv", "season"),
        uses_future_weather_features=_bool(summary.get("uses_future_weather_features", future.get("uses_future_weather_features"))),
        future_feature_source=future_source_hint,
        operational_valid=operational_valid,
        backtest_only=backtest_only,
        leakage_risk_note=_str(summary.get("leakage_risk_note") or future.get("leakage_risk_note")),
        bias_correction_enabled=_bool(bias.get("enabled")) if bias else None,
        bias_correction_mode=_str(bias.get("mode")) if bias else None,
        bias_correction_method=_str(bias.get("method")) if bias else None,
        bias_correction_accepted=_bool((bias.get("selection") or {}).get("accepted")) if bias else None,
        rmse_goal=rmse_goal,
        rmse_goal_met=None if not goal_eligible else bool(rmse <= rmse_goal),
        rmse_gap_to_goal=None if rmse_goal is None or rmse is None else float(rmse - rmse_goal),
        goal_eligible=goal_eligible,
        is_diagnostic=is_diagnostic,
        is_alias_artifact=is_alias_artifact,
        canonical_experiment_id=_canonical_experiment_id(experiment_name),
        run_timestamp=run_timestamp,
        is_representative_run=not is_alias_artifact,
        included_in_main_leaderboard=False,
        artifact_dir=str(exp_dir),
        created_at_or_modified_at=_mtime(exp_dir),
        complete=bool(summary and pred_path.exists()),
        warnings=warnings,
    )
    for plot in PLOT_FILES:
        attr = plot.replace(".png", "_path")
        if hasattr(record, attr):
            setattr(record, attr, str(exp_dir / plot) if (exp_dir / plot).exists() else None)
    return record


def collect_all(root: Path) -> list[ExperimentRecord]:
    return _mark_main_leaderboard_flags(_mark_representative_runs([collect_experiment(path) for path in find_experiment_dirs(root)]))


def _load_experiment_config(exp_dir: Path) -> dict[str, Any]:
    """Load V2/V3 unified config, or merge V1 snapshot configs into one view."""

    config = load_yaml_if_exists(exp_dir / "experiment_config.yaml")
    if config:
        return config
    data_config = load_yaml_if_exists(exp_dir / "data_config.yaml")
    model_config = load_yaml_if_exists(exp_dir / "model_config.yaml")
    train_config = load_yaml_if_exists(exp_dir / "train_config.yaml")
    merged: dict[str, Any] = {}
    if data_config:
        merged.update(data_config)
        merged["data_config"] = data_config
    if model_config:
        merged["model"] = model_config.get("model", model_config)
        merged["model_config"] = model_config
    if train_config:
        merged["experiment"] = train_config.get("experiment", {})
        merged["training"] = train_config.get("training", {})
        merged["train_config"] = train_config
    return merged


def _target_name(summary: dict[str, Any], config: dict[str, Any]) -> str:
    target = summary.get("target_name") or config.get("data", {}).get("target_name")
    if target:
        return str(target)
    targets = config.get("targets")
    if isinstance(targets, list) and targets:
        return str(targets[0]) if len(targets) == 1 else "multi"
    target_columns = config.get("model", {}).get("target_columns")
    if isinstance(target_columns, list) and target_columns:
        names = [str(col).replace("target_", "", 1) for col in target_columns]
        return names[0] if len(set(names)) == 1 else "multi"
    by_target = (summary.get("metrics") or {}).get("by_target")
    if isinstance(by_target, dict) and by_target:
        return next(iter(by_target)) if len(by_target) == 1 else "multi"
    return "unknown"


def _normalize_track(track: str, *, target_name: str, future_source: str) -> str:
    normalized = track.strip() or "observation_only"
    if normalized == "nwp_assisted" and target_name == "temp":
        return "nwp_assisted_mos"
    if normalized == "observation_only" and future_source not in {"", "none"} and target_name == "temp":
        return "nwp_assisted_mos"
    return normalized


def _infer_version(exp_dir: Path, summary: dict[str, Any], config: dict[str, Any]) -> str:
    explicit = summary.get("version") or config.get("experiment", {}).get("version")
    if explicit:
        return str(explicit)
    parts = {part.lower() for part in exp_dir.parts}
    name = exp_dir.name.lower()
    if "v3_experiments" in parts or name.startswith("v3_"):
        return "v3"
    if "v2_experiments" in parts or name.startswith("v2_"):
        return "v2"
    if "experiments" in parts:
        return "v1"
    return "unknown"



def main_leaderboard_records(records: list[ExperimentRecord]) -> list[ExperimentRecord]:
    """Rows eligible for main KPI, charts, leaderboard, and best-model ranking."""

    return [
        record
        for record in records
        if record.included_in_main_leaderboard
        or (
            record.complete
            and record.rmse is not None
            and not record.is_diagnostic
            and not record.is_alias_artifact
            and record.is_representative_run
            and float(record.rmse) > 0.0
        )
    ]


def diagnostic_records(records: list[ExperimentRecord]) -> list[ExperimentRecord]:
    return [record for record in records if record.is_diagnostic]


def _mark_representative_runs(records: list[ExperimentRecord]) -> list[ExperimentRecord]:
    groups: dict[tuple[str, str, str, str, float | None, float | None, float | None], list[ExperimentRecord]] = {}
    for record in records:
        key = (
            record.canonical_experiment_id or _canonical_experiment_id(record.experiment_name),
            record.version,
            record.target_name,
            record.track,
            _rounded(record.rmse),
            _rounded(record.mae),
            _rounded(record.bias),
        )
        groups.setdefault(key, []).append(record)
    for group in groups.values():
        canonical = [record for record in group if not record.is_alias_artifact]
        candidates = canonical or group
        representative = max(candidates, key=_representative_sort_key)
        for record in group:
            record.is_representative_run = record is representative and not record.is_alias_artifact
    return records


def _mark_main_leaderboard_flags(records: list[ExperimentRecord]) -> list[ExperimentRecord]:
    for record in records:
        record.included_in_main_leaderboard = bool(
            record.complete
            and record.rmse is not None
            and not record.is_diagnostic
            and not record.is_alias_artifact
            and record.is_representative_run
            and float(record.rmse) > 0.0
        )
    return records


def _representative_sort_key(record: ExperimentRecord) -> tuple[str, str]:
    return (record.run_timestamp or "", record.created_at_or_modified_at or "")


def _rounded(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 9)


def _is_alias_artifact(exp_dir: Path) -> bool:
    return exp_dir.name.lower() in {"best", "latest"}


def _run_timestamp(exp_dir: Path) -> str | None:
    match = re.search(r"(20\d{6}T\d{6}Z)", exp_dir.name)
    return match.group(1) if match else None


def _canonical_experiment_id(experiment_name: str) -> str:
    return re.sub(r"_20\d{6}T\d{6}Z$", "", str(experiment_name))


def _is_diagnostic(
    *,
    track: str,
    model_type: str,
    future_feature_source: str,
    rmse: float | None,
    backtest_only: bool | None,
    exp_dir: Path,
    summary: dict[str, Any],
) -> bool:
    haystack = " ".join(
        str(value).lower()
        for value in (
            track,
            model_type,
            future_feature_source,
            exp_dir.name,
            summary.get("experiment_name", ""),
            summary.get("notes", ""),
        )
    )
    return bool(
        "oracle" in str(track).lower()
        or "oracle" in str(future_feature_source).lower()
        or "oracle" in exp_dir.name.lower()
        or "diagnostic" in haystack
        or model_type == "decoder_feature_baseline"
        or future_feature_source == "observed_target_oracle"
        or (rmse is not None and float(rmse) == 0.0 and backtest_only is True)
    )

def write_summary_csv(records: list[ExperimentRecord], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([record.to_dict() for record in records])
    if frame.empty:
        frame = pd.DataFrame(columns=CSV_COLUMNS)
    else:
        frame = frame.reindex(columns=CSV_COLUMNS)
        frame = frame.sort_values(["target_name", "track", "rmse"], na_position="last")
    frame.to_csv(output_path, index=False)
    return output_path


def write_summary_json(records: list[ExperimentRecord], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([r.to_dict() for r in records], ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def write_best_models_csv(records: list[ExperimentRecord], output_path: Path) -> Path:
    rows = _best_model_rows(main_leaderboard_records(records))
    if rows:
        best = pd.DataFrame(rows)
        ordered_columns = ["best_model_category", *CSV_COLUMNS]
        best = best.reindex(columns=ordered_columns)
    else:
        best = pd.DataFrame(columns=["best_model_category", *CSV_COLUMNS])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best.to_csv(output_path, index=False)
    return output_path


def _best_model_rows(records: list[ExperimentRecord]) -> list[dict[str, Any]]:
    categories = {
        "best_temp_observation_only": lambda r: r.target_name == "temp" and r.track == "observation_only",
        "best_temp_nwp_assisted": lambda r: r.target_name == "temp" and r.track == "nwp_assisted_mos",
        "best_humidity_observation_only": lambda r: r.target_name == "humidity" and r.track == "observation_only",
        "best_humidity_nwp_assisted": lambda r: r.target_name == "humidity" and r.track == "nwp_assisted_mos",
        "best_operational_valid": lambda r: r.operational_valid is True,
        "best_backtest_only": lambda r: r.backtest_only is True,
        "best_tft": lambda r: "tft" in r.model_type.lower(),
        "best_ridge": lambda r: "ridge" in r.model_type.lower(),
        "best_lgbm": lambda r: "lightgbm" in r.model_type.lower() or "lgbm" in r.model_type.lower(),
    }
    rows: list[dict[str, Any]] = []
    for category, predicate in categories.items():
        candidates = [record for record in records if record.rmse is not None and predicate(record)]
        if not candidates:
            continue
        row = min(candidates, key=lambda record: record.rmse or float("inf")).to_dict()
        row["best_model_category"] = category
        rows.append(row)
    return rows


def write_failed_csv(records: list[ExperimentRecord], output_path: Path) -> Path:
    frame = pd.DataFrame([r.to_dict() for r in records if not r.complete or r.error])
    if frame.empty:
        frame = pd.DataFrame(columns=CSV_COLUMNS)
    else:
        frame = frame.reindex(columns=CSV_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def _metrics_from_files(exp_dir: Path, summary: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    metrics = dict(summary.get("metrics") or {})
    for filename in ["metrics_summary.json", "metrics_test.json"]:
        path = exp_dir / filename
        if not path.exists():
            warnings.append(f"{filename} missing")
            continue
        try:
            payload = load_json_if_exists(path)
            if "metrics" in payload and isinstance(payload["metrics"], dict):
                metrics = {**payload["metrics"], **metrics}
            else:
                metrics = {**payload, **metrics}
        except Exception as exc:
            warnings.append(f"{filename} parse failed: {exc}")
    return metrics


def _raw_metrics_from_files(exp_dir: Path, summary: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    raw = dict(summary.get("raw_metrics") or {})
    path = exp_dir / "metrics_test.json"
    if not path.exists():
        return raw
    try:
        payload = load_json_if_exists(path)
    except Exception as exc:
        warnings.append(f"metrics_test.json raw_metrics parse failed: {exc}")
        return raw
    if isinstance(payload.get("raw_metrics"), dict):
        return {**payload["raw_metrics"], **raw}
    return raw


def _rmse_goal(target: str, track: str) -> float | None:
    if target == "temp" and track == "nwp_assisted_mos":
        return 1.0
    if target == "temp" and track == "observation_only":
        return 2.0
    if target == "humidity":
        return 10.0
    return None


def _breakdown_rmse(exp_dir: Path, filename: str, key: str) -> float | None:
    row = _worst_breakdown_row(exp_dir, filename)
    return _float(row.get("rmse")) if row else None


def _breakdown_metric(exp_dir: Path, filename: str, metric: str) -> float | None:
    row = _worst_breakdown_row(exp_dir, filename)
    return _float(row.get(metric)) if row else None


def _breakdown_key(exp_dir: Path, filename: str, key: str) -> str | None:
    row = _worst_breakdown_row(exp_dir, filename)
    if not row:
        return None
    if key in row:
        return _str(row.get(key))
    if key == "region" and "region_class" in row:
        return _str(row.get("region_class"))
    return None


def _worst_breakdown_row(exp_dir: Path, filename: str) -> dict[str, Any] | None:
    path = exp_dir / filename
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    if frame.empty or "rmse" not in frame.columns:
        return None
    return frame.sort_values("rmse", ascending=False).iloc[0].to_dict()


def _sample_count(exp_dir: Path, metrics: dict[str, Any]) -> int | None:
    if metrics.get("sample_count") is not None:
        return _int(metrics.get("sample_count"))
    path = exp_dir / "predictions_test.csv"
    if not path.exists():
        return None
    try:
        return int(sum(1 for _ in path.open("r", encoding="utf-8")) - 1)
    except Exception:
        return None


def _unknown_region(exp_dir: Path) -> bool:
    path = exp_dir / "metrics_target_name_region.csv"
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path)
    except Exception:
        return False
    return bool(any(frame.astype(str).apply(lambda col: col.str.lower().eq("unknown")).any()))


def _mtime(path: Path) -> str:
    import datetime as dt
    return dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc).isoformat()

def _float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None

def _int(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(float(value))
    except Exception:
        return None

def _bool(value: Any) -> bool | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}

def _str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
