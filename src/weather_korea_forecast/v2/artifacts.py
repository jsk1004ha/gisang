from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from weather_korea_forecast.utils.config import dump_yaml
from weather_korea_forecast.utils.io import write_json, write_table
from weather_korea_forecast.utils.paths import ensure_dir, timestamp_slug
from weather_korea_forecast.v2.future_features import build_future_feature_metadata


def create_experiment_dir(config: dict) -> Path:
    root_dir = ensure_dir(config["artifacts"]["root_dir"])
    experiment_name = config["experiment"]["name"]
    experiment_dir = root_dir / f"{experiment_name}_{timestamp_slug()}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir


def snapshot_config(experiment_dir: Path, config: dict) -> None:
    dump_yaml(experiment_dir / "experiment_config.yaml", config)


def write_scaler_artifact(experiment_dir: Path, scaler) -> Path:
    return write_json(scaler.to_dict(), experiment_dir / "scaler.json")


def write_experiment_summary(
    experiment_dir: Path,
    config: dict,
    metrics: dict[str, object],
    raw_metrics: dict[str, object] | None,
    val_metrics: dict[str, object] | None,
    best_val_loss: float,
    training_history: list[dict[str, object]],
) -> tuple[Path, Path]:
    data_config = config["data"]
    future_features = build_future_feature_metadata(config)
    summary = {
        "experiment_name": config["experiment"]["name"],
        "version": config["experiment"].get("version", "v2"),
        "target_name": data_config["target_name"],
        "model_name": config["model"]["name"],
        "model_type": config["model"]["type"],
        "model_family": _model_family(config["model"]["type"]),
        "artifact_profile": _artifact_profile(config),
        "encoder_length": data_config["window"]["encoder_length"],
        "prediction_length": data_config["window"]["prediction_length"],
        "train_start": _to_min_datetime(data_config["split"]),
        "train_end": data_config["split"]["train_end"],
        "val_start": data_config["split"].get("val_start"),
        "val_end": data_config["split"]["val_end"],
        "test_start": data_config["split"].get("test_start"),
        "test_end": data_config["split"]["test_end"],
        "metrics": metrics,
        "raw_metrics": raw_metrics,
        "val_metrics": val_metrics,
        "best_val_loss": best_val_loss,
        "best_epoch": _infer_best_epoch(training_history),
        "future_features": future_features,
        "forecast_track": future_features["forecast_track"],
        "track": future_features["forecast_track"],
        "uses_future_weather_features": future_features["uses_future_weather_features"],
        "uses_future_nwp_features": future_features["uses_future_nwp_features"],
        "future_feature_source": future_features["future_feature_source"],
        "operational_valid": future_features["operational_valid"],
        "leakage_risk_note": future_features.get("leakage_risk_note"),
        "backtest_only": future_features["backtest_only"],
        "forecast_schema_version": future_features.get("forecast_schema_version"),
        "forecast_schema_valid": future_features.get("forecast_schema_valid"),
        "forecast_source_schema_valid": future_features.get("forecast_source_schema_valid"),
        "forecast_source_path": future_features.get("forecast_source_path"),
        "uses_patch_features": future_features.get("uses_patch_features"),
        "patch_features_enabled": future_features.get("patch_features_enabled"),
        "patch_size": future_features.get("patch_size"),
        "patch_feature_set": future_features.get("patch_feature_set"),
        "patch_feature_mode": future_features.get("patch_feature_mode"),
        **_goal_status(config["data"]["target_name"], future_features["forecast_track"], metrics, {}, {}),
        "notes": config["experiment"].get("notes", ""),
    }
    json_path = write_json(summary, experiment_dir / "experiment_summary.json")
    markdown = _summary_markdown(summary)
    markdown_path = experiment_dir / "experiment_summary.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    return json_path, markdown_path


def update_leaderboard(experiment_dir: Path, config: dict, metrics: dict[str, object], raw_metrics: dict[str, object] | None = None) -> Path:
    leaderboard_path = Path(config["artifacts"]["leaderboard_path"])
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    split = config["data"]["split"]
    horizon_extrema = _leaderboard_horizon_extrema(experiment_dir)
    station_extrema = _leaderboard_station_extrema(experiment_dir)
    num_stations = _leaderboard_station_count(experiment_dir)
    scaling_config = config["data"].get("scaling", {})
    scaling_mode = str(scaling_config.get("mode", "global"))
    future_features = build_future_feature_metadata(config)
    row = {
        "experiment_name": config["experiment"]["name"],
        "version": config["experiment"].get("version", "v2"),
        "target": config["data"]["target_name"],
        "target_name": config["data"]["target_name"],
        "model_name": config["model"]["name"],
        "model_type": config["model"]["type"],
        "model_family": _model_family(config["model"]["type"]),
        "artifact_profile": _artifact_profile(config),
        "encoder_length": config["data"]["window"]["encoder_length"],
        "prediction_length": config["data"]["window"]["prediction_length"],
        "scaling_mode": scaling_mode,
        "scaling_group_column": scaling_config.get("group_column", "station_id"),
        "forecast_track": future_features["forecast_track"],
        "track": future_features["forecast_track"],
        "uses_future_nwp_features": future_features["uses_future_nwp_features"],
        "uses_future_weather_features": future_features["uses_future_weather_features"],
        "future_feature_source": future_features["future_feature_source"],
        "operational_valid": future_features["operational_valid"],
        "leakage_risk_note": future_features.get("leakage_risk_note"),
        "backtest_only": future_features["backtest_only"],
        "forecast_schema_version": future_features.get("forecast_schema_version"),
        "forecast_schema_valid": future_features.get("forecast_schema_valid"),
        "forecast_source_schema_valid": future_features.get("forecast_source_schema_valid"),
        "forecast_source_path": future_features.get("forecast_source_path"),
        "uses_patch_features": future_features.get("uses_patch_features"),
        "patch_features_enabled": future_features.get("patch_features_enabled"),
        "patch_size": future_features.get("patch_size"),
        "patch_feature_set": future_features.get("patch_feature_set"),
        "patch_feature_mode": future_features.get("patch_feature_mode"),
        "future_weather_feature_columns": "|".join(future_features["future_weather_feature_columns"]),
        "num_stations": num_stations,
        "train_start": split.get("train_start"),
        "train_end": split["train_end"],
        "val_start": split.get("val_start"),
        "val_end": split["val_end"],
        "test_start": split.get("test_start"),
        "test_end": split["test_end"],
        "train_period": _format_period(split.get("train_start"), split.get("train_end")),
        "val_period": _format_period(split.get("val_start"), split.get("val_end")),
        "test_period": _format_period(split.get("test_start"), split.get("test_end")),
        "rmse": metrics.get("rmse"),
        "mae": metrics.get("mae"),
        "bias": metrics.get("bias"),
        "raw_rmse": raw_metrics.get("rmse") if raw_metrics else metrics.get("rmse"),
        "corrected_rmse": metrics.get("rmse"),
        "rmse_raw": raw_metrics.get("rmse") if raw_metrics else metrics.get("rmse"),
        "rmse_corrected": metrics.get("rmse"),
        "raw_mae": raw_metrics.get("mae") if raw_metrics else metrics.get("mae"),
        "corrected_mae": metrics.get("mae"),
        "mae_raw": raw_metrics.get("mae") if raw_metrics else metrics.get("mae"),
        "mae_corrected": metrics.get("mae"),
        "raw_bias": raw_metrics.get("bias") if raw_metrics else metrics.get("bias"),
        "corrected_bias": metrics.get("bias"),
        "bias_raw": raw_metrics.get("bias") if raw_metrics else metrics.get("bias"),
        "bias_corrected": metrics.get("bias"),
        "best_horizon": horizon_extrema.get("best_horizon"),
        "worst_horizon": horizon_extrema.get("worst_horizon"),
        "best_horizon_rmse": horizon_extrema.get("best_horizon_rmse"),
        "worst_horizon_rmse": horizon_extrema.get("worst_horizon_rmse"),
        "best_station": station_extrema.get("best_station"),
        "worst_station": station_extrema.get("worst_station"),
        "best_station_rmse": station_extrema.get("best_station_rmse"),
        "worst_station_rmse": station_extrema.get("worst_station_rmse"),
        "mape": metrics.get("mape"),
        "daily_max_temp_mae": metrics.get("daily_max_temp_mae"),
        "daily_min_temp_mae": metrics.get("daily_min_temp_mae"),
        "diurnal_range_mae": metrics.get("diurnal_range_mae"),
        "diurnal_range_bias": metrics.get("diurnal_range_bias"),
        "daily_score": metrics.get("daily_score"),
        **_goal_status(config["data"]["target_name"], future_features["forecast_track"], metrics, horizon_extrema, station_extrema),
        "notes": config["experiment"].get("notes", ""),
        "experiment_dir": str(experiment_dir),
    }
    if leaderboard_path.exists():
        leaderboard = pd.read_csv(leaderboard_path)
        row_frame = pd.DataFrame([row])
        for column in leaderboard.columns:
            if column not in row_frame.columns:
                row_frame[column] = pd.NA
        for column in row_frame.columns:
            if column not in leaderboard.columns:
                leaderboard[column] = pd.NA
        leaderboard = pd.concat([leaderboard, row_frame[leaderboard.columns]], ignore_index=True)
    else:
        leaderboard = pd.DataFrame([row])
    leaderboard = leaderboard.sort_values(["target_name", "rmse_corrected", "mae_corrected"], na_position="last").reset_index(drop=True)
    write_table(leaderboard, leaderboard_path)
    for target_name, target_frame in leaderboard.groupby("target_name", dropna=False):
        if pd.isna(target_name):
            continue
        write_table(target_frame.reset_index(drop=True), leaderboard_path.with_name(f"leaderboard_{target_name}.csv"))
    if "forecast_track" in leaderboard.columns:
        for track_name, track_frame in leaderboard.groupby("forecast_track", dropna=False):
            if pd.isna(track_name):
                continue
            safe_track = str(track_name).replace("/", "_").replace("\\", "_").replace(" ", "_")
            write_table(track_frame.reset_index(drop=True), leaderboard_path.with_name(f"leaderboard_{safe_track}.csv"))
    _write_v3_umbrella_leaderboards(leaderboard, leaderboard_path)
    return leaderboard_path


def refresh_aliases(experiment_dir: Path) -> None:
    _refresh_alias_pointer(experiment_dir, alias_name="latest", manifest_key="latest_experiment")
    best_dir = experiment_dir.parent / "best"
    current_summary = _read_summary(experiment_dir / "experiment_summary.json")
    previous_summary = _read_summary(best_dir / "experiment_summary.json") if best_dir.exists() else None
    if previous_summary is None or _is_better_experiment(current_summary, previous_summary):
        _refresh_alias_pointer(experiment_dir, alias_name="best", manifest_key="best_experiment")


def write_feature_importance(experiment_dir: Path, frame: pd.DataFrame | None) -> Path | None:
    if frame is None or frame.empty:
        return None
    return write_table(frame, experiment_dir / "feature_importance.csv")


def _refresh_alias_pointer(experiment_dir: Path, alias_name: str, manifest_key: str) -> None:
    alias_dir = experiment_dir.parent / alias_name
    alias_dir.mkdir(parents=True, exist_ok=True)
    write_json({manifest_key: experiment_dir.name}, alias_dir / "manifest.json")
    for source_name in (
        "predictions_test.csv",
        "metrics_test.json",
        "metrics_summary.json",
        "experiment_summary.json",
        "experiment_summary.md",
        "forecast_vs_actual.png",
        "horizon_error.png",
        "prediction_scatter.png",
        "raw_vs_corrected.png",
        "experiment_config.yaml",
        "model.pt",
        "training_history.json",
        "bias_correction.json",
        "scaler.json",
        "future_feature_metadata.json",
        "feature_importance.csv",
        "horizon_model_metrics.csv",
        "predictions_test_components.csv",
        "worst_case_samples.csv",
        "worst_case_summary.json",
        "metrics_target_name.csv",
        "metrics_target_name_horizon_step.csv",
        "metrics_target_name_station_id.csv",
        "metrics_target_name_region.csv",
        "metrics_target_name_season.csv",
        "metrics_daily_target.csv",
        "daily_target_errors.csv",
        "horizon_station_heatmap.png",
        "station_rmse_bar.png",
        "region_rmse_bar.png",
        "daily_max_min_error.png",
        "extreme_target_scatter.png",
        "metrics_humidity_extremes.csv",
        "metrics_target_name_rolling_origin_fold.csv",
        "metrics_raw_target_name.csv",
        "metrics_raw_target_name_horizon_step.csv",
        "metrics_raw_target_name_station_id.csv",
        "metrics_raw_target_name_region.csv",
        "metrics_raw_target_name_season.csv",
    ):
        source = experiment_dir / source_name
        if source.exists():
            (alias_dir / source_name).write_bytes(source.read_bytes())


def _summary_markdown(summary: dict[str, object]) -> str:
    metrics = dict(summary.get("metrics", {}))
    future_features = dict(summary.get("future_features") or {})
    lines = [
        f"# {summary['experiment_name']}",
        "",
        f"- Version: {summary['version']}",
        f"- Target: {summary['target_name']}",
        f"- Model: {summary['model_name']} ({summary['model_type']})",
        f"- Model family: {summary.get('model_family', 'unknown')}",
        f"- Window: encoder={summary['encoder_length']} / prediction={summary['prediction_length']}",
        f"- Forecast track: {summary.get('forecast_track', 'observation_only')}",
        f"- Uses future NWP/weather features: {summary.get('uses_future_weather_features', summary.get('uses_future_nwp_features', False))}",
        f"- Future feature source: {summary.get('future_feature_source', 'none')}",
        f"- Operational valid: {summary.get('operational_valid', False)}",
        f"- Backtest only: {summary.get('backtest_only', False)}",
        f"- Leakage risk note: {summary.get('leakage_risk_note') or future_features.get('leakage_risk_note', '')}",
        f"- RMSE: {_fmt(metrics.get('rmse'))}",
        f"- MAE: {_fmt(metrics.get('mae'))}",
        f"- Bias: {_fmt(metrics.get('bias'))}",
        f"- MAPE: {_fmt(metrics.get('mape'))}",
        f"- Daily max temp MAE: {_fmt(metrics.get('daily_max_temp_mae'))}",
        f"- Daily min temp MAE: {_fmt(metrics.get('daily_min_temp_mae'))}",
        f"- Diurnal range MAE: {_fmt(metrics.get('diurnal_range_mae'))}",
        f"- Diurnal range Bias: {_fmt(metrics.get('diurnal_range_bias'))}",
        f"- Raw RMSE: {_fmt(dict(summary.get('raw_metrics') or {}).get('rmse'))}",
        f"- Raw MAE: {_fmt(dict(summary.get('raw_metrics') or {}).get('mae'))}",
        f"- Raw Bias: {_fmt(dict(summary.get('raw_metrics') or {}).get('bias'))}",
        f"- RMSE goal: {_fmt(summary.get('rmse_goal'))}",
        f"- RMSE goal met: {summary.get('rmse_goal_met')}",
        f"- RMSE gap to goal: {_fmt(summary.get('rmse_gap_to_goal'))}",
        f"- Worst horizon goal met: {summary.get('worst_horizon_goal_met')}",
        f"- Worst station goal met: {summary.get('worst_station_goal_met')}",
        f"- Best val loss: {_fmt(summary.get('best_val_loss'))}",
        f"- Best epoch: {summary.get('best_epoch')}",
    ]
    if summary.get("rmse_goal_met") is False:
        lines.extend(["", "## Next improvement suggestions", "", "- Tune residual learner hyperparameters on validation only.", "- Compare 72h vs 168h encoders and horizon-wise residual heads.", "- Try ridge/LGBM ensemble weights from validation predictions."])
    warnings = future_features.get("warnings") or []
    if warnings:
        lines.extend(["", "## Future feature warnings", ""])
        lines.extend([f"- {warning}" for warning in warnings])
    notes = str(summary.get("notes", "")).strip()
    if notes:
        lines.extend(["", "## Notes", "", notes])
    return "\n".join(lines) + "\n"


def _infer_best_epoch(history: list[dict[str, object]]) -> int | None:
    best_epoch = None
    best_val_loss = None
    for row in history:
        if "val_loss" not in row:
            continue
        try:
            val_loss = float(row["val_loss"])
        except (TypeError, ValueError):
            continue
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            epoch_value = row.get("epoch")
            best_epoch = int(epoch_value) if isinstance(epoch_value, (int, float)) else None
    return best_epoch


def _to_min_datetime(split_config: dict[str, object]) -> str | None:
    return str(split_config.get("train_start")) if split_config.get("train_start") else None


def _read_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _is_better_experiment(current_summary: dict, previous_summary: dict) -> bool:
    current_val = current_summary.get("best_val_loss")
    previous_val = previous_summary.get("best_val_loss")
    if _is_finite_number(current_val) and _is_finite_number(previous_val):
        return float(current_val) < float(previous_val)
    current_rmse = current_summary.get("metrics", {}).get("rmse")
    previous_rmse = previous_summary.get("metrics", {}).get("rmse")
    if _is_finite_number(current_rmse) and _is_finite_number(previous_rmse):
        return float(current_rmse) < float(previous_rmse)
    return previous_summary is None


def _is_finite_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _fmt(value) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _goal_status(
    target_name: object,
    track: object,
    metrics: dict[str, object],
    horizon_extrema: dict[str, object],
    station_extrema: dict[str, object],
) -> dict[str, object]:
    target = str(target_name)
    forecast_track = str(track or "observation_only")
    if target == "temp" and forecast_track == "nwp_assisted_mos":
        rmse_goal = 1.0
        worst_horizon_goal = 1.2
        worst_station_goal = 1.4
        bias_goal = 0.2
    elif target == "temp" and forecast_track == "observation_only":
        rmse_goal = 2.0
        worst_horizon_goal = None
        worst_station_goal = None
        bias_goal = None
    elif target == "humidity":
        rmse_goal = 10.0
        worst_horizon_goal = None
        worst_station_goal = None
        bias_goal = 3.0
    else:
        rmse_goal = None
        worst_horizon_goal = None
        worst_station_goal = None
        bias_goal = None
    rmse = _float_or_none(metrics.get("rmse"))
    bias = _float_or_none(metrics.get("bias"))
    worst_horizon_rmse = _float_or_none(horizon_extrema.get("worst_horizon_rmse"))
    worst_station_rmse = _float_or_none(station_extrema.get("worst_station_rmse"))
    return {
        "rmse_goal": rmse_goal,
        "rmse_goal_met": None if rmse_goal is None or rmse is None else bool(rmse <= rmse_goal),
        "rmse_gap_to_goal": None if rmse_goal is None or rmse is None else float(rmse - rmse_goal),
        "worst_horizon_goal_met": None if worst_horizon_goal is None or worst_horizon_rmse is None else bool(worst_horizon_rmse <= worst_horizon_goal),
        "worst_station_goal_met": None if worst_station_goal is None or worst_station_rmse is None else bool(worst_station_rmse <= worst_station_goal),
        "bias_goal_met": None if bias_goal is None or bias is None else bool(abs(bias) <= bias_goal),
    }


def _float_or_none(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _model_family(model_type: object) -> str:
    normalized = str(model_type).strip().lower().replace("-", "_")
    if "ensemble" in normalized:
        return "ensemble"
    if "ridge" in normalized:
        return "ridge"
    if "lightgbm" in normalized or normalized == "lgbm":
        return "lightgbm"
    if "catboost" in normalized:
        return "catboost"
    if normalized == "tft":
        return "tft"
    if "persistence" in normalized:
        return "persistence"
    if normalized == "residual":
        return "residual"
    if "decoder_feature" in normalized or "future_feature" in normalized:
        return "decoder_feature"
    return normalized or "unknown"


def _artifact_profile(config: dict) -> str:
    artifacts = config.get("artifacts", {})
    profile = str(artifacts.get("profile") or artifacts.get("artifact_profile") or "full").strip().lower().replace("-", "_")
    if profile in {"minimal", "slim", "lean", "report_only"}:
        return "minimal"
    return "full"


def _write_v3_umbrella_leaderboards(leaderboard: pd.DataFrame, leaderboard_path: Path) -> None:
    """Write stable V3 track files independent of exact forecast_track names."""

    if "uses_future_weather_features" not in leaderboard.columns:
        return
    uses_future = leaderboard["uses_future_weather_features"].map(_truthy)
    observation_only = leaderboard.loc[~uses_future].reset_index(drop=True)
    nwp_assisted = leaderboard.loc[uses_future].reset_index(drop=True)
    if not observation_only.empty:
        write_table(observation_only, leaderboard_path.with_name("leaderboard_observation_only.csv"))
    if not nwp_assisted.empty:
        write_table(nwp_assisted, leaderboard_path.with_name("leaderboard_nwp_assisted.csv"))


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _format_period(start, end) -> str:
    return f"{start or 'begin'}..{end or 'open'}"


def _leaderboard_horizon_extrema(experiment_dir: Path) -> dict[str, object]:
    horizon_path = experiment_dir / "metrics_target_name_horizon_step.csv"
    if not horizon_path.exists():
        return {}
    try:
        horizon_metrics = pd.read_csv(horizon_path)
    except Exception:
        return {}
    if horizon_metrics.empty or "horizon_step" not in horizon_metrics.columns or "rmse" not in horizon_metrics.columns:
        return {}
    best_row = horizon_metrics.sort_values("rmse", ascending=True).iloc[0]
    worst_row = horizon_metrics.sort_values("rmse", ascending=False).iloc[0]
    return {
        "best_horizon": int(best_row["horizon_step"]),
        "best_horizon_rmse": float(best_row["rmse"]),
        "worst_horizon": int(worst_row["horizon_step"]),
        "worst_horizon_rmse": float(worst_row["rmse"]),
    }


def _leaderboard_station_extrema(experiment_dir: Path) -> dict[str, object]:
    station_path = experiment_dir / "metrics_target_name_station_id.csv"
    if not station_path.exists():
        return {}
    try:
        station_metrics = pd.read_csv(station_path)
    except Exception:
        return {}
    if station_metrics.empty or "station_id" not in station_metrics.columns or "rmse" not in station_metrics.columns:
        return {}
    best_row = station_metrics.sort_values("rmse", ascending=True).iloc[0]
    worst_row = station_metrics.sort_values("rmse", ascending=False).iloc[0]
    return {
        "best_station": str(best_row["station_id"]),
        "best_station_rmse": float(best_row["rmse"]),
        "worst_station": str(worst_row["station_id"]),
        "worst_station_rmse": float(worst_row["rmse"]),
    }


def _leaderboard_station_count(experiment_dir: Path) -> int | None:
    predictions_path = experiment_dir / "predictions_test.csv"
    if not predictions_path.exists():
        return None
    try:
        predictions = pd.read_csv(predictions_path, usecols=["station_id"])
    except Exception:
        return None
    if "station_id" not in predictions.columns:
        return None
    return int(predictions["station_id"].nunique())
