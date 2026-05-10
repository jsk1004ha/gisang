from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from weather_korea_forecast.utils.config import dump_yaml
from weather_korea_forecast.utils.io import write_json, write_table
from weather_korea_forecast.utils.paths import ensure_dir, timestamp_slug


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
    summary = {
        "experiment_name": config["experiment"]["name"],
        "version": config["experiment"].get("version", "v2"),
        "target_name": data_config["target_name"],
        "model_name": config["model"]["name"],
        "model_type": config["model"]["type"],
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
    num_stations = _leaderboard_station_count(experiment_dir)
    scaling_config = config["data"].get("scaling", {})
    scaling_mode = str(scaling_config.get("mode", "global"))
    row = {
        "experiment_name": config["experiment"]["name"],
        "version": config["experiment"].get("version", "v2"),
        "target": config["data"]["target_name"],
        "target_name": config["data"]["target_name"],
        "model_name": config["model"]["name"],
        "model_type": config["model"]["type"],
        "encoder_length": config["data"]["window"]["encoder_length"],
        "prediction_length": config["data"]["window"]["prediction_length"],
        "scaling_mode": scaling_mode,
        "scaling_group_column": scaling_config.get("group_column", "station_id"),
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
        "mape": metrics.get("mape"),
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
        "metrics_daily_temperature.csv",
        "daily_temperature_errors.csv",
        "horizon_station_heatmap.png",
        "station_rmse_bar.png",
        "region_rmse_bar.png",
        "daily_max_min_error.png",
        "extreme_temperature_scatter.png",
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
    lines = [
        f"# {summary['experiment_name']}",
        "",
        f"- Version: {summary['version']}",
        f"- Target: {summary['target_name']}",
        f"- Model: {summary['model_name']} ({summary['model_type']})",
        f"- Window: encoder={summary['encoder_length']} / prediction={summary['prediction_length']}",
        f"- RMSE: {_fmt(metrics.get('rmse'))}",
        f"- MAE: {_fmt(metrics.get('mae'))}",
        f"- Bias: {_fmt(metrics.get('bias'))}",
        f"- MAPE: {_fmt(metrics.get('mape'))}",
        f"- Raw RMSE: {_fmt(dict(summary.get('raw_metrics') or {}).get('rmse'))}",
        f"- Raw MAE: {_fmt(dict(summary.get('raw_metrics') or {}).get('mae'))}",
        f"- Raw Bias: {_fmt(dict(summary.get('raw_metrics') or {}).get('bias'))}",
        f"- Best val loss: {_fmt(summary.get('best_val_loss'))}",
        f"- Best epoch: {summary.get('best_epoch')}",
    ]
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
