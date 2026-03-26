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
        "val_end": data_config["split"]["val_end"],
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


def update_leaderboard(experiment_dir: Path, config: dict, metrics: dict[str, object]) -> Path:
    leaderboard_path = Path(config["artifacts"]["leaderboard_path"])
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "experiment_name": config["experiment"]["name"],
        "version": config["experiment"].get("version", "v2"),
        "target_name": config["data"]["target_name"],
        "model_name": config["model"]["name"],
        "model_type": config["model"]["type"],
        "encoder_length": config["data"]["window"]["encoder_length"],
        "prediction_length": config["data"]["window"]["prediction_length"],
        "train_end": config["data"]["split"]["train_end"],
        "val_end": config["data"]["split"]["val_end"],
        "test_end": config["data"]["split"]["test_end"],
        "rmse": metrics.get("rmse"),
        "mae": metrics.get("mae"),
        "bias": metrics.get("bias"),
        "mape": metrics.get("mape"),
        "notes": config["experiment"].get("notes", ""),
        "experiment_dir": str(experiment_dir),
    }
    if leaderboard_path.exists():
        leaderboard = pd.read_csv(leaderboard_path)
        leaderboard = pd.concat([leaderboard, pd.DataFrame([row])], ignore_index=True)
    else:
        leaderboard = pd.DataFrame([row])
    leaderboard = leaderboard.sort_values(["target_name", "rmse", "mae"], na_position="last").reset_index(drop=True)
    write_table(leaderboard, leaderboard_path)
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
        "experiment_config.yaml",
        "model.pt",
        "training_history.json",
        "bias_correction.json",
        "scaler.json",
        "feature_importance.csv",
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
