from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from weather_korea_forecast.utils.io import read_table, write_json, write_table
from weather_korea_forecast.utils.paths import ensure_dir, resolve_path, timestamp_slug
from weather_korea_forecast.v2.artifacts import refresh_aliases, update_leaderboard
from weather_korea_forecast.v2.evaluate import evaluate_prediction_frame
from weather_korea_forecast.v2.future_features import build_future_feature_metadata


DEFAULT_KEY_COLUMNS = ["station_id", "prediction_start", "valid_time", "horizon_step", "target_name"]


def build_prediction_ensemble(
    experiment_dirs: Sequence[str | Path],
    output_root: str | Path,
    name: str,
    method: str = "mean",
    weights: Sequence[float] | None = None,
    clip_min: float | None = None,
    clip_max: float | None = None,
    leaderboard_path: str | Path | None = None,
    version: str = "v3",
    forecast_track: str = "observation_only",
    notes: str | None = None,
) -> Path:
    """Average aligned ``predictions_test.csv`` files into a stronger ensemble artifact.

    This is deliberately artifact-driven: it can combine direct-RH, dew-point,
    dew-point-depression, or MOS model outputs without retraining, then runs the
    same V2 evaluator so station/region/horizon/daily reports stay comparable.
    """

    if len(experiment_dirs) < 2:
        raise ValueError("At least two experiment directories are required for an ensemble.")
    normalized_method = method.strip().lower()
    if normalized_method not in {"mean", "median", "inverse_rmse", "validation_inverse_rmse", "horizon_linear", "constrained_least_squares"}:
        raise ValueError("method must be one of: mean, median, inverse_rmse, validation_inverse_rmse, horizon_linear, constrained_least_squares")

    component_frames = [_load_component_predictions(path) for path in experiment_dirs]
    key_columns = _shared_key_columns(component_frames)
    component_table = _aligned_component_table(component_frames, [resolve_path(path) for path in experiment_dirs], key_columns)
    prediction_columns = [column for column in component_table.columns if column.startswith("prediction_component_")]

    prediction_values = component_table[prediction_columns].astype(float).to_numpy()
    learned_weights = _learn_ensemble_weights(
        experiment_dirs=[resolve_path(path) for path in experiment_dirs],
        method=normalized_method,
        fallback_weights=_normalized_weights(weights, len(prediction_columns)),
        key_columns=key_columns,
        test_component_table=component_table,
    )
    if normalized_method == "median":
        ensemble_prediction = np.median(prediction_values, axis=1)
    elif isinstance(learned_weights, dict):
        ensemble_prediction = _apply_horizon_weights(component_table, prediction_columns, learned_weights)
    else:
        ensemble_prediction = np.average(prediction_values, axis=1, weights=learned_weights)
    if clip_min is not None or clip_max is not None:
        lower = -np.inf if clip_min is None else float(clip_min)
        upper = np.inf if clip_max is None else float(clip_max)
        ensemble_prediction = np.clip(ensemble_prediction, lower, upper)

    output_dir = ensure_dir(output_root) / f"{name}_{timestamp_slug()}"
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions = _base_prediction_frame(component_frames[0], key_columns)
    predictions["prediction_raw"] = ensemble_prediction
    predictions["prediction"] = ensemble_prediction
    write_table(predictions, output_dir / "predictions_test.csv")
    write_table(predictions, output_dir / "ensemble_predictions_test.csv")
    write_table(component_table, output_dir / "predictions_test_components.csv")
    write_json(_jsonable_weights(learned_weights, prediction_columns), output_dir / "ensemble_weights.json")
    write_table(_component_metrics(component_table, prediction_columns), output_dir / "ensemble_component_metrics.csv")

    metric_payload = evaluate_prediction_frame(predictions, output_dir)
    config = _ensemble_config(
        name=name,
        version=version,
        method=normalized_method,
        component_dirs=[resolve_path(path) for path in experiment_dirs],
        metrics=metric_payload["metrics"],
        output_root=output_root,
        leaderboard_path=leaderboard_path,
        forecast_track=forecast_track,
        notes=notes,
        first_component_dir=resolve_path(experiment_dirs[0]),
    )
    write_json(config, output_dir / "ensemble_config.json")
    write_json(build_future_feature_metadata(config), output_dir / "future_feature_metadata.json")
    _write_ensemble_summary(output_dir, config, metric_payload["metrics"], metric_payload.get("raw_metrics"))
    if leaderboard_path is not None:
        update_leaderboard(output_dir, config, metric_payload["metrics"], metric_payload.get("raw_metrics"))
    refresh_aliases(output_dir)
    return output_dir


def build_prediction_ensemble_from_config(config: dict[str, object]) -> Path:
    ensemble_config = dict(config.get("ensemble", {}))
    output_root = config.get("artifacts", {}).get("root_dir", "data/artifacts/v3_experiments")
    components = [str(value) for value in ensemble_config.get("components", [])]
    resolved_components = [_resolve_component_dir(component, Path(str(output_root))) for component in components]
    return build_prediction_ensemble(
        experiment_dirs=resolved_components,
        output_root=output_root,
        name=str(config.get("experiment", {}).get("name", "v3_ensemble")),
        method=str(ensemble_config.get("method", "mean")),
        clip_min=ensemble_config.get("clip_min"),
        clip_max=ensemble_config.get("clip_max"),
        leaderboard_path=config.get("artifacts", {}).get("leaderboard_path"),
        version=str(config.get("experiment", {}).get("version", "v3")),
        forecast_track=str(ensemble_config.get("forecast_track", "observation_only")),
        notes=config.get("experiment", {}).get("notes"),
    )


def _resolve_component_dir(component: str, output_root: Path) -> Path:
    if "<" not in component and ">" not in component:
        return resolve_path(component)
    name = component.strip("<>").replace(" experiment_dir", "")
    candidates = sorted(output_root.glob(f"{name}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"Could not resolve ensemble component placeholder {component!r} under {output_root}.")
    return candidates[0]


def _learn_ensemble_weights(
    experiment_dirs: Sequence[Path],
    method: str,
    fallback_weights: np.ndarray | None,
    key_columns: list[str],
    test_component_table: pd.DataFrame,
):
    prediction_columns = [column for column in test_component_table.columns if column.startswith("prediction_component_")]
    if method in {"mean", "median"}:
        return fallback_weights
    val_frames = []
    for path in experiment_dirs:
        val_path = path / "predictions_val.csv"
        if not val_path.exists():
            val_frames = []
            break
        val_frames.append(read_table(val_path))
    if val_frames:
        val_table = _aligned_component_table(val_frames, experiment_dirs, _shared_key_columns(val_frames))
    else:
        val_table = test_component_table
    val_prediction_columns = [column for column in val_table.columns if column.startswith("prediction_component_")]
    if method in {"inverse_rmse", "validation_inverse_rmse"}:
        rmses = []
        actual = val_table["actual"].astype(float).to_numpy()
        for column in val_prediction_columns:
            pred = val_table[column].astype(float).to_numpy()
            rmses.append(float(np.sqrt(np.mean(np.square(pred - actual)))))
        inv = 1.0 / np.maximum(np.asarray(rmses, dtype=float), 1e-9)
        return inv / inv.sum()
    if method in {"horizon_linear", "constrained_least_squares"}:
        weights_by_horizon = {}
        for horizon_step, group in val_table.groupby("horizon_step") if "horizon_step" in val_table.columns else [("all", val_table)]:
            X = group[val_prediction_columns].astype(float).to_numpy()
            y = group["actual"].astype(float).to_numpy()
            try:
                weights = np.linalg.lstsq(X, y, rcond=None)[0]
                weights = np.clip(weights, 0.0, None)
                if weights.sum() <= 0:
                    weights = np.ones(len(val_prediction_columns)) / len(val_prediction_columns)
                else:
                    weights = weights / weights.sum()
            except Exception:
                weights = np.ones(len(val_prediction_columns)) / len(val_prediction_columns)
            weights_by_horizon[str(horizon_step)] = [float(value) for value in weights]
        return weights_by_horizon
    return fallback_weights


def _apply_horizon_weights(table: pd.DataFrame, prediction_columns: list[str], weights_by_horizon: dict[str, list[float]]) -> np.ndarray:
    output = np.zeros(len(table), dtype=float)
    for output_index, (_, row) in enumerate(table.iterrows()):
        key = str(row.get("horizon_step", "all"))
        weights = np.asarray(weights_by_horizon.get(key) or weights_by_horizon.get("all") or [1 / len(prediction_columns)] * len(prediction_columns), dtype=float)
        values = row[prediction_columns].astype(float).to_numpy()
        output[output_index] = float(np.average(values, weights=weights))
    return output


def _jsonable_weights(weights, prediction_columns: list[str]) -> dict[str, object]:
    if isinstance(weights, dict):
        return {"mode": "horizon", "weights_by_horizon": weights, "components": prediction_columns}
    if weights is None:
        weights = np.ones(len(prediction_columns), dtype=float) / len(prediction_columns)
    return {"mode": "global", "weights": [float(value) for value in weights], "components": prediction_columns}


def _component_metrics(table: pd.DataFrame, prediction_columns: list[str]) -> pd.DataFrame:
    rows = []
    actual = table["actual"].astype(float).to_numpy()
    for column in prediction_columns:
        pred = table[column].astype(float).to_numpy()
        error = pred - actual
        rows.append({"component": column, "rmse": float(np.sqrt(np.mean(np.square(error)))), "mae": float(np.mean(np.abs(error))), "bias": float(np.mean(error))})
    return pd.DataFrame(rows)


def _load_component_predictions(experiment_dir: str | Path) -> pd.DataFrame:
    path = resolve_path(experiment_dir) / "predictions_test.csv"
    frame = read_table(path)
    required = {"prediction", "actual"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required prediction columns: {sorted(missing)}")
    return frame


def _shared_key_columns(component_frames: Sequence[pd.DataFrame]) -> list[str]:
    key_columns = [column for column in DEFAULT_KEY_COLUMNS if all(column in frame.columns for frame in component_frames)]
    if not key_columns:
        raise ValueError("Component predictions do not share any supported key columns.")
    return key_columns


def _aligned_component_table(component_frames: Sequence[pd.DataFrame], experiment_dirs: Sequence[Path], key_columns: list[str]) -> pd.DataFrame:
    base = component_frames[0][key_columns + ["actual", "prediction"]].copy()
    _assert_unique_keys(base, key_columns, str(experiment_dirs[0]))
    base = base.rename(columns={"actual": "actual_component_0", "prediction": "prediction_component_0"})
    for index, (frame, experiment_dir) in enumerate(zip(component_frames[1:], experiment_dirs[1:]), start=1):
        candidate = frame[key_columns + ["actual", "prediction"]].copy()
        _assert_unique_keys(candidate, key_columns, str(experiment_dir))
        candidate = candidate.rename(columns={"actual": f"actual_component_{index}", "prediction": f"prediction_component_{index}"})
        base = base.merge(candidate, on=key_columns, how="inner", validate="one_to_one")
    if base.empty:
        raise ValueError("No overlapping prediction keys were found across component experiments.")
    actual_columns = [column for column in base.columns if column.startswith("actual_component_")]
    reference_actual = base[actual_columns[0]].astype(float).to_numpy()
    for column in actual_columns[1:]:
        if not np.allclose(reference_actual, base[column].astype(float).to_numpy(), equal_nan=True):
            raise ValueError("Component predictions have inconsistent actual values for the same keys.")
    base["actual"] = reference_actual
    base["component_experiments"] = "|".join(path.name for path in experiment_dirs)
    return base


def _assert_unique_keys(frame: pd.DataFrame, key_columns: list[str], label: str) -> None:
    duplicated = frame.duplicated(key_columns).sum()
    if duplicated:
        raise ValueError(f"{label} contains {duplicated} duplicate prediction key rows.")


def _normalized_weights(weights: Sequence[float] | None, count: int) -> np.ndarray | None:
    if weights is None:
        return None
    values = np.asarray([float(weight) for weight in weights], dtype=float)
    if len(values) != count:
        raise ValueError(f"Expected {count} ensemble weights, got {len(values)}.")
    total = values.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Ensemble weights must sum to a positive finite value.")
    return values / total


def _base_prediction_frame(first_component: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    metadata_columns = [
        column
        for column in ["region_class", "region", "season"]
        if column in first_component.columns and column not in key_columns
    ]
    return first_component[key_columns + ["actual"] + metadata_columns].copy()


def _ensemble_config(
    name: str,
    version: str,
    method: str,
    component_dirs: Sequence[Path],
    metrics: dict[str, object],
    output_root: str | Path,
    leaderboard_path: str | Path | None,
    forecast_track: str,
    notes: str | None,
    first_component_dir: Path,
) -> dict[str, object]:
    component_summary = _read_json(first_component_dir / "experiment_summary.json")
    target_name = str(component_summary.get("target_name") or "target")
    encoder_length = int(component_summary.get("encoder_length") or 0)
    prediction_length = int(component_summary.get("prediction_length") or 0)
    split = {
        "train_start": component_summary.get("train_start"),
        "train_end": component_summary.get("train_end"),
        "val_start": component_summary.get("val_start"),
        "val_end": component_summary.get("val_end"),
        "test_start": component_summary.get("test_start"),
        "test_end": component_summary.get("test_end"),
    }
    component_future = dict(component_summary.get("future_features") or {})
    return {
        "experiment": {
            "name": name,
            "version": version,
            "notes": notes
            or f"{version.upper()} {target_name} prediction ensemble using {method} over {len(component_dirs)} component experiments.",
        },
        "data": {
            "target_name": target_name,
            "window": {"encoder_length": encoder_length, "prediction_length": prediction_length},
            "split": split,
            "scaling": {"mode": "ensemble", "group_column": "station_id"},
            "features": {"decoder_known": []},
            "future_features": {
                "track": forecast_track,
                "source": component_future.get("future_feature_source", "none"),
                "operational_valid": bool(component_future.get("operational_valid", False)),
                "backtest_only": bool(component_future.get("backtest_only", False)),
                "uses_future_weather_features": bool(component_future.get("uses_future_weather_features", forecast_track != "observation_only")),
                "leakage_risk_note": component_future.get("leakage_risk_note", "Ensemble inherits future-feature validity from component experiments."),
            },
        },
        "model": {
            "name": name,
            "type": f"ensemble_{method}",
            "components": [str(path) for path in component_dirs],
        },
        "artifacts": {
            "root_dir": str(output_root),
            "leaderboard_path": str(leaderboard_path or Path(output_root) / "leaderboard.csv"),
        },
        "metrics": metrics,
    }


def _write_ensemble_summary(
    output_dir: Path,
    config: dict[str, object],
    metrics: dict[str, object],
    raw_metrics: dict[str, object] | None,
) -> None:
    future_features = build_future_feature_metadata(config)
    model_config = dict(config["model"])
    data_config = dict(config["data"])
    summary = {
        "experiment_name": config["experiment"]["name"],
        "version": config["experiment"].get("version", "v3"),
        "target_name": data_config["target_name"],
        "model_name": model_config["name"],
        "model_type": model_config["type"],
        "model_family": "ensemble",
        "encoder_length": data_config["window"]["encoder_length"],
        "prediction_length": data_config["window"]["prediction_length"],
        "train_start": data_config["split"].get("train_start"),
        "train_end": data_config["split"].get("train_end"),
        "val_start": data_config["split"].get("val_start"),
        "val_end": data_config["split"].get("val_end"),
        "test_start": data_config["split"].get("test_start"),
        "test_end": data_config["split"].get("test_end"),
        "metrics": metrics,
        "raw_metrics": raw_metrics,
        "future_features": future_features,
        "forecast_track": future_features["forecast_track"],
        "uses_future_nwp_features": future_features["uses_future_nwp_features"],
        "future_feature_source": future_features["future_feature_source"],
        "operational_valid": future_features["operational_valid"],
        "backtest_only": future_features["backtest_only"],
        "component_experiments": model_config.get("components", []),
        "notes": config["experiment"].get("notes", ""),
    }
    write_json(summary, output_dir / "experiment_summary.json")
    lines = [
        f"# {summary['experiment_name']}",
        "",
        f"- Version: {summary['version']}",
        f"- Target: {summary['target_name']}",
        f"- Model: {summary['model_name']} ({summary['model_type']})",
        "- Model family: ensemble",
        f"- Forecast track: {summary['forecast_track']}",
        f"- RMSE: {_fmt(metrics.get('rmse'))}",
        f"- MAE: {_fmt(metrics.get('mae'))}",
        f"- Bias: {_fmt(metrics.get('bias'))}",
        "",
        "## Components",
        "",
        *[f"- {component}" for component in model_config.get("components", [])],
    ]
    output_dir.joinpath("experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fmt(value: object) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _parse_weights(raw: str | None) -> list[float] | None:
    if raw is None or not raw.strip():
        return None
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a V2/V3 prediction ensemble from experiment artifacts.")
    parser.add_argument("--experiment-dir", action="append", required=True, help="Component experiment dir; repeat at least twice.")
    parser.add_argument("--output-root", default="data/artifacts/v3_experiments")
    parser.add_argument("--name", default="v3_humidity_mean_ensemble_72to24")
    parser.add_argument("--method", choices=["mean", "median", "inverse_rmse", "validation_inverse_rmse", "horizon_linear", "constrained_least_squares"], default="mean")
    parser.add_argument("--weights", help="Optional comma-separated weights for mean ensembles.")
    parser.add_argument("--clip-min", type=float)
    parser.add_argument("--clip-max", type=float)
    parser.add_argument("--leaderboard-path", default="data/artifacts/v3_experiments/leaderboard.csv")
    parser.add_argument("--version", default="v3")
    parser.add_argument("--forecast-track", default="observation_only")
    parser.add_argument("--notes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = build_prediction_ensemble(
        experiment_dirs=args.experiment_dir,
        output_root=args.output_root,
        name=args.name,
        method=args.method,
        weights=_parse_weights(args.weights),
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        leaderboard_path=args.leaderboard_path,
        version=args.version,
        forecast_track=args.forecast_track,
        notes=args.notes,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
