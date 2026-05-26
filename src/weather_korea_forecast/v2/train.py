from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from weather_korea_forecast.models.registry import build_model, resolve_model_config
from weather_korea_forecast.training.metrics import compute_prediction_metrics
from weather_korea_forecast.utils.config import load_yaml
from weather_korea_forecast.utils.io import write_json, write_table
from weather_korea_forecast.v2.artifacts import (
    create_experiment_dir,
    refresh_aliases,
    snapshot_config,
    update_leaderboard,
    write_experiment_summary,
    write_feature_importance,
    write_scaler_artifact,
)
from weather_korea_forecast.v2.data import load_or_prepare_v2_training_table
from weather_korea_forecast.v2.dataset import V2DatasetBundle, build_v2_dataset_bundle
from weather_korea_forecast.v2.evaluate import evaluate_prediction_frame
from weather_korea_forecast.v2.future_features import build_future_feature_metadata
from weather_korea_forecast.v2.target_transforms import (
    inverse_target_transform_value,
    normalize_target_transform_config,
)


def train_v2_experiment(config: dict) -> Path:
    if "ensemble" in config and ("data" not in config or "model" not in config):
        from weather_korea_forecast.v2.ensemble import build_prediction_ensemble_from_config

        return build_prediction_ensemble_from_config(config)

    resolved_model_config = resolve_model_config({"model": dict(config["model"])})
    training_table = load_or_prepare_v2_training_table(config)
    bundle = build_v2_dataset_bundle(training_table, config, backend=resolved_model_config["model"].get("backend", "fallback_torch"))

    batch_size = int(config["training"].get("batch_size", 32))
    num_workers = int(config["training"].get("num_workers", 0))
    device = str(config["training"].get("device", "cpu"))
    train_loader = bundle.make_dataloader("train", batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = bundle.make_dataloader("val", batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = bundle.make_dataloader("test", batch_size=batch_size, num_workers=num_workers, shuffle=False)

    experiment_dir = create_experiment_dir(config)
    snapshot_config(experiment_dir, config)
    write_scaler_artifact(experiment_dir, bundle.scaler)
    write_json(build_future_feature_metadata(config), experiment_dir / "future_feature_metadata.json")

    model = build_model(resolved_model_config, bundle)
    model_type = resolved_model_config["model"]["type"]
    if model_type in {"persistence", "seasonal_persistence"}:
        model.save(experiment_dir / "model.pt", extra_state={"bundle_metadata": bundle.metadata})
        history = []
        best_val_loss = float("nan")
        val_predictions = _predict_baseline(model, val_loader)
        test_predictions = _predict_baseline(model, test_loader)
    else:
        fit_kwargs = {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "max_epochs": int(config["training"].get("max_epochs", 10)),
            "learning_rate": float(resolved_model_config["model"].get("learning_rate", config["training"].get("learning_rate", 1e-3))),
            "device": device,
            "early_stopping_patience": int(config["training"].get("early_stopping_patience", 3)),
        }
        if model_type == "tft":
            fit_kwargs["gradient_clip_val"] = float(config["training"].get("gradient_clip_val", 0.0))
        train_result = model.fit(**fit_kwargs)
        history = train_result.history
        best_val_loss = train_result.best_val_loss
        model.save(experiment_dir / "model.pt", extra_state={"bundle_metadata": bundle.metadata})
        val_predictions = model.predict_loader(val_loader, device=device)
        test_predictions = model.predict_loader(test_loader, device=device)

    write_json(history, experiment_dir / "training_history.json")
    val_frame = build_v2_prediction_frame(*val_predictions, bundle=bundle)
    test_frame = build_v2_prediction_frame(*test_predictions, bundle=bundle)

    bias_payload = compute_bias_correction(val_frame, config)
    write_json(bias_payload, experiment_dir / "bias_correction.json")

    val_frame = apply_postprocessing(val_frame, config, bias_payload)
    test_frame = apply_postprocessing(test_frame, config, bias_payload)
    write_table(val_frame, experiment_dir / "predictions_val.csv")
    write_table(test_frame, experiment_dir / "predictions_test.csv")
    _write_residual_component_predictions(model, test_loader, bundle, experiment_dir, config, bias_payload)

    raw_metrics = None
    if "prediction_raw" in test_frame.columns:
        raw_metrics = compute_prediction_metrics(test_frame.rename(columns={"prediction_raw": "_prediction_raw"}), predicted_column="_prediction_raw")
    summary = evaluate_prediction_frame(test_frame, experiment_dir)

    feature_importance = export_feature_importance(model, bundle)
    write_feature_importance(experiment_dir, feature_importance)
    export_lgbm_grid_search_artifacts(model, experiment_dir)
    export_horizon_model_artifacts(model, experiment_dir, bundle)
    horizon_model_metrics = export_horizon_model_metrics(model)
    if horizon_model_metrics is not None and not horizon_model_metrics.empty:
        write_table(horizon_model_metrics, experiment_dir / "horizon_model_metrics.csv")
    write_experiment_summary(
        experiment_dir=experiment_dir,
        config=config,
        metrics=summary["metrics"],
        raw_metrics=raw_metrics,
        val_metrics=compute_prediction_metrics(val_frame) if not val_frame.empty else None,
        best_val_loss=best_val_loss,
        training_history=history,
    )
    update_leaderboard(experiment_dir, config, summary["metrics"], raw_metrics=raw_metrics)
    refresh_aliases(experiment_dir)
    print(experiment_dir)
    return experiment_dir


def build_v2_prediction_frame(
    prediction_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    metadata: dict[str, list],
    bundle: V2DatasetBundle,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    station_lookup = (
        bundle.full_frame[["station_id", "region_class", "region"]]
        .drop_duplicates("station_id")
        .set_index("station_id")
        .to_dict("index")
    )
    scaler_group_column = getattr(bundle.scaler, "group_column", "station_id")
    target_transform = normalize_target_transform_config(bundle.metadata.get("target_transform"))
    for sample_index in range(prediction_tensor.shape[0]):
        station_id = str(metadata["station_id"][sample_index])
        prediction_start = _ensure_utc_timestamp(metadata["prediction_start"][sample_index])
        station_meta = station_lookup.get(station_id, {})
        scaler_group = _scaler_group_for_sample(
            station_id=station_id,
            sample_index=sample_index,
            metadata=metadata,
            station_meta=station_meta,
            group_column=scaler_group_column,
        )
        for horizon_index in range(prediction_tensor.shape[1]):
            valid_time = prediction_start + pd.Timedelta(hours=horizon_index)
            scaled_prediction = float(prediction_tensor[sample_index, horizon_index, 0].item())
            scaled_actual = float(target_tensor[sample_index, horizon_index, 0].item())
            prediction_model_value = float(bundle.scaler.inverse_values("target_value", [scaled_prediction], groups=[scaler_group])[0])
            actual_model_value = float(bundle.scaler.inverse_values("target_value", [scaled_actual], groups=[scaler_group])[0])
            target_context = _target_transform_context_for_row(metadata, bundle, sample_index, horizon_index)
            prediction = inverse_target_transform_value(prediction_model_value, target_transform, target_context)
            actual = inverse_target_transform_value(actual_model_value, target_transform, target_context)
            row = {
                "station_id": station_id,
                "prediction_start": prediction_start,
                "valid_time": valid_time,
                "horizon_step": horizon_index + 1,
                "target_name": bundle.target_name,
                "prediction": prediction,
                "actual": actual,
                "prediction_model_value": prediction_model_value,
                "actual_model_value": actual_model_value,
                "region_class": station_meta.get("region_class", "unknown"),
                "region": station_meta.get("region", station_meta.get("region_class", "unknown")),
                "season": _season_from_timestamp(valid_time),
            }
            if target_transform.get("type") == "residual_from_feature" and target_context:
                baseline = float(target_context.get("_target_context_baseline_value"))
                row.update(
                    {
                        "baseline_prediction": baseline,
                        "predicted_residual": prediction_model_value,
                        "actual_residual": actual_model_value,
                        "prediction_corrected": prediction,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _target_transform_context_for_row(
    metadata: dict[str, list],
    bundle: V2DatasetBundle,
    sample_index: int,
    horizon_index: int,
) -> dict[str, float] | None:
    context_columns = list(bundle.metadata.get("target_context_columns", []))
    if not context_columns:
        return None
    contexts = metadata.get("target_context", [])
    if len(contexts) <= sample_index:
        return None
    sample_context = contexts[sample_index]
    if hasattr(sample_context, "detach"):
        sample_context = sample_context.detach().cpu()
    values = sample_context[horizon_index]
    if hasattr(values, "tolist"):
        values = values.tolist()
    return {column: float(value) for column, value in zip(context_columns, values)}


def _scaler_group_for_sample(
    station_id: str,
    sample_index: int,
    metadata: dict[str, list],
    station_meta: dict[str, object],
    group_column: str,
) -> str:
    if group_column == "station_id":
        return station_id
    values = metadata.get(group_column)
    if values and sample_index < len(values):
        return str(values[sample_index])
    if group_column in station_meta:
        return str(station_meta[group_column])
    if group_column == "region":
        return str(station_meta.get("region", station_meta.get("region_class", "unknown")))
    if group_column == "region_class":
        return str(station_meta.get("region_class", "unknown"))
    return station_id


def compute_bias_correction(prediction_frame: pd.DataFrame, config: dict) -> dict[str, object]:
    bias_config = config.get("evaluation", {}).get("bias_correction", {})
    enabled = bool(bias_config.get("enabled", False))
    mode = str(bias_config.get("mode", "global"))
    method = str(bias_config.get("method", "mean_bias"))
    if not enabled or prediction_frame.empty:
        return {"enabled": False, "mode": mode, "method": method, "values": []}

    frame = prediction_frame.copy()
    calibration_frame, selection_frame = _split_bias_calibration_frame(
        frame,
        calibration_fraction=float(bias_config.get("calibration_fraction", 1.0)),
    )
    shrinkage = float(bias_config.get("shrinkage", 1.0))
    if (mode == "auto" or method == "auto") and not selection_frame.empty:
        return _select_auto_correction_payload(
            calibration_frame=calibration_frame,
            selection_frame=selection_frame,
            full_frame=frame,
            config=config,
            bias_config=bias_config,
            shrinkage=shrinkage,
        )

    payload = _fit_bias_payload(calibration_frame, mode, method)
    payload["calibration"] = {
        "source": "validation",
        "calibration_sample_count": int(len(calibration_frame)),
        "selection_sample_count": int(len(selection_frame)),
        "calibration_fraction": float(bias_config.get("calibration_fraction", 1.0)),
    }
    if shrinkage != 1.0:
        payload = _scale_bias_payload(payload, shrinkage)
        payload["calibration"]["shrinkage"] = shrinkage

    if not selection_frame.empty and str(bias_config.get("apply_when", "always")) == "improves_on_holdout":
        payload = _disable_if_bias_correction_hurts_holdout(selection_frame, config, payload, bias_config)
        if bool(payload.get("enabled", False)) and bool(bias_config.get("refit_after_accept", True)):
            payload = _refit_bias_payload_after_acceptance(
                full_frame=frame,
                mode=mode,
                previous_payload=payload,
                shrinkage=shrinkage,
            )
    return payload


def _select_auto_correction_payload(
    calibration_frame: pd.DataFrame,
    selection_frame: pd.DataFrame,
    full_frame: pd.DataFrame,
    config: dict,
    bias_config: dict,
    shrinkage: float,
) -> dict[str, object]:
    metric_name = str(bias_config.get("selection_metric", "rmse"))
    min_improvement = float(bias_config.get("min_improvement", 0.0))
    candidate_modes = _candidate_correction_modes(bias_config)
    candidate_methods = _candidate_correction_methods(bias_config)
    raw_metrics = compute_prediction_metrics(selection_frame)
    raw_value = float(raw_metrics[metric_name])
    candidate_results: list[dict[str, object]] = []
    best_payload: dict[str, object] | None = None
    best_value = raw_value

    for candidate_mode in candidate_modes:
        for candidate_method in candidate_methods:
            payload = _fit_correction_payload(calibration_frame, candidate_mode, candidate_method)
            payload["calibration"] = {
                "source": "validation",
                "calibration_sample_count": int(len(calibration_frame)),
                "selection_sample_count": int(len(selection_frame)),
                "calibration_fraction": float(bias_config.get("calibration_fraction", 1.0)),
                "auto_selection": True,
            }
            if shrinkage != 1.0:
                payload = _scale_bias_payload(payload, shrinkage)
                payload["calibration"]["shrinkage"] = shrinkage
            corrected_frame = apply_postprocessing(selection_frame, config, payload)
            corrected_metrics = compute_prediction_metrics(corrected_frame)
            corrected_value = float(corrected_metrics[metric_name])
            candidate_results.append(
                {
                    "mode": candidate_mode,
                    "method": candidate_method,
                    "selection_metric": metric_name,
                    "corrected_metric": corrected_value,
                    "raw_metric": raw_value,
                }
            )
            if corrected_value < best_value - min_improvement:
                best_value = corrected_value
                best_payload = payload

    candidate_results = sorted(candidate_results, key=lambda row: float(row["corrected_metric"]))
    if best_payload is None:
        return {
            "enabled": False,
            "mode": "auto",
            "method": "auto",
            "values": [],
            "disabled_reason": "no_auto_correction_candidate_improved_holdout",
            "selection": {
                "apply_when": "auto_best_improves_on_holdout",
                "selection_metric": metric_name,
                "min_improvement": min_improvement,
                "raw_metric": raw_value,
                "corrected_metric": raw_value,
                "accepted": False,
                "candidates": candidate_results,
            },
        }

    best_payload["selection"] = {
        "apply_when": "auto_best_improves_on_holdout",
        "selection_metric": metric_name,
        "min_improvement": min_improvement,
        "raw_metric": raw_value,
        "corrected_metric": best_value,
        "accepted": True,
        "selected_mode": best_payload["mode"],
        "selected_method": best_payload["method"],
        "candidates": candidate_results,
    }
    if bool(bias_config.get("refit_after_accept", True)):
        best_payload = _refit_bias_payload_after_acceptance(full_frame, str(best_payload["mode"]), best_payload, shrinkage)
    return best_payload


def _candidate_correction_modes(bias_config: dict) -> list[str]:
    configured_mode = str(bias_config.get("mode", "global"))
    if configured_mode != "auto":
        return [configured_mode]
    modes = bias_config.get("candidate_modes") or [
        "global",
        "per_horizon",
        "per_station",
        "per_station_horizon",
        "per_season_horizon",
        "per_region_horizon",
    ]
    return [str(mode) for mode in modes]


def _candidate_correction_methods(bias_config: dict) -> list[str]:
    configured_method = str(bias_config.get("method", "mean_bias"))
    if configured_method != "auto":
        return [configured_method]
    methods = bias_config.get("candidate_methods") or ["mean_bias", "affine", "quantile"]
    return [str(method) for method in methods]


def _fit_bias_payload(frame: pd.DataFrame, mode: str, method: str = "mean_bias") -> dict[str, object]:
    return _fit_correction_payload(frame, mode, method=method)


def _fit_correction_payload(frame: pd.DataFrame, mode: str, method: str) -> dict[str, object]:
    frame = frame.copy()
    if method not in {"mean_bias", "affine", "quantile"}:
        raise ValueError(f"Unsupported bias correction method: {method}")
    if mode == "global":
        return {"enabled": True, "mode": mode, "method": method, "values": [_fit_correction_row(frame, method)]}
    group_columns = _bias_correction_group_columns(mode)
    if group_columns:
        grouped = _fit_grouped_correction_rows(frame, group_columns, method)
        return {"enabled": True, "mode": mode, "method": method, "values": grouped.to_dict(orient="records")}
    raise ValueError(f"Unsupported bias correction mode: {mode}")


def _bias_correction_group_columns(mode: str) -> list[str]:
    if mode == "per_horizon":
        return ["horizon_step"]
    if mode == "per_station":
        return ["station_id"]
    if mode == "per_station_horizon":
        return ["station_id", "horizon_step"]
    if mode == "per_season_horizon":
        return ["season", "horizon_step"]
    if mode == "per_region_horizon":
        region_column = "region" if mode == "per_region_horizon" else "region_class"
        return [region_column, "horizon_step"]
    return []


def _fit_grouped_correction_rows(frame: pd.DataFrame, group_columns: list[str], method: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groupby_arg = group_columns[0] if len(group_columns) == 1 else group_columns
    for group_key, group in frame.groupby(groupby_arg):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {column: value for column, value in zip(group_columns, group_key)}
        row.update(_fit_correction_row(group, method))
        rows.append(row)
    return pd.DataFrame(rows)


def _fit_correction_row(frame: pd.DataFrame, method: str) -> dict[str, object]:
    prediction = frame["prediction"].astype(float).to_numpy()
    actual = frame["actual"].astype(float).to_numpy()
    if method == "mean_bias":
        return {"bias": float(np.mean(prediction - actual))}

    if method == "quantile":
        levels = np.linspace(0.0, 1.0, 21)
        prediction_quantiles = np.quantile(prediction, levels)
        actual_quantiles = np.quantile(actual, levels)
        prediction_knots, actual_knots = _deduplicate_quantile_knots(prediction_quantiles, actual_quantiles)
        return {
            "quantile_levels": [float(value) for value in levels],
            "prediction_quantiles": [float(value) for value in prediction_knots],
            "actual_quantiles": [float(value) for value in actual_knots],
        }

    prediction_mean = float(np.mean(prediction))
    actual_mean = float(np.mean(actual))
    prediction_variance = float(np.mean(np.square(prediction - prediction_mean)))
    if prediction_variance <= 1e-12:
        return {"slope": 1.0, "intercept": actual_mean - prediction_mean}
    covariance = float(np.mean((prediction - prediction_mean) * (actual - actual_mean)))
    slope = covariance / prediction_variance
    intercept = actual_mean - slope * prediction_mean
    return {"slope": float(slope), "intercept": float(intercept)}


def _deduplicate_quantile_knots(prediction_quantiles: np.ndarray, actual_quantiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(prediction_quantiles)
    prediction_sorted = np.asarray(prediction_quantiles, dtype=float)[order]
    actual_sorted = np.asarray(actual_quantiles, dtype=float)[order]
    unique_predictions: list[float] = []
    unique_actuals: list[float] = []
    for prediction_value, actual_value in zip(prediction_sorted, actual_sorted):
        if unique_predictions and abs(prediction_value - unique_predictions[-1]) <= 1e-12:
            unique_actuals[-1] = float(actual_value)
            continue
        unique_predictions.append(float(prediction_value))
        unique_actuals.append(float(actual_value))
    if len(unique_predictions) == 1:
        value = unique_predictions[0]
        unique_predictions = [value - 1e-6, value + 1e-6]
        unique_actuals = [unique_actuals[0], unique_actuals[0]]
    return np.asarray(unique_predictions, dtype=float), np.asarray(unique_actuals, dtype=float)


def _split_bias_calibration_frame(frame: pd.DataFrame, calibration_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if calibration_fraction <= 0.0 or calibration_fraction > 1.0:
        raise ValueError("bias_correction.calibration_fraction must be in the interval (0, 1].")
    ordered = frame.sort_values(["prediction_start", "station_id", "horizon_step"]).reset_index(drop=True)
    if calibration_fraction >= 1.0 or len(ordered) < 2:
        return ordered, ordered.iloc[0:0].copy()

    prediction_starts = pd.Series(pd.to_datetime(ordered["prediction_start"], utc=True).dropna().sort_values().unique())
    if len(prediction_starts) < 2:
        split_index = max(1, min(len(ordered) - 1, int(round(len(ordered) * calibration_fraction))))
        return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()

    split_start_count = max(1, min(len(prediction_starts) - 1, int(round(len(prediction_starts) * calibration_fraction))))
    cutoff = prediction_starts.iloc[split_start_count - 1]
    starts = pd.to_datetime(ordered["prediction_start"], utc=True)
    calibration = ordered.loc[starts <= cutoff].copy()
    selection = ordered.loc[starts > cutoff].copy()
    return calibration, selection


def _scale_bias_payload(payload: dict[str, object], shrinkage: float) -> dict[str, object]:
    scaled = {**payload, "values": []}
    for value in payload.get("values", []):
        row = dict(value)
        method = payload.get("method", "mean_bias")
        if method == "affine":
            row["slope"] = 1.0 + (float(row["slope"]) - 1.0) * shrinkage
            row["intercept"] = float(row["intercept"]) * shrinkage
        elif method == "quantile":
            prediction_quantiles = [float(item) for item in row.get("prediction_quantiles", [])]
            actual_quantiles = [float(item) for item in row.get("actual_quantiles", [])]
            row["actual_quantiles"] = [
                prediction + (actual - prediction) * shrinkage
                for prediction, actual in zip(prediction_quantiles, actual_quantiles)
            ]
        else:
            row["bias"] = float(row["bias"]) * shrinkage
        scaled["values"].append(row)
    return scaled


def _disable_if_bias_correction_hurts_holdout(
    selection_frame: pd.DataFrame,
    config: dict,
    payload: dict[str, object],
    bias_config: dict,
) -> dict[str, object]:
    metric_name = str(bias_config.get("selection_metric", "rmse"))
    min_improvement = float(bias_config.get("min_improvement", 0.0))
    raw_metrics = compute_prediction_metrics(selection_frame)
    corrected_frame = apply_postprocessing(selection_frame, config, payload)
    corrected_metrics = compute_prediction_metrics(corrected_frame)
    raw_value = float(raw_metrics[metric_name])
    corrected_value = float(corrected_metrics[metric_name])
    improved = corrected_value < raw_value - min_improvement
    selection_payload = {
        "apply_when": "improves_on_holdout",
        "selection_metric": metric_name,
        "min_improvement": min_improvement,
        "raw_metric": raw_value,
        "corrected_metric": corrected_value,
        "accepted": bool(improved),
    }
    if improved:
        payload["selection"] = selection_payload
        return payload

    return {
        **payload,
        "enabled": False,
        "disabled_reason": "bias_correction_did_not_improve_holdout",
        "selection": selection_payload,
    }


def _refit_bias_payload_after_acceptance(
    full_frame: pd.DataFrame,
    mode: str,
    previous_payload: dict[str, object],
    shrinkage: float,
) -> dict[str, object]:
    method = str(previous_payload.get("method", "mean_bias"))
    refit_payload = _fit_correction_payload(full_frame, mode, method)
    if shrinkage != 1.0:
        refit_payload = _scale_bias_payload(refit_payload, shrinkage)
    refit_payload["calibration"] = {
        **dict(previous_payload.get("calibration", {})),
        "final_refit": "full_validation_after_holdout_acceptance",
        "final_refit_sample_count": int(len(full_frame)),
    }
    refit_payload["selection"] = previous_payload.get("selection", {})
    return refit_payload


def apply_postprocessing(prediction_frame: pd.DataFrame, config: dict, bias_payload: dict[str, object]) -> pd.DataFrame:
    frame = prediction_frame.copy()
    frame["prediction_raw"] = frame["prediction"]
    if bool(bias_payload.get("enabled", False)):
        mode = str(bias_payload.get("mode", "global"))
        method = str(bias_payload.get("method", "mean_bias"))
        if mode == "global":
            frame["prediction"] = _apply_correction_values(frame["prediction"], bias_payload["values"][0], method)
        else:
            group_columns = _bias_correction_group_columns(mode)
            if not group_columns:
                raise ValueError(f"Unsupported bias correction mode: {mode}")
            correction_frame = pd.DataFrame(bias_payload["values"])
            frame = frame.merge(correction_frame, on=group_columns, how="left")
            frame["prediction"] = _apply_correction_columns(frame, method)
            frame = frame.drop(
                columns=[
                    column
                    for column in (
                        "bias",
                        "slope",
                        "intercept",
                        "quantile_levels",
                        "prediction_quantiles",
                        "actual_quantiles",
                    )
                    if column in frame.columns
                ]
            )

    clip_range = config["data"].get("postprocess", {}).get("clip_prediction")
    if clip_range:
        frame["prediction"] = frame["prediction"].clip(lower=float(clip_range[0]), upper=float(clip_range[1]))
    frame["prediction_corrected"] = frame["prediction"]
    if "actual" in frame.columns:
        frame["error"] = frame["prediction"].astype(float) - frame["actual"].astype(float)
        frame["abs_error"] = frame["error"].abs()
    return frame


def _apply_correction_values(prediction: pd.Series, values: dict[str, object], method: str) -> pd.Series:
    if method == "affine":
        return prediction.astype(float) * float(values.get("slope", 1.0)) + float(values.get("intercept", 0.0))
    if method == "quantile":
        return pd.Series(
            _apply_quantile_array(
                prediction.astype(float).to_numpy(),
                values.get("prediction_quantiles", []),
                values.get("actual_quantiles", []),
            ),
            index=prediction.index,
        )
    return prediction.astype(float) - float(values.get("bias", 0.0))


def _apply_correction_columns(frame: pd.DataFrame, method: str) -> pd.Series:
    if method == "affine":
        slope = frame.get("slope", pd.Series(1.0, index=frame.index)).fillna(1.0).astype(float)
        intercept = frame.get("intercept", pd.Series(0.0, index=frame.index)).fillna(0.0).astype(float)
        return frame["prediction"].astype(float) * slope + intercept
    if method == "quantile":
        corrected = frame["prediction"].astype(float).copy()
        for index, row in frame.iterrows():
            prediction_quantiles = row.get("prediction_quantiles", [])
            actual_quantiles = row.get("actual_quantiles", [])
            if not isinstance(prediction_quantiles, list) or not isinstance(actual_quantiles, list):
                continue
            corrected.loc[index] = float(
                _apply_quantile_array(
                    np.asarray([float(row["prediction"])]),
                    prediction_quantiles,
                    actual_quantiles,
                )[0]
            )
        return corrected
    bias = frame.get("bias", pd.Series(0.0, index=frame.index)).fillna(0.0).astype(float)
    return frame["prediction"].astype(float) - bias


def _apply_quantile_array(values: np.ndarray, prediction_quantiles, actual_quantiles) -> np.ndarray:
    prediction_knots = np.asarray(prediction_quantiles, dtype=float)
    actual_knots = np.asarray(actual_quantiles, dtype=float)
    if len(prediction_knots) < 2 or len(prediction_knots) != len(actual_knots):
        return values
    return np.interp(values, prediction_knots, actual_knots, left=actual_knots[0], right=actual_knots[-1])


def export_lgbm_grid_search_artifacts(model, experiment_dir: Path) -> None:
    frames: list[pd.DataFrame] = []
    best_payload: dict[str, object] = {}
    for component_name, candidate in _iter_model_components(model):
        if hasattr(candidate, "grid_search_results_frame"):
            frame = candidate.grid_search_results_frame()
            if frame is not None and not frame.empty:
                frame = frame.copy()
                frame.insert(0, "component", component_name)
                frames.append(frame)
        if hasattr(candidate, "best_params_dict"):
            best_payload[component_name] = candidate.best_params_dict()
    if frames:
        write_table(pd.concat(frames, ignore_index=True), experiment_dir / "lgbm_grid_search_results.csv")
    if best_payload:
        write_json(best_payload, experiment_dir / "best_lgbm_params.json")


def export_horizon_model_artifacts(model, experiment_dir: Path, bundle: V2DatasetBundle) -> None:
    model_dir = experiment_dir / "model"
    wrote = False
    for component_name, candidate in _iter_model_components(model):
        if hasattr(candidate, "horizon_metrics_frame"):
            summary = candidate.horizon_metrics_frame()
            if summary is not None and not summary.empty:
                summary = summary.copy()
                if "component" not in summary.columns:
                    summary.insert(0, "component", component_name)
                write_table(summary, experiment_dir / "horizon_model_summary.csv")
        if hasattr(candidate, "feature_importance_frame"):
            importance = candidate.feature_importance_frame(flattened_feature_names(bundle))
            if importance is not None and not importance.empty and "horizon_step" in importance.columns:
                importance = importance.copy()
                if "component" not in importance.columns:
                    importance.insert(0, "component", component_name)
                write_table(importance, experiment_dir / "horizon_feature_importance.csv")
        if candidate.__class__.__name__.lower().startswith("horizonwise"):
            model_dir.mkdir(parents=True, exist_ok=True)
            import pickle

            for horizon_step in range(1, bundle.prediction_length + 1):
                with (model_dir / f"horizon_{horizon_step:02d}.pkl").open("wb") as handle:
                    pickle.dump({"component": component_name, "horizon_step": horizon_step, "model_class": candidate.__class__.__name__}, handle)
                wrote = True
    if wrote:
        write_json({"horizon_model_count": bundle.prediction_length, "directory": str(model_dir)}, experiment_dir / "horizon_model_manifest.json")


def _iter_model_components(model):
    yield "main", model
    for name in ("baseline_model", "residual_model"):
        if hasattr(model, name):
            yield name.replace("_model", ""), getattr(model, name)


def export_feature_importance(model, bundle: V2DatasetBundle) -> pd.DataFrame | None:
    if not hasattr(model, "feature_importance_frame"):
        return None
    return model.feature_importance_frame(flattened_feature_names(bundle))


def export_horizon_model_metrics(model) -> pd.DataFrame | None:
    if not hasattr(model, "horizon_metrics_frame"):
        return None
    return model.horizon_metrics_frame()


def _write_residual_component_predictions(model, test_loader, bundle: V2DatasetBundle, experiment_dir: Path, config: dict, bias_payload: dict[str, object]) -> None:
    if not hasattr(model, "predict_components_loader"):
        return
    components, target_tensor, metadata = model.predict_components_loader(test_loader, device=str(config["training"].get("device", "cpu")))
    frames: list[pd.DataFrame] = []
    for component_name, prediction_tensor in components.items():
        frame = build_v2_prediction_frame(prediction_tensor, target_tensor, metadata, bundle=bundle)
        frame.insert(0, "component", component_name)
        if component_name == "final":
            frame = apply_postprocessing(frame, config, bias_payload)
        frames.append(frame)
    if frames:
        write_table(pd.concat(frames, ignore_index=True), experiment_dir / "predictions_test_components.csv")


def flattened_feature_names(bundle: V2DatasetBundle) -> list[str]:
    names: list[str] = []
    for encoder_step in range(bundle.encoder_length):
        for column in bundle.encoder_columns:
            names.append(f"encoder_t-{bundle.encoder_length - encoder_step}:{column}")
    for horizon_step in range(1, bundle.prediction_length + 1):
        for column in bundle.decoder_columns:
            names.append(f"decoder_t+{horizon_step}:{column}")
    for column in bundle.static_baseline_columns:
        names.append(f"static:{column}")
    return names


def _predict_baseline(model, loader) -> tuple[torch.Tensor, torch.Tensor, dict[str, list]]:
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    metadata = {"station_id": [], "prediction_start": [], "region_class": [], "target_context": []}
    for batch in loader:
        predictions.append(model.predict_batch(batch))
        targets.append(batch["target"])
        metadata["station_id"].extend(batch["station_id"])
        metadata["prediction_start"].extend(batch["prediction_start"])
        metadata["region_class"].extend(batch.get("region_class", []))
        context = batch.get("target_context")
        if context is not None:
            metadata["target_context"].extend(context.cpu())
    return torch.cat(predictions), torch.cat(targets), metadata


def _ensure_utc_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _season_from_timestamp(timestamp: pd.Timestamp) -> str:
    month = timestamp.month
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "autumn"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a V2 experiment from a unified config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--update-report", dest="update_report", action="store_true", default=True, help="Regenerate reports/experiment_report.html after training (default).")
    parser.add_argument("--no-update-report", dest="update_report", action="store_false", help="Skip automatic unified report generation.")
    parser.add_argument("--report-output-dir", default="reports")
    parser.add_argument("--report-title", default="기상 V1-V3 실험 리포트")
    args = parser.parse_args()

    config = load_yaml(args.config)
    experiment_dir = train_v2_experiment(config)
    if args.update_report:
        from weather_korea_forecast.reporting.generate_report import build_report

        build_report(experiments_root=Path("data/artifacts"), output_dir=Path(args.report_output_dir), title=args.report_title)
    return experiment_dir


if __name__ == "__main__":
    main()
