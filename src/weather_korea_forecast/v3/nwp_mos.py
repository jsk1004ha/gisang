from __future__ import annotations

import argparse
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from weather_korea_forecast.training.metrics import compute_prediction_metrics
from weather_korea_forecast.utils.config import load_yaml
from weather_korea_forecast.utils.io import write_json, write_table
from weather_korea_forecast.utils.paths import resolve_path
from weather_korea_forecast.v2.data import load_or_prepare_v2_training_table
from weather_korea_forecast.v2.future_features import build_future_feature_metadata, load_future_weather_archive


DEFAULT_INIT_FEATURES = [
    "obs_temp",
    "obs_humidity",
    "obs_dew_point_c",
    "obs_dew_point_depression",
    "obs_pressure",
    "obs_wind_speed",
    "obs_precipitation",
    "obs_vapor_pressure_hpa",
    "obs_absolute_humidity_g_m3",
    "target_value",
    "target_value_lag_1",
    "target_value_lag_3",
    "target_value_lag_6",
    "target_value_lag_12",
    "target_value_lag_24",
    "target_value_lag_48",
    "target_value_lag_72",
    "target_value_lag_168",
    "target_value_roll_mean_3",
    "target_value_roll_mean_6",
    "target_value_roll_mean_12",
    "target_value_roll_mean_24",
    "target_value_delta_1",
    "target_value_delta_6",
    "target_value_delta_24",
    "obs_humidity_delta_1",
    "obs_humidity_delta_6",
    "obs_humidity_delta_24",
    "obs_temp_delta_1",
    "obs_temp_delta_6",
    "obs_temp_delta_24",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "month_sin",
    "month_cos",
    "is_daytime",
]

DEFAULT_STATIC_FEATURES = ["lat", "lon", "elevation", "coastal_distance_km"]

DEFAULT_NWP_FEATURE_PREFIXES = ("nwp_", "gfs_", "kma_", "ecmwf_", "gdps_", "um_")


def run_nwp_mos_experiment(config: dict[str, Any]) -> Path:
    """Train/evaluate an honest issue-time-aligned NWP MOS humidity model.

    Unlike the sequence V2/V3 table, this path keeps each forecast row keyed by
    ``station_id, issue_time, valid_time, lead_hour``.  That prevents using a
    later forecast run for an earlier forecast-init sample.
    """

    frame, feature_columns, audit = build_nwp_mos_frame(config)
    model_config = dict(config.get("model", {}))
    target_name = str(config.get("data", {}).get("target_name", "humidity"))
    if not feature_columns:
        raise ValueError("NWP MOS frame has no feature columns after applying config.")

    split_masks = {name: frame["split"].astype(str).eq(name) for name in ("train", "val", "test")}
    if int(split_masks["train"].sum()) == 0:
        raise ValueError("NWP MOS training requires at least one train row in the prepared forecast archive.")
    if int(split_masks["test"].sum()) == 0:
        raise ValueError("NWP MOS evaluation requires at least one test row in the prepared forecast archive.")

    baseline_column = str(model_config.get("baseline_column") or config.get("data", {}).get("baseline_column") or "")
    residual_mode = bool(model_config.get("residual", bool(baseline_column)))
    if residual_mode and baseline_column not in frame.columns:
        raise ValueError(f"Configured baseline_column={baseline_column!r} is missing from the NWP MOS frame.")

    X, feature_names, imputer = _build_feature_matrix(frame, feature_columns, split_masks["train"])
    y_actual = frame["actual"].astype(float).to_numpy()
    if residual_mode:
        y_train_target = y_actual - frame[baseline_column].astype(float).to_numpy()
    else:
        y_train_target = y_actual

    model = _build_lightgbm_model(model_config)
    train_mask = split_masks["train"].to_numpy()
    val_mask = split_masks["val"].to_numpy()
    test_mask = split_masks["test"].to_numpy()
    model.fit(X[train_mask], y_train_target[train_mask])

    predictions = _predict_actual(model, X, frame, baseline_column=baseline_column if residual_mode else "")
    clip = config.get("data", {}).get("postprocess", {}).get("clip_prediction")
    if clip is not None:
        predictions = np.clip(predictions, float(clip[0]), float(clip[1]))

    experiment_dir = _create_experiment_dir(config)
    _snapshot_config(experiment_dir, config)
    payload = {
        "model": model,
        "feature_columns": feature_columns,
        "feature_names": feature_names,
        "imputer": imputer,
        "baseline_column": baseline_column if residual_mode else "",
        "residual_mode": residual_mode,
    }
    with (experiment_dir / "model.pkl").open("wb") as handle:
        pickle.dump(payload, handle)

    metrics_by_split: dict[str, dict[str, float]] = {}
    prediction_frames: dict[str, pd.DataFrame] = {}
    for split_name, mask in (("val", val_mask), ("test", test_mask)):
        if not bool(mask.any()):
            continue
        pred_frame = _prediction_frame(frame.loc[mask].copy(), predictions[mask], target_name=target_name)
        prediction_frames[split_name] = pred_frame
        metrics_by_split[split_name] = compute_prediction_metrics(pred_frame)
        write_table(pred_frame, experiment_dir / f"predictions_{split_name}.csv")
        write_json(metrics_by_split[split_name], experiment_dir / f"metrics_{split_name}.json")

    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame(
            {"feature_name": feature_names, "importance": [float(v) for v in model.feature_importances_]}
        ).sort_values("importance", ascending=False)
        write_table(importance, experiment_dir / "feature_importance.csv")

    metadata = build_future_feature_metadata(config)
    metadata.update(
        {
            "issue_time_aligned": True,
            "training_rows": int(train_mask.sum()),
            "validation_rows": int(val_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "baseline_column": baseline_column if residual_mode else None,
            "feature_columns": feature_columns,
        }
    )
    write_json(metadata, experiment_dir / "future_feature_metadata.json")
    write_json(audit, experiment_dir / "nwp_mos_frame_audit.json")
    summary = {
        "experiment": config.get("experiment", {}),
        "metrics": metrics_by_split,
        "future_features": metadata,
        "forecast_schema_version": metadata.get("forecast_schema_version"),
        "forecast_schema_valid": metadata.get("forecast_schema_valid"),
        "forecast_source_schema_valid": metadata.get("forecast_source_schema_valid"),
        "forecast_source_path": metadata.get("forecast_source_path"),
        "uses_patch_features": metadata.get("uses_patch_features"),
        "patch_features_enabled": metadata.get("patch_features_enabled"),
        "patch_size": metadata.get("patch_size"),
        "patch_feature_set": metadata.get("patch_feature_set"),
        "patch_feature_mode": metadata.get("patch_feature_mode"),
    }
    write_json(summary, experiment_dir / "experiment_summary.json")
    print(experiment_dir)
    return experiment_dir


def build_nwp_mos_frame(config: dict[str, Any]) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    paths = config.get("paths", {})
    nwp_path = paths.get("nwp_forecast_csv") or paths.get("future_weather_csv")
    if not nwp_path:
        raise ValueError("NWP MOS config requires paths.nwp_forecast_csv or paths.future_weather_csv.")

    base_config_path = config.get("data", {}).get("base_training_config")
    if base_config_path:
        base_config = load_yaml(base_config_path)
        observations = load_or_prepare_v2_training_table(base_config)
    else:
        observations = load_or_prepare_v2_training_table(config)
    observations = observations.copy()
    observations["station_id"] = observations["station_id"].astype(str)
    observations["datetime"] = pd.to_datetime(observations["datetime"], utc=True)
    if "target_value" not in observations.columns:
        target = str(config.get("data", {}).get("target_name", "humidity"))
        observations["target_value"] = observations[target].astype(float)

    nwp = load_future_weather_archive(nwp_path, config)
    max_lead = int(config.get("data", {}).get("window", {}).get("prediction_length", config.get("data", {}).get("max_lead_hour", 24)))
    min_lead = int(config.get("data", {}).get("min_lead_hour", 1))
    nwp = nwp.loc[nwp["lead_hour"].between(min_lead, max_lead)].copy()

    target_name = str(config.get("data", {}).get("target_name", "humidity"))
    valid_columns = ["station_id", "datetime", "target_value", target_name, "split"]
    valid_columns = [column for column in valid_columns if column in observations.columns]
    valid = observations[valid_columns].drop_duplicates(["station_id", "datetime"]).rename(columns={"datetime": "valid_time"})
    if target_name in valid.columns:
        valid = valid.rename(columns={target_name: "actual"})
    else:
        valid = valid.rename(columns={"target_value": "actual"})

    feature_config = config.get("data", {}).get("features", {})
    init_features = [str(c) for c in feature_config.get("init_features", DEFAULT_INIT_FEATURES) if c in observations.columns]
    static_features = [str(c) for c in feature_config.get("static_features", DEFAULT_STATIC_FEATURES) if c in observations.columns]
    init_columns = ["station_id", "datetime", *init_features, *static_features]
    init_columns = list(dict.fromkeys(init_columns))
    init = observations[init_columns].drop_duplicates(["station_id", "datetime"]).rename(columns={"datetime": "issue_time"})
    init = init.rename(columns={column: f"init_{column}" for column in init_features})

    frame = nwp.merge(valid, on=["station_id", "valid_time"], how="inner")
    frame = frame.merge(init, on=["station_id", "issue_time"], how="inner")
    frame = frame.dropna(subset=["actual", "split"]).reset_index(drop=True)

    nwp_features = [str(c) for c in feature_config.get("nwp_features", []) if c in frame.columns]
    if not nwp_features:
        excluded = {"station_id", "issue_time", "valid_time", "lead_hour", "actual", "split", "target_value"}
        nwp_features = [
            column
            for column in frame.columns
            if column not in excluded and column.startswith(DEFAULT_NWP_FEATURE_PREFIXES)
        ]
    issue_time_features = _time_features_from_series(frame["issue_time"], prefix="issue")
    valid_time_features = _time_features_from_series(frame["valid_time"], prefix="valid")
    frame = pd.concat([frame, issue_time_features, valid_time_features], axis=1)
    time_features = list(issue_time_features.columns) + list(valid_time_features.columns)
    init_feature_columns = [f"init_{column}" for column in init_features if f"init_{column}" in frame.columns]
    static_feature_columns = [column for column in static_features if column in frame.columns]
    feature_columns = ["lead_hour", *nwp_features, *init_feature_columns, *static_feature_columns, *time_features]
    feature_columns = [column for column in dict.fromkeys(feature_columns) if column in frame.columns]

    audit = {
        "nwp_rows": int(len(nwp)),
        "joined_rows": int(len(frame)),
        "stations": sorted(frame["station_id"].astype(str).unique().tolist()) if not frame.empty else [],
        "lead_hour_min": int(frame["lead_hour"].min()) if not frame.empty else None,
        "lead_hour_max": int(frame["lead_hour"].max()) if not frame.empty else None,
        "feature_columns": feature_columns,
        "honesty_rule": "Each sample is keyed by station_id + issue_time + valid_time + lead_hour; no forecast issued after issue_time is used.",
    }
    return frame, feature_columns, audit


def _time_features_from_series(series: pd.Series, prefix: str) -> pd.DataFrame:
    timestamps = pd.to_datetime(series, utc=True)
    hour = timestamps.dt.hour.astype(float)
    doy = timestamps.dt.dayofyear.astype(float)
    month = timestamps.dt.month.astype(float)
    return pd.DataFrame(
        {
            f"{prefix}_hour_sin": np.sin(2.0 * np.pi * hour / 24.0),
            f"{prefix}_hour_cos": np.cos(2.0 * np.pi * hour / 24.0),
            f"{prefix}_doy_sin": np.sin(2.0 * np.pi * doy / 366.0),
            f"{prefix}_doy_cos": np.cos(2.0 * np.pi * doy / 366.0),
            f"{prefix}_month_sin": np.sin(2.0 * np.pi * month / 12.0),
            f"{prefix}_month_cos": np.cos(2.0 * np.pi * month / 12.0),
        },
        index=series.index,
    )


def _build_feature_matrix(frame: pd.DataFrame, feature_columns: list[str], train_mask: pd.Series) -> tuple[np.ndarray, list[str], dict[str, float]]:
    raw = frame[feature_columns].copy()
    categorical_columns = [column for column in raw.columns if raw[column].dtype == "object"]
    encoded = pd.get_dummies(raw, columns=categorical_columns, dummy_na=True, dtype=float)
    imputer = encoded.loc[train_mask].median(numeric_only=True).fillna(0.0).to_dict()
    encoded = encoded.fillna(imputer).fillna(0.0)
    encoded = encoded.replace([np.inf, -np.inf], 0.0)
    return encoded.to_numpy(dtype=np.float32), encoded.columns.tolist(), {str(k): float(v) for k, v in imputer.items()}


def _build_lightgbm_model(model_config: dict[str, Any]):
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:  # pragma: no cover - dependency is installed in the normal dev env.
        raise RuntimeError("NWP MOS requires lightgbm to be installed.") from exc

    params = {
        "n_estimators": 800,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 20260518,
        "n_jobs": -1,
        "verbosity": -1,
    }
    params.update(dict(model_config.get("params", {})))
    return LGBMRegressor(**params)


def _predict_actual(model: Any, X: np.ndarray, frame: pd.DataFrame, baseline_column: str = "") -> np.ndarray:
    model_prediction = np.asarray(model.predict(X), dtype=float)
    if baseline_column:
        return frame[baseline_column].astype(float).to_numpy() + model_prediction
    return model_prediction


def _prediction_frame(frame: pd.DataFrame, prediction: np.ndarray, target_name: str) -> pd.DataFrame:
    result = frame[["station_id", "issue_time", "valid_time", "lead_hour", "actual", "split"]].copy()
    result["prediction"] = prediction.astype(float)
    result["target_name"] = target_name
    result = result.rename(columns={"valid_time": "timestamp", "lead_hour": "horizon_step"})
    return result[["station_id", "issue_time", "timestamp", "horizon_step", "target_name", "prediction", "actual", "split"]]


def _create_experiment_dir(config: dict[str, Any]) -> Path:
    root = resolve_path(config.get("artifacts", {}).get("root_dir", "data/artifacts/v3_nwp_mos_experiments"))
    name = str(config.get("experiment", {}).get("name", "v3_nwp_mos"))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root / f"{name}_{timestamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _snapshot_config(experiment_dir: Path, config: dict[str, Any]) -> None:
    import yaml

    with (experiment_dir / "experiment_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate issue-time-aligned V3 NWP MOS humidity model.")
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()
    config = load_yaml(args.config)
    run_nwp_mos_experiment(config)


if __name__ == "__main__":
    main()
