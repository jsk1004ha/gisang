from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from weather_korea_forecast.training.metrics import compute_prediction_metrics
from weather_korea_forecast.utils.config import load_yaml
from weather_korea_forecast.utils.io import write_json, write_table
from weather_korea_forecast.utils.paths import resolve_path
from weather_korea_forecast.v2.future_features import build_future_feature_metadata
from weather_korea_forecast.service.export_forecast import is_blocked_provenance

KEY_COLUMNS = ["station_id", "issue_time", "timestamp", "horizon_step", "target_name"]
SUPPORTED_METHODS = ["simple_average", "inverse_rmse_weight", "horizonwise_linear_weight", "constrained_least_squares"]


def run_operational_ensemble(config: dict[str, Any]) -> Path:
    components = _load_components(config)
    if len(components) < 2:
        raise ValueError("Operational ensemble requires at least two component experiment directories.")
    component_audit = _audit_components(components, config)
    expected_target = str(config.get("data", {}).get("target_name", "temp"))
    val = _merge_predictions(components, "val", expected_target=expected_target)
    test = _merge_predictions(components, "test", expected_target=expected_target)
    if val.empty or test.empty:
        raise ValueError("Operational ensemble requires non-empty val and test component predictions.")

    method_payloads = _fit_methods(val, components, _configured_methods(config))
    selected = min(method_payloads, key=lambda payload: payload["val_metrics"]["rmse"])
    test_predictions = _predict_with_payload(test, selected, components)
    prediction_frame = _prediction_output_frame(test, test_predictions)
    metrics = compute_prediction_metrics(prediction_frame)

    experiment_dir = _create_experiment_dir(config)
    _snapshot_config(experiment_dir, config)
    write_table(prediction_frame, experiment_dir / "predictions_test.csv")
    write_table(prediction_frame, experiment_dir / "ensemble_predictions_test.csv")
    write_json(metrics, experiment_dir / "metrics_test.json")
    component_metrics = _component_metrics(test, components)
    write_table(component_metrics, experiment_dir / "ensemble_component_metrics.csv")
    write_json(component_audit, experiment_dir / "ensemble_component_audit.json")
    weights_payload = {
        "selected_method": selected["method"],
        "methods": method_payloads,
        "components": [component["name"] for component in components],
    }
    write_json(weights_payload, experiment_dir / "ensemble_weights.json")
    _write_scatter(prediction_frame, experiment_dir / "baseline_vs_final_scatter.png")

    metadata = build_future_feature_metadata(config)
    summary = {
        "experiment": config.get("experiment", {}),
        "experiment_name": config.get("experiment", {}).get("name"),
        "version": config.get("experiment", {}).get("version", "v4"),
        "model_name": config.get("model", {}).get("name", config.get("experiment", {}).get("name")),
        "model_type": "operational_ensemble",
        "target_name": config.get("data", {}).get("target_name", "temp"),
        "track": config.get("data", {}).get("future_features", {}).get("track", "nwp_assisted_mos"),
        "metrics": {"test": metrics},
        "future_features": metadata,
        "uses_future_weather_features": metadata.get("uses_future_weather_features"),
        "future_feature_source": metadata.get("future_feature_source"),
        "operational_valid": metadata.get("operational_valid"),
        "backtest_only": metadata.get("backtest_only"),
        "forecast_schema_version": metadata.get("forecast_schema_version"),
        "forecast_schema_valid": metadata.get("forecast_schema_valid"),
        "forecast_source_schema_valid": metadata.get("forecast_source_schema_valid"),
        "forecast_archive_adequate": metadata.get("forecast_archive_adequate"),
        "forecast_archive_row_count": metadata.get("forecast_archive_row_count"),
        "forecast_archive_station_count": metadata.get("forecast_archive_station_count"),
        "forecast_archive_issue_time_count": metadata.get("forecast_archive_issue_time_count"),
        "forecast_source_path": metadata.get("forecast_source_path"),
        "v4_stage": config.get("experiment", {}).get("v4_stage", "v4_operational_ensemble"),
        "ensemble_method": selected["method"],
        "ensemble_component_count": len(components),
        "ensemble_component_metrics_path": "ensemble_component_metrics.csv",
        "ensemble_component_audit_path": "ensemble_component_audit.json",
        "ensemble_weights_path": "ensemble_weights.json",
    }
    write_json(summary, experiment_dir / "experiment_summary.json")
    print(experiment_dir)
    return experiment_dir


def _configured_methods(config: dict[str, Any]) -> list[str]:
    methods = [str(method) for method in config.get("ensemble", {}).get("methods", SUPPORTED_METHODS)]
    invalid = sorted(set(methods) - set(SUPPORTED_METHODS))
    if invalid:
        raise ValueError(f"Unsupported ensemble methods: {invalid}; supported={SUPPORTED_METHODS}")
    if not methods:
        raise ValueError("ensemble.methods must contain at least one method.")
    return methods


def _load_components(config: dict[str, Any]) -> list[dict[str, Any]]:
    raw_components = config.get("ensemble", {}).get("components", [])
    components: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_components):
        if isinstance(raw, str):
            name = Path(raw).name
            path = raw
        else:
            name = str(raw.get("name") or Path(str(raw.get("experiment_dir", idx))).name)
            path = raw.get("experiment_dir") or raw.get("path")
        if not path:
            raise ValueError(f"Ensemble component {idx} is missing experiment_dir/path.")
        components.append({"name": name, "dir": resolve_path(path)})
    return components


def _audit_components(components: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Validate component honesty before an ensemble can claim operational status."""

    ensemble_config = dict(config.get("ensemble", {}))
    allow_diagnostic = bool(ensemble_config.get("allow_diagnostic_components", False))
    expected_source = str(ensemble_config.get("expected_future_feature_source", "prepared_forecast_csv"))
    expected_target = str(config.get("data", {}).get("target_name", "temp"))
    rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    for component in components:
        summary = _load_component_summary(component)
        future = dict(summary.get("future_features") or {})
        experiment = summary.get("experiment") if isinstance(summary.get("experiment"), dict) else {}
        source = str(summary.get("future_feature_source") or future.get("future_feature_source") or future.get("source") or "none")
        row = {
            "name": component["name"],
            "experiment_name": str(summary.get("experiment_name") or experiment.get("name") or component["name"]),
            "experiment_dir": str(component["dir"]),
            "target_name": summary.get("target_name"),
            "future_feature_source": source,
            "operational_valid": summary.get("operational_valid", future.get("operational_valid")),
            "backtest_only": summary.get("backtest_only", future.get("backtest_only")),
            "forecast_source_schema_valid": summary.get(
                "forecast_source_schema_valid",
                summary.get("forecast_schema_valid", future.get("forecast_source_schema_valid", future.get("forecast_schema_valid"))),
            ),
            "forecast_archive_adequate": summary.get(
                "forecast_archive_adequate",
                summary.get("real_forecast_archive_adequate", future.get("forecast_archive_adequate", (future.get("archive") or {}).get("adequate") if isinstance(future.get("archive"), dict) else None)),
            ),
            "forecast_source_path": summary.get(
                "forecast_source_path",
                future.get("forecast_source_path", future.get("prepared_forecast_archive", future.get("prepared_forecast_csv"))),
            ),
            "diagnostic": _looks_diagnostic(component, summary),
            "valid": True,
            "reasons": [],
        }
        if row["target_name"] is None:
            row["valid"] = False
            row["reasons"].append("target_name is missing")
        elif str(row["target_name"]) != expected_target:
            row["valid"] = False
            row["reasons"].append(f"target_name {row['target_name']!r} != expected {expected_target!r}")
        if row["operational_valid"] is not True:
            row["valid"] = False
            row["reasons"].append("operational_valid is not true")
        if row["backtest_only"] is not False:
            row["valid"] = False
            row["reasons"].append("backtest_only is not explicitly false")
        if source != expected_source:
            row["valid"] = False
            row["reasons"].append(f"future_feature_source {source!r} != {expected_source!r}")
        if row["forecast_source_schema_valid"] is not True:
            row["valid"] = False
            row["reasons"].append("forecast schema is not valid")
        if row["forecast_archive_adequate"] is not True:
            row["valid"] = False
            row["reasons"].append("forecast archive is not adequate")
        if not row["forecast_source_path"]:
            row["valid"] = False
            row["reasons"].append("forecast source path is missing")
        if row["diagnostic"]:
            row["valid"] = False
            row["reasons"].append("diagnostic component is not allowed")
        rows.append(row)
        if not row["valid"]:
            invalid_rows.append(row)
    audit = {
        "valid": not invalid_rows,
        "allow_diagnostic_components": allow_diagnostic,
        "expected_future_feature_source": expected_source,
        "expected_target_name": expected_target,
        "components": rows,
    }
    if invalid_rows:
        audit["invalid_components"] = invalid_rows
        preview = "; ".join(f"{row['name']}: {', '.join(row['reasons'])}" for row in invalid_rows[:5])
        raise ValueError(f"Operational ensemble component audit failed: {preview}")
    return audit


def _load_component_summary(component: dict[str, Any]) -> dict[str, Any]:
    summary_path = component["dir"] / "experiment_summary.json"
    if not summary_path.exists():
        raise ValueError(f"Ensemble component {component['name']} is missing experiment_summary.json: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"Ensemble component {component['name']} experiment_summary.json must be an object.")
    metadata_path = component["dir"] / "future_feature_metadata.json"
    if metadata_path.exists() and not summary.get("future_features"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if isinstance(metadata, dict):
            summary["future_features"] = metadata
    return summary


def _looks_diagnostic(component: dict[str, Any], summary: dict[str, Any]) -> bool:
    experiment = summary.get("experiment") if isinstance(summary.get("experiment"), dict) else {}
    haystack = " ".join(
        str(value).lower()
        for value in (
            component["dir"].parent.name,
            component["dir"].name,
            component["name"],
            summary.get("experiment_name", ""),
            summary.get("track", ""),
            summary.get("future_feature_source", ""),
            summary.get("notes", ""),
            experiment.get("name", ""),
            experiment.get("notes", ""),
        )
    )
    return is_blocked_provenance(summary) or any(
        token in haystack for token in ("diagnostic", "synthetic", "smoke", "fixture", "generated", "oracle")
    )


def _read_prediction(component: dict[str, Any], split: str, *, expected_target: str) -> pd.DataFrame:
    path = component["dir"] / f"predictions_{split}.csv"
    if not path.exists():
        raise ValueError(f"Missing {split} predictions for component {component['name']}: {path}")
    frame = pd.read_csv(path)
    missing = sorted(set(KEY_COLUMNS + ["prediction", "actual"]) - set(frame.columns))
    if missing:
        raise ValueError(f"Component {component['name']} predictions_{split}.csv missing columns: {missing}")
    frame = frame.copy()
    _validate_raw_key_columns(component, split, frame)
    frame["station_id"] = frame["station_id"].astype(str).str.strip()
    frame["issue_time"] = pd.to_datetime(frame["issue_time"], utc=True, errors="coerce")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["horizon_step"] = pd.to_numeric(frame["horizon_step"], errors="coerce")
    frame["target_name"] = frame["target_name"].astype(str).str.strip()
    _raise_if_missing_values(component, split, frame, KEY_COLUMNS)
    if frame["horizon_step"].mod(1).ne(0).any():
        raise ValueError(f"Component {component['name']} predictions_{split}.csv contains non-integer horizon_step values.")
    frame["issue_time"] = frame["issue_time"].astype(str)
    frame["timestamp"] = frame["timestamp"].astype(str)
    frame["horizon_step"] = frame["horizon_step"].astype(int)
    targets = sorted(set(frame["target_name"].dropna().astype(str)))
    if targets != [expected_target]:
        raise ValueError(
            f"Component {component['name']} predictions_{split}.csv target_name values {targets} "
            f"do not match expected target {expected_target!r}."
        )
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    frame["actual"] = pd.to_numeric(frame["actual"], errors="coerce")
    _raise_if_missing_values(component, split, frame, KEY_COLUMNS + ["prediction", "actual"])
    duplicate_count = int(frame.duplicated(KEY_COLUMNS).sum())
    if duplicate_count:
        raise ValueError(f"Component {component['name']} predictions_{split}.csv contains {duplicate_count} duplicate key rows.")
    return frame[KEY_COLUMNS + ["prediction", "actual"]]


def _validate_raw_key_columns(component: dict[str, Any], split: str, frame: pd.DataFrame) -> None:
    """Reject missing key values before coercion can stringify them as 'nan'/'NaT'."""

    station = frame["station_id"]
    target = frame["target_name"]
    blank_station = station.astype("string").str.strip().isin(["", "nan", "NaN", "None", "NaT"])
    blank_target = target.astype("string").str.strip().isin(["", "nan", "NaN", "None", "NaT"])
    missing_parts = {
        "station_id": int(station.isna().sum() + blank_station.fillna(False).sum()),
        "target_name": int(target.isna().sum() + blank_target.fillna(False).sum()),
        "issue_time": int(pd.to_datetime(frame["issue_time"], utc=True, errors="coerce").isna().sum()),
        "timestamp": int(pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").isna().sum()),
        "horizon_step": int(pd.to_numeric(frame["horizon_step"], errors="coerce").isna().sum()),
    }
    missing_parts = {column: count for column, count in missing_parts.items() if count > 0}
    if missing_parts:
        details = ", ".join(f"{column}={count}" for column, count in missing_parts.items())
        raise ValueError(f"Component {component['name']} predictions_{split}.csv contains invalid key values before coercion: {details}.")


def _merge_predictions(components: list[dict[str, Any]], split: str, *, expected_target: str) -> pd.DataFrame:
    frames = [(component, _read_prediction(component, split, expected_target=expected_target)) for component in components]
    if not frames:
        return pd.DataFrame()
    reference_component, reference = frames[0]
    reference_keys = set(map(tuple, reference[KEY_COLUMNS].to_numpy()))
    reference_actual = reference.set_index(KEY_COLUMNS)["actual"].astype(float).sort_index()
    merged = reference[KEY_COLUMNS + ["actual"]].copy()
    merged[f"prediction__{reference_component['name']}"] = reference["prediction"].astype(float).to_numpy()
    for component, frame in frames[1:]:
        keys = set(map(tuple, frame[KEY_COLUMNS].to_numpy()))
        if keys != reference_keys:
            missing = len(reference_keys - keys)
            extra = len(keys - reference_keys)
            raise ValueError(
                f"Component {component['name']} predictions_{split}.csv key coverage mismatch: "
                f"{missing} missing and {extra} extra rows versus {reference_component['name']}."
            )
        actual = frame.set_index(KEY_COLUMNS)["actual"].astype(float).sort_index()
        if not np.allclose(reference_actual.to_numpy(), actual.to_numpy(), rtol=1e-9, atol=1e-9):
            raise ValueError(f"Component {component['name']} predictions_{split}.csv actual values differ from {reference_component['name']}.")
        pred_col = f"prediction__{component['name']}"
        component_predictions = frame[KEY_COLUMNS + ["prediction"]].rename(columns={"prediction": pred_col})
        merged = merged.merge(component_predictions, on=KEY_COLUMNS, how="inner")
        if len(merged) != len(reference):
            raise ValueError(
                f"Component {component['name']} predictions_{split}.csv merge changed row count "
                f"from {len(reference)} to {len(merged)} despite matching key coverage."
            )
    _raise_if_missing_values({"name": "merged_ensemble"}, split, merged, ["actual", *_component_columns(components)])
    return merged.reset_index(drop=True)


def _raise_if_missing_values(component: dict[str, Any], split: str, frame: pd.DataFrame, columns: list[str]) -> None:
    missing = frame[columns].isna().sum()
    missing = missing.loc[missing.gt(0)]
    if not missing.empty:
        details = ", ".join(f"{column}={int(count)}" for column, count in missing.items())
        raise ValueError(f"Component {component['name']} predictions_{split}.csv contains missing values: {details}.")


def _component_columns(components: list[dict[str, Any]]) -> list[str]:
    return [f"prediction__{component['name']}" for component in components]


def _fit_methods(frame: pd.DataFrame, components: list[dict[str, Any]], methods: list[str]) -> list[dict[str, Any]]:
    payloads = []
    for method in methods:
        payload = _fit_method(frame, components, method)
        predictions = _predict_with_payload(frame, payload, components)
        metrics = compute_prediction_metrics(_prediction_output_frame(frame, predictions))
        payload["val_metrics"] = metrics
        payloads.append(payload)
    return payloads


def _fit_method(frame: pd.DataFrame, components: list[dict[str, Any]], method: str) -> dict[str, Any]:
    cols = _component_columns(components)
    if method == "simple_average":
        weights = {component["name"]: 1.0 / len(components) for component in components}
        return {"method": method, "weights": weights}
    if method == "inverse_rmse_weight":
        rmses = []
        for component, col in zip(components, cols):
            err = frame[col].astype(float) - frame["actual"].astype(float)
            rmses.append((component["name"], float(np.sqrt(np.mean(np.square(err))))))
        inv = np.asarray([1.0 / max(rmse, 1e-9) for _, rmse in rmses], dtype=float)
        inv = inv / inv.sum()
        return {"method": method, "weights": {name: float(weight) for (name, _), weight in zip(rmses, inv)}}
    if method == "constrained_least_squares":
        weights = _positive_linear_weights(frame[cols].to_numpy(dtype=float), frame["actual"].to_numpy(dtype=float))
        return {"method": method, "weights": {component["name"]: float(weight) for component, weight in zip(components, weights)}}
    if method == "horizonwise_linear_weight":
        horizon_weights: dict[str, dict[str, float]] = {}
        global_weights = _positive_linear_weights(frame[cols].to_numpy(dtype=float), frame["actual"].to_numpy(dtype=float))
        for horizon, group in frame.groupby("horizon_step"):
            if len(group) < len(components):
                weights = global_weights
            else:
                weights = _positive_linear_weights(group[cols].to_numpy(dtype=float), group["actual"].to_numpy(dtype=float))
            horizon_weights[str(int(horizon))] = {component["name"]: float(weight) for component, weight in zip(components, weights)}
        return {"method": method, "weights": {component["name"]: float(weight) for component, weight in zip(components, global_weights)}, "horizon_weights": horizon_weights}
    raise ValueError(f"Unsupported ensemble method: {method}")


def _positive_linear_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LinearRegression

    model = LinearRegression(fit_intercept=False, positive=True)
    model.fit(X, y)
    weights = np.asarray(model.coef_, dtype=float)
    weights = np.clip(weights, 0.0, None)
    if float(weights.sum()) <= 1e-12:
        weights = np.ones(X.shape[1], dtype=float)
    return weights / weights.sum()


def _predict_with_payload(frame: pd.DataFrame, payload: dict[str, Any], components: list[dict[str, Any]]) -> np.ndarray:
    cols = _component_columns(components)
    if payload["method"] == "horizonwise_linear_weight" and payload.get("horizon_weights"):
        output = np.zeros(len(frame), dtype=float)
        default = payload.get("weights", {})
        for idx, row in frame.reset_index(drop=True).iterrows():
            weights = payload["horizon_weights"].get(str(int(row["horizon_step"])), default)
            output[idx] = sum(float(weights.get(component["name"], 0.0)) * float(row[col]) for component, col in zip(components, cols))
        return output
    weights = payload.get("weights", {})
    return sum(float(weights.get(component["name"], 0.0)) * frame[col].astype(float).to_numpy() for component, col in zip(components, cols))


def _prediction_output_frame(frame: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    output = frame[KEY_COLUMNS + ["actual"]].copy()
    output["prediction"] = predictions.astype(float)
    return output[["station_id", "issue_time", "timestamp", "horizon_step", "target_name", "prediction", "actual"]]


def _component_metrics(frame: pd.DataFrame, components: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for component in components:
        pred = frame[f"prediction__{component['name']}"].astype(float).to_numpy()
        metrics = compute_prediction_metrics(_prediction_output_frame(frame, pred))
        rows.append({"component": component["name"], **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}})
    return pd.DataFrame(rows)


def _write_scatter(frame: pd.DataFrame, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(frame["actual"], frame["prediction"], s=8, alpha=0.5)
        lo = float(min(frame["actual"].min(), frame["prediction"].min()))
        hi = float(max(frame["actual"].max(), frame["prediction"].max()))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Ensemble prediction")
        ax.set_title("Baseline vs final ensemble")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    except Exception:
        return


def _create_experiment_dir(config: dict[str, Any]) -> Path:
    root = resolve_path(config.get("artifacts", {}).get("root_dir", "data/artifacts/v4_experiments"))
    name = str(config.get("experiment", {}).get("name", "v4_temp_operational_ensemble_72to24"))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root / f"{name}_{timestamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _snapshot_config(experiment_dir: Path, config: dict[str, Any]) -> None:
    import yaml

    with (experiment_dir / "experiment_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a V4 operational MOS ensemble from component experiment predictions.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_operational_ensemble(load_yaml(args.config))


if __name__ == "__main__":
    main()
