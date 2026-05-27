from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from weather_korea_forecast.v4.operational_ensemble import run_operational_ensemble


def _write_component(
    root: Path,
    name: str,
    offset: float,
    *,
    operational_valid: bool = True,
    backtest_only: bool = False,
    future_feature_source: str = "prepared_forecast_csv",
    schema_valid: bool = True,
    diagnostic: bool = False,
    drop_last_test_row: bool = False,
    actual_offset: float = 0.0,
    target_name: str | None = "temp",
    prediction_target_name: str = "temp",
    nan_prediction: bool = False,
) -> Path:
    exp = root / name
    exp.mkdir(parents=True)
    rows_val = []
    rows_test = []
    for split_rows, start, count in ((rows_val, "2024-01-01T00:00:00Z", 6), (rows_test, "2024-01-02T00:00:00Z", 6)):
        issue_times = pd.date_range(start, periods=count, freq="3h", tz="UTC")
        for issue in issue_times:
            for horizon in (1, 2, 3):
                timestamp = issue + pd.Timedelta(hours=horizon)
                actual = 10.0 + 0.1 * horizon + 0.01 * len(split_rows) + actual_offset
                split_rows.append(
                    {
                        "station_id": "108",
                        "issue_time": issue.isoformat(),
                        "timestamp": timestamp.isoformat(),
                        "horizon_step": horizon,
                        "target_name": prediction_target_name,
                        "prediction": actual + offset,
                        "actual": actual,
                    }
                )
    if drop_last_test_row:
        rows_test = rows_test[:-1]
    if nan_prediction:
        rows_test[0]["prediction"] = None
    pd.DataFrame(rows_val).to_csv(exp / "predictions_val.csv", index=False)
    pd.DataFrame(rows_test).to_csv(exp / "predictions_test.csv", index=False)
    summary = {
        "experiment_name": name,
        "experiment": {"name": name, "notes": "DIAGNOSTIC" if diagnostic else ""},
        "future_feature_source": future_feature_source,
        "operational_valid": operational_valid,
        "backtest_only": backtest_only,
        "forecast_source_schema_valid": schema_valid,
        "metrics": {"test": {"rmse": abs(offset), "mae": abs(offset), "bias": offset}},
    }
    if target_name is not None:
        summary["target_name"] = target_name
    (exp / "experiment_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return exp


def _ensemble_config(tmp_path: Path, components: list[Path]) -> dict:
    return {
        "experiment": {"name": "v4_temp_operational_ensemble_test", "version": "v4", "v4_stage": "v4_operational_ensemble"},
        "paths": {"prepared_forecast_csv": "prepared.csv"},
        "data": {
            "target_name": "temp",
            "future_features": {
                "track": "nwp_assisted_mos",
                "source": "prepared_forecast_csv",
                "operational_valid": True,
                "backtest_only": False,
                "schema": {"version": "v4-prepared-forecast-v1", "valid": True},
                "weather_columns": ["nwp_t2m"],
            },
        },
        "ensemble": {
            "methods": ["simple_average", "inverse_rmse_weight"],
            "components": [{"name": path.name, "experiment_dir": str(path)} for path in components],
        },
        "artifacts": {"root_dir": str(tmp_path / "artifacts"), "profile": "minimal"},
    }


def test_operational_ensemble_writes_weights_metrics_predictions_and_audit(tmp_path: Path) -> None:
    c1 = _write_component(tmp_path, "ridge", 0.4)
    c2 = _write_component(tmp_path, "lgbm", 0.05)
    config = _ensemble_config(tmp_path, [c1, c2])

    out = run_operational_ensemble(config)

    assert (out / "ensemble_weights.json").exists()
    assert (out / "ensemble_component_metrics.csv").exists()
    assert (out / "ensemble_component_audit.json").exists()
    assert (out / "ensemble_predictions_test.csv").exists()
    assert (out / "metrics_test.json").exists()
    summary = json.loads((out / "experiment_summary.json").read_text(encoding="utf-8"))
    weights = json.loads((out / "ensemble_weights.json").read_text(encoding="utf-8"))
    audit = json.loads((out / "ensemble_component_audit.json").read_text(encoding="utf-8"))
    metrics = json.loads((out / "metrics_test.json").read_text(encoding="utf-8"))
    assert summary["operational_valid"] is True
    assert summary["future_feature_source"] == "prepared_forecast_csv"
    assert weights["selected_method"] in {"simple_average", "inverse_rmse_weight"}
    assert audit["valid"] is True
    assert metrics["rmse"] <= 0.4


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"operational_valid": False}, "operational_valid is not true"),
        ({"backtest_only": True}, "backtest_only is not explicitly false"),
        ({"backtest_only": None}, "backtest_only is not explicitly false"),
        ({"future_feature_source": "era5_reanalysis"}, "future_feature_source"),
        ({"schema_valid": False}, "forecast schema is not valid"),
        ({"diagnostic": True}, "diagnostic component is not allowed"),
        ({"target_name": None}, "target_name is missing"),
        ({"target_name": "humidity"}, "target_name"),
    ],
)
def test_operational_ensemble_rejects_invalid_component_provenance(tmp_path: Path, kwargs: dict, match: str) -> None:
    c1 = _write_component(tmp_path, "ridge", 0.4)
    c2 = _write_component(tmp_path, "bad", 0.05, **kwargs)

    with pytest.raises(ValueError, match=match):
        run_operational_ensemble(_ensemble_config(tmp_path, [c1, c2]))


def test_operational_ensemble_rejects_component_key_coverage_mismatch(tmp_path: Path) -> None:
    c1 = _write_component(tmp_path, "ridge", 0.4)
    c2 = _write_component(tmp_path, "lgbm", 0.05, drop_last_test_row=True)

    with pytest.raises(ValueError, match="key coverage mismatch"):
        run_operational_ensemble(_ensemble_config(tmp_path, [c1, c2]))


def test_operational_ensemble_rejects_component_actual_mismatch(tmp_path: Path) -> None:
    c1 = _write_component(tmp_path, "ridge", 0.4)
    c2 = _write_component(tmp_path, "lgbm", 0.05, actual_offset=1.0)

    with pytest.raises(ValueError, match="actual values differ"):
        run_operational_ensemble(_ensemble_config(tmp_path, [c1, c2]))


def test_operational_ensemble_rejects_missing_prediction_values(tmp_path: Path) -> None:
    c1 = _write_component(tmp_path, "ridge", 0.4)
    c2 = _write_component(tmp_path, "lgbm", 0.05, nan_prediction=True)

    with pytest.raises(ValueError, match="missing values"):
        run_operational_ensemble(_ensemble_config(tmp_path, [c1, c2]))


def test_operational_ensemble_rejects_prediction_target_mismatch(tmp_path: Path) -> None:
    c1 = _write_component(tmp_path, "ridge", 0.4)
    c2 = _write_component(tmp_path, "lgbm", 0.05, prediction_target_name="humidity")

    with pytest.raises(ValueError, match="target_name values"):
        run_operational_ensemble(_ensemble_config(tmp_path, [c1, c2]))


@pytest.mark.parametrize(
    "column,value",
    [
        ("station_id", None),
        ("station_id", ""),
        ("issue_time", None),
        ("issue_time", "not-a-time"),
        ("timestamp", None),
        ("timestamp", "not-a-time"),
        ("horizon_step", None),
        ("horizon_step", "not-a-number"),
        ("target_name", None),
        ("target_name", ""),
    ],
)
def test_operational_ensemble_rejects_invalid_key_values_before_coercion(tmp_path: Path, column: str, value: object) -> None:
    c1 = _write_component(tmp_path, "ridge", 0.4)
    c2 = _write_component(tmp_path, "lgbm", 0.05)
    path = c2 / "predictions_test.csv"
    frame = pd.read_csv(path)
    frame.loc[0, column] = value
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="invalid key values before coercion"):
        run_operational_ensemble(_ensemble_config(tmp_path, [c1, c2]))


def test_operational_ensemble_rejects_non_integer_horizon_step(tmp_path: Path) -> None:
    c1 = _write_component(tmp_path, "ridge", 0.4)
    c2 = _write_component(tmp_path, "lgbm", 0.05)
    path = c2 / "predictions_test.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "horizon_step"] = 1.5
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="non-integer horizon_step"):
        run_operational_ensemble(_ensemble_config(tmp_path, [c1, c2]))
