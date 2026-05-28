from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import weather_korea_forecast.v4.operational_performance as operational_performance
from weather_korea_forecast.v4.operational_performance import (
    build_ensemble_results,
    classify_benchmark_reliability,
    evaluate_site_readiness_gate,
    evaluate_v4c_gate,
    fit_calibration_candidates,
    residual_debug_summary,
    summarize_patch_ablation_results,
    validate_lgbm_grid_results,
    write_operational_performance_html,
)


def test_benchmark_reliability_classifies_short_medium_strong_and_seasonal() -> None:
    assert classify_benchmark_reliability(forecast_cycle_count=30, date_span_days=8, season_count=1) == "short"
    assert classify_benchmark_reliability(forecast_cycle_count=90, date_span_days=23, season_count=1) == "medium"
    assert classify_benchmark_reliability(forecast_cycle_count=180, date_span_days=46, season_count=2) == "strong"
    assert classify_benchmark_reliability(forecast_cycle_count=1200, date_span_days=366, season_count=4) == "seasonal"


def test_time_ordered_splits_use_full_medium_archive() -> None:
    cycles = pd.date_range("2026-05-01T00:00Z", periods=90, freq="6h")
    frame = pd.DataFrame(
        {
            "issue_time": np.repeat(cycles, 2),
            "station_id": ["108", "112"] * len(cycles),
        }
    )

    split = operational_performance.add_time_ordered_splits(frame, train_cycles=20, val_cycles=5, test_cycles=5)

    split_cycles = split.groupby("split")["issue_time"].nunique().to_dict()
    assert split_cycles == {"test": 14, "train": 62, "val": 14}
    assert split["issue_time"].nunique() == 90


def test_calibration_selection_uses_holdout_and_only_applies_improving_candidate() -> None:
    calibration = pd.DataFrame(
        {
            "station_id": ["108", "108", "112", "112"],
            "horizon_step": [1, 2, 1, 2],
            "actual": [10.0, 11.0, 20.0, 21.0],
            "prediction": [8.0, 9.0, 18.0, 19.0],
        }
    )
    holdout = pd.DataFrame(
        {
            "station_id": ["108", "108", "112", "112"],
            "horizon_step": [1, 2, 1, 2],
            "actual": [12.0, 13.0, 22.0, 23.0],
            "prediction": [10.0, 11.0, 20.0, 21.0],
        }
    )
    result = fit_calibration_candidates(
        calibration,
        holdout,
        actual_column="actual",
        prediction_column="prediction",
        candidates=["none", "global_mean_bias", "per_horizon_mean_bias"],
    )

    assert result.selected_name == "global_mean_bias"
    assert result.selected_holdout_metrics["rmse"] == 0.0
    corrected = result.selected.apply(holdout)
    np.testing.assert_allclose(corrected, holdout["actual"].to_numpy())


def test_isotonic_calibration_alias_applies_model() -> None:
    calibration = pd.DataFrame(
        {
            "actual": [0.0, 10.0, 20.0, 30.0],
            "prediction": [0.0, 1.0, 2.0, 3.0],
        }
    )
    holdout = pd.DataFrame(
        {
            "actual": [10.0, 20.0],
            "prediction": [1.0, 2.0],
        }
    )

    result = fit_calibration_candidates(
        calibration,
        holdout,
        actual_column="actual",
        prediction_column="prediction",
        candidates=["isotonic_calibration"],
    )

    np.testing.assert_allclose(result.selected.apply(holdout), holdout["actual"].to_numpy())


def test_residual_debug_summary_verifies_sign_and_final_prediction() -> None:
    frame = pd.DataFrame(
        {
            "actual": [12.0, 15.0, 18.0],
            "baseline": [10.0, 14.0, 20.0],
            "predicted_residual": [2.0, 1.0, -2.0],
            "prediction": [12.0, 15.0, 18.0],
        }
    )

    debug = residual_debug_summary(
        frame,
        actual_column="actual",
        baseline_column="baseline",
        predicted_residual_column="predicted_residual",
        prediction_column="prediction",
    )

    assert debug["residual_target_sign_ok"] is True
    assert debug["final_prediction_formula_ok"] is True
    assert debug["residual_target_mean"] == 1 / 3


def test_patch_ablation_improvement_metrics_use_no_patch_baseline() -> None:
    summary = summarize_patch_ablation_results(
        [
            {"mode": "no_patch", "rmse": 1.8, "worst_station_rmse": 2.6, "late_horizon_rmse": 2.0},
            {"mode": "patch3", "rmse": 1.7, "worst_station_rmse": 2.7, "late_horizon_rmse": 1.8},
            {"mode": "patch5", "rmse": 1.6, "worst_station_rmse": 2.5, "late_horizon_rmse": 1.7},
        ]
    )

    assert summary["best_mode"] == "patch5"
    assert summary["patch_improvement_rmse"] == 0.19999999999999996
    assert summary["patch_improvement_worst_station"] == 0.10000000000000009
    assert summary["patch_improvement_late_horizon"] == 0.30000000000000004


def test_patch_ablation_status_requires_true_grid_patch_modes() -> None:
    short_status = operational_performance.patch_ablation_status(["no_patch", "proxy_patch5"])
    complete_status = operational_performance.patch_ablation_status(["no_patch", "proxy_patch5", "true_patch3", "true_patch5"])

    assert short_status["patch_ablation_completed"] is False
    assert short_status["patch_ablation_missing_modes"] == ["true_patch3", "true_patch5"]
    assert complete_status["patch_ablation_completed"] is True
    assert complete_status["patch_ablation_missing_modes"] == []


def test_true_patch_feature_loader_rejects_non_true_grid_provenance(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "station_id": ["108"],
            "forecast_init_time": pd.to_datetime(["2026-05-01T00:00Z"]),
            "valid_time": pd.to_datetime(["2026-05-01T01:00Z"]),
            "horizon_step": [1],
        }
    )
    feature_csv = tmp_path / "bad_patch.csv"
    pd.DataFrame(
        {
            "station_id": ["108"],
            "forecast_init_time": ["2026-05-01T00:00Z"],
            "valid_time": ["2026-05-01T01:00Z"],
            "horizon_step": [1],
            "patch_size": [3],
            "patch_feature_mode": ["station_neighborhood_proxy"],
            "nwp_t2m_patch_mean": [12.0],
        }
    ).to_csv(feature_csv, index=False)

    with pytest.raises(ValueError, match="non-true-grid"):
        operational_performance._with_true_grid_patch_features(frame, feature_csv)  # noqa: SLF001


def test_true_patch_feature_loader_rejects_metadata_only_patch_file(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "station_id": ["108"],
            "forecast_init_time": pd.to_datetime(["2026-05-01T00:00Z"]),
            "valid_time": pd.to_datetime(["2026-05-01T01:00Z"]),
            "horizon_step": [1],
        }
    )
    feature_csv = tmp_path / "metadata_only_patch.csv"
    pd.DataFrame(
        {
            "station_id": ["108"],
            "forecast_init_time": ["2026-05-01T00:00Z"],
            "valid_time": ["2026-05-01T01:00Z"],
            "horizon_step": [1],
            "patch_size": [3],
            "patch_feature_mode": ["true_gfs_grid_patch"],
        }
    ).to_csv(feature_csv, index=False)

    with pytest.raises(ValueError, match="no numeric true-grid patch feature"):
        operational_performance._with_true_grid_patch_features(frame, feature_csv)  # noqa: SLF001


def test_lgbm_grid_result_schema_requires_tuning_columns() -> None:
    validate_lgbm_grid_results(
        [
            {
                "grid_index": 0,
                "num_leaves": 31,
                "max_depth": -1,
                "learning_rate": 0.03,
                "n_estimators": 500,
                "min_child_samples": 20,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_alpha": 0.0,
                "reg_lambda": 0.0,
                "min_split_gain": 0.0,
                "val_rmse": 1.2,
                "val_mae": 0.9,
                "val_bias": 0.1,
                "val_n": 100,
            }
        ]
    )


def test_ensemble_weights_schema_and_best_selection(tmp_path) -> None:
    val = pd.DataFrame(
        {
            "station_id": ["108", "108", "112", "112"],
            "issue_time": pd.to_datetime(["2026-05-01T00:00Z"] * 4),
            "valid_time": pd.to_datetime(["2026-05-01T01:00Z", "2026-05-01T02:00Z"] * 2),
            "horizon_step": [1, 2, 1, 2],
            "actual": [10.0, 11.0, 20.0, 21.0],
        }
    )
    test = val.copy()
    components = {
        "weak": {
            "val": val.assign(prediction=[9.0, 10.0, 18.0, 19.0]),
            "test": test.assign(prediction=[9.0, 10.0, 18.0, 19.0]),
        },
        "strong": {
            "val": val.assign(prediction=[10.0, 11.0, 20.0, 21.0]),
            "test": test.assign(prediction=[10.0, 11.0, 20.0, 21.0]),
        },
    }

    result = build_ensemble_results(components, actual_column="actual", output_dir=tmp_path, target_name="temp")

    assert result["summary"]["status"] == "trained"
    assert result["summary"]["best"]["rmse"] == 0.0
    assert (tmp_path / "temp_ensemble_weights.json").exists()
    assert (tmp_path / "temp_ensemble_component_metrics.csv").exists()


def test_catboost_optional_skip(monkeypatch) -> None:
    monkeypatch.setattr(operational_performance, "CatBoostRegressor", None)
    frame = pd.DataFrame({"split": ["train"], "actual": [1.0], "baseline": [0.0], "station_id": ["108"]})

    result = operational_performance._fit_optional_catboost(  # noqa: SLF001
        frame,
        actual_column="actual",
        baseline_column="baseline",
        feature_columns=["station_id"],
    )

    assert result["summary"]["status"] == "skipped"


def test_v4c_and_site_readiness_keep_short_benchmark_caveat() -> None:
    summary = {
        "benchmark_reliability": "short",
        "patch_ablation_completed": False,
        "official_baselines": {
            "temp": {"operational_residual_lgbm_temp": {"rmse": 1.49}},
            "humidity": {"operational_residual_lgbm_humidity": {"rmse": 9.3}},
        },
    }

    v4c = evaluate_v4c_gate(summary)
    site = evaluate_site_readiness_gate(summary)

    assert v4c["status"] == "FAIL"
    assert "benchmark_reliability >= medium" in v4c["missing_conditions"]
    assert "patch_ablation_completed" in v4c["missing_conditions"]
    assert site["status"] == "WARN"
    assert "benchmark_reliability >= medium" in site["missing_conditions"]


def test_site_readiness_passes_only_for_medium_reliable_best_models() -> None:
    summary = {
        "benchmark_reliability": "medium",
        "best_operational_models": {
            "temp": {"rmse": 1.49},
            "humidity": {"rmse": 9.8},
        },
        "patch_ablation_completed": True,
        "artifacts": {"html_report": "report.html"},
    }

    assert evaluate_v4c_gate(summary)["status"] == "PASS"
    assert evaluate_site_readiness_gate(summary)["status"] == "PASS"


def test_site_readiness_requires_patch_ablation_and_report() -> None:
    summary = {
        "benchmark_reliability": "medium",
        "best_operational_models": {
            "temp": {"rmse": 1.49},
            "humidity": {"rmse": 9.8},
        },
        "patch_ablation_completed": False,
        "artifacts": {},
    }

    site = evaluate_site_readiness_gate(summary)

    assert site["status"] == "WARN"
    assert "patch_ablation_completed" in site["missing_conditions"]
    assert "report_generated" in site["missing_conditions"]


def test_operational_performance_html_contains_metrics_and_paths(tmp_path) -> None:
    summary = {
        "benchmark_reliability": "short",
        "official_baselines": {
            "temp": {
                "raw_gfs_t2m": {"rmse": 2.8, "mae": 2.2, "bias": -1.8},
                "operational_residual_lgbm_temp": {"rmse": 1.78, "mae": 1.38, "bias": -0.2},
            },
            "humidity": {
                "raw_gfs_rh": {"rmse": 12.4, "mae": 9.6, "bias": 3.4},
                "operational_residual_lgbm_humidity": {"rmse": 9.38, "mae": 7.37, "bias": -4.0},
            },
        },
        "v4c_gate": {"status": "FAIL", "missing_conditions": ["temp_rmse <= 1.5"]},
        "site_readiness": {"status": "WARN", "missing_conditions": ["benchmark_reliability >= medium"]},
        "artifacts": {"summary_json": "experiment_summary.json"},
    }

    html_path = write_operational_performance_html(summary, tmp_path / "report.html")
    html = html_path.read_text(encoding="utf-8")

    assert "Operational Performance Report" in html
    assert "operational_residual_lgbm_temp" in html
    assert "benchmark_reliability" in html
    assert "temp_rmse &lt;= 1.5" in html
    payload = json.loads((tmp_path / "summary_snapshot.json").read_text(encoding="utf-8"))
    assert payload["benchmark_reliability"] == "short"
