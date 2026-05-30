from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import weather_korea_forecast.v4.operational_report as operational_report
import weather_korea_forecast.v4.operational_performance as operational_performance
from weather_korea_forecast.v4.beta_targets import build_beta_targets_summary, circular_direction_mae, select_precip_probability_descriptor
from weather_korea_forecast.v4.operational_performance import (
    build_production_model_freeze_record,
    build_production_model_manifest,
    build_ensemble_results,
    classify_benchmark_reliability,
    evaluate_site_readiness_gate,
    evaluate_v4c_gate,
    fit_calibration_candidates,
    residual_debug_summary,
    summarize_variable_coverage,
    summarize_patch_ablation_results,
    validate_lgbm_grid_results,
    write_operational_performance_html,
)


_ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def test_benchmark_reliability_classifies_short_medium_strong_and_seasonal() -> None:
    assert classify_benchmark_reliability(forecast_cycle_count=30, date_span_days=8, season_count=1) == "short"
    assert classify_benchmark_reliability(forecast_cycle_count=90, date_span_days=23, season_count=1) == "medium"
    assert classify_benchmark_reliability(forecast_cycle_count=180, date_span_days=46, season_count=2) == "strong"
    assert classify_benchmark_reliability(forecast_cycle_count=1200, date_span_days=366, season_count=4) == "seasonal"


def test_full_variable_coverage_reports_available_and_missing_columns() -> None:
    frame = pd.DataFrame(
        {
            "nwp_t2m": [10.0, 11.0],
            "nwp_gust": [4.0, None],
            "nwp_specific_humidity": [0.004, 0.005],
        }
    )

    summary = summarize_variable_coverage(frame, variables=["nwp_t2m", "nwp_gust", "nwp_pwat", "nwp_specific_humidity"])

    by_variable = {row["variable"]: row for row in summary["variables"]}
    assert summary["full_variable_count"] == 3
    assert by_variable["nwp_t2m"]["coverage"] == 1.0
    assert by_variable["nwp_gust"]["coverage"] == 0.5
    assert by_variable["nwp_pwat"]["present"] is False
    assert "nwp_pwat" in summary["missing_variables"]


def test_full_variable_coverage_reports_split_specific_train_availability() -> None:
    frame = pd.DataFrame(
        {
            "split": ["train", "train", "val", "test"],
            "nwp_t2m": [10.0, 11.0, 12.0, 13.0],
            "nwp_pwat": [None, None, 18.0, 19.0],
        }
    )

    summary = summarize_variable_coverage(frame, variables=["nwp_t2m", "nwp_pwat"])

    by_variable = {row["variable"]: row for row in summary["variables"]}
    assert by_variable["nwp_t2m"]["train_coverage"] == 1.0
    assert by_variable["nwp_pwat"]["coverage"] == 0.5
    assert by_variable["nwp_pwat"]["train_coverage"] == 0.0
    assert by_variable["nwp_pwat"]["val_coverage"] == 1.0
    assert by_variable["nwp_pwat"]["test_coverage"] == 1.0


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


def test_joined_operational_frame_preserves_optional_wind_and_cloud_labels(tmp_path) -> None:
    nwp = tmp_path / "nwp.csv"
    pd.DataFrame(
        {
            "station_id": ["108"],
            "forecast_init_time": ["2026-05-01T00:00:00Z"],
            "valid_time": ["2026-05-01T01:00:00Z"],
            "horizon_step": [1],
            "nwp_t2m": [10.0],
            "nwp_humidity": [60.0],
            "nwp_wind_speed": [2.0],
            "nwp_cloud_cover": [70.0],
        }
    ).to_csv(nwp, index=False)
    obs = tmp_path / "obs.csv"
    pd.DataFrame(
        {
            "station_id": ["108"],
            "datetime": ["2026-05-01 10:00:00"],
            "temp": [11.0],
            "humidity": [65.0],
            "pressure": [1000.0],
            "wind_speed": [2.5],
            "precipitation": [0.0],
            "wd": [280.0],
            "cloud_class": ["cloudy"],
        }
    ).to_csv(obs, index=False)
    stations = tmp_path / "stations.csv"
    pd.DataFrame({"station_id": ["108"], "lat": [37.5], "lon": [127.0], "region_class": ["metro"]}).to_csv(stations, index=False)

    frame = operational_performance.load_joined_operational_frame(nwp_archive=nwp, observations=obs, station_metadata=stations)

    assert frame.loc[0, "wind_direction"] == 280.0
    assert frame.loc[0, "cloud_class"] == "cloudy"


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


def test_issue_hour_horizon_calibration_corrects_issue_time_bias() -> None:
    calibration = pd.DataFrame(
        {
            "station_id": ["108", "108", "112", "112"],
            "issue_time": pd.to_datetime(
                [
                    "2026-05-01T00:00Z",
                    "2026-05-01T06:00Z",
                    "2026-05-02T00:00Z",
                    "2026-05-02T06:00Z",
                ]
            ),
            "horizon_step": [1, 1, 1, 1],
            "actual": [12.0, 9.0, 22.0, 19.0],
            "prediction": [10.0, 10.0, 20.0, 20.0],
        }
    )
    holdout = pd.DataFrame(
        {
            "station_id": ["108", "108", "112", "112"],
            "issue_time": pd.to_datetime(
                [
                    "2026-05-03T00:00Z",
                    "2026-05-03T06:00Z",
                    "2026-05-04T00:00Z",
                    "2026-05-04T06:00Z",
                ]
            ),
            "horizon_step": [1, 1, 1, 1],
            "actual": [32.0, 29.0, 42.0, 39.0],
            "prediction": [30.0, 30.0, 40.0, 40.0],
        }
    )

    result = fit_calibration_candidates(
        calibration,
        holdout,
        actual_column="actual",
        prediction_column="prediction",
        candidates=["none", "per_horizon_mean_bias", "per_issue_hour_horizon_mean_bias"],
    )

    assert result.selected_name == "per_issue_hour_horizon_mean_bias"
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


def test_humidity_patch_policy_keeps_no_patch_and_filters_moisture_features() -> None:
    columns = [
        "true_patch5_nwp_t2m_patch_mean",
        "true_patch5_nwp_humidity_patch_mean",
        "true_patch5_nwp_pwat_patch_mean",
        "true_patch5_nwp_u10_patch_mean",
        "true_patch5_nwp_sp_patch_mean",
    ]

    assert operational_performance._target_patch_feature_columns("humidity", "no_patch", columns) == columns  # noqa: SLF001
    filtered = operational_performance._target_patch_feature_columns("humidity", "true_patch5", columns)  # noqa: SLF001

    assert "true_patch5_nwp_humidity_patch_mean" in filtered
    assert "true_patch5_nwp_pwat_patch_mean" in filtered
    assert "true_patch5_nwp_u10_patch_mean" not in filtered
    assert "true_patch5_nwp_sp_patch_mean" not in filtered


def test_humidity_official_lgbm_candidate_is_pinned_to_no_patch() -> None:
    assert operational_performance._is_official_lgbm_candidate("humidity", "no_patch") is True  # noqa: SLF001
    assert operational_performance._is_official_lgbm_candidate("humidity", "true_patch5") is False  # noqa: SLF001
    assert operational_performance._is_official_lgbm_candidate("temp", "true_patch5") is True  # noqa: SLF001


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

    assert result["summary"]["status"] == "accepted"
    assert result["summary"]["best"]["rmse"] == 0.0
    assert (tmp_path / "temp_ensemble_weights.json").exists()
    assert (tmp_path / "temp_ensemble_component_metrics.csv").exists()


def test_humidity_ensemble_rejects_when_holdout_or_bias_worsens(tmp_path) -> None:
    val = pd.DataFrame(
        {
            "station_id": ["108", "108", "112", "112"],
            "issue_time": pd.to_datetime(["2026-05-01T00:00Z"] * 4),
            "valid_time": pd.to_datetime(["2026-05-01T01:00Z", "2026-05-01T02:00Z"] * 2),
            "horizon_step": [1, 2, 1, 2],
            "actual": [50.0, 50.0, 50.0, 50.0],
        }
    )
    components = {
        "baseline_like": {"val": val.assign(prediction=[50.2, 49.8, 50.1, 49.9]), "test": val.assign(prediction=[50.2, 49.8, 50.1, 49.9])},
        "biased": {"val": val.assign(prediction=[45.0, 45.0, 45.0, 45.0]), "test": val.assign(prediction=[45.0, 45.0, 45.0, 45.0])},
    }

    result = build_ensemble_results(
        components,
        actual_column="actual",
        output_dir=tmp_path,
        target_name="humidity",
        baseline_validation_metrics={"rmse": 0.2, "bias": 0.0},
        baseline_test_metrics={"rmse": 0.2, "bias": 0.0},
        reject_if_bias_worse=True,
    )

    assert result["summary"]["selection_status"] == "rejected"
    assert result["best_test_prediction"] is None
    assert result["summary"]["rejected_reasons"]


def test_humidity_ensemble_selection_does_not_use_test_bias_for_rejection(tmp_path) -> None:
    val = pd.DataFrame(
        {
            "station_id": ["108", "108", "112", "112"],
            "issue_time": pd.to_datetime(["2026-05-01T00:00Z"] * 4),
            "valid_time": pd.to_datetime(["2026-05-01T01:00Z", "2026-05-01T02:00Z"] * 2),
            "horizon_step": [1, 2, 1, 2],
            "actual": [50.0, 51.0, 52.0, 53.0],
        }
    )
    test = val.copy()
    components = {
        "perfect_holdout_bad_test_a": {
            "val": val.assign(prediction=[50.0, 51.0, 52.0, 53.0]),
            "test": test.assign(prediction=[40.0, 41.0, 42.0, 43.0]),
        },
        "perfect_holdout_bad_test_b": {
            "val": val.assign(prediction=[50.0, 51.0, 52.0, 53.0]),
            "test": test.assign(prediction=[40.0, 41.0, 42.0, 43.0]),
        },
    }

    result = build_ensemble_results(
        components,
        actual_column="actual",
        output_dir=tmp_path,
        target_name="humidity",
        baseline_validation_metrics={"rmse": 0.5, "bias": 0.0},
        baseline_test_metrics={"rmse": 0.5, "bias": 0.0},
        reject_if_bias_worse=True,
    )

    assert result["summary"]["selection_status"] == "accepted"
    assert result["summary"]["best"]["bias"] == -10.0
    assert result["best_test_prediction"] is not None


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


def test_production_model_manifest_uses_target_specific_best_models(tmp_path: Path) -> None:
    html = tmp_path / "operational_performance_report.html"
    html.write_text("<html></html>", encoding="utf-8")
    summary = {
        "benchmark_reliability": "strong",
        "best_operational_models": {
            "temp": {"model": "ensemble_horizonwise_inverse_rmse", "rmse": 1.49, "mae": 1.1, "bias": 0.05},
            "humidity": {"model": "operational_residual_lgbm_humidity", "rmse": 9.8, "mae": 7.5, "bias": -1.0},
        },
        "beta_targets": {
            "precip_probability": {"descriptor": {"model_key": "nwp_direct", "source": "nwp_direct", "status": "direct", "confidence": "low"}},
            "wind": {"descriptor": {"model_key": "gfs_direct", "source": "gfs_direct", "status": "direct", "confidence": "low"}},
            "cloud": {"descriptor": {"model_key": "nwp_direct_or_lgbm_beta", "source": "nwp_direct", "status": "direct", "confidence": "low"}},
            "weather_code": {"descriptor": {"model_key": "rule_based_beta", "source": "rule_based_beta", "status": "beta", "confidence": "low"}},
        },
        "v4c_gate": {"status": "PASS"},
        "site_readiness": {"status": "PASS"},
        "report_generated": True,
        "artifacts": {"models": "models.pkl", "html_report": str(html)},
        "ensembles": {
            "temp": {
                "artifacts": {
                    "weights": "temp_ensemble_weights.json",
                    "metrics": "temp_ensemble_component_metrics.csv",
                }
            }
        },
    }

    manifest = build_production_model_manifest(summary)

    assert manifest["target_specific_models"] is True
    assert manifest["temp_model"] == "ensemble_horizonwise_inverse_rmse"
    assert manifest["humidity_model"] == "operational_residual_lgbm_humidity"
    assert manifest["temp_artifact_type"] == "ensemble"
    assert manifest["temp_model_artifact"] == "temp_ensemble_weights.json"
    assert manifest["temp_artifacts"]["component_model_artifact"] == "models.pkl"
    assert manifest["humidity_artifact_type"] == "model"
    assert manifest["humidity_model_artifact"] == "models.pkl"
    assert manifest["operational_beta_allowed"] is True
    assert manifest["operational_valid"] is True
    assert manifest["temperature_model"] == "ensemble_horizonwise_inverse_rmse"
    assert manifest["precip_probability_model"] == "nwp_direct"
    assert manifest["wind_model"] == "gfs_direct"
    assert manifest["cloud_model"] == "nwp_direct_or_lgbm_beta"
    assert manifest["weather_code_model"] == "rule_based_beta"


def test_production_model_manifest_reflects_final_gate_status() -> None:
    summary = {
        "benchmark_reliability": "strong",
        "best_operational_models": {
            "temp": {"model": "operational_residual_lgbm_temp", "rmse": 1.49, "mae": 1.1, "bias": 0.05},
            "humidity": {"model": "operational_residual_lgbm_humidity", "rmse": 9.8, "mae": 7.5, "bias": -1.0},
        },
        "v4c_gate": {"status": "PASS"},
        "site_readiness": {"status": "WARN"},
        "artifacts": {"models": "models.pkl"},
    }

    manifest = build_production_model_manifest(summary)

    assert manifest["v4c_gate_status"] == summary["v4c_gate"]["status"]
    assert manifest["site_readiness_status"] == summary["site_readiness"]["status"]
    assert manifest["operational_beta_allowed"] is False


def test_production_model_manifest_records_near_pass_and_caveats(tmp_path: Path) -> None:
    html = tmp_path / "operational_performance_report.html"
    html.write_text("<html></html>", encoding="utf-8")
    summary = {
        "benchmark_reliability": "strong",
        "report_generated": True,
        "model_improvement_frozen": True,
        "best_operational_models": {
            "temp": {"model": "ensemble_stationwise_inverse_rmse", "rmse": 1.694, "mae": 1.3, "bias": 0.1},
            "humidity": {"model": "operational_residual_lgbm_humidity", "rmse": 10.396, "mae": 8.0, "bias": 0.6},
        },
        "v4c_gate": {"status": "FAIL"},
        "site_readiness": {"status": "WARN"},
        "artifacts": {"models": str(tmp_path / "models.pkl"), "html_report": str(html)},
    }

    manifest = build_production_model_manifest(summary)

    assert manifest["temperature_status"] == "FAIL"
    assert manifest["humidity_status"] == "NEAR_PASS"
    assert manifest["humidity_bias_status"] == "PASS"
    assert manifest["temperature_rmse"] == 1.694
    assert manifest["temperature_mae"] == 1.3
    assert manifest["temperature_bias"] == 0.1
    assert manifest["beta_label_required"] == {"temperature": True, "humidity": True, "weather_code": True}
    assert "temperature_accuracy_caveat" in manifest["site_caveats"]
    assert "humidity_beta" in manifest["site_caveats"]
    assert manifest["forecast_schema_version"] == "forecast_points.v2-beta-sources"
    assert manifest["operational_valid"] is False
    assert manifest["model_improvement_frozen"] is True


def test_production_model_freeze_record_hashes_manifest_and_artifacts(tmp_path: Path) -> None:
    model = tmp_path / "models.pkl"
    model.write_text("model-bytes", encoding="utf-8")
    summary_json = tmp_path / "experiment_summary.json"
    summary_json.write_text("{}", encoding="utf-8")
    source_summary_json = tmp_path / "source_experiment_summary.json"
    source_summary_json.write_text('{"source": true}', encoding="utf-8")
    manifest = {
        "temp_model_artifact": str(model),
        "humidity_model_artifact": str(model),
        "forecast_schema_version": "forecast_points.v2-beta-sources",
        "temperature_status": "FAIL",
        "humidity_status": "NEAR_PASS",
        "humidity_bias_status": "PASS",
        "site_caveats": ["humidity_beta"],
    }
    summary = {
        "artifacts": {
            "summary_json": str(summary_json),
            "source_benchmark_summary_json": str(source_summary_json),
            "models": str(model),
        }
    }

    record = build_production_model_freeze_record(summary, manifest)

    assert record["manifest_sha256"]
    assert record["manifest_hash_convention"].startswith("sha256 over canonical JSON")
    assert record["benchmark_summary_path"] == str(source_summary_json)
    assert record["benchmark_summary_sha256"]
    assert record["model_artifact_sha256"]["temperature"]["sha256"]
    assert record["forecast_schema_version"] == "forecast_points.v2-beta-sources"
    assert "training" in record["forbidden_post_freeze_change_scopes"]


def test_production_model_freeze_record_requires_existing_benchmark_summary(tmp_path: Path) -> None:
    manifest = {"forecast_schema_version": "forecast_points.v2-beta-sources"}
    summary = {"artifacts": {"summary_json": str(tmp_path / "missing_experiment_summary.json")}}

    with pytest.raises(FileNotFoundError, match="benchmark summary"):
        build_production_model_freeze_record(summary, manifest)


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


def test_site_readiness_passes_only_for_medium_reliable_best_models(tmp_path) -> None:
    report = tmp_path / "report.html"
    report.write_text("<html></html>", encoding="utf-8")
    summary = {
        "benchmark_reliability": "medium",
        "best_operational_models": {
            "temp": {"rmse": 1.49},
            "humidity": {"rmse": 9.8},
        },
        "patch_ablation_completed": True,
        "report_generated": True,
        "artifacts": {"html_report": str(report)},
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


def test_site_readiness_requires_report_file_when_artifact_path_is_declared(tmp_path) -> None:
    summary = {
        "benchmark_reliability": "medium",
        "best_operational_models": {
            "temp": {"rmse": 1.49},
            "humidity": {"rmse": 9.8},
        },
        "patch_ablation_completed": True,
        "report_generated": True,
        "artifacts": {"html_report": str(tmp_path / "missing.html")},
    }

    assert evaluate_site_readiness_gate(summary)["status"] == "WARN"
    (tmp_path / "missing.html").write_text("<html></html>", encoding="utf-8")
    assert evaluate_site_readiness_gate(summary)["status"] == "PASS"


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


def test_operational_performance_html_dashboard_cards_plots_badges_and_units(tmp_path) -> None:
    plots = tmp_path / "plots"
    plots.mkdir()
    (plots / "temp_forecast_vs_actual.png").write_bytes(_ONE_PIXEL_PNG)
    summary = {
        "data": {"forecast_cycles": 90, "stations": 30, "joined_rows": 64800},
        "benchmark_reliability": "medium",
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
        "best_operational_models": {
            "temp": {"model": "operational_residual_lgbm_temp", "rmse": 1.78, "mae": 1.38, "bias": -0.2},
            "humidity": {"model": "operational_residual_lgbm_humidity", "rmse": 9.38, "mae": 7.37, "bias": -4.0},
        },
        "v4c_gate": {"status": "FAIL", "conditions": {"temp_rmse <= 1.5": False}, "missing_conditions": ["temp_rmse <= 1.5"]},
        "site_readiness": {"status": "WARN", "conditions": {"benchmark_reliability >= medium": True}, "missing_conditions": ["temp RMSE <= 1.5"]},
        "patch_ablation": {
            "temp": [{"target": "temp", "mode": "no_patch", "rmse": 1.9}, {"target": "temp", "mode": "true_patch5", "rmse": 1.78}],
            "humidity": [{"target": "humidity", "mode": "no_patch", "rmse": 9.38}, {"target": "humidity", "mode": "true_patch5", "rmse": 10.2}],
        },
        "calibration": {
            "temp": {"mode": "true_patch5", "raw_test_metrics": {"rmse": 1.9}, "corrected_test_metrics": {"rmse": 1.78}},
            "humidity": {"mode": "no_patch", "raw_test_metrics": {"rmse": 9.7}, "corrected_test_metrics": {"rmse": 9.38}},
        },
        "ensembles": {"temp": {"best": {"method": "simple_average", "rmse": 1.7}}, "humidity": {"selection_status": "rejected"}},
    }

    html_path = write_operational_performance_html(summary, tmp_path / "operational_performance_report.html")
    page = html_path.read_text(encoding="utf-8")

    assert "kpi-grid" in page
    assert "Temp Best Model" in page
    assert "Humidity Best Model" in page
    assert "Section 3: Temperature Analysis" in page
    assert "Section 4: Humidity Analysis" in page
    assert "temp_forecast_vs_actual.png" in page
    assert "humidity_forecast_vs_actual.png 사용할 수 없음" in page
    assert "data:image/png;base64" in page
    assert 'href="plots/temp_forecast_vs_actual.png"' in page
    assert 'class="badge badge-bad">FAIL</span>' in page
    assert 'class="badge badge-warn">WARN</span>' in page
    assert "°C" in page
    assert "%p" in page


def test_operational_performance_html_embed_modes_and_escapes_metric_text(tmp_path) -> None:
    plots = tmp_path / "plots"
    plots.mkdir()
    (plots / "temp_forecast_vs_actual.png").write_bytes(_ONE_PIXEL_PNG)
    summary = {
        "official_baselines": {
            "temp": {
                "raw_gfs_t2m": {"rmse": 2.0, "mae": 1.0, "bias": 0.0},
                "operational_residual_lgbm_temp": {"rmse": "<script>alert(1)</script>", "mae": 1.0, "bias": 0.0},
            }
        },
        "best_operational_models": {
            "temp": {"model": "operational_residual_lgbm_temp", "rmse": "<script>alert(1)</script>", "mae": 1.0, "bias": 0.0}
        },
    }

    full = write_operational_performance_html(summary, tmp_path / "full.html", embed_images="full").read_text(encoding="utf-8")
    external = write_operational_performance_html(summary, tmp_path / "external.html", embed_images="external-assets").read_text(encoding="utf-8")

    assert "data:image/png;base64" in full
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in full
    assert "<script>alert(1)</script>" not in full
    assert 'src="plots/temp_forecast_vs_actual.png"' in external
    assert "data:image/png;base64" not in external
    with pytest.raises(ValueError, match="embed_images"):
        write_operational_performance_html(summary, tmp_path / "bad.html", embed_images="bad-mode")


def test_operational_performance_summary_artifacts_are_written(tmp_path) -> None:
    summary = {
        "official_baselines": {
            "temp": {
                "raw_gfs_t2m": {"rmse": 2.0, "mae": 1.5, "bias": -0.5},
                "operational_residual_lgbm_temp": {"rmse": 1.5, "mae": 1.2, "bias": 0.1},
            },
            "humidity": {
                "raw_gfs_rh": {"rmse": 12.0, "mae": 9.0, "bias": 1.0},
                "operational_residual_lgbm_humidity": {"rmse": 10.0, "mae": 8.0, "bias": 0.2},
            },
        },
        "beta_targets": {
            "precip_probability": {"descriptor": {"model_key": "nwp_direct", "source": "nwp_direct", "status": "direct", "confidence": "low"}},
            "cloud": {"descriptor": {"model_key": "rule_based_beta", "source": "rule_based_beta", "status": "beta", "confidence": "low"}},
        },
    }

    artifacts = operational_performance.write_operational_performance_summary_artifacts(summary, tmp_path)

    assert artifacts["summary_json"].endswith("operational_performance_summary.json")
    assert artifacts["summary_csv"].endswith("operational_performance_summary.csv")
    rows = pd.read_csv(tmp_path / "operational_performance_summary.csv")
    assert {"temp", "humidity"}.issubset(set(rows["target"]))
    assert {"°C", "%p"}.issubset(set(rows["unit"]))
    assert rows.loc[rows["target"].eq("temp") & rows["model"].eq("operational_residual_lgbm_temp"), "rmse_improvement_pct"].iloc[0] == 25.0
    beta_rows = pd.read_csv(tmp_path / "operational_performance_summary.csv")
    assert {"precip_probability", "cloud"}.issubset(set(beta_rows["target"]))


def test_precip_probability_selection_uses_validation_brier_not_test_metric() -> None:
    descriptor = select_precip_probability_descriptor(
        raw_validation_metrics={"brier": 0.20},
        model_validation_metrics={"brier": 0.25},
        raw_test_metrics={"brier": 0.40},
        model_test_metrics={"brier": 0.10},
    )

    assert descriptor["source"] == "nwp_direct"
    assert descriptor["model_key"] == "nwp_direct"
    assert descriptor["fallback_reason"] == "validation_brier_not_improved"


def test_precip_probability_selection_does_not_adopt_ai_beta_after_freeze_by_default() -> None:
    descriptor = select_precip_probability_descriptor(
        raw_validation_metrics={"brier": 0.20},
        model_validation_metrics={"brier": 0.10},
        raw_test_metrics={"brier": 0.40},
        model_test_metrics={"brier": 0.05},
    )

    assert descriptor["source"] == "nwp_direct"
    assert descriptor["model_key"] == "nwp_direct"
    assert descriptor["fallback_reason"] == "ai_beta_training_frozen"

    pre_freeze_descriptor = select_precip_probability_descriptor(
        raw_validation_metrics={"brier": 0.20},
        model_validation_metrics={"brier": 0.10},
        allow_ai_beta=True,
    )
    assert pre_freeze_descriptor["source"] == "ai_beta"
    assert pre_freeze_descriptor["selection_reason"] == "validation_brier_improved"


def test_beta_target_summary_records_three_stage_trace_and_honest_fallbacks() -> None:
    frame = pd.DataFrame(
        {
            "split": ["train", "val", "test"],
            "station_id": ["108", "108", "108"],
            "horizon_step": [1, 1, 1],
            "precipitation": [0.0, 0.2, 0.0],
            "nwp_tp": [0.0, 5.0, 0.0],
            "wind_speed": [2.0, 4.0, 3.0],
            "nwp_wind_speed": [2.5, 4.5, 3.5],
            "nwp_cloud_cover": [20.0, 70.0, 90.0],
        }
    )

    summary = build_beta_targets_summary(frame)

    assert set(summary) == {"precip_probability", "wind", "cloud", "weather_code"}
    for target in summary.values():
        assert set(target["iterations"]) == {"1_raw_baseline", "2_calibration", "3_final_selection"}
    assert summary["precip_probability"]["descriptor"]["source"] == "unavailable"
    assert summary["precip_probability"]["descriptor"]["fallback_reason"] == "nwp_precip_probability_missing"
    assert summary["cloud"]["descriptor"]["source"] == "nwp_direct"
    assert summary["cloud"]["descriptor"]["fallback_reason"] == "observed_cloud_label_missing"
    assert summary["weather_code"]["descriptor"]["model_key"] == "rule_based_beta"


def test_beta_target_summary_keeps_precip_probability_direct_after_freeze() -> None:
    frame = pd.DataFrame(
        {
            "split": ["train", "val", "test"],
            "station_id": ["108", "108", "108"],
            "horizon_step": [1, 1, 1],
            "precipitation": [0.0, 0.2, 0.0],
            "nwp_precip_probability": [10.0, 60.0, 30.0],
        }
    )

    summary = build_beta_targets_summary(frame)
    precip = summary["precip_probability"]

    assert precip["descriptor"]["source"] == "nwp_direct"
    assert precip["descriptor"]["status"] == "direct"
    assert precip["descriptor"]["fallback_reason"] == "ai_beta_unavailable"
    assert precip["iterations"]["2_calibration"] == {"status": "not_applied", "reason": "model_improvement_frozen"}


def test_circular_direction_mae_wraps_across_zero_degrees() -> None:
    assert circular_direction_mae([350], [10]) == 20.0
    assert circular_direction_mae([10], [350]) == 20.0


def test_operational_dashboard_plot_generation_writes_required_pngs(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    predictions = pd.DataFrame(
        {
            "station_id": ["108", "108", "112", "112", "108", "112"],
            "region": ["capital", "capital", "south", "south", "capital", "south"],
            "valid_time": pd.date_range("2026-05-01T00:00Z", periods=6, freq="h"),
            "horizon_step": [1, 2, 1, 2, 3, 3],
            "temp": [10.0, 11.0, 15.0, 16.0, 12.0, 17.0],
            "humidity": [30.0, 85.0, 55.0, 90.0, 35.0, 82.0],
            "raw_gfs_t2m_prediction": [9.0, 10.0, 14.0, 15.0, 11.0, 16.0],
            "operational_residual_lgbm_temp_prediction": [10.5, 10.7, 15.2, 16.1, 12.3, 16.8],
            "temp_ensemble_prediction": [10.2, 10.9, 15.1, 15.9, 12.1, 17.1],
            "raw_gfs_rh_prediction": [40.0, 75.0, 50.0, 80.0, 45.0, 78.0],
            "operational_residual_lgbm_humidity_prediction": [32.0, 83.0, 56.0, 88.0, 36.0, 84.0],
        }
    )
    summary = {
        "best_operational_models": {
            "temp": {"model": "ensemble_simple_average"},
            "humidity": {"model": "operational_residual_lgbm_humidity"},
        },
        "patch_ablation": {
            "temp": [{"mode": "no_patch", "rmse": 1.9}, {"mode": "true_patch5", "rmse": 1.7}],
            "humidity": [{"mode": "no_patch", "rmse": 9.8}, {"mode": "true_patch5", "rmse": 10.5}],
        },
        "calibration": {
            "temp": {"raw_test_metrics": {"rmse": 1.9, "mae": 1.5, "bias": 0.4}, "corrected_test_metrics": {"rmse": 1.7, "mae": 1.3, "bias": 0.1}},
            "humidity": {"raw_test_metrics": {"rmse": 10.2, "mae": 8.5, "bias": -2.0}, "corrected_test_metrics": {"rmse": 9.8, "mae": 8.0, "bias": -0.5}},
        },
        "ensembles": {
            "temp": {"methods": [{"method": "simple_average", "test_rmse": 1.7}, {"method": "stationwise_inverse_rmse", "test_rmse": 1.6}]},
            "humidity": {"methods": [{"method": "simple_average", "test_rmse": 10.1}], "selection_status": "rejected"},
        },
    }

    plot_paths = operational_performance.write_operational_dashboard_plots(summary, predictions, tmp_path / "plots")

    assert (tmp_path / "plots" / "temp_forecast_vs_actual.png").exists()
    assert (tmp_path / "plots" / "temp_patch_ablation_bar.png").exists()
    assert (tmp_path / "plots" / "humidity_dry_humid_event_error.png").exists()
    assert (tmp_path / "plots" / "humidity_model_comparison.png").exists()
    assert "temp_forecast_vs_actual.png" in {path.name for path in plot_paths}
    assert {"written"} <= {row["status"] for row in summary["plot_diagnostics"]}


def test_operational_dashboard_plot_generation_records_failures(monkeypatch, tmp_path) -> None:
    pytest.importorskip("matplotlib")
    predictions = pd.DataFrame(
        {
            "station_id": ["108"],
            "valid_time": pd.date_range("2026-05-01T00:00Z", periods=1, freq="h"),
            "horizon_step": [1],
            "temp": [10.0],
            "humidity": [55.0],
            "operational_residual_lgbm_temp_prediction": [10.1],
            "operational_residual_lgbm_humidity_prediction": [54.0],
        }
    )
    summary = {
        "patch_ablation": {"temp": [{"mode": "no_patch", "rmse": 1.0}]},
        "best_operational_models": {
            "temp": {"model": "operational_residual_lgbm_temp"},
            "humidity": {"model": "operational_residual_lgbm_humidity"},
        },
    }

    def fail_plot(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(operational_report, "_plot_patch_ablation", fail_plot)

    operational_performance.write_operational_dashboard_plots(summary, predictions, tmp_path / "plots")

    failed = [row for row in summary["plot_diagnostics"] if row["status"] == "failed"]
    assert any(row["plot"] == "temp_patch_ablation_bar.png" and "boom" in row["reason"] for row in failed)


def test_operational_performance_parser_defaults_to_thumbnail_embed() -> None:
    args = operational_performance.build_arg_parser().parse_args(
        [
            "--nwp-archive",
            "nwp.csv",
            "--archive-quality-report",
            "quality.json",
            "--observations",
            "obs.csv",
            "--station-metadata",
            "stations.csv",
        ]
    )

    assert args.embed_images == "thumbnail"
