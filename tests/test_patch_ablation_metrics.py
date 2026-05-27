from weather_korea_forecast.v4.performance_metrics import compute_patch_ablation_metrics


def test_patch_ablation_metrics_compute_improvements() -> None:
    metrics = compute_patch_ablation_metrics(
        {"rmse": 1.5, "worst_station_rmse": 2.0, "late_horizon_rmse": 1.8},
        {"rmse": 1.3, "worst_station_rmse": 1.7, "late_horizon_rmse": 1.6},
    )

    assert metrics["patch_baseline_rmse"] == 1.5
    assert round(metrics["patch_improvement_rmse"], 6) == 0.2
    assert round(metrics["patch_improvement_worst_station"], 6) == 0.3
    assert round(metrics["patch_improvement_late_horizon"], 6) == 0.2
