from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PLOT_FILES = [
    "forecast_vs_actual.png",
    "horizon_error.png",
    "prediction_scatter.png",
    "raw_vs_corrected.png",
    "station_rmse_bar.png",
    "region_rmse_bar.png",
    "horizon_station_heatmap.png",
    "daily_max_min_error.png",
    "residual_scatter.png",
    "baseline_vs_final_scatter.png",
]

@dataclass
class ExperimentRecord:
    experiment_name: str = "unknown"
    version: str = "unknown"
    track: str = "observation_only"
    target_name: str = "unknown"
    model_name: str = "unknown"
    model_type: str = "unknown"
    artifact_profile: str = "full"
    encoder_length: int | None = None
    prediction_length: int | None = None
    train_start: str | None = None
    train_end: str | None = None
    val_start: str | None = None
    val_end: str | None = None
    test_start: str | None = None
    test_end: str | None = None
    rmse: float | None = None
    mae: float | None = None
    bias: float | None = None
    mape: float | None = None
    raw_rmse: float | None = None
    raw_mae: float | None = None
    raw_bias: float | None = None
    val_rmse: float | None = None
    best_val_loss: float | None = None
    best_epoch: int | None = None
    sample_count: int | None = None
    worst_horizon_step: int | None = None
    worst_horizon_rmse: float | None = None
    worst_horizon_mae: float | None = None
    worst_horizon_bias: float | None = None
    worst_station_id: str | None = None
    worst_station_rmse: float | None = None
    worst_region: str | None = None
    worst_region_rmse: float | None = None
    worst_season: str | None = None
    uses_future_weather_features: bool | None = None
    future_feature_source: str = "none"
    operational_valid: bool | None = None
    backtest_only: bool | None = None
    leakage_risk_note: str | None = None
    v4_stage: str = "pre_v4"
    forecast_schema_version: str | None = None
    forecast_schema_valid: bool | None = None
    forecast_source_schema_valid: bool | None = None
    forecast_archive_adequate: bool | None = None
    forecast_archive_row_count: int | None = None
    forecast_archive_station_count: int | None = None
    forecast_archive_issue_time_count: int | None = None
    forecast_archive_horizon_coverage: float | None = None
    forecast_archive_missing_rate: float | None = None
    forecast_archive_blocking_reasons: list[str] = field(default_factory=list)
    forecast_source_path: str | None = None
    patch_features_enabled: bool | None = None
    uses_patch_features: bool | None = None
    patch_size: int | None = None
    patch_feature_set: str | None = None
    patch_feature_mode: str | None = None
    backtest_baseline_rmse: float | None = None
    operational_gap: float | None = None
    operational_gap_status: str | None = None
    patch_baseline_rmse: float | None = None
    patch_improvement_rmse: float | None = None
    patch_improvement_worst_station: float | None = None
    patch_improvement_late_horizon: float | None = None
    bias_correction_enabled: bool | None = None
    bias_correction_mode: str | None = None
    bias_correction_method: str | None = None
    bias_correction_accepted: bool | None = None
    rmse_goal: float | None = None
    rmse_goal_met: bool | None = None
    rmse_gap_to_goal: float | None = None
    goal_eligible: bool = True
    is_diagnostic: bool = False
    is_alias_artifact: bool = False
    canonical_experiment_id: str = ""
    run_timestamp: str | None = None
    is_representative_run: bool = True
    included_in_main_leaderboard: bool = False
    artifact_dir: str = ""
    forecast_vs_actual_path: str | None = None
    horizon_error_path: str | None = None
    prediction_scatter_path: str | None = None
    raw_vs_corrected_path: str | None = None
    station_rmse_bar_path: str | None = None
    region_rmse_bar_path: str | None = None
    horizon_station_heatmap_path: str | None = None
    daily_max_min_error_path: str | None = None
    residual_scatter_path: str | None = None
    baseline_vs_final_scatter_path: str | None = None
    created_at_or_modified_at: str | None = None
    complete: bool = True
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def path_for_plot(self, name: str) -> Path | None:
        path = Path(self.artifact_dir) / name
        return path if path.exists() else None

CSV_COLUMNS = list(ExperimentRecord().__dict__.keys())
