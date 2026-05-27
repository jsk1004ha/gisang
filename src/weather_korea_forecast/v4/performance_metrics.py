from __future__ import annotations

from typing import Any


def compute_patch_ablation_metrics(baseline: dict[str, Any], patch: dict[str, Any]) -> dict[str, float | None]:
    return {
        "patch_baseline_rmse": _float(baseline.get("rmse")),
        "patch_improvement_rmse": _delta(baseline.get("rmse"), patch.get("rmse")),
        "patch_improvement_worst_station": _delta(baseline.get("worst_station_rmse"), patch.get("worst_station_rmse")),
        "patch_improvement_late_horizon": _delta(
            baseline.get("late_horizon_rmse", baseline.get("worst_horizon_rmse")),
            patch.get("late_horizon_rmse", patch.get("worst_horizon_rmse")),
        ),
    }


def _delta(before: Any, after: Any) -> float | None:
    before_float = _float(before)
    after_float = _float(after)
    if before_float is None or after_float is None:
        return None
    return before_float - after_float


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
