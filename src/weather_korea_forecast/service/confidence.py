from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConfidenceResult:
    confidence: str
    confidence_score: float
    confidence_reason: str


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def estimate_confidence(
    *,
    target_name: str = "temp",
    horizon_step: Any = None,
    rmse: Any = None,
    operational_valid: bool | None = None,
    backtest_only: bool | None = None,
    humidity_beta: bool = False,
) -> ConfidenceResult:
    horizon = _float(horizon_step) or 999.0
    metric = _float(rmse)
    target = str(target_name or "").lower()

    if operational_valid is not True or backtest_only is True:
        return ConfidenceResult("research", 0.35, "운영 검증 전 연구/백테스트 기반 예측입니다.")
    if target == "humidity" or humidity_beta:
        if metric is None or metric > 10:
            return ConfidenceResult("low", 0.35, "습도 모델은 beta이며 RMSE가 목표보다 큽니다.")
        return ConfidenceResult("medium", 0.6, "습도 beta 모델이지만 목표 범위에 접근했습니다.")
    if target == "temp":
        if metric is not None and metric <= 1.2 and horizon <= 12:
            return ConfidenceResult("high", 0.85, "운영 검증 모델이며 단기 기온 RMSE가 낮습니다.")
        if metric is not None and metric <= 1.5 and horizon <= 24:
            return ConfidenceResult("medium", 0.68, "운영 검증 모델이며 24시간 기온 RMSE가 허용 범위입니다.")
    return ConfidenceResult("low", 0.45, "모델 신뢰도 정보가 부족하거나 장기 예측입니다.")
