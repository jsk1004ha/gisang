from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any


BETA_TARGET_PREFIXES = ("precip_probability", "wind", "cloud", "weather_code")
DIRECT_SOURCES = {"gfs_direct", "kma_direct", "nwp_direct"}
BETA_FORECAST_POINTS_SCHEMA_VERSION = "forecast_points.v2-beta-sources"


@dataclass(frozen=True)
class BetaSourceDescriptor:
    target: str
    source: str
    status: str
    confidence: str
    unit: str | None = None
    model_key: str | None = None
    selection_reason: str | None = None
    fallback_reason: str | None = None
    validation_metrics: dict[str, Any] = field(default_factory=dict)
    test_metrics: dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value not in (None, {}, [])}


def normalize_probability_fraction(value: Any, *, input_unit: str = "auto") -> float | None:
    number = _finite_float(value)
    if number is None:
        return None
    if number < 0:
        return None
    normalized_unit = str(input_unit or "auto").lower()
    if normalized_unit == "percent":
        return number / 100.0 if number <= 100.0 else None
    if normalized_unit == "fraction":
        return number if number <= 1.0 else None
    if number <= 1.0:
        return number
    if number <= 100.0:
        return number / 100.0
    return None


def normalize_probability_percent(value: Any, *, input_unit: str = "auto") -> float | None:
    fraction = normalize_probability_fraction(value, input_unit=input_unit)
    if fraction is None:
        return None
    percent = fraction * 100.0
    return int(percent) if float(percent).is_integer() else float(percent)


def direct_source_for_provider(provider: Any) -> str:
    source = str(provider or "").lower()
    if "kma" in source:
        return "kma_direct"
    if "gfs" in source:
        return "gfs_direct"
    return "nwp_direct"


def default_confidence_for_source(source: str, status: str | None = None) -> str:
    normalized_source = str(source or "")
    normalized_status = str(status or "")
    if normalized_source == "unavailable" or normalized_status == "unavailable":
        return "unavailable"
    if normalized_source == "ai_beta":
        return "medium"
    if normalized_source in {"ai_mos_model", "ai_mos"}:
        return "medium"
    if normalized_source in {"ai_mos_model_beta", "ai_mos_beta"}:
        return "medium"
    if normalized_source in DIRECT_SOURCES:
        return "low"
    if normalized_source == "rule_based_beta":
        return "low"
    return "low"


def descriptor_fields(
    *,
    prefix: str,
    source: str,
    status: str | None = None,
    confidence: str | None = None,
) -> dict[str, str]:
    resolved_status = status or (
        "direct"
        if source in DIRECT_SOURCES
        else "model"
        if source in {"ai_mos_model", "ai_mos"}
        else "beta"
        if source in {"ai_beta", "rule_based_beta", "ai_mos_model_beta", "ai_mos_beta"}
        else "unavailable"
    )
    resolved_confidence = confidence or default_confidence_for_source(source, resolved_status)
    return {
        f"{prefix}_source": source,
        f"{prefix}_status": resolved_status,
        f"{prefix}_confidence": resolved_confidence,
    }


def summarize_beta_targets_from_point(point: dict[str, Any]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for prefix in BETA_TARGET_PREFIXES:
        source = point.get(f"{prefix}_source")
        status = point.get(f"{prefix}_status")
        confidence = point.get(f"{prefix}_confidence")
        if source is None and status is None and confidence is None:
            continue
        summary[prefix] = {
            "source": source,
            "status": status,
            "confidence": confidence,
        }
    return summary


def summarize_beta_targets_from_points(points: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    aggregate: dict[str, dict[str, Any]] = {}
    for prefix in BETA_TARGET_PREFIXES:
        sources: list[str] = []
        statuses: list[str] = []
        confidences: list[str] = []
        counts_by_source: dict[str, int] = {}
        counts_by_status: dict[str, int] = {}
        for point in points:
            source = point.get(f"{prefix}_source")
            status = point.get(f"{prefix}_status")
            confidence = point.get(f"{prefix}_confidence")
            if source is None and status is None and confidence is None:
                continue
            source_str = str(source or "unavailable")
            status_str = str(status or "unavailable")
            confidence_str = str(confidence or default_confidence_for_source(source_str, status_str))
            if source_str not in sources:
                sources.append(source_str)
            if status_str not in statuses:
                statuses.append(status_str)
            if confidence_str not in confidences:
                confidences.append(confidence_str)
            counts_by_source[source_str] = counts_by_source.get(source_str, 0) + 1
            counts_by_status[status_str] = counts_by_status.get(status_str, 0) + 1
        if not sources and not statuses and not confidences:
            continue
        aggregate[prefix] = {
            "source": sources[0] if len(sources) == 1 else "mixed",
            "status": statuses[0] if len(statuses) == 1 else "mixed",
            "confidence": confidences[0] if len(confidences) == 1 else "mixed",
            "sources": sources,
            "statuses": statuses,
            "confidences": confidences,
            "counts_by_source": counts_by_source,
            "counts_by_status": counts_by_status,
        }
    return aggregate


def _finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None
