from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Any


@dataclass(frozen=True)
class SiteReadiness:
    status: str
    temp_operational_valid: bool
    temp_rmse_le_1_5: bool
    humidity_beta_allowed: bool
    diagnostic_export_forbidden: bool
    best_operational_temp_rmse: float | None
    best_humidity_rmse: float | None
    missing_conditions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_site_readiness(records: Iterable[Any]) -> SiteReadiness:
    record_list = list(records)
    operational_temp = [
        r for r in record_list
        if _is_trusted_operational_temp_record(r)
    ]
    best_temp = min((float(r.rmse) for r in operational_temp), default=None)
    humidity = [
        r for r in record_list
        if getattr(r, "target_name", None) == "humidity"
        and getattr(r, "rmse", None) is not None
        and getattr(r, "is_diagnostic", False) is not True
    ]
    best_humidity = min((float(r.rmse) for r in humidity), default=None)
    conditions = {
        "trusted_operational_temp_model": best_temp is not None,
        "temp_rmse_le_1_5": best_temp is not None and best_temp <= 1.5,
        "humidity_beta_allowed": True,
        "diagnostic_export_forbidden": all(
            not getattr(r, "included_in_main_leaderboard", False) for r in record_list if getattr(r, "is_diagnostic", False)
        ),
    }
    missing = [name for name, ok in conditions.items() if not ok]
    return SiteReadiness(
        status="PASS" if not missing else "WARN",
        temp_operational_valid=conditions["trusted_operational_temp_model"],
        temp_rmse_le_1_5=conditions["temp_rmse_le_1_5"],
        humidity_beta_allowed=conditions["humidity_beta_allowed"],
        diagnostic_export_forbidden=conditions["diagnostic_export_forbidden"],
        best_operational_temp_rmse=best_temp,
        best_humidity_rmse=best_humidity,
        missing_conditions=missing,
    )


def _is_trusted_operational_temp_record(record: Any) -> bool:
    if getattr(record, "target_name", None) != "temp" or getattr(record, "rmse", None) is None:
        return False
    if getattr(record, "operational_valid", None) is not True or getattr(record, "backtest_only", None) is not False:
        return False
    if getattr(record, "future_feature_source", None) != "prepared_forecast_csv":
        return False
    if getattr(record, "forecast_source_schema_valid", None) is not True:
        return False
    if getattr(record, "forecast_archive_adequate", None) is not True:
        return False
    if not getattr(record, "forecast_source_path", None):
        return False
    if getattr(record, "is_diagnostic", False) is True or getattr(record, "is_alias_artifact", False) is True:
        return False
    text = " ".join(
        str(getattr(record, attr, "") or "").lower()
        for attr in ("experiment_name", "model_name", "artifact_profile", "v4_stage", "forecast_source_path", "artifact_dir", "leakage_risk_note")
    )
    return not any(token in text for token in ("synthetic", "smoke", "fixture", "generated", "oracle", "diagnostic"))
