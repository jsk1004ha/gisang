from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from weather_korea_forecast.service.beta_sources import summarize_beta_targets_from_points

try:  # optional runtime dependency
    from fastapi import FastAPI
except Exception:  # pragma: no cover - exercised when dependency is absent
    FastAPI = None  # type: ignore[assignment]


DEFAULT_FORECAST_DIR = Path("data/forecasts/latest")
DEFAULT_REPORTS_DIR = Path("reports")
DEFAULT_PRODUCTION_MANIFEST = Path("data/artifacts/g030_production_freeze_final/production_model_manifest.json")


class SimpleAPI:
    def __init__(self, title: str = "Weather Korea Forecast API") -> None:
        self.title = title
        self.routes: dict[tuple[str, str], Callable[..., Any]] = {}

    def get(self, path: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.routes[("GET", path)] = func
            return func
        return decorator


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _forecast_run(forecast_dir: Path = DEFAULT_FORECAST_DIR) -> dict[str, Any]:
    return _read_json(forecast_dir / "forecast_run.json", {})


def _forecast_points(forecast_dir: Path = DEFAULT_FORECAST_DIR) -> list[dict[str, Any]]:
    payload = _read_json(forecast_dir / "forecast_points.json", [])
    return payload if isinstance(payload, list) else []


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left == right or left.resolve() == right.resolve()
    except OSError:
        return left == right


def _first_existing_manifest(
    reports_dir: Path,
    manifest_path: Path | None = None,
    *,
    allow_default_manifest: bool = False,
) -> dict[str, Any]:
    candidates = [
        manifest_path,
        reports_dir / "production_model_manifest.json",
        reports_dir / "g030_production_freeze_final" / "production_model_manifest.json",
    ]
    if allow_default_manifest:
        candidates.append(DEFAULT_PRODUCTION_MANIFEST)
    for candidate in candidates:
        if candidate and candidate.exists():
            payload = _read_json(candidate, {})
            if isinstance(payload, dict):
                return payload
    return {}


def _warnings(run: dict[str, Any]) -> list[str]:
    warnings = list(run.get("warnings") or [])
    if run.get("operational_valid") is False and "operational_valid=false" not in " ".join(warnings):
        warnings.append("operational_valid=false: 연구/백테스트 기반 예측입니다.")
    if run.get("backtest_only") is True and "backtest_only=true" not in " ".join(warnings):
        warnings.append("backtest_only=true: 실제 운영 예보로 사용하지 마세요.")
    return warnings


def _manifest_warnings(manifest: dict[str, Any]) -> list[str]:
    if not manifest:
        return []
    warnings: list[str] = []
    caveats = manifest.get("site_caveats") if isinstance(manifest.get("site_caveats"), list) else []
    caveat_messages = {
        "temperature_accuracy_caveat": "temperature_accuracy_caveat: 기온 정확도 주의 표시가 필요합니다.",
        "humidity_beta": "humidity_beta: 습도 예측은 beta로 표시해야 합니다.",
        "humidity_bias_caveat": "humidity_bias_caveat: 습도 bias 주의 표시가 필요합니다.",
        "site_readiness_not_pass": "site_readiness_not_pass: 운영 준비 상태가 PASS가 아닙니다.",
        "v4c_gate_not_pass": "v4c_gate_not_pass: V4-C gate가 PASS가 아닙니다.",
        "weather_code_rule_based_beta": "weather_code_rule_based_beta: 날씨 상태는 rule-based beta입니다.",
        "benchmark_reliability_low": "benchmark_reliability_low: benchmark reliability warning required.",
    }

    def add_warning(message: str) -> None:
        if message not in warnings:
            warnings.append(message)

    if _manifest_operational_valid(manifest) is not True:
        add_warning("operational_valid=false: frozen production manifest is not operational-valid.")
    if manifest.get("humidity_status") != "PASS" or "humidity_beta" in caveats:
        add_warning(caveat_messages["humidity_beta"])
    if manifest.get("weather_code_model") == "rule_based_beta" or "weather_code_rule_based_beta" in caveats:
        add_warning(caveat_messages["weather_code_rule_based_beta"])
    if manifest.get("site_readiness") != "PASS":
        add_warning(f"site_readiness={manifest.get('site_readiness', 'UNKNOWN')}: 운영 준비 상태를 표시해야 합니다.")
    if manifest.get("benchmark_reliability") not in {"medium", "strong", "seasonal"}:
        add_warning(caveat_messages["benchmark_reliability_low"])
    for caveat in caveats:
        add_warning(caveat_messages.get(str(caveat), f"{caveat}: production manifest caveat must be shown."))
    return warnings


def _manifest_operational_valid(manifest: dict[str, Any]) -> bool | None:
    if not manifest:
        return None
    declared = manifest.get("operational_valid")
    status_fields = {
        "temperature_status": manifest.get("temperature_status") or manifest.get("temp_status"),
        "humidity_status": manifest.get("humidity_status"),
        "humidity_bias_status": manifest.get("humidity_bias_status"),
        "site_readiness": manifest.get("site_readiness_status") or manifest.get("site_readiness"),
        "v4c_gate_status": manifest.get("v4c_gate_status"),
        "benchmark_reliability": manifest.get("benchmark_reliability"),
    }
    known_gate_fields = [value for value in status_fields.values() if value is not None]
    if known_gate_fields:
        return bool(
            declared is True
            and status_fields["temperature_status"] == "PASS"
            and status_fields["humidity_status"] == "PASS"
            and status_fields["humidity_bias_status"] == "PASS"
            and status_fields["site_readiness"] == "PASS"
            and status_fields["v4c_gate_status"] == "PASS"
            and status_fields["benchmark_reliability"] in {"medium", "strong", "seasonal"}
        )
    return bool(declared) if declared is not None else None


def _beta_targets(run: dict[str, Any], points: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    point_summary = summarize_beta_targets_from_points(points or [])
    if point_summary:
        return point_summary
    beta = run.get("beta_targets")
    if isinstance(beta, dict):
        return beta
    return {}


def _target_has_source(beta_targets: dict[str, Any], target: str, source: str) -> bool:
    payload = beta_targets.get(target) if isinstance(beta_targets, dict) else None
    if not isinstance(payload, dict):
        return False
    sources = payload.get("sources")
    descriptor = payload.get("descriptor") if isinstance(payload.get("descriptor"), dict) else {}
    return payload.get("source") == source or source in (sources or []) or descriptor.get("source") == source


def _humidity_beta(run: dict[str, Any], points: list[dict[str, Any]]) -> bool:
    if run.get("humidity_beta") is True:
        return True
    beta_sources = {"ai_mos_model_beta", "ai_mos_beta", "ai_beta"}
    has_explicit_humidity_source = False
    for point in points:
        source = point.get("humidity_source")
        if source not in (None, ""):
            has_explicit_humidity_source = True
            if str(source) in beta_sources:
                return True
    if has_explicit_humidity_source:
        return False
    return any(point.get("humidity_percent") not in (None, "") for point in points)


def health() -> dict[str, Any]:
    return {"status": "ok"}


def stations(forecast_dir: Path = DEFAULT_FORECAST_DIR) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for point in _forecast_points(forecast_dir):
        sid = str(point.get("station_id"))
        seen.setdefault(sid, {k: point.get(k) for k in ["station_id", "station_name", "lat", "lon", "region_class"]})
    return list(seen.values())


def latest_forecast(forecast_dir: Path = DEFAULT_FORECAST_DIR) -> dict[str, Any]:
    run = _forecast_run(forecast_dir)
    points = _forecast_points(forecast_dir)
    return {"run": run, "warnings": _warnings(run), "points": points, "humidity_beta": _humidity_beta(run, points), "beta_targets": _beta_targets(run, points)}


def station_forecast(station_id: str, forecast_dir: Path = DEFAULT_FORECAST_DIR) -> dict[str, Any]:
    run = _forecast_run(forecast_dir)
    points = [p for p in _forecast_points(forecast_dir) if str(p.get("station_id")) == str(station_id)]
    return {"run": run, "warnings": _warnings(run), "station_id": str(station_id), "points": points, "humidity_beta": _humidity_beta(run, points), "beta_targets": _beta_targets(run, points)}


def station_hourly(station_id: str, forecast_dir: Path = DEFAULT_FORECAST_DIR) -> list[dict[str, Any]]:
    return station_forecast(station_id, forecast_dir)["points"]


def station_daily(station_id: str, forecast_dir: Path = DEFAULT_FORECAST_DIR) -> dict[str, Any]:
    points = station_hourly(station_id, forecast_dir)
    frame = pd.DataFrame(points)
    if frame.empty:
        return {"station_id": str(station_id), "daily": []}
    frame["date"] = pd.to_datetime(frame["valid_time"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
    grouped = frame.groupby("date", dropna=True).agg(temp_max_c=("temperature_c", "max"), temp_min_c=("temperature_c", "min"), humidity_mean_percent=("humidity_percent", "mean")).reset_index()
    return {"station_id": str(station_id), "daily": grouped.to_dict(orient="records")}


def model_status(
    forecast_dir: Path = DEFAULT_FORECAST_DIR,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    run = _forecast_run(forecast_dir)
    allow_default_manifest = (
        manifest_path is None
        and _same_path(forecast_dir, DEFAULT_FORECAST_DIR)
        and _same_path(reports_dir, DEFAULT_REPORTS_DIR)
    )
    manifest = _first_existing_manifest(reports_dir, manifest_path, allow_default_manifest=allow_default_manifest)
    summary_path = reports_dir / "experiment_summary.csv"
    best_temp_rmse = None
    best_humidity_rmse = None
    if summary_path.exists():
        frame = pd.read_csv(summary_path)
        main = frame[frame.get("included_in_main_leaderboard", False).astype(bool)] if "included_in_main_leaderboard" in frame else frame
        temp = main[main.get("target_name", "") == "temp"] if "target_name" in main else pd.DataFrame()
        humidity = main[main.get("target_name", "") == "humidity"] if "target_name" in main else pd.DataFrame()
        if not temp.empty and "rmse" in temp:
            best_temp_rmse = float(temp["rmse"].min())
        if not humidity.empty and "rmse" in humidity:
            best_humidity_rmse = float(humidity["rmse"].min())
    temp_rmse = manifest.get("temperature_rmse", manifest.get("temp_rmse", best_temp_rmse)) if manifest else best_temp_rmse
    humidity_rmse = manifest.get("humidity_rmse", best_humidity_rmse) if manifest else best_humidity_rmse
    manifest_operational_valid = _manifest_operational_valid(manifest) if manifest else None
    operational_valid = manifest_operational_valid if manifest_operational_valid is not None else run.get("operational_valid", False)
    warnings = [*_warnings(run), *_manifest_warnings(manifest)]
    points = _forecast_points(forecast_dir)
    beta_targets = _beta_targets(run, points)
    humidity_beta = bool(manifest and (manifest.get("humidity_status") != "PASS")) or _humidity_beta(run, points)
    weather_code_rule_based_beta = bool(manifest and manifest.get("weather_code_model") == "rule_based_beta") or _target_has_source(beta_targets, "weather_code", "rule_based_beta")
    return {
        "model_version": run.get("model_version") or manifest.get("freeze_iteration") or manifest.get("temperature_model"),
        "operational_valid": operational_valid,
        "backtest_only": run.get("backtest_only", True),
        "latest_forecast_run_time": run.get("created_at"),
        "temp_rmse": temp_rmse,
        "temperature_rmse": temp_rmse,
        "temperature_status": manifest.get("temperature_status") if manifest else None,
        "humidity_rmse": humidity_rmse,
        "humidity_bias": manifest.get("humidity_bias") if manifest else None,
        "humidity_status": manifest.get("humidity_status") if manifest else None,
        "humidity_bias_status": manifest.get("humidity_bias_status") if manifest else None,
        "benchmark_reliability": manifest.get("benchmark_reliability") if manifest else None,
        "site_readiness": manifest.get("site_readiness") if manifest else None,
        "v4_c_gate_status": manifest.get("v4c_gate_status") or ("FAIL" if operational_valid is not True else "CHECK_REPORT"),
        "forecast_schema_version": manifest.get("forecast_schema_version") if manifest else run.get("forecast_schema_version"),
        "site_caveats": manifest.get("site_caveats", []) if manifest else [],
        "beta_label_required": manifest.get("beta_label_required", {}) if manifest else {},
        "humidity_beta": humidity_beta,
        "weather_code_rule_based_beta": weather_code_rule_based_beta,
        "model_improvement_frozen": bool(manifest.get("model_improvement_frozen")) if manifest else False,
        "beta_targets": beta_targets,
        "warnings": warnings,
    }


def evaluation_summary(reports_dir: Path = DEFAULT_REPORTS_DIR) -> dict[str, Any]:
    path = reports_dir / "experiment_summary.csv"
    if not path.exists():
        return {"available": False}
    frame = pd.read_csv(path)
    return {"available": True, "experiment_count": int(len(frame)), "columns": list(frame.columns)}


def create_app(
    forecast_dir: Path = DEFAULT_FORECAST_DIR,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    manifest_path: Path | None = None,
) -> Any:
    app = FastAPI(title="Weather Korea Forecast API") if FastAPI else SimpleAPI()

    @app.get("/health")
    def _health() -> dict[str, Any]:
        return health()

    @app.get("/api/stations")
    def _stations() -> list[dict[str, Any]]:
        return stations(forecast_dir)

    @app.get("/api/forecast/latest")
    def _latest() -> dict[str, Any]:
        return latest_forecast(forecast_dir)

    @app.get("/api/forecast/station/{station_id}")
    def _station(station_id: str) -> dict[str, Any]:
        return station_forecast(station_id, forecast_dir)

    @app.get("/api/forecast/station/{station_id}/hourly")
    def _hourly(station_id: str) -> list[dict[str, Any]]:
        return station_hourly(station_id, forecast_dir)

    @app.get("/api/forecast/station/{station_id}/daily")
    def _daily(station_id: str) -> dict[str, Any]:
        return station_daily(station_id, forecast_dir)

    @app.get("/api/model/status")
    def _status() -> dict[str, Any]:
        return model_status(forecast_dir, reports_dir, manifest_path)

    @app.get("/api/evaluation/summary")
    def _eval() -> dict[str, Any]:
        return evaluation_summary(reports_dir)

    return app


app = create_app()
