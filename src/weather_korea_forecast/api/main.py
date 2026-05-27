from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

try:  # optional runtime dependency
    from fastapi import FastAPI
except Exception:  # pragma: no cover - exercised when dependency is absent
    FastAPI = None  # type: ignore[assignment]


DEFAULT_FORECAST_DIR = Path("data/forecasts/latest")
DEFAULT_REPORTS_DIR = Path("reports")


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


def _warnings(run: dict[str, Any]) -> list[str]:
    warnings = list(run.get("warnings") or [])
    if run.get("operational_valid") is False and "operational_valid=false" not in " ".join(warnings):
        warnings.append("operational_valid=false: 연구/백테스트 기반 예측입니다.")
    if run.get("backtest_only") is True and "backtest_only=true" not in " ".join(warnings):
        warnings.append("backtest_only=true: 실제 운영 예보로 사용하지 마세요.")
    return warnings


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
    return {"run": run, "warnings": _warnings(run), "points": _forecast_points(forecast_dir), "humidity_beta": True}


def station_forecast(station_id: str, forecast_dir: Path = DEFAULT_FORECAST_DIR) -> dict[str, Any]:
    run = _forecast_run(forecast_dir)
    points = [p for p in _forecast_points(forecast_dir) if str(p.get("station_id")) == str(station_id)]
    return {"run": run, "warnings": _warnings(run), "station_id": str(station_id), "points": points, "humidity_beta": True}


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


def model_status(forecast_dir: Path = DEFAULT_FORECAST_DIR, reports_dir: Path = DEFAULT_REPORTS_DIR) -> dict[str, Any]:
    run = _forecast_run(forecast_dir)
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
    return {
        "model_version": run.get("model_version"),
        "operational_valid": run.get("operational_valid", False),
        "backtest_only": run.get("backtest_only", True),
        "latest_forecast_run_time": run.get("created_at"),
        "temp_rmse": best_temp_rmse,
        "humidity_rmse": best_humidity_rmse,
        "v4_c_gate_status": "FAIL" if run.get("operational_valid") is not True else "CHECK_REPORT",
        "warnings": _warnings(run),
    }


def evaluation_summary(reports_dir: Path = DEFAULT_REPORTS_DIR) -> dict[str, Any]:
    path = reports_dir / "experiment_summary.csv"
    if not path.exists():
        return {"available": False}
    frame = pd.read_csv(path)
    return {"available": True, "experiment_count": int(len(frame)), "columns": list(frame.columns)}


def create_app(forecast_dir: Path = DEFAULT_FORECAST_DIR, reports_dir: Path = DEFAULT_REPORTS_DIR) -> Any:
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
        return model_status(forecast_dir, reports_dir)

    @app.get("/api/evaluation/summary")
    def _eval() -> dict[str, Any]:
        return evaluation_summary(reports_dir)

    return app


app = create_app()
