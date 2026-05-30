from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from weather_korea_forecast.v4.operational_metadata import TARGETS, TARGET_GATE_RMSE, TARGET_NEAR_PASS_RMSE, TARGET_UNITS

EMBED_IMAGE_MODES = {"full", "thumbnail", "external-assets"}
OPERATIONAL_REPORT_PLOTS = {
    "temp": [
        "temp_forecast_vs_actual.png",
        "temp_prediction_scatter.png",
        "temp_residual_scatter.png",
        "temp_horizon_error.png",
        "temp_station_rmse_bar.png",
        "temp_region_rmse_bar.png",
        "temp_horizon_station_heatmap.png",
        "temp_daily_max_min_error.png",
        "temp_patch_ablation_bar.png",
        "temp_calibration_raw_vs_corrected.png",
        "temp_ensemble_component_comparison.png",
    ],
    "humidity": [
        "humidity_forecast_vs_actual.png",
        "humidity_prediction_scatter.png",
        "humidity_residual_scatter.png",
        "humidity_horizon_error.png",
        "humidity_station_rmse_bar.png",
        "humidity_region_rmse_bar.png",
        "humidity_horizon_station_heatmap.png",
        "humidity_dry_humid_event_error.png",
        "humidity_calibration_raw_vs_corrected.png",
        "humidity_model_comparison.png",
    ],
}


def metric_dict(actual: Iterable[float], prediction: Iterable[float]) -> dict[str, float | int]:
    y = np.asarray(list(actual), dtype=float)
    p = np.asarray(list(prediction), dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "bias": float("nan"), "n": 0}
    error = p - y
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "n": int(len(y)),
    }


def write_operational_performance_html(summary: dict[str, Any], output_path: str | Path, embed_images: str = "thumbnail") -> Path:
    if embed_images not in EMBED_IMAGE_MODES:
        raise ValueError(f"embed_images must be one of {sorted(EMBED_IMAGE_MODES)}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = output.parent / "summary_snapshot.json"
    snapshot.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    report_title = str(summary.get("report_title") or "Operational Performance Report")
    report_phase = str(summary.get("report_phase") or "Operational Benchmark Dashboard")
    rows = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(report_title)}</title>",
        f"<style>{_operational_report_css()}</style>",
        "</head><body>",
        f"<header class='hero'><p class='eyebrow'>{html.escape(report_phase)}</p><h1>{html.escape(report_title)}</h1>",
        "<p>성능 개선 결과를 사람이 이해할 수 있도록 KPI, gate, plot, experiment table을 한 화면에서 확인하는 운영 성능 dashboard입니다.</p></header>",
        _html_overview_section(summary),
        _html_official_baselines_section(summary),
        _html_target_analysis_section("temp", summary, output.parent, embed_images),
        _html_target_analysis_section("humidity", summary, output.parent, embed_images),
        _html_gate_readiness_section(summary),
        _html_experiment_tables_section(summary),
        _html_section("Artifacts", summary.get("artifacts", {})),
        "</body></html>",
    ]
    output.write_text("\n".join(rows), encoding="utf-8")
    return output



def write_operational_dashboard_plots(summary: dict[str, Any], predictions: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    diagnostics: list[dict[str, str]] = []
    summary["plot_diagnostics"] = diagnostics
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional in minimal environments.
        for plot in [filename for filenames in OPERATIONAL_REPORT_PLOTS.values() for filename in filenames]:
            diagnostics.append({"plot": plot, "status": "skipped", "reason": f"matplotlib unavailable: {exc}"})
        return []

    written: list[Path] = []
    frame = predictions.copy()
    if "valid_time" in frame.columns:
        frame["valid_time"] = pd.to_datetime(frame["valid_time"], utc=True, errors="coerce")
    for target in TARGETS:
        target_frame = _prediction_plot_frame(frame, summary, target)
        if not target_frame.empty:
            _run_plot(diagnostics, output / f"{target}_forecast_vs_actual.png", lambda: _plot_forecast_vs_actual(plt, target_frame, target, output / f"{target}_forecast_vs_actual.png", written))
            _run_plot(diagnostics, output / f"{target}_prediction_scatter.png", lambda: _plot_prediction_scatter(plt, target_frame, target, output / f"{target}_prediction_scatter.png", written))
            _run_plot(diagnostics, output / f"{target}_residual_scatter.png", lambda: _plot_residual_scatter(plt, target_frame, target, output / f"{target}_residual_scatter.png", written))
            _run_plot(diagnostics, output / f"{target}_horizon_error.png", lambda: _plot_horizon_error(plt, target_frame, target, output / f"{target}_horizon_error.png", written))
            _run_plot(diagnostics, output / f"{target}_station_rmse_bar.png", lambda: _plot_group_rmse_bar(plt, target_frame, target, "station_id", output / f"{target}_station_rmse_bar.png", written))
            region_column = "region_class" if "region_class" in target_frame.columns else "region" if "region" in target_frame.columns else None
            if region_column:
                _run_plot(diagnostics, output / f"{target}_region_rmse_bar.png", lambda: _plot_group_rmse_bar(plt, target_frame, target, region_column, output / f"{target}_region_rmse_bar.png", written))
            else:
                diagnostics.append({"plot": f"{target}_region_rmse_bar.png", "status": "skipped", "reason": "region column unavailable"})
            _run_plot(diagnostics, output / f"{target}_horizon_station_heatmap.png", lambda: _plot_horizon_station_heatmap(plt, target_frame, target, output / f"{target}_horizon_station_heatmap.png", written))
            if target == "temp":
                _run_plot(diagnostics, output / "temp_daily_max_min_error.png", lambda: _plot_daily_max_min_error(plt, target_frame, target, output / "temp_daily_max_min_error.png", written))
            else:
                _run_plot(diagnostics, output / "humidity_dry_humid_event_error.png", lambda: _plot_dry_humid_event_error(plt, target_frame, target, output / "humidity_dry_humid_event_error.png", written))
        else:
            for plot in [
                f"{target}_forecast_vs_actual.png",
                f"{target}_prediction_scatter.png",
                f"{target}_residual_scatter.png",
                f"{target}_horizon_error.png",
                f"{target}_station_rmse_bar.png",
                f"{target}_region_rmse_bar.png",
                f"{target}_horizon_station_heatmap.png",
                "temp_daily_max_min_error.png" if target == "temp" else "humidity_dry_humid_event_error.png",
            ]:
                diagnostics.append({"plot": plot, "status": "skipped", "reason": "prediction data unavailable"})
        _run_plot(diagnostics, output / f"{target}_patch_ablation_bar.png", lambda: _plot_patch_ablation(plt, summary, target, output / f"{target}_patch_ablation_bar.png", written))
        _run_plot(diagnostics, output / f"{target}_calibration_raw_vs_corrected.png", lambda: _plot_calibration_comparison(plt, summary, target, output / f"{target}_calibration_raw_vs_corrected.png", written))
    _run_plot(diagnostics, output / "temp_ensemble_component_comparison.png", lambda: _plot_ensemble_comparison(plt, summary, "temp", output / "temp_ensemble_component_comparison.png", written))
    _run_plot(diagnostics, output / "humidity_model_comparison.png", lambda: _plot_humidity_model_comparison(plt, summary, output / "humidity_model_comparison.png", written))
    return written


def _run_plot(diagnostics: list[dict[str, str]], path: Path, plotter: Any) -> None:
    try:
        plotter()
    except Exception as exc:  # noqa: BLE001
        diagnostics.append({"plot": path.name, "status": "failed", "reason": str(exc)})
        return
    if path.exists():
        diagnostics.append({"plot": path.name, "status": "written", "reason": ""})
    else:
        diagnostics.append({"plot": path.name, "status": "skipped", "reason": "insufficient data"})


def _prediction_plot_frame(predictions: pd.DataFrame, summary: dict[str, Any], target: str) -> pd.DataFrame:
    actual = TARGETS[target]["actual"]
    prediction_col = _best_prediction_column(predictions, summary, target)
    if actual not in predictions.columns or prediction_col is None:
        return pd.DataFrame()
    requested_columns = [
        column
        for column in ["station_id", "region", "region_class", "valid_time", "horizon_step", actual, prediction_col, f"{TARGETS[target]['official_raw']}_prediction", f"{TARGETS[target]['official_lgbm']}_prediction"]
        if column in predictions.columns
    ]
    columns = list(dict.fromkeys(requested_columns))
    frame = predictions[columns].copy()
    frame = frame.rename(columns={actual: "actual", prediction_col: "prediction"})
    frame["actual"] = pd.to_numeric(frame["actual"], errors="coerce")
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    if "horizon_step" in frame.columns:
        frame["horizon_step"] = pd.to_numeric(frame["horizon_step"], errors="coerce")
    return frame.dropna(subset=["actual", "prediction"])


def _best_prediction_column(predictions: pd.DataFrame, summary: dict[str, Any], target: str) -> str | None:
    best = _best_metrics(summary, target)
    model = str(best.get("model", ""))
    candidates: list[str] = []
    if model.startswith("ensemble_"):
        candidates.append(f"{target}_ensemble_prediction")
    candidates.extend(
        [
            f"{TARGETS[target]['official_lgbm']}_prediction",
            f"{target}_ensemble_prediction",
            f"{TARGETS[target]['official_raw']}_prediction",
        ]
    )
    for column in candidates:
        if column in predictions.columns:
            return column
    return None


def _plot_forecast_vs_actual(plt: Any, frame: pd.DataFrame, target: str, path: Path, written: list[Path]) -> None:
    try:
        plot_frame = frame.sort_values("valid_time" if "valid_time" in frame.columns else "horizon_step").head(240)
        fig, ax = plt.subplots(figsize=(10, 4))
        x = plot_frame["valid_time"] if "valid_time" in plot_frame.columns else range(len(plot_frame))
        ax.plot(x, plot_frame["actual"], label="actual", linewidth=1.8)
        ax.plot(x, plot_frame["prediction"], label="prediction", linewidth=1.5)
        ax.set_title(f"{target} forecast vs actual")
        ax.set_ylabel(_unit(target))
        ax.legend()
        _save_plot(fig, path, written)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"failed to write {path.name}") from exc


def _plot_prediction_scatter(plt: Any, frame: pd.DataFrame, target: str, path: Path, written: list[Path]) -> None:
    try:
        sample = frame.sample(min(len(frame), 5000), random_state=20260529) if len(frame) > 5000 else frame
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(sample["actual"], sample["prediction"], s=10, alpha=0.45)
        lo = float(np.nanmin([sample["actual"].min(), sample["prediction"].min()]))
        hi = float(np.nanmax([sample["actual"].max(), sample["prediction"].max()]))
        ax.plot([lo, hi], [lo, hi], color="#334155", linewidth=1)
        ax.set_xlabel(f"actual ({_unit(target)})")
        ax.set_ylabel(f"prediction ({_unit(target)})")
        ax.set_title(f"{target} prediction scatter")
        _save_plot(fig, path, written)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"failed to write {path.name}") from exc


def _plot_residual_scatter(plt: Any, frame: pd.DataFrame, target: str, path: Path, written: list[Path]) -> None:
    try:
        sample = frame.sample(min(len(frame), 5000), random_state=20260529) if len(frame) > 5000 else frame
        residual = sample["prediction"] - sample["actual"]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(sample["prediction"], residual, s=10, alpha=0.45)
        ax.axhline(0.0, color="#334155", linewidth=1)
        ax.set_xlabel(f"prediction ({_unit(target)})")
        ax.set_ylabel(f"residual ({_unit(target)})")
        ax.set_title(f"{target} residual scatter")
        _save_plot(fig, path, written)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"failed to write {path.name}") from exc


def _plot_horizon_error(plt: Any, frame: pd.DataFrame, target: str, path: Path, written: list[Path]) -> None:
    if "horizon_step" not in frame.columns:
        return
    try:
        rows = []
        for horizon, group in frame.groupby("horizon_step"):
            metrics = metric_dict(group["actual"], group["prediction"])
            rows.append({"horizon_step": horizon, **metrics})
        metrics_frame = pd.DataFrame(rows).sort_values("horizon_step")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(metrics_frame["horizon_step"], metrics_frame["rmse"], marker="o", label="RMSE")
        axes[0].plot(metrics_frame["horizon_step"], metrics_frame["mae"], marker="o", label="MAE")
        axes[0].set_ylabel(_unit(target))
        axes[0].legend()
        axes[1].bar(metrics_frame["horizon_step"].astype(str), metrics_frame["bias"], color="#f59e0b")
        axes[1].axhline(0.0, color="#334155", linewidth=1)
        axes[1].set_ylabel(f"Bias ({_unit(target)})")
        fig.suptitle(f"{target} horizon error")
        _save_plot(fig, path, written)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"failed to write {path.name}") from exc


def _plot_group_rmse_bar(plt: Any, frame: pd.DataFrame, target: str, group_column: str, path: Path, written: list[Path]) -> None:
    if group_column not in frame.columns:
        return
    try:
        rows = [{group_column: group, "rmse": metric_dict(payload["actual"], payload["prediction"])["rmse"]} for group, payload in frame.groupby(group_column)]
        metrics = pd.DataFrame(rows).sort_values("rmse", ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(max(6, len(metrics) * 0.5), 4))
        ax.bar(metrics[group_column].astype(str), metrics["rmse"], color="#3b82f6")
        ax.set_ylabel(f"RMSE ({_unit(target)})")
        ax.set_title(f"{target} {group_column} RMSE")
        ax.tick_params(axis="x", rotation=45)
        _save_plot(fig, path, written)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"failed to write {path.name}") from exc


def _plot_horizon_station_heatmap(plt: Any, frame: pd.DataFrame, target: str, path: Path, written: list[Path]) -> None:
    if "station_id" not in frame.columns or "horizon_step" not in frame.columns:
        return
    try:
        rows = []
        for (station, horizon), group in frame.groupby(["station_id", "horizon_step"]):
            rows.append({"station_id": station, "horizon_step": int(horizon), "rmse": metric_dict(group["actual"], group["prediction"])["rmse"]})
        heatmap = pd.DataFrame(rows).pivot(index="station_id", columns="horizon_step", values="rmse")
        if heatmap.empty:
            return
        fig, ax = plt.subplots(figsize=(max(8, heatmap.shape[1] * 0.35), max(4, heatmap.shape[0] * 0.28)))
        image = ax.imshow(heatmap.to_numpy(dtype=float), aspect="auto", cmap="Blues")
        ax.set_xticks(range(len(heatmap.columns)))
        ax.set_xticklabels([str(c) for c in heatmap.columns], fontsize=8)
        ax.set_yticks(range(len(heatmap.index)))
        ax.set_yticklabels([str(i) for i in heatmap.index], fontsize=8)
        ax.set_xlabel("horizon")
        ax.set_ylabel("station")
        ax.set_title(f"{target} horizon × station RMSE")
        fig.colorbar(image, ax=ax, label=f"RMSE ({_unit(target)})")
        _save_plot(fig, path, written)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"failed to write {path.name}") from exc


def _plot_daily_max_min_error(plt: Any, frame: pd.DataFrame, target: str, path: Path, written: list[Path]) -> None:
    if "valid_time" not in frame.columns:
        return
    try:
        daily = frame.assign(valid_date=frame["valid_time"].dt.date).groupby("valid_date").agg(actual_max=("actual", "max"), actual_min=("actual", "min"), pred_max=("prediction", "max"), pred_min=("prediction", "min")).reset_index()
        daily["max_error"] = daily["pred_max"] - daily["actual_max"]
        daily["min_error"] = daily["pred_min"] - daily["actual_min"]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(daily["valid_date"].astype(str), daily["max_error"], marker="o", label="daily max error")
        ax.plot(daily["valid_date"].astype(str), daily["min_error"], marker="o", label="daily min error")
        ax.axhline(0.0, color="#334155", linewidth=1)
        ax.set_ylabel(_unit(target))
        ax.set_title("temperature daily max/min error")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        _save_plot(fig, path, written)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"failed to write {path.name}") from exc


def _plot_dry_humid_event_error(plt: Any, frame: pd.DataFrame, target: str, path: Path, written: list[Path]) -> None:
    try:
        event = frame.copy()
        event["event"] = np.select([event["actual"] <= 40.0, event["actual"] >= 80.0], ["dry<=40", "humid>=80"], default="normal")
        rows = [{"event": label, **metric_dict(group["actual"], group["prediction"])} for label, group in event.groupby("event")]
        metrics = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(metrics["event"], metrics["rmse"], color=["#60a5fa", "#f59e0b", "#22c55e"][: len(metrics)])
        ax.set_ylabel(f"RMSE ({_unit(target)})")
        ax.set_title("humidity dry/humid event error")
        _save_plot(fig, path, written)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"failed to write {path.name}") from exc


def _plot_patch_ablation(plt: Any, summary: dict[str, Any], target: str, path: Path, written: list[Path]) -> None:
    rows = [row for row in _patch_rows(summary) if row.get("target") == target]
    if not rows:
        return
    try:
        frame = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(max(6, len(frame) * 1.2), 4))
        ax.bar(frame["mode"].astype(str), pd.to_numeric(frame["rmse"], errors="coerce"), color="#2563eb")
        ax.set_ylabel(f"RMSE ({_unit(target)})")
        ax.set_title(f"{target} patch ablation")
        ax.tick_params(axis="x", rotation=20)
        _save_plot(fig, path, written)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"failed to write {path.name}") from exc


def _plot_calibration_comparison(plt: Any, summary: dict[str, Any], target: str, path: Path, written: list[Path]) -> None:
    rows = [row for row in _calibration_rows(summary) if row.get("target") == target]
    if not rows:
        return
    try:
        row = rows[0]
        labels = ["raw_rmse", "corrected_rmse", "raw_bias", "corrected_bias"]
        values = [_float_or_none(row.get(label)) or 0.0 for label in labels]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, values, color=["#94a3b8", "#2563eb", "#fbbf24", "#22c55e"])
        ax.set_ylabel(_unit(target))
        ax.set_title(f"{target} calibration raw vs corrected")
        ax.tick_params(axis="x", rotation=20)
        _save_plot(fig, path, written)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"failed to write {path.name}") from exc


def _plot_ensemble_comparison(plt: Any, summary: dict[str, Any], target: str, path: Path, written: list[Path]) -> None:
    rows = [row for row in _ensemble_rows(summary) if row.get("target") == target and row.get("method") != "n/a"]
    if not rows:
        rows = [row for row in _patch_rows(summary) if row.get("target") == target]
    if not rows:
        return
    try:
        frame = pd.DataFrame(rows)
        value_col = "test_rmse" if "test_rmse" in frame.columns else "rmse"
        label_col = "method" if "method" in frame.columns else "mode"
        fig, ax = plt.subplots(figsize=(max(7, len(frame) * 1.2), 4))
        ax.bar(frame[label_col].astype(str), pd.to_numeric(frame[value_col], errors="coerce"), color="#2563eb")
        ax.set_ylabel(f"RMSE ({_unit(target)})")
        ax.set_title(f"{target} ensemble component comparison")
        ax.tick_params(axis="x", rotation=25)
        _save_plot(fig, path, written)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"failed to write {path.name}") from exc


def _plot_humidity_model_comparison(plt: Any, summary: dict[str, Any], path: Path, written: list[Path]) -> None:
    rows = [row for row in _patch_rows(summary) if row.get("target") == "humidity"]
    rows.extend({"mode": f"ensemble:{row.get('method')}", "rmse": row.get("test_rmse")} for row in _ensemble_rows(summary) if row.get("target") == "humidity" and row.get("test_rmse") is not None)
    if not rows:
        return
    try:
        frame = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(max(7, len(frame) * 1.1), 4))
        ax.bar(frame["mode"].astype(str), pd.to_numeric(frame["rmse"], errors="coerce"), color="#0ea5e9")
        ax.set_ylabel(f"RMSE ({_unit('humidity')})")
        ax.set_title("humidity no_patch / patch / ensemble comparison")
        ax.tick_params(axis="x", rotation=25)
        _save_plot(fig, path, written)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"failed to write {path.name}") from exc


def _save_plot(fig: Any, path: Path, written: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    fig.clf()
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:  # noqa: BLE001
        pass
    written.append(path)


def write_operational_performance_summary_artifacts(summary: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "operational_performance_summary.json"
    csv_path = output / "operational_performance_summary.csv"
    rows = _operational_summary_rows(summary)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return {"summary_json": str(json_path), "summary_csv": str(csv_path)}


def _operational_summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    baselines = summary.get("official_baselines", {}) if isinstance(summary.get("official_baselines"), dict) else {}
    rows: list[dict[str, Any]] = []
    for target, cfg in TARGETS.items():
        models = baselines.get(target, {}) if isinstance(baselines.get(target), dict) else {}
        raw_metrics = models.get(cfg["official_raw"], {}) if isinstance(models.get(cfg["official_raw"]), dict) else {}
        raw_rmse = _float_or_none(raw_metrics.get("rmse"))
        for model, metrics_any in models.items():
            metrics = metrics_any if isinstance(metrics_any, dict) else {}
            rmse = _float_or_none(metrics.get("rmse"))
            rows.append(
                {
                    "target": target,
                    "unit": _unit(target),
                    "model": model,
                    "rmse": rmse,
                    "mae": _float_or_none(metrics.get("mae")),
                    "bias": _float_or_none(metrics.get("bias")),
                    "n": int(metrics.get("n", 0) or 0),
                    "patch_mode": metrics.get("patch_mode", ""),
                    "selected_calibration": metrics.get("selected_calibration", ""),
                    "rmse_improvement_pct": _improvement_pct(raw_rmse, rmse),
                    "target_gate_rmse": TARGET_GATE_RMSE[target],
                    "gate_status": _target_status_text(summary, target, rmse),
                }
            )
    beta_targets = summary.get("beta_targets", {}) if isinstance(summary.get("beta_targets"), dict) else {}
    for target, payload_any in beta_targets.items():
        payload = payload_any if isinstance(payload_any, dict) else {}
        descriptor = payload.get("descriptor", payload)
        descriptor = descriptor if isinstance(descriptor, dict) else {}
        rows.append(
            {
                "target": target,
                "unit": str(descriptor.get("unit") or ""),
                "model": descriptor.get("model_key", descriptor.get("source", "")),
                "rmse": None,
                "mae": None,
                "bias": None,
                "n": int((descriptor.get("validation_metrics") or {}).get("n", 0) or 0) if isinstance(descriptor.get("validation_metrics"), dict) else 0,
                "patch_mode": "",
                "selected_calibration": descriptor.get("selection_reason", ""),
                "rmse_improvement_pct": None,
                "target_gate_rmse": None,
                "gate_status": str(descriptor.get("status") or "unknown").upper(),
                "source": descriptor.get("source", ""),
                "confidence": descriptor.get("confidence", ""),
                "fallback_reason": descriptor.get("fallback_reason", ""),
            }
        )
    return rows


def _html_overview_section(summary: dict[str, Any]) -> str:
    data = summary.get("data", {}) if isinstance(summary.get("data"), dict) else {}
    reliability = str(summary.get("benchmark_reliability", "n/a"))
    temp_best = _best_metrics(summary, "temp")
    humidity_best = _best_metrics(summary, "humidity")
    cards = [
        _kpi_card("Benchmark reliability", reliability, _badge(_reliability_status(reliability)), "archive 신뢰도"),
        _kpi_card("Archive cycles", _fmt(summary.get("forecast_cycle_count", data.get("forecast_cycles"))), "", "forecast cycles"),
        _kpi_card("Stations", _fmt(summary.get("station_count", data.get("stations"))), "", "station count"),
        _kpi_card("Rows", _fmt(data.get("joined_rows")), "", "joined rows"),
        _kpi_card("Temp best RMSE", _metric_text(temp_best.get("rmse"), "temp"), _target_badge(summary, "temp"), str(temp_best.get("model", "n/a"))),
        _kpi_card("Humidity best RMSE", _metric_text(humidity_best.get("rmse"), "humidity"), _target_badge(summary, "humidity"), str(humidity_best.get("model", "n/a"))),
        _kpi_card("V4-C gate", str(_gate(summary, "v4c_gate").get("status", "n/a")), _badge(str(_gate(summary, "v4c_gate").get("status", "n/a"))), "operational gate"),
        _kpi_card("Site readiness", str(_gate(summary, "site_readiness").get("status", "n/a")), _badge(str(_gate(summary, "site_readiness").get("status", "n/a"))), "site beta readiness"),
    ]
    target_badges = "".join(f"<span>{html.escape(target)} {_target_badge(summary, target)}</span>" for target in TARGETS)
    return "\n".join(
        [
            "<section id='overview' class='section'><h2>Section 1: Overview</h2>",
            f"<div class='target-badges'>{target_badges}</div>",
            f"<div class='kpi-grid'>{''.join(cards)}</div>",
            f"<div class='insight'>{html.escape(_summary_sentence(summary))}</div>",
            "</section>",
        ]
    )


def _html_official_baselines_section(summary: dict[str, Any]) -> str:
    rows = [
        "<section id='official-baselines' class='section'><h2>Section 2: Official Baselines</h2>",
        "<p>Raw GFS와 Operational Residual LGBM을 target별로 비교하고 RMSE 개선율을 표시합니다.</p>",
        _html_metric_table(summary.get("official_baselines", {})),
        "</section>",
    ]
    return "\n".join(rows)


def _html_target_analysis_section(target: str, summary: dict[str, Any], output_dir: Path, embed_images: str) -> str:
    title = "Temperature" if target == "temp" else "Humidity"
    section_number = "3" if target == "temp" else "4"
    best = _best_metrics(summary, target)
    plot_cards = _plot_cards(target, output_dir, embed_images)
    if target == "temp":
        bullets = [
            "forecast vs actual 및 horizon error로 시간축 오차를 확인합니다.",
            "station/region breakdown과 heatmap으로 취약 지점을 분리합니다.",
            "patch ablation, calibration, ensemble component 비교를 함께 표시합니다.",
        ]
    else:
        bullets = [
            "forecast vs actual 및 scatter로 습도 예측 안정성을 확인합니다.",
            "dry/humid event 성능과 calibration 전후를 분리해 봅니다.",
            "no_patch / patch / ensemble 비교와 humidity beta 여부를 표시합니다.",
        ]
    rows = [
        f"<section id='{target}-analysis' class='section'><h2>Section {section_number}: {title} Analysis</h2>",
        _best_model_card(target, best),
        f"<div class='insight'>{html.escape(_target_interpretation(summary, target))}</div>",
        "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in bullets) + "</ul>",
        f"<div class='plots'>{plot_cards}</div>",
        _target_detail_tables(target, summary),
        "</section>",
    ]
    return "\n".join(rows)


def _html_gate_readiness_section(summary: dict[str, Any]) -> str:
    v4c = _gate(summary, "v4c_gate")
    site = _gate(summary, "site_readiness")
    missing = [*list(v4c.get("missing_conditions", []) or []), *list(site.get("missing_conditions", []) or [])]
    next_actions = _next_actions(summary)
    return "\n".join(
        [
            "<section id='gate-readiness' class='section'><h2>Section 5: Gate &amp; Readiness</h2>",
            "<div class='two-col'>",
            _html_gate("V4-C Gate", v4c),
            _html_gate("Site Readiness", site),
            "</div>",
            "<h3>Beta target/source readiness</h3>",
            "<p>강수확률, 바람, 구름/날씨상태는 source/status/confidence를 명시하고, 검증 개선이 없으면 direct/rule-based fallback으로 유지합니다.</p>",
            _beta_targets_table(summary),
            f"<p><b>부족한 조건:</b> {html.escape(', '.join(str(x) for x in missing) if missing else '없음')}</p>",
            "<ul>" + "".join(f"<li>{html.escape(action)}</li>" for action in next_actions) + "</ul>",
            "</section>",
        ]
    )


def _beta_targets_table(summary: dict[str, Any]) -> str:
    beta_targets = summary.get("beta_targets", {}) if isinstance(summary.get("beta_targets"), dict) else {}
    rows = []
    for target, payload_any in beta_targets.items():
        payload = payload_any if isinstance(payload_any, dict) else {}
        descriptor = payload.get("descriptor", payload)
        descriptor = descriptor if isinstance(descriptor, dict) else {}
        rows.append(
            {
                "target": target,
                "model": descriptor.get("model_key", ""),
                "source": descriptor.get("source", ""),
                "status": descriptor.get("status", ""),
                "confidence": descriptor.get("confidence", ""),
                "fallback_reason": descriptor.get("fallback_reason", ""),
            }
        )
    return _records_table(rows, ["target", "model", "source", "status", "confidence", "fallback_reason"])


def _html_experiment_tables_section(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "<section id='experiment-tables' class='section'><h2>Section 6: Experiment Table</h2>",
            "<h3>Target metrics</h3>",
            _records_table(_target_metric_rows(summary), ["target", "unit", "best_model", "rmse", "mae", "bias", "gate_status"]),
            "<h3>Model metrics</h3>",
            _records_table(_operational_summary_rows(summary), ["target", "unit", "model", "rmse", "mae", "bias", "rmse_improvement_pct", "gate_status"]),
            "<h3>Patch ablation</h3>",
            _records_table(_patch_rows(summary), ["target", "mode", "rmse", "mae", "bias", "worst_station_rmse", "late_horizon_rmse", "selected_calibration"]),
            "<h3>Calibration candidates</h3>",
            _records_table(_calibration_rows(summary), ["target", "mode", "raw_rmse", "corrected_rmse", "selected", "corrected_applied_to_test"]),
            "<h3>Ensemble candidates</h3>",
            _records_table(_ensemble_rows(summary), ["target", "method", "validation_rmse", "validation_bias", "test_rmse", "test_mae", "test_bias", "selection_status"]),
            "</section>",
        ]
    )


def _html_metric_table(baselines: Any) -> str:
    if not isinstance(baselines, dict):
        return "<div class='card'>No baselines.</div>"
    rows = ["<table><tr><th>Target</th><th>Model</th><th>RMSE</th><th>MAE</th><th>Bias</th><th>Improvement</th><th>Notes</th></tr>"]
    for target, models in baselines.items():
        if not isinstance(models, dict):
            continue
        raw_name = TARGETS.get(str(target), {}).get("official_raw")
        raw_metrics = models.get(raw_name, {}) if raw_name and isinstance(models.get(raw_name), dict) else {}
        raw_rmse = _float_or_none(raw_metrics.get("rmse"))
        for model, metrics_any in models.items():
            metrics = metrics_any if isinstance(metrics_any, dict) else {}
            improvement = _improvement_pct(raw_rmse, _float_or_none(metrics.get("rmse")))
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(target))}</td><td><code>{html.escape(str(model))}</code></td>"
                f"<td>{html.escape(_metric_text(metrics.get('rmse'), str(target)))}</td><td>{html.escape(_metric_text(metrics.get('mae'), str(target)))}</td><td>{html.escape(_metric_text(metrics.get('bias'), str(target)))}</td>"
                f"<td>{html.escape(_fmt(improvement))}%</td>"
                f"<td>{html.escape(str({k: v for k, v in metrics.items() if k not in {'rmse', 'mae', 'bias', 'n'}}))}</td>"
                "</tr>"
            )
    rows.append("</table>")
    return "\n".join(rows)


def _html_gate(title: str, gate: Any) -> str:
    gate = gate if isinstance(gate, dict) else {}
    status = str(gate.get("status", "n/a"))
    conditions = gate.get("conditions", {}) if isinstance(gate.get("conditions"), dict) else {}
    condition_rows = [
        f"<tr><td>{html.escape(str(name))}</td><td>{_condition_badge(bool(ok))}</td></tr>"
        for name, ok in conditions.items()
    ]
    missing = gate.get("missing_conditions", [])
    return "".join(
        [
            "<div class='card'>",
            f"<h3>{html.escape(title)}</h3><p>Status: {_badge(status)}</p>",
            f"<p><b>Missing:</b> {html.escape(str(missing))}</p>",
            "<table><tr><th>Condition</th><th>Status</th></tr>",
            "".join(condition_rows) or "<tr><td colspan='2'>No conditions.</td></tr>",
            "</table></div>",
        ]
    )


def _html_section(title: str, payload: Any) -> str:
    return f"<section class='section'><h2>{html.escape(title)}</h2><pre>{html.escape(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))}</pre></section>"


def _kpi_card(title: str, value: str, badge: str, note: str) -> str:
    return f"<div class='kpi-card'><div class='kpi-title'>{html.escape(title)}</div><div class='kpi-value'>{html.escape(value)}</div><div>{badge}</div><p>{html.escape(note)}</p></div>"


def _best_model_card(target: str, best: dict[str, Any]) -> str:
    label = "Temp Best Model" if target == "temp" else "Humidity Best Model"
    return "".join(
        [
            "<div class='card best-model-card'>",
            f"<h3>{label} {_target_status_from_metrics(target, best)}</h3>",
            f"<p class='mono'>{html.escape(str(best.get('model', 'n/a')))}</p>",
            f"<p>RMSE {html.escape(_metric_text(best.get('rmse'), target))} · MAE {html.escape(_metric_text(best.get('mae'), target))} · Bias {html.escape(_metric_text(best.get('bias'), target))}</p>",
            "</div>",
        ]
    )


def _plot_cards(target: str, output_dir: Path, embed_images: str) -> str:
    cards = []
    plot_dir = output_dir / "plots"
    for filename in OPERATIONAL_REPORT_PLOTS[target]:
        image = _embed_plot_image(plot_dir / filename, output_dir, filename, embed_images)
        cards.append(f"<div class='plot-card'><h4>{html.escape(filename)}</h4>{image}</div>")
    return "".join(cards)


def _embed_plot_image(path: Path, output_dir: Path, alt: str, mode: str) -> str:
    href = _relative_asset_path(path, output_dir)
    if not path.exists():
        return f'<div class="plot-placeholder">{html.escape(alt)} 사용할 수 없음 (not available)</div>'
    if mode == "external-assets":
        return f'<a href="{html.escape(href, quote=True)}"><img alt="{html.escape(alt)}" src="{html.escape(href, quote=True)}"></a>'
    warning = ""
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > 5:
            warning = f'<div class="small warn">큰 이미지: {size_mb:.1f}MB</div>'
        image_bytes = _thumbnail_png_bytes(path) if mode == "thumbnail" else path.read_bytes()
        encoded = base64.b64encode(image_bytes).decode("ascii")
    except Exception as exc:  # noqa: BLE001
        return f'<div class="plot-placeholder">{html.escape(alt)} 이미지를 포함할 수 없음: {html.escape(str(exc))}</div>'
    cls = " class=\"thumb\"" if mode == "thumbnail" else ""
    return f'{warning}<a href="{html.escape(href, quote=True)}" title="원본 PNG 열기"><img{cls} alt="{html.escape(alt)}" src="data:image/png;base64,{encoded}"></a>'


def _thumbnail_png_bytes(path: Path, max_size: tuple[int, int] = (900, 520)) -> bytes:
    from PIL import Image

    with Image.open(path) as image:
        thumbnail = image.convert("RGBA")
        thumbnail.thumbnail(max_size)
        buffer = io.BytesIO()
        thumbnail.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()


def _relative_asset_path(path: Path, output_dir: Path) -> str:
    try:
        return path.resolve().relative_to(output_dir.resolve()).as_posix()
    except Exception:  # noqa: BLE001
        return path.as_posix()


def _target_detail_tables(target: str, summary: dict[str, Any]) -> str:
    patch = [row for row in _patch_rows(summary) if row.get("target") == target]
    calibration = [row for row in _calibration_rows(summary) if row.get("target") == target]
    ensemble = [row for row in _ensemble_rows(summary) if row.get("target") == target]
    worst = _worst_summary(target, summary)
    return "\n".join(
        [
            f"<div class='card'><h3>{html.escape(target)} worst station / worst horizon</h3><p>{html.escape(worst)}</p></div>",
            "<h3>Patch ablation 결과</h3>",
            _records_table(patch, ["mode", "rmse", "mae", "bias", "worst_station_rmse", "late_horizon_rmse", "selected_calibration"]),
            "<h3>Calibration 전후 비교</h3>",
            _records_table(calibration, ["mode", "raw_rmse", "corrected_rmse", "raw_bias", "corrected_bias", "selected"]),
            "<h3>Ensemble / model comparison</h3>",
            _records_table(ensemble, ["method", "validation_rmse", "validation_bias", "test_rmse", "test_mae", "test_bias", "selection_status"]),
        ]
    )


def _records_table(records: list[dict[str, Any]], columns: list[str]) -> str:
    if not records:
        return "<div class='table-empty'>No rows available.</div>"
    rows = ["<table><tr>" + "".join(f"<th>{html.escape(column)}</th>" for column in columns) + "</tr>"]
    for record in records:
        rows.append("<tr>" + "".join(f"<td>{_html_value(record.get(column))}</td>" for column in columns) + "</tr>")
    rows.append("</table>")
    return "\n".join(rows)


def _html_value(value: Any) -> str:
    if isinstance(value, float):
        return html.escape(_fmt(value))
    return html.escape(str(value if value is not None else ""))


def _target_metric_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for target in TARGETS:
        best = _best_metrics(summary, target)
        rmse = _float_or_none(best.get("rmse"))
        rows.append(
            {
                "target": target,
                "unit": _unit(target),
                "best_model": best.get("model", "n/a"),
                "rmse": rmse,
                "mae": _float_or_none(best.get("mae")),
                "bias": _float_or_none(best.get("bias")),
                "gate_status": _target_status_text(summary, target, rmse),
            }
        )
    return rows


def _patch_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    payload = summary.get("patch_ablation", {}) if isinstance(summary.get("patch_ablation"), dict) else {}
    rows: list[dict[str, Any]] = []
    for target, target_rows in payload.items():
        if isinstance(target_rows, list):
            for row in target_rows:
                if isinstance(row, dict):
                    rows.append({"target": target, **row})
    return rows


def _calibration_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    payload = summary.get("calibration", {}) if isinstance(summary.get("calibration"), dict) else {}
    rows: list[dict[str, Any]] = []
    for target, item_any in payload.items():
        item = item_any if isinstance(item_any, dict) else {}
        raw = item.get("raw_test_metrics", {}) if isinstance(item.get("raw_test_metrics"), dict) else {}
        corrected = item.get("corrected_test_metrics", {}) if isinstance(item.get("corrected_test_metrics"), dict) else {}
        selected = item.get("selected", {}) if isinstance(item.get("selected"), dict) else {}
        rows.append(
            {
                "target": target,
                "mode": item.get("mode", ""),
                "raw_rmse": raw.get("rmse"),
                "corrected_rmse": corrected.get("rmse"),
                "raw_bias": raw.get("bias"),
                "corrected_bias": corrected.get("bias"),
                "selected": selected.get("name", item.get("selected_calibration", "")),
                "corrected_applied_to_test": item.get("corrected_applied_to_test", ""),
            }
        )
    return rows


def _ensemble_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    payload = summary.get("ensembles", {}) if isinstance(summary.get("ensembles"), dict) else {}
    rows: list[dict[str, Any]] = []
    for target, item_any in payload.items():
        item = item_any if isinstance(item_any, dict) else {}
        selection = item.get("selection_status", item.get("status", ""))
        for method in item.get("methods", []) if isinstance(item.get("methods"), list) else []:
            if isinstance(method, dict):
                rows.append({"target": target, "selection_status": selection, **method})
        best = item.get("best", {}) if isinstance(item.get("best"), dict) else {}
        if best and not any(row.get("target") == target for row in rows):
            rows.append({"target": target, "selection_status": selection, "method": best.get("method", "best"), **best})
        if not best and not item.get("methods"):
            rows.append({"target": target, "selection_status": selection, "method": "n/a"})
    return rows


def _best_metrics(summary: dict[str, Any], target: str) -> dict[str, Any]:
    best = summary.get("best_operational_models", {}) if isinstance(summary.get("best_operational_models"), dict) else {}
    if isinstance(best.get(target), dict):
        return dict(best[target])
    baselines = summary.get("official_baselines", {}) if isinstance(summary.get("official_baselines"), dict) else {}
    target_baselines = baselines.get(target, {}) if isinstance(baselines.get(target), dict) else {}
    official = TARGETS[target]["official_lgbm"]
    metrics = target_baselines.get(official, {}) if isinstance(target_baselines.get(official), dict) else {}
    return {"model": official, **metrics}


def _target_interpretation(summary: dict[str, Any], target: str) -> str:
    baselines = summary.get("official_baselines", {}) if isinstance(summary.get("official_baselines"), dict) else {}
    models = baselines.get(target, {}) if isinstance(baselines.get(target), dict) else {}
    raw = models.get(TARGETS[target]["official_raw"], {}) if isinstance(models.get(TARGETS[target]["official_raw"]), dict) else {}
    best = _best_metrics(summary, target)
    raw_rmse = _float_or_none(raw.get("rmse"))
    best_rmse = _float_or_none(best.get("rmse"))
    improvement = _improvement_pct(raw_rmse, best_rmse)
    gap = None if best_rmse is None else max(0.0, best_rmse - TARGET_GATE_RMSE[target])
    if target == "temp":
        return f"Temperature는 Raw GFS 대비 RMSE가 {_fmt(improvement)}% 개선되었지만, V4-C gate {TARGET_GATE_RMSE[target]:.1f}°C에는 아직 {_fmt(gap)}°C 부족합니다. Patch5/true-grid 계열은 temperature에 유효한 후보로 유지합니다."
    beta = "PASS" if best_rmse is not None and best_rmse <= TARGET_GATE_RMSE[target] else "BETA/WARN"
    return f"Humidity는 no_patch LGBM이 가장 안정적인 기준이며, ensemble은 validation holdout에서 개선될 때만 채택됩니다. 현재 humidity beta 상태는 {beta}이고 gate까지 차이는 {_fmt(gap)}%p입니다."


def _summary_sentence(summary: dict[str, Any]) -> str:
    temp = _target_interpretation(summary, "temp")
    humidity = _target_interpretation(summary, "humidity")
    return f"{temp} {humidity} Site readiness는 temp/humidity gate와 benchmark reliability를 모두 만족해야 PASS입니다."


def _worst_summary(target: str, summary: dict[str, Any]) -> str:
    patch_rows = [row for row in _patch_rows(summary) if row.get("target") == target]
    best_row = min(patch_rows, key=lambda row: float(row.get("rmse", float("inf")))) if patch_rows else {}
    worst_station = best_row.get("worst_station_rmse")
    late = best_row.get("late_horizon_rmse")
    return f"Worst station RMSE {_metric_text(worst_station, target)}, late/worst horizon RMSE {_metric_text(late, target)} 기준으로 다음 액션을 우선순위화합니다."


def _next_actions(summary: dict[str, Any]) -> list[str]:
    actions = []
    for target in TARGETS:
        best = _best_metrics(summary, target)
        rmse = _float_or_none(best.get("rmse"))
        if rmse is None or rmse > TARGET_GATE_RMSE[target]:
            actions.append(f"{target} RMSE를 {TARGET_GATE_RMSE[target]}{_unit(target)} 이하로 낮추는 feature/calibration 후보를 추가 검증합니다.")
    if str(summary.get("benchmark_reliability", "")) not in {"medium", "strong", "seasonal"}:
        actions.append("benchmark reliability를 medium 이상으로 올릴 수 있는 real archive coverage를 확보합니다.")
    if not actions:
        actions.append("현재 gate 조건은 충족했으므로 운영 beta 절차와 regression monitoring을 준비합니다.")
    return actions


def _target_badge(summary: dict[str, Any], target: str) -> str:
    return _badge(_target_status_text(summary, target, _float_or_none(_best_metrics(summary, target).get("rmse"))))


def _target_status_from_metrics(target: str, metrics: dict[str, Any]) -> str:
    return _badge(_target_status_from_rmse(target, _float_or_none(metrics.get("rmse"))))


def _target_status_text(summary: dict[str, Any], target: str, rmse: float | None) -> str:
    manifest = summary.get("production_model_manifest", {}) if isinstance(summary.get("production_model_manifest"), dict) else {}
    key = "temperature_status" if target == "temp" else "humidity_status"
    if key in manifest:
        return str(manifest[key])
    return _target_status_from_rmse(target, rmse)


def _target_status_from_rmse(target: str, rmse: float | None) -> str:
    if rmse is None:
        return "FAIL"
    if rmse <= TARGET_GATE_RMSE[target]:
        return "PASS"
    if rmse <= TARGET_NEAR_PASS_RMSE[target]:
        return "NEAR_PASS"
    return "FAIL"


def _badge(status: str) -> str:
    normalized = status.upper()
    css = "badge-ok" if normalized == "PASS" else "badge-warn" if normalized in {"WARN", "NEAR_PASS", "BETA", "BETA/WARN"} else "badge-bad" if normalized == "FAIL" else "badge-muted"
    return f'<span class="badge {css}">{html.escape(normalized)}</span>'


def _condition_badge(ok: bool) -> str:
    return _badge("PASS" if ok else "FAIL")


def _gate(summary: dict[str, Any], key: str) -> dict[str, Any]:
    gate = summary.get(key, {}) if isinstance(summary.get(key), dict) else {}
    return gate


def _reliability_status(reliability: str) -> str:
    return "PASS" if reliability in {"medium", "strong", "seasonal"} else "WARN"


def _metric_text(value: Any, target: str) -> str:
    return f"{_fmt(value)} {_unit(target)}"


def _unit(target: str) -> str:
    return TARGET_UNITS.get(str(target), "")


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
        return number if np.isfinite(number) else None
    except Exception:  # noqa: BLE001
        return None


def _improvement_pct(raw_rmse: float | None, candidate_rmse: float | None) -> float | None:
    if raw_rmse is None or candidate_rmse is None or raw_rmse == 0:
        return None
    return float((raw_rmse - candidate_rmse) / raw_rmse * 100.0)


def _operational_report_css() -> str:
    return """
:root { --text:#0f172a; --muted:#64748b; --line:#d8dee9; --panel:#ffffff; --bg:#f6f8fb; --blue:#2563eb; }
body { font-family: Inter, Segoe UI, Arial, sans-serif; margin:0; background:var(--bg); color:var(--text); }
.hero { padding:34px 42px; background:linear-gradient(135deg,#0f172a,#1e3a8a); color:white; }
.hero h1 { margin:4px 0 8px; font-size:34px; } .eyebrow { text-transform:uppercase; letter-spacing:.08em; opacity:.8; font-size:12px; }
.section { margin:24px auto; max-width:1280px; padding:0 24px; }
.card, .kpi-card { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:16px; box-shadow:0 8px 24px rgba(15,23,42,.04); }
.kpi-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:14px; margin:14px 0; }
.kpi-title { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; } .kpi-value { font-size:24px; font-weight:800; margin:6px 0; }
.best-model-card { margin:12px 0; border-left:5px solid var(--blue); }
.target-badges { display:flex; gap:14px; flex-wrap:wrap; margin:12px 0; }
.badge { display:inline-block; padding:4px 9px; border-radius:999px; font-size:12px; font-weight:800; }
.badge-ok { color:#05603a; background:#d1fadf; } .badge-warn { color:#93370d; background:#fef0c7; } .badge-bad { color:#912018; background:#fee4e2; } .badge-muted { color:#475467; background:#eaecf0; }
.insight { background:#eef6ff; border:1px solid #bfdbfe; border-radius:14px; padding:14px; margin:12px 0; line-height:1.55; }
.two-col { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:14px; }
table { width:100%; border-collapse:collapse; background:white; margin:12px 0; font-size:13px; } th, td { border:1px solid var(--line); padding:8px; text-align:left; vertical-align:top; } th { background:#edf2f7; }
.plots { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:14px; }
.plot-card { background:white; border:1px solid var(--line); border-radius:14px; padding:12px; } .plot-card h4 { margin:0 0 8px; font-size:14px; }
.plot-card img { max-width:100%; border-radius:10px; border:1px solid #e5e7eb; background:white; } .plot-card img.thumb { max-height:220px; object-fit:contain; display:block; margin:auto; }
.plot-placeholder, .table-empty { color:var(--muted); background:#f2f4f7; border:1px dashed #cbd5e1; padding:28px; border-radius:10px; text-align:center; }
.mono, code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; } pre { white-space:pre-wrap; background:#0f172a; color:#e2e8f0; border-radius:12px; padding:14px; overflow:auto; }
.small { font-size:12px; } .warn { color:#b45309; }
"""



def _fmt(value: Any) -> str:
    try:
        if value is None:
            return "n/a"
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)
