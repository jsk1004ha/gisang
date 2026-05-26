from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from weather_korea_forecast.utils.paths import resolve_path


def load_dashboard_state(artifact_root: str | Path = "data/artifacts/v3_experiments") -> dict[str, Any]:
    """Read artifact files into a compact web-dashboard state object."""

    root = resolve_path(artifact_root)
    leaderboard = _read_leaderboard(root / "leaderboard.csv")
    rows = _records(leaderboard)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_root": str(root),
        "leaderboard_path": str(root / "leaderboard.csv"),
        "counts": _counts(leaderboard),
        "best_by_target": _best_by(leaderboard, "target_name"),
        "best_by_track": _best_by(leaderboard, "forecast_track"),
        "recent_experiments": _recent_experiments(root),
        "aliases": {
            "latest": _alias_state(root / "latest", "latest_experiment"),
            "best": _alias_state(root / "best", "best_experiment"),
        },
        "leaderboard": rows,
        "comparisons": _comparison_reports(root),
    }


def render_dashboard_html(state: dict[str, Any]) -> str:
    """Render a dependency-free HTML dashboard for local use."""

    title = "Weather Korea Forecast Dashboard"
    leaderboard_rows = "\n".join(_leaderboard_row(row) for row in state.get("leaderboard", [])[:100])
    best_target_rows = "\n".join(_summary_row(row) for row in state.get("best_by_target", []))
    best_track_rows = "\n".join(_summary_row(row) for row in state.get("best_by_track", []))
    recent_rows = "\n".join(
        f"<li><code>{_esc(item.get('name'))}</code> — {_esc(item.get('updated_at'))}</li>"
        for item in state.get("recent_experiments", [])[:12]
    )
    comparison_rows = "\n".join(
        f"<li><a href='/artifact/{_esc(item.get('relative_path'))}'>{_esc(item.get('name'))}</a> — {_esc(item.get('updated_at'))}</li>"
        for item in state.get("comparisons", [])[:10]
    )
    latest = state.get("aliases", {}).get("latest", {})
    best = state.get("aliases", {}).get("best", {})
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="60">
  <title>{title}</title>
  <style>
    :root {{ color-scheme: light dark; --accent: #2563eb; --good: #059669; --warn: #d97706; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }}
    header {{ padding: 24px 32px; background: linear-gradient(135deg, #1e3a8a, #0f172a); }}
    main {{ padding: 24px 32px; }}
    section {{ background: rgba(15, 23, 42, 0.86); border: 1px solid #334155; border-radius: 14px; margin-bottom: 20px; padding: 18px; }}
    h1, h2 {{ margin: 0 0 12px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .card {{ background: #111827; border: 1px solid #334155; border-radius: 12px; padding: 14px; }}
    .metric {{ font-size: 24px; font-weight: 700; color: #93c5fd; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #334155; text-align: left; vertical-align: top; }}
    th {{ color: #bfdbfe; position: sticky; top: 0; background: #111827; }}
    code {{ color: #bae6fd; }}
    a {{ color: #93c5fd; }}
    .ok {{ color: var(--good); }}
    .warn {{ color: var(--warn); }}
    .small {{ color: #94a3b8; font-size: 13px; }}
    .scroll {{ overflow-x: auto; }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <p>Artifact root: <code>{_esc(state.get('artifact_root'))}</code></p>
    <p class="small">Generated: {_esc(state.get('generated_at'))} · Auto-refresh: 60s · JSON: <a href="/api/status">/api/status</a></p>
  </header>
  <main>
    <section class="cards">
      <div class="card"><div class="small">Experiments</div><div class="metric">{_esc(state.get('counts', {}).get('experiments', 0))}</div></div>
      <div class="card"><div class="small">Targets</div><div class="metric">{_esc(state.get('counts', {}).get('targets', 0))}</div></div>
      <div class="card"><div class="small">Tracks</div><div class="metric">{_esc(state.get('counts', {}).get('tracks', 0))}</div></div>
      <div class="card"><div class="small">Latest</div><code>{_esc(latest.get('experiment'))}</code></div>
      <div class="card"><div class="small">Best</div><code>{_esc(best.get('experiment'))}</code></div>
    </section>

    <section>
      <h2>Best by target</h2>
      <div class="scroll"><table><thead><tr><th>group</th><th>experiment</th><th>target</th><th>track</th><th>RMSE</th><th>MAE</th><th>Bias</th><th>dir</th></tr></thead><tbody>{best_target_rows}</tbody></table></div>
    </section>

    <section>
      <h2>Best by forecast track</h2>
      <div class="scroll"><table><thead><tr><th>group</th><th>experiment</th><th>target</th><th>track</th><th>RMSE</th><th>MAE</th><th>Bias</th><th>dir</th></tr></thead><tbody>{best_track_rows}</tbody></table></div>
    </section>

    <section>
      <h2>Recent experiments</h2>
      <ul>{recent_rows}</ul>
    </section>

    <section>
      <h2>Comparison reports</h2>
      <ul>{comparison_rows}</ul>
    </section>

    <section>
      <h2>Leaderboard</h2>
      <div class="scroll"><table><thead><tr><th>experiment</th><th>target</th><th>model</th><th>family</th><th>track</th><th>future</th><th>operational</th><th>RMSE</th><th>MAE</th><th>Bias</th><th>dir</th></tr></thead><tbody>{leaderboard_rows}</tbody></table></div>
    </section>
  </main>
</body>
</html>
"""


def _read_leaderboard(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    normalized = frame.copy()
    metric_column = "rmse_corrected" if "rmse_corrected" in normalized.columns else "rmse"
    normalized[metric_column] = pd.to_numeric(normalized[metric_column], errors="coerce")
    normalized = normalized.sort_values(metric_column, na_position="last")
    return [_json_safe(row) for row in normalized.to_dict(orient="records")]


def _counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {"experiments": 0, "targets": 0, "tracks": 0}
    return {
        "experiments": int(len(frame)),
        "targets": int(frame["target_name"].nunique()) if "target_name" in frame.columns else 0,
        "tracks": int(frame["forecast_track"].nunique()) if "forecast_track" in frame.columns else 0,
    }


def _best_by(frame: pd.DataFrame, group_column: str) -> list[dict[str, Any]]:
    if frame.empty or group_column not in frame.columns:
        return []
    metric_column = "rmse_corrected" if "rmse_corrected" in frame.columns else "rmse"
    normalized = frame.copy()
    normalized[metric_column] = pd.to_numeric(normalized[metric_column], errors="coerce")
    rows = []
    for group, group_frame in normalized.groupby(group_column, dropna=False):
        best = group_frame.sort_values(metric_column, na_position="last").iloc[0].to_dict()
        best["group"] = group
        rows.append(_json_safe(best))
    return rows


def _recent_experiments(root: Path) -> list[dict[str, str]]:
    rows = []
    if not root.exists():
        return rows
    for path in root.iterdir():
        if not path.is_dir() or path.name in {"latest", "best"}:
            continue
        summary_path = path / "experiment_summary.json"
        metrics_path = path / "metrics_test.json"
        if not summary_path.exists() and not metrics_path.exists():
            continue
        rows.append(
            {
                "name": path.name,
                "path": str(path),
                "updated_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    return sorted(rows, key=lambda row: row["updated_at"], reverse=True)


def _alias_state(alias_dir: Path, manifest_key: str) -> dict[str, Any]:
    manifest = _read_json(alias_dir / "manifest.json")
    summary = _read_json(alias_dir / "experiment_summary.json")
    metrics = _read_json(alias_dir / "metrics_summary.json")
    return {
        "experiment": manifest.get(manifest_key),
        "summary": summary,
        "metrics": metrics,
    }


def _comparison_reports(root: Path) -> list[dict[str, str]]:
    rows = []
    if not root.exists():
        return rows
    for path in root.glob("*comparison*.md"):
        rows.append(
            {
                "name": path.name,
                "path": str(path),
                "relative_path": path.name,
                "updated_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    return sorted(rows, key=lambda row: row["updated_at"], reverse=True)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_safe(row: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in row.items():
        if pd.isna(value):
            safe[key] = None
        else:
            safe[key] = value.item() if hasattr(value, "item") else value
    return safe


def _leaderboard_row(row: dict[str, Any]) -> str:
    future = "yes" if _truthy(row.get("uses_future_weather_features")) else "no"
    operational = "yes" if _truthy(row.get("operational_valid")) else "no"
    return (
        "<tr>"
        f"<td>{_esc(row.get('experiment_name'))}</td>"
        f"<td>{_esc(row.get('target_name'))}</td>"
        f"<td>{_esc(row.get('model_type'))}</td>"
        f"<td>{_esc(row.get('model_family'))}</td>"
        f"<td>{_esc(row.get('forecast_track'))}</td>"
        f"<td>{future}</td>"
        f"<td>{operational}</td>"
        f"<td>{_fmt(row.get('rmse_corrected') or row.get('rmse'))}</td>"
        f"<td>{_fmt(row.get('mae_corrected') or row.get('mae'))}</td>"
        f"<td>{_fmt(row.get('bias_corrected') or row.get('bias'))}</td>"
        f"<td><code>{_esc(_short_path(row.get('experiment_dir')))}</code></td>"
        "</tr>"
    )


def _summary_row(row: dict[str, Any]) -> str:
    return (
        "<tr>"
        f"<td>{_esc(row.get('group'))}</td>"
        f"<td>{_esc(row.get('experiment_name'))}</td>"
        f"<td>{_esc(row.get('target_name'))}</td>"
        f"<td>{_esc(row.get('forecast_track'))}</td>"
        f"<td>{_fmt(row.get('rmse_corrected') or row.get('rmse'))}</td>"
        f"<td>{_fmt(row.get('mae_corrected') or row.get('mae'))}</td>"
        f"<td>{_fmt(row.get('bias_corrected') or row.get('bias'))}</td>"
        f"<td><code>{_esc(_short_path(row.get('experiment_dir')))}</code></td>"
        "</tr>"
    )


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _short_path(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\\", "/").split("/data/artifacts/")[-1]


def _esc(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
