from __future__ import annotations

import html
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from weather_korea_forecast.reporting.plot_embed import embed_image_tag
from weather_korea_forecast.reporting.schema import PLOT_FILES, ExperimentRecord
from weather_korea_forecast.reporting.site_readiness import evaluate_site_readiness


_BADGE_CLASS = {
    True: "badge badge-ok",
    False: "badge badge-bad",
    None: "badge badge-muted",
}


def render_report(
    records: list[ExperimentRecord],
    *,
    title: str,
    experiments_root: Path,
    include_images: bool = True,
    image_mode: str = "full",
) -> str:
    complete = [record for record in records if record.complete and record.rmse is not None]
    incomplete = [record for record in records if not record.complete or record.error]
    main_records = _main_records(records)
    diagnostic = [record for record in records if record.is_diagnostic]
    alias_records = [record for record in records if record.is_alias_artifact]
    generated_at = datetime.now(timezone.utc).isoformat()
    best = _best_records(main_records)
    effective_image_mode = image_mode if include_images else "none"
    table_rows = "\n".join(_leaderboard_row(record) for record in _sort_records(main_records))
    detail_sections = "\n".join(_detail_section(record, image_mode=effective_image_mode) for record in _sort_records(records))
    warnings = _warnings_section(records)
    charts = _summary_charts(main_records)
    kpis = _kpi_cards(main_records, best, diagnostic_count=len(diagnostic), alias_count=len(alias_records))
    filters_payload = _json_for_inline_script(_filter_options(main_records))
    return f"""<!doctype html>
<html lang=\"ko\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{html.escape(title)}</title>
  <style>{_css()}</style>
</head>
<body>
  <header class=\"hero\">
    <div>
      <h1>{html.escape(title)}</h1>
      <p>생성 시각 {html.escape(generated_at)} · 실험 루트 <code>{html.escape(str(experiments_root))}</code></p>
      {_header_best_summary(best)}
    </div>
    <div class=\"hero-stats\">
      <span><b>{len(records)}</b>개 실험</span>
      <span><b>{len(complete)}</b>개 완료</span>
      <span><b>{len(incomplete)}</b>개 불완전</span>
    </div>
  </header>
  {_operational_banner(main_records)}
  {kpis}
  {_readiness_section(records, main_records)}
  {_v4_validation_section(records, main_records)}
  {_g022_performance_section(records, main_records)}
  {_v4_c_gate_section(main_records)}
  <section class=\"card\">
    <h2>리더보드</h2>
    <div class=\"filters\">
      <input id=\"searchBox\" placeholder=\"실험 이름 검색\" oninput=\"filterTable()\">
      {_select('versionFilter', '버전')}
      {_select('targetFilter', '타깃')}
      {_select('trackFilter', '트랙')}
      {_select('modelFilter', '모델 유형')}
      {_select('sourceFilter', '미래 feature 소스')}
      {_select('opFilter', '운영 가능')}
      {_select('stageFilter', 'V4 단계')}
      {_select('schemaFilter', 'Schema 검증')}
      {_select('patchFilter', 'Patch feature')}
      {_select('goalFilter', '목표 달성')}
    </div>
    <div class=\"table-wrap\">
      <table id=\"leaderboard\">
        <thead><tr>
          <th onclick=\"sortTable(0)\">실험</th><th onclick=\"sortTable(1)\">버전</th><th onclick=\"sortTable(2)\">타깃</th><th onclick=\"sortTable(3)\">트랙</th><th onclick=\"sortTable(4)\">모델</th><th onclick=\"sortTable(5)\">RMSE</th><th onclick=\"sortTable(6)\">MAE</th><th onclick=\"sortTable(7)\">Bias</th><th onclick=\"sortTable(8)\">최악 Horizon RMSE</th><th onclick=\"sortTable(9)\">최악 관측소 RMSE</th><th onclick=\"sortTable(10)\">목표</th><th>V4</th><th>상태</th><th>Artifact 경로</th>
        </tr></thead>
        <tbody>{table_rows}</tbody>
      </table>
    </div>
  </section>
  <section class=\"grid two\">
    {charts}
  </section>
  {_best_model_section(best, image_mode=effective_image_mode)}
  {_diagnostic_section(diagnostic, image_mode=effective_image_mode)}
  {_alias_section(alias_records)}
  {warnings}
  <section class=\"card\">
    <h2>실험별 상세</h2>
    {detail_sections or '<p>수집된 실험이 없습니다.</p>'}
  </section>
  <script>const FILTER_OPTIONS = {filters_payload};\n{_js()}</script>
</body>
</html>
"""


def _json_for_inline_script(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False)
    return (
        payload.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _select(element_id: str, label: str) -> str:
    return f'<label>{html.escape(label)}<select id="{element_id}" onchange="filterTable()"><option value="">전체</option></select></label>'


def _header_best_summary(best: dict[str, ExperimentRecord]) -> str:
    items = [
        ("기온 observation-only", best.get("best temp observation-only")),
        ("기온 NWP-assisted", best.get("best temp NWP-assisted")),
        ("습도 observation-only", best.get("best humidity observation-only")),
        ("습도 NWP-assisted", best.get("best humidity NWP-assisted")),
        ("운영 가능", best.get("best operational-valid")),
        ("백테스트", best.get("best backtest-only")),
    ]
    rendered = []
    for label, record in items:
        if record is None:
            rendered.append(f"<span>{html.escape(label)}: n/a</span>")
        else:
            rendered.append(f"<span>{html.escape(label)}: <b>{_fmt(record.rmse)}</b> <code>{html.escape(record.experiment_name)}</code></span>")
    return '<div class="hero-best">' + ''.join(rendered) + '</div>'


def _main_records(records: Iterable[ExperimentRecord]) -> list[ExperimentRecord]:
    return [
        record
        for record in records
        if record.included_in_main_leaderboard
        or (
            record.complete
            and record.rmse is not None
            and not record.is_diagnostic
            and not record.is_alias_artifact
            and record.is_representative_run
            and float(record.rmse) > 0.0
        )
    ]


def _sort_records(records: Iterable[ExperimentRecord]) -> list[ExperimentRecord]:
    return sorted(records, key=lambda r: (r.target_name or "", r.track or "", float("inf") if r.rmse is None else r.rmse, r.experiment_name))


def _operational_banner(records: list[ExperimentRecord]) -> str:
    if any(record.operational_valid is True for record in records):
        return ""
    return """<section class="banner banner-warn">
      <strong>운영 가능 모델 없음</strong>
      <span>현재 main leaderboard의 대표 모델은 operational_valid=true가 없습니다. ERA5 reanalysis 기반 최고 성능 모델은 backtest-only로 해석해야 하며, 실제 forecast NWP 입력으로 별도 검증이 필요합니다.</span>
    </section>"""


def _leaderboard_row(record: ExperimentRecord) -> str:
    goal_class = _goal_class(record)
    status = []
    if record.is_diagnostic:
        status.append('<span class="badge badge-purple">Diagnostic</span>')
    if record.is_alias_artifact:
        status.append('<span class="badge badge-muted">alias</span>')
    if record.is_representative_run and not record.is_alias_artifact:
        status.append('<span class="badge badge-muted">대표 run</span>')
    if not record.goal_eligible:
        status.append('<span class="badge badge-muted">목표 집계 제외</span>')
    elif record.rmse_goal_met is True:
        status.append('<span class="badge badge-ok">목표 달성</span>')
    elif record.rmse_goal_met is False:
        status.append('<span class="badge badge-bad">목표 차이 {}</span>'.format(_fmt(record.rmse_gap_to_goal)))
    if record.backtest_only:
        status.append('<span class="badge badge-blue">백테스트</span>')
    if record.operational_valid is True:
        status.append('<span class="badge badge-ok">운영 가능</span>')
    elif record.operational_valid is False:
        status.append('<span class="badge badge-muted">운영 불가</span>')
    v4_badges = _v4_badges(record)
    return f"""<tr data-version=\"{_attr(record.version)}\" data-target=\"{_attr(record.target_name)}\" data-track=\"{_attr(record.track)}\" data-model=\"{_attr(record.model_type)}\" data-source=\"{_attr(record.future_feature_source)}\" data-op=\"{_attr(str(record.operational_valid))}\" data-stage=\"{_attr(record.v4_stage)}\" data-schema=\"{_attr(str(record.forecast_schema_valid))}\" data-patch=\"{_attr(str(record.patch_features_enabled))}\" data-goal=\"{_attr(str(record.rmse_goal_met))}\" data-name=\"{_attr(record.experiment_name.lower())}\">
      <td class=\"mono\">{html.escape(record.experiment_name)}</td><td>{html.escape(record.version)}</td><td>{html.escape(record.target_name)}</td><td>{html.escape(record.track)}</td><td>{html.escape(record.model_type)}</td>
      <td class=\"num {goal_class}\">{_fmt(record.rmse)}</td><td class=\"num\">{_fmt(record.mae)}</td><td class=\"num\">{_fmt(record.bias)}</td><td class=\"num\">{_fmt(record.worst_horizon_rmse)}</td><td class=\"num\">{_fmt(record.worst_station_rmse)}</td><td class=\"num\">{_fmt(record.rmse_goal)}</td>
      <td>{v4_badges}</td><td>{''.join(status) or '<span class="badge badge-muted">n/a</span>'}</td><td class=\"mono small\">{html.escape(record.artifact_dir)}</td>
    </tr>"""


def _v4_badges(record: ExperimentRecord) -> str:
    badges = [f'<span class="badge badge-blue">{html.escape(record.v4_stage)}</span>']
    if record.forecast_schema_valid is True:
        label = "schema ok" if not record.forecast_schema_version else f"schema {record.forecast_schema_version}"
        badges.append(f'<span class="badge badge-ok">{html.escape(label)}</span>')
    elif record.forecast_schema_valid is False:
        badges.append('<span class="badge badge-bad">schema invalid</span>')
    if record.patch_features_enabled is True:
        patch_label = "patch"
        if record.patch_size:
            patch_label += f" {record.patch_size}x{record.patch_size}"
        if record.patch_feature_set:
            patch_label += f" {record.patch_feature_set}"
        badges.append(f'<span class="badge badge-purple">{html.escape(patch_label)}</span>')
    elif record.patch_features_enabled is False:
        badges.append('<span class="badge badge-muted">no patch</span>')
    return ''.join(badges)


def _detail_section(record: ExperimentRecord, *, image_mode: str = "full") -> str:
    metrics = f"RMSE {_fmt(record.rmse)} · MAE {_fmt(record.mae)} · Bias {_fmt(record.bias)}"
    if record.is_diagnostic:
        goal = "진단/oracle 실험: 목표 달성 집계 제외"
    else:
        goal = "n/a" if record.rmse_goal is None else ("달성" if record.rmse_goal_met else f"목표 차이 {_fmt(record.rmse_gap_to_goal)}")
    images = ""
    if image_mode != "none":
        cards = []
        for plot in PLOT_FILES:
            path = Path(record.artifact_dir) / plot
            cards.append(f'<div class="plot-card"><h4>{html.escape(plot)}</h4>{embed_image_tag(path, plot, mode=image_mode)}</div>')
        images = f'<div class="plots">{"".join(cards)}</div>'
    warnings = "".join(f"<li>{html.escape(_warning_label(w))}</li>" for w in record.warnings)
    warnings_block = f"<ul class=\"warnings\">{warnings}</ul>" if warnings else "<p>수집 경고가 없습니다.</p>"
    return f"""<details class=\"detail\">
      <summary><span class=\"mono\">{html.escape(record.experiment_name)}</span> <span>{html.escape(metrics)}</span></summary>
      <div class=\"detail-body\">
        <div class=\"grid two\">
          <div><h3>성능 지표</h3><dl>{_dl({'타깃': record.target_name, '트랙': record.track, '모델': record.model_type, 'Encoder 길이': record.encoder_length, '예측 길이': record.prediction_length, 'RMSE': record.rmse, 'MAE': record.mae, 'Bias': record.bias, 'MAPE': record.mape, 'Raw RMSE': record.raw_rmse, 'Raw MAE': record.raw_mae, 'Raw Bias': record.raw_bias, 'Val RMSE': record.val_rmse, 'RMSE 목표': record.rmse_goal, '목표 상태': goal, '최악 horizon': record.worst_horizon_step, '최악 horizon RMSE': record.worst_horizon_rmse, '최악 horizon MAE': record.worst_horizon_mae, '최악 horizon Bias': record.worst_horizon_bias, '최악 관측소': record.worst_station_id, '최악 관측소 RMSE': record.worst_station_rmse, 'Backtest baseline RMSE': record.backtest_baseline_rmse, 'Operational gap': record.operational_gap, 'Operational gap status': record.operational_gap_status, 'Patch baseline RMSE': record.patch_baseline_rmse, 'Patch improvement RMSE': record.patch_improvement_rmse, 'Patch improvement worst station': record.patch_improvement_worst_station, 'Patch improvement late horizon': record.patch_improvement_late_horizon})}</dl></div>
          <div><h3>운영/대표 run 상태</h3><dl>{_dl({'미래 feature 소스': record.future_feature_source, '미래 기상 feature 사용': record.uses_future_weather_features, '운영 가능': record.operational_valid, '백테스트 전용': record.backtest_only, '진단/oracle': record.is_diagnostic, 'artifact profile': record.artifact_profile, 'alias artifact': record.is_alias_artifact, '대표 run': record.is_representative_run, 'canonical id': record.canonical_experiment_id, 'run timestamp': record.run_timestamp, '누수 위험 메모': record.leakage_risk_note, 'V4 단계': record.v4_stage, 'forecast schema version': record.forecast_schema_version, 'forecast schema valid': record.forecast_schema_valid, 'forecast source schema valid': record.forecast_source_schema_valid, 'forecast archive adequate': record.forecast_archive_adequate, 'forecast archive rows': record.forecast_archive_row_count, 'forecast archive stations': record.forecast_archive_station_count, 'forecast archive issue times': record.forecast_archive_issue_time_count, 'forecast source path': record.forecast_source_path, 'patch features': record.patch_features_enabled, 'uses patch features': record.uses_patch_features, 'patch size': record.patch_size, 'patch feature set': record.patch_feature_set, 'patch feature mode': record.patch_feature_mode, 'Artifact 경로': record.artifact_dir})}</dl></div>
        </div>
        <h3>경고 / 메모</h3>{warnings_block}
        {images}
      </div>
    </details>"""


def _dl(values: dict[str, object]) -> str:
    return "".join(f"<dt>{html.escape(k)}</dt><dd>{html.escape(str(v)) if v is not None else 'n/a'}</dd>" for k, v in values.items())


def _kpi_cards(records: list[ExperimentRecord], best: dict[str, ExperimentRecord], *, diagnostic_count: int, alias_count: int) -> str:
    cards = [
        ("기온 observation-only Best", _fmt(_best_rmse(records, target="temp", track="observation_only"))),
        ("기온 NWP-assisted Best", _fmt(_best_rmse(records, target="temp", track="nwp_assisted_mos"))),
        ("습도 observation-only Best", _fmt(_best_rmse(records, target="humidity", track="observation_only"))),
        ("습도 NWP-assisted Best", _fmt(_best_rmse(records, target="humidity", track="nwp_assisted_mos"))),
        ("운영 가능 Best", _fmt(_best_operational_rmse(records))),
        ("백테스트 전용 Best", _fmt(_best_backtest_rmse(records))),
        ("RMSE 목표 달성", str(sum(1 for r in records if r.goal_eligible and r.rmse_goal_met is True))),
        ("main 대표 run", str(len(records))),
        ("Diagnostic/oracle", str(diagnostic_count)),
        ("Alias artifact", str(alias_count)),
    ]
    return '<section class="kpis">' + ''.join(f'<div class="kpi"><span>{html.escape(k)}</span><b>{html.escape(v)}</b></div>' for k, v in cards) + '</section>'


def _readiness_section(all_records: list[ExperimentRecord], main_records: list[ExperimentRecord]) -> str:
    operational_temp = [
        r
        for r in main_records
        if r.target_name == "temp" and r.track == "nwp_assisted_mos" and r.operational_valid is True and r.rmse is not None
    ]
    best_operational_temp = min((float(r.rmse) for r in operational_temp), default=None)
    patch_candidates = [
        r
        for r in main_records
        if r.target_name == "temp" and r.track == "nwp_assisted_mos" and r.uses_patch_features is True
    ]
    patch_improved = any(
        (r.patch_improvement_rmse is not None and r.patch_improvement_rmse > 0)
        or (r.patch_improvement_worst_station is not None and r.patch_improvement_worst_station > 0)
        for r in patch_candidates
    )
    checks = [
        (
            "Diagnostic/oracle 분리",
            "pass" if all(not r.included_in_main_leaderboard for r in all_records if r.is_diagnostic) else "fail",
            "oracle/diagnostic row가 main leaderboard/KPI/best 산정에서 제외되어야 합니다.",
        ),
        (
            "Alias artifact dedupe",
            "pass" if all(not r.included_in_main_leaderboard for r in all_records if r.is_alias_artifact) else "fail",
            "best/latest alias는 별도 섹션과 상세에서만 표시되어야 합니다.",
        ),
        (
            "Temp NWP-assisted ≤ 1.1°C",
            "pass" if (_best_rmse(main_records, target="temp", track="nwp_assisted_mos") or float("inf")) <= 1.1 else "fail",
            f"현재 {_fmt(_best_rmse(main_records, target='temp', track='nwp_assisted_mos'))}; V3.5 안정화 기준입니다.",
        ),
        (
            "Temp NWP-assisted ≤ 1.0°C",
            "pass" if (_best_rmse(main_records, target="temp", track="nwp_assisted_mos") or float("inf")) <= 1.0 else "warn",
            "목표 RMSE 1.0°C 이하. 미달이면 residual LGBM/ensemble/calibration 후보를 유지합니다.",
        ),
        (
            "Prepared forecast CSV adapter implemented",
            "pass" if any(r.future_feature_source == "prepared_forecast_csv" or r.forecast_schema_version for r in all_records) else "warn",
            "prepared_forecast_csv schema/metadata가 report에 수집되어야 합니다.",
        ),
        (
            "V4 forecast schema validation",
            "pass" if any(r.forecast_schema_valid is True for r in main_records) else "warn",
            "prepared forecast CSV schema가 검증된 대표 run이 있어야 operational_valid=true를 신뢰할 수 있습니다.",
        ),
        (
            "Operational-valid temp model exists",
            "pass" if operational_temp else "warn",
            "operational_valid=true 기온 MOS 대표 모델이 있어야 V4-C 운영 검증으로 넘어갈 수 있습니다.",
        ),
        (
            "Operational temp RMSE ≤ 1.5°C",
            "pass" if (best_operational_temp if best_operational_temp is not None else float("inf")) <= 1.5 else "warn",
            f"현재 {_fmt(best_operational_temp)}; V4-C 진입 1차 gate입니다.",
        ),
        (
            "Operational temp RMSE ≤ 1.2°C",
            "pass" if (best_operational_temp if best_operational_temp is not None else float("inf")) <= 1.2 else "warn",
            "좋은 operational 모델 기준입니다.",
        ),
        (
            "Operational temp RMSE ≤ 1.0°C",
            "pass" if (best_operational_temp if best_operational_temp is not None else float("inf")) <= 1.0 else "warn",
            "최종 운영 목표입니다.",
        ),
        (
            "Patch ablation completed",
            "pass" if any(r.uses_patch_features is True for r in main_records) else "warn",
            "no patch / 3x3 / 5x5 비교 대표 run이 필요합니다. Diagnostic smoke는 성능 gate를 충족하지 않습니다.",
        ),
        (
            "V4 patch feature readiness",
            "pass" if any(r.patch_features_enabled is True for r in main_records) else "warn",
            "3x3/5x5 NWP grid patch summary feature run이 main report에 들어와야 spatial V4 비교가 가능합니다.",
        ),
        (
            "Patch improves RMSE or worst station",
            "pass" if patch_improved else "warn",
            "patch_improvement_rmse 또는 patch_improvement_worst_station 중 하나가 양수이면 PASS입니다.",
        ),
        (
            "Humidity honest RMSE ≤ 10%p",
            "pass" if (_best_rmse(main_records, target="humidity") or float("inf")) <= 10.0 else "warn",
            f"현재 {_fmt(_best_rmse(main_records, target='humidity'))}; 습도는 V3.5 개선 트랙에 유지합니다.",
        ),
        (
            "Observation-only temp ≤ 2.0°C",
            "pass" if (_best_rmse(main_records, target="temp", track="observation_only") or float("inf")) <= 2.0 else "warn",
            f"현재 {_fmt(_best_rmse(main_records, target='temp', track='observation_only'))}; anomaly/station-wise/horizon-wise 실험이 필요합니다.",
        ),
    ]
    rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{_readiness_badge(status)}</td><td>{html.escape(note)}</td></tr>"
        for name, status, note in checks
    )
    return f"""<section class="card readiness">
      <h2>V3.5 / V4 Readiness Checklist</h2>
      <p class="muted">V4 진입 전 temp MOS 운영성, humidity 목표, observation-only baseline, report 신뢰도를 동시에 확인합니다.</p>
      <table><thead><tr><th>항목</th><th>상태</th><th>근거 / 다음 작업</th></tr></thead><tbody>{rows}</tbody></table>
    </section>"""


def _v4_validation_section(all_records: list[ExperimentRecord], main_records: list[ExperimentRecord]) -> str:
    prepared = [r for r in all_records if r.future_feature_source == "prepared_forecast_csv"]
    operational = [r for r in main_records if r.operational_valid is True]
    patch = [r for r in all_records if r.uses_patch_features is True or r.patch_features_enabled is True]
    operational_temp = [
        r
        for r in main_records
        if r.target_name == "temp" and r.track == "nwp_assisted_mos" and r.operational_valid is True and r.rmse is not None
    ]
    backtest_temp = [
        r
        for r in main_records
        if r.target_name == "temp" and r.track == "nwp_assisted_mos" and r.backtest_only is True and r.rmse is not None
    ]
    best_operational = min(operational_temp, key=lambda r: r.rmse or float("inf"), default=None)
    best_backtest = min(backtest_temp, key=lambda r: r.rmse or float("inf"), default=None)
    operational_gap = None
    gap_status = None
    if best_operational and best_backtest and best_operational.rmse is not None and best_backtest.rmse is not None:
        operational_gap = float(best_operational.rmse) - float(best_backtest.rmse)
        gap_status = _gap_label(operational_gap)

    patch_rows = _patch_ablation_rows(main_records)
    metrics = {
        "V4 실험 수": sum(1 for r in all_records if str(r.version).startswith("v4") or str(r.v4_stage).startswith("v4")),
        "operational_valid=true 대표 실험 수": len(operational),
        "prepared_forecast_csv 실험 수": len(prepared),
        "patch feature 사용 실험 수": len(patch),
        "best operational temp model": best_operational.experiment_name if best_operational else "n/a",
        "best operational temp RMSE": best_operational.rmse if best_operational else None,
        "best backtest temp model": best_backtest.experiment_name if best_backtest else "n/a",
        "backtest RMSE": best_backtest.rmse if best_backtest else None,
        "operational_gap": operational_gap,
        "operational_gap 해석": gap_status,
    }
    return f"""<section class="card v4-validation">
      <h2>V4 Validation Sprint</h2>
      <p class="muted">prepared forecast CSV operational 경로, backtest vs operational gap, patch ablation 상태를 한 곳에서 확인합니다. Synthetic/diagnostic smoke는 main 성능 판정에서 제외됩니다.</p>
      <dl>{_dl(metrics)}</dl>
      <h3>Patch feature ablation</h3>
      <table><thead><tr><th>범주</th><th>Best RMSE</th><th>Worst station RMSE</th><th>Late/worst horizon RMSE</th><th>실험</th></tr></thead><tbody>{patch_rows}</tbody></table>
    </section>"""


def _is_trusted_v4_operational_forecast_record(record: ExperimentRecord) -> bool:
    """Return True only for real prepared-forecast operational temp MOS records.

    The V4-C gate is stricter than the main leaderboard: a row must prove
    prepared forecast provenance, schema validity, and non-backtest status
    directly. This prevents self-reported synthetic/generated/smoke artifacts
    from satisfying the operational expansion gate.
    """

    if record.target_name != "temp" or record.track != "nwp_assisted_mos" or record.rmse is None:
        return False
    if record.operational_valid is not True or record.backtest_only is not False:
        return False
    if record.future_feature_source != "prepared_forecast_csv":
        return False
    if record.forecast_source_schema_valid is not True:
        return False
    if record.forecast_archive_adequate is not True:
        return False
    if not record.forecast_source_path:
        return False
    if record.is_diagnostic or record.is_alias_artifact or not record.is_representative_run:
        return False
    blocked_tokens = ("synthetic", "smoke", "oracle", "diagnostic", "fixture", "generated")
    artifact_profile = str(record.artifact_profile or "").lower()
    if any(token in artifact_profile for token in blocked_tokens):
        return False
    provenance_text = " ".join(
        str(value or "")
        for value in (
            record.experiment_name,
            record.model_name,
            record.track,
            record.v4_stage,
            record.forecast_source_path,
            record.artifact_dir,
            record.leakage_risk_note,
        )
    ).lower()
    return not any(token in provenance_text for token in blocked_tokens)


def _g022_performance_section(all_records: list[ExperimentRecord], main_records: list[ExperimentRecord]) -> str:
    def is_g022(record: ExperimentRecord) -> bool:
        return (
            str(record.version).lower() == "g022"
            or str(record.v4_stage).startswith("g022")
            or str(record.experiment_name).startswith("g022_")
        )

    g022_records = [r for r in all_records if is_g022(r)]
    g022_main_records = [r for r in main_records if is_g022(r)]
    operational_temp = [
        r for r in g022_main_records
        if _is_trusted_v4_operational_forecast_record(r) and r.rmse is not None
    ]
    backtest_temp = [
        r for r in g022_main_records
        if r.target_name == "temp" and r.track == "nwp_assisted_mos" and r.backtest_only is True and r.rmse is not None
    ]
    humidity = [r for r in g022_main_records if r.target_name == "humidity" and r.rmse is not None]
    best_operational = min(operational_temp, key=lambda r: r.rmse or float("inf"), default=None)
    best_backtest = min(backtest_temp, key=lambda r: r.rmse or float("inf"), default=None)
    best_humidity = min(humidity, key=lambda r: r.rmse or float("inf"), default=None)
    calibration = [r for r in g022_main_records if r.bias_correction_accepted is True or r.bias_correction_enabled is True]
    ensemble = [r for r in g022_main_records if "ensemble" in str(r.model_type).lower() or "ensemble" in str(r.experiment_name).lower()]
    patch = [r for r in g022_main_records if r.uses_patch_features is True]
    readiness = evaluate_site_readiness(g022_main_records)
    gap = None
    if best_operational and best_backtest and best_operational.rmse is not None and best_backtest.rmse is not None:
        gap = float(best_operational.rmse) - float(best_backtest.rmse)
    metrics = {
        "G022 experiment count": len(g022_records),
        "operational temp best model": best_operational.experiment_name if best_operational else "n/a",
        "operational temp best RMSE": best_operational.rmse if best_operational else None,
        "backtest best RMSE": best_backtest.rmse if best_backtest else None,
        "operational gap vs backtest": gap,
        "patch ablation runs": len(patch),
        "calibration candidate runs": len(calibration),
        "ensemble candidate runs": len(ensemble),
        "humidity best honest model": best_humidity.experiment_name if best_humidity else "n/a",
        "humidity best honest RMSE": best_humidity.rmse if best_humidity else None,
        "site readiness status": readiness.status,
        "site readiness missing": ", ".join(readiness.missing_conditions) if readiness.missing_conditions else "none",
    }
    return f"""<section class="card g022-performance">
      <h2>G022 Model Performance Sprint</h2>
      <p class="muted">기능 추가보다 성능 개선을 추적합니다. real NWP archive가 부족하면 operational RMSE를 계산하지 않고 gate를 WARN/FAIL로 유지합니다.</p>
      <dl>{_dl(metrics)}</dl>
    </section>"""

def _v4_c_gate_section(main_records: list[ExperimentRecord]) -> str:
    operational_temp = [r for r in main_records if _is_trusted_v4_operational_forecast_record(r)]
    best_operational = min(operational_temp, key=lambda r: r.rmse or float("inf"), default=None)
    best_operational_rmse = float(best_operational.rmse) if best_operational and best_operational.rmse is not None else None
    patch_candidates = [r for r in operational_temp if r.uses_patch_features is True and r.rmse is not None]
    patch_completed = bool(patch_candidates)
    patch_improves = any(
        (r.patch_improvement_rmse is not None and r.patch_improvement_rmse > 0)
        or (r.patch_improvement_worst_station is not None and r.patch_improvement_worst_station > 0)
        or (r.patch_improvement_late_horizon is not None and r.patch_improvement_late_horizon > 0)
        for r in patch_candidates
    )
    gap_ok = any(r.operational_gap is not None and r.operational_gap <= 0.5 for r in operational_temp)
    required = [
        ("trusted prepared forecast operational temp model exists", bool(operational_temp)),
        ("prepared forecast schema valid", bool(operational_temp)),
        ("real forecast archive adequate", bool(operational_temp)),
        ("backtest_only=false and source=prepared_forecast_csv", bool(operational_temp)),
        ("operational temp RMSE ≤ 1.5°C", best_operational_rmse is not None and best_operational_rmse <= 1.5),
        ("real patch ablation completed", patch_completed),
        ("report generated", True),
    ]
    missing = [name for name, ok in required if not ok]
    if not missing:
        status = "PASS"
        badge = "pass"
        next_action = "V4-C 전국 확장으로 넘어갈 수 있습니다. 좋은 조건(≤1.2°C, gap≤0.5, patch worst-station 개선)도 함께 확인하세요."
    elif operational_temp:
        status = "WARN"
        badge = "warn"
        next_action = "부족한 V4-C 조건을 보강한 뒤 전국 확장을 시작하세요."
    else:
        status = "FAIL"
        badge = "fail"
        next_action = "real prepared forecast NWP archive로 operational_valid temp MOS 모델을 먼저 생성해야 합니다."
    good = {
        "operational temp RMSE ≤ 1.2°C": best_operational_rmse is not None and best_operational_rmse <= 1.2,
        "patch improves worst/late/overall": patch_improves,
        "operational_gap ≤ 0.5°C": gap_ok,
    }
    rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{_readiness_badge('pass' if ok else 'fail')}</td></tr>"
        for name, ok in required
    )
    good_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{_readiness_badge('pass' if ok else 'warn')}</td></tr>"
        for name, ok in good.items()
    )
    summary = {
        "V4-C gate status": status,
        "best operational temp model": best_operational.experiment_name if best_operational else "n/a",
        "best operational temp RMSE": best_operational_rmse,
        "missing required conditions": ", ".join(missing) if missing else "none",
        "next action": next_action,
    }
    return f"""<section class="card v4-c-gate">
      <h2>V4-C Gate</h2>
      <p>{_readiness_badge(badge)} <b>{html.escape(status)}</b></p>
      <dl>{_dl(summary)}</dl>
      <h3>필수 조건</h3>
      <table><thead><tr><th>조건</th><th>상태</th></tr></thead><tbody>{rows}</tbody></table>
      <h3>추가 좋은 조건</h3>
      <table><thead><tr><th>조건</th><th>상태</th></tr></thead><tbody>{good_rows}</tbody></table>
    </section>"""


def _patch_ablation_rows(records: list[ExperimentRecord]) -> str:
    candidates = [
        r
        for r in records
        if r.complete
        and r.rmse is not None
        and r.target_name == "temp"
        and r.track == "nwp_assisted_mos"
        and (r.operational_valid is True or str(r.v4_stage).startswith("v4"))
    ]
    groups: list[tuple[str, callable]] = [
        ("no patch", lambda r: r.uses_patch_features is not True),
        ("3x3 patch", lambda r: r.uses_patch_features is True and r.patch_size == 3),
        ("5x5 patch", lambda r: r.uses_patch_features is True and r.patch_size == 5),
    ]
    rows = []
    for label, predicate in groups:
        group = [r for r in candidates if predicate(r)]
        best = min(group, key=lambda r: r.rmse or float("inf"), default=None)
        rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td class='num'>{_fmt(best.rmse if best else None)}</td>"
            f"<td class='num'>{_fmt(best.worst_station_rmse if best else None)}</td>"
            f"<td class='num'>{_fmt(best.worst_horizon_rmse if best else None)}</td>"
            f"<td class='mono'>{html.escape(best.experiment_name if best else 'n/a')}</td>"
            "</tr>"
        )
    return "".join(rows)


def _gap_label(gap: float | None) -> str:
    if gap is None:
        return "n/a"
    if gap <= 0.2:
        return "매우 좋음 (≤0.2°C)"
    if gap <= 0.5:
        return "허용 가능 (≤0.5°C)"
    return "NWP source / feature mismatch 점검 필요 (>0.5°C)"


def _readiness_badge(status: str) -> str:
    if status == "pass":
        return '<span class="badge badge-ok">PASS</span>'
    if status == "warn":
        return '<span class="badge badge-warn">WARN</span>'
    return '<span class="badge badge-bad">FAIL</span>'


def _best_records(records: list[ExperimentRecord]) -> dict[str, ExperimentRecord]:
    categories = {
        "best temp observation-only": lambda r: r.target_name == "temp" and r.track == "observation_only",
        "best temp NWP-assisted": lambda r: r.target_name == "temp" and r.track == "nwp_assisted_mos",
        "best humidity observation-only": lambda r: r.target_name == "humidity" and r.track == "observation_only",
        "best humidity NWP-assisted": lambda r: r.target_name == "humidity" and r.track == "nwp_assisted_mos",
        "best operational-valid": lambda r: r.operational_valid is True,
        "best backtest-only": lambda r: r.backtest_only is True,
        "best residual MOS": lambda r: "residual" in r.model_type.lower() or "residual" in r.experiment_name.lower(),
        "best TFT": lambda r: "tft" in r.model_type.lower(),
        "best ridge": lambda r: "ridge" in r.model_type.lower(),
        "best LGBM": lambda r: "lightgbm" in r.model_type.lower() or "lgbm" in r.model_type.lower(),
    }
    best: dict[str, ExperimentRecord] = {}
    for name, predicate in categories.items():
        candidates = [r for r in records if r.rmse is not None and predicate(r)]
        if candidates:
            best[name] = min(candidates, key=lambda r: r.rmse or float("inf"))
    return best


def _best_model_section(best: dict[str, ExperimentRecord], *, image_mode: str = "full") -> str:
    rows = []
    for label in _best_category_order():
        record = best.get(label)
        if record is None:
            rows.append(f"<div class='best-card'><h3>{html.escape(_best_label(label))}</h3><p class='muted'>해당 조건의 대표 run이 없습니다.</p></div>")
            continue
        plot = embed_image_tag(Path(record.artifact_dir) / "forecast_vs_actual.png", "forecast_vs_actual", mode=image_mode) if image_mode != "none" else ""
        pros = "이 범주에서 diagnostic/oracle 및 alias를 제외한 대표 run 중 RMSE가 가장 낮습니다."
        cons = "운영 사용 전 operational/backtest 배지를 확인해야 합니다." if record.backtest_only or record.operational_valid is False else "수집기 기준 운영 차단 플래그가 없습니다."
        rows.append(f"<div class='best-card'><h3>{html.escape(_best_label(label))}</h3><p class='mono'>{html.escape(record.experiment_name)}</p><p>{_fmt(record.rmse)} RMSE · {_fmt(record.mae)} MAE · {_fmt(record.bias)} Bias</p><p><b>장점:</b> {html.escape(pros)}<br><b>주의:</b> {html.escape(cons)}</p>{plot}</div>")
    return '<section class="card"><h2>Best 모델 비교</h2><div class="grid two">' + ''.join(rows) + '</div></section>'


def _best_category_order() -> list[str]:
    return [
        "best temp observation-only",
        "best temp NWP-assisted",
        "best humidity observation-only",
        "best humidity NWP-assisted",
        "best operational-valid",
        "best backtest-only",
        "best residual MOS",
        "best TFT",
        "best ridge",
        "best LGBM",
    ]


def _best_label(label: str) -> str:
    return {
        "best temp observation-only": "기온 observation-only Best",
        "best temp NWP-assisted": "기온 NWP-assisted Best",
        "best humidity observation-only": "습도 observation-only Best",
        "best humidity NWP-assisted": "습도 NWP-assisted Best",
        "best operational-valid": "운영 가능 Best",
        "best backtest-only": "백테스트 전용 Best",
        "best residual MOS": "Residual MOS Best",
        "best TFT": "TFT Best",
        "best ridge": "Ridge Best",
        "best LGBM": "LGBM Best",
    }.get(label, label)


def _diagnostic_section(records: list[ExperimentRecord], *, image_mode: str = "full") -> str:
    if not records:
        return ""
    rows = "\n".join(_leaderboard_row(record) for record in _sort_records(records))
    details = "\n".join(_detail_section(record, image_mode=image_mode) for record in _sort_records(records))
    return f"""<section class="card diagnostic">
      <h2>Diagnostic / Oracle Checks</h2>
      <p class="warning-text">이 섹션의 실험은 pipeline sanity check, oracle decoder feature 검증, 누수 진단 목적입니다. 실제 예측 모델 성능이나 best RMSE로 해석하지 마십시오.</p>
      <div class="table-wrap compact"><table><thead><tr><th>실험</th><th>버전</th><th>타깃</th><th>트랙</th><th>모델</th><th>RMSE</th><th>MAE</th><th>Bias</th><th>최악 Horizon RMSE</th><th>최악 관측소 RMSE</th><th>목표</th><th>V4</th><th>상태</th><th>Artifact 경로</th></tr></thead><tbody>{rows}</tbody></table></div>
      <h3>Diagnostic 상세</h3>
      {details}
    </section>"""


def _alias_section(records: list[ExperimentRecord]) -> str:
    if not records:
        return ""
    rows = "\n".join(_leaderboard_row(record) for record in _sort_records(records))
    return f"""<section class="card alias">
      <h2>Alias Artifacts</h2>
      <p class="muted"><code>best</code>, <code>latest</code> alias artifact는 상세 확인용으로만 표시하며 main leaderboard, KPI, best model 산정에서는 제외됩니다.</p>
      <div class="table-wrap compact"><table><thead><tr><th>실험</th><th>버전</th><th>타깃</th><th>트랙</th><th>모델</th><th>RMSE</th><th>MAE</th><th>Bias</th><th>최악 Horizon RMSE</th><th>최악 관측소 RMSE</th><th>목표</th><th>V4</th><th>상태</th><th>Artifact 경로</th></tr></thead><tbody>{rows}</tbody></table></div>
    </section>"""


def _warnings_section(records: list[ExperimentRecord]) -> str:
    rows = []
    for record in records:
        for warning in record.warnings:
            rows.append(f"<tr><td class='mono'>{html.escape(record.experiment_name)}</td><td>{html.escape(_warning_label(warning))}</td></tr>")
        if record.error:
            rows.append(f"<tr><td class='mono'>{html.escape(record.experiment_name)}</td><td>{html.escape(_warning_label(record.error))}</td></tr>")
    if not rows:
        return '<section class="card"><h2>경고 / 데이터 품질</h2><p>경고가 없습니다.</p></section>'
    return '<section class="card"><h2>경고 / 데이터 품질</h2><table><thead><tr><th>실험</th><th>경고</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></section>'


def _warning_label(message: str) -> str:
    if message.endswith(" missing"):
        return f"{message.removesuffix(' missing')} 없음"
    if "parse failed:" in message:
        head, _, tail = message.partition(" parse failed:")
        return f"{head} 파싱 실패:{tail}"
    if message.startswith("summary_json_error:"):
        return "experiment_summary.json 파싱 실패:" + message.split(":", 1)[1]
    translations = {
        "predictions_test.csv missing": "predictions_test.csv 없음",
        "metrics file missing or empty": "metrics 파일이 없거나 비어 있음",
        "ERA5 reanalysis future features are backtest-only": "ERA5 reanalysis 미래 feature 사용: 백테스트 전용",
        "operational_valid=false": "운영 가능 플래그가 false",
        "region_class contains unknown": "region_class에 unknown 포함",
        "test sample count is very small": "테스트 sample 수가 너무 적음",
        "temperature MAPE unreliable near zero": "기온 MAPE는 0°C 근처에서 신뢰하기 어려움",
    }
    if message.startswith("bias correction disabled:"):
        return "bias correction 비활성화: " + message.split(":", 1)[1].strip()
    return translations.get(message, message)


def _summary_charts(records: list[ExperimentRecord]) -> str:
    return "".join([
        _bar_chart("실험별 RMSE", [(r.experiment_name, r.rmse) for r in sorted(records, key=lambda r: r.rmse or 999)[:30]]),
        _bar_chart("타깃/트랙별 RMSE", _group_mean(records, lambda r: f"{r.target_name}/{r.track}")),
        _scatter_chart("RMSE vs MAE", [(r.experiment_name, r.rmse, r.mae) for r in records if r.rmse is not None and r.mae is not None]),
        _bar_chart("Bias 분포", _bias_bins(records)),
        _bar_chart("모델 유형별 평균 RMSE", _group_mean(records, lambda r: r.model_type)),
        _bar_chart("버전별 Best RMSE", _group_min(records, lambda r: r.version)),
    ])


def _bar_chart(title: str, values: list[tuple[str, float | None]]) -> str:
    values = [(label, val) for label, val in values if val is not None]
    if not values:
        return f'<div class="card"><h2>{html.escape(title)}</h2><p>데이터가 없습니다.</p></div>'
    width, height = 620, max(180, 24 * len(values) + 40)
    max_val = max(float(v) for _, v in values) or 1.0
    bars = []
    for idx, (label, value) in enumerate(values):
        val = float(value)
        y = 30 + idx * 24
        bar_w = int((val / max_val) * (width - 220))
        bars.append(f'<text x="8" y="{y+14}" class="svg-label">{html.escape(label[:34])}</text><rect x="210" y="{y}" width="{bar_w}" height="16" rx="3" class="bar"/><text x="{215+bar_w}" y="{y+13}" class="svg-val">{val:.3f}</text>')
    return f'<div class="card"><h2>{html.escape(title)}</h2><svg viewBox="0 0 {width} {height}" role="img">{"".join(bars)}</svg></div>'


def _scatter_chart(title: str, points: list[tuple[str, float | None, float | None]]) -> str:
    points = [(name, float(x), float(y)) for name, x, y in points if x is not None and y is not None]
    if not points:
        return f'<div class="card"><h2>{html.escape(title)}</h2><p>데이터가 없습니다.</p></div>'
    width, height = 620, 320
    max_x = max(x for _, x, _ in points) or 1.0
    max_y = max(y for _, _, y in points) or 1.0
    dots = []
    for name, x, y in points:
        cx = 50 + (x / max_x) * (width - 90)
        cy = height - 40 - (y / max_y) * (height - 80)
        dots.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4"><title>{html.escape(name)} RMSE={x:.3f} MAE={y:.3f}</title></circle>')
    return f'<div class="card"><h2>{html.escape(title)}</h2><svg viewBox="0 0 {width} {height}" role="img"><line x1="50" y1="{height-40}" x2="{width-20}" y2="{height-40}"/><line x1="50" y1="20" x2="50" y2="{height-40}"/>{"".join(dots)}</svg></div>'


def _group_mean(records: list[ExperimentRecord], key_fn) -> list[tuple[str, float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if record.rmse is not None:
            groups[str(key_fn(record))].append(float(record.rmse))
    return sorted([(k, sum(v) / len(v)) for k, v in groups.items()], key=lambda item: item[1])


def _group_min(records: list[ExperimentRecord], key_fn) -> list[tuple[str, float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if record.rmse is not None:
            groups[str(key_fn(record))].append(float(record.rmse))
    return sorted([(k, min(v)) for k, v in groups.items()], key=lambda item: item[1])


def _bias_bins(records: list[ExperimentRecord]) -> list[tuple[str, float]]:
    bins = {"<-0.5": 0, "-0.5..0": 0, "0..0.5": 0, ">0.5": 0}
    for record in records:
        if record.bias is None:
            continue
        if record.bias < -0.5:
            bins["<-0.5"] += 1
        elif record.bias < 0:
            bins["-0.5..0"] += 1
        elif record.bias <= 0.5:
            bins["0..0.5"] += 1
        else:
            bins[">0.5"] += 1
    return [(k, float(v)) for k, v in bins.items()]


def _filter_options(records: list[ExperimentRecord]) -> dict[str, list[str]]:
    def values(attr: str) -> list[str]:
        return sorted({str(getattr(r, attr)) for r in records if getattr(r, attr) is not None})
    return {
        "versionFilter": values("version"),
        "targetFilter": values("target_name"),
        "trackFilter": values("track"),
        "modelFilter": values("model_type"),
        "sourceFilter": values("future_feature_source"),
        "opFilter": sorted({str(r.operational_valid) for r in records}),
        "stageFilter": values("v4_stage"),
        "schemaFilter": sorted({str(r.forecast_schema_valid) for r in records}),
        "patchFilter": sorted({str(r.patch_features_enabled) for r in records}),
        "goalFilter": sorted({str(r.rmse_goal_met) for r in records}),
    }


def _goal_class(record: ExperimentRecord) -> str:
    if not record.goal_eligible or record.rmse_goal is None or record.rmse is None:
        return ""
    if record.rmse <= record.rmse_goal:
        return "goal-ok"
    if record.rmse <= record.rmse_goal * 1.10:
        return "goal-warn"
    return "goal-bad"


def _best_rmse(records: list[ExperimentRecord], *, target: str | None = None, track: str | None = None) -> float | None:
    vals = [r.rmse for r in records if r.rmse is not None and (target is None or r.target_name == target) and (track is None or r.track == track)]
    return min(vals) if vals else None


def _best_operational_rmse(records: list[ExperimentRecord]) -> float | None:
    vals = [r.rmse for r in records if r.rmse is not None and r.operational_valid is True]
    return min(vals) if vals else None


def _best_backtest_rmse(records: list[ExperimentRecord]) -> float | None:
    vals = [r.rmse for r in records if r.rmse is not None and r.backtest_only is True]
    return min(vals) if vals else None


def _fmt(value: object) -> str:
    try:
        if value is None:
            return "n/a"
        if isinstance(value, bool):
            return str(value)
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _attr(value: object) -> str:
    return html.escape(str(value), quote=True)


def _css() -> str:
    return """
:root { --bg:#f7f8fa; --card:#fff; --text:#172033; --muted:#667085; --ok:#17803d; --warn:#b7791f; --bad:#b42318; --blue:#175cd3; --border:#d8dde6; }
* { box-sizing:border-box; } body { margin:0; font-family:Inter, Segoe UI, Arial, sans-serif; color:var(--text); background:var(--bg); }
.hero { display:flex; justify-content:space-between; gap:1rem; align-items:center; padding:28px 36px; background:#0f172a; color:white; }
.hero h1 { margin:0 0 6px; } .hero p { margin:0; color:#cbd5e1; } code, .mono { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; } .small { font-size:12px; }
.hero-best { display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; font-size:12px; } .hero-best span { background:rgba(255,255,255,.10); padding:6px 8px; border-radius:10px; }
.hero-stats { display:flex; gap:12px; flex-wrap:wrap; } .hero-stats span { background:rgba(255,255,255,.12); padding:10px 12px; border-radius:12px; }
.kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:14px; padding:22px 28px 0; }
.kpi, .card, .best-card { background:var(--card); border:1px solid var(--border); border-radius:16px; box-shadow:0 1px 2px rgba(16,24,40,.06); }
.kpi { padding:16px; } .kpi span { color:var(--muted); display:block; font-size:13px; } .kpi b { font-size:24px; }
.banner { margin:18px 28px 0; padding:14px 18px; border-radius:14px; border:1px solid var(--border); display:flex; gap:10px; align-items:center; }
.banner-warn { color:#7a2e0e; background:#fffaeb; border-color:#fedf89; }
.muted { color:var(--muted); } .warning-text { color:#7a2e0e; background:#fffaeb; border:1px solid #fedf89; padding:10px 12px; border-radius:10px; }
.card { margin:22px 28px; padding:20px; } .grid { display:grid; gap:18px; } .grid.two { grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); }
.filters { display:flex; flex-wrap:wrap; gap:10px; align-items:end; margin:12px 0 16px; } .filters label { font-size:12px; color:var(--muted); display:flex; flex-direction:column; gap:4px; }
input, select { border:1px solid var(--border); border-radius:10px; padding:8px 10px; background:white; min-width:130px; }
.table-wrap { max-height:640px; overflow:auto; border:1px solid var(--border); border-radius:12px; }
table { width:100%; border-collapse:collapse; font-size:13px; } th, td { padding:9px 10px; border-bottom:1px solid #edf0f5; vertical-align:top; } th { position:sticky; top:0; background:#f1f5f9; text-align:left; z-index:1; } th[onclick] { cursor:pointer; } th[onclick]::after { content:' ↕'; color:#98a2b3; font-weight:400; } .num { text-align:right; font-variant-numeric:tabular-nums; }
.goal-ok { color:var(--ok); font-weight:700; } .goal-warn { color:var(--warn); font-weight:700; } .goal-bad { color:var(--bad); font-weight:700; }
.badge { display:inline-block; padding:3px 7px; border-radius:999px; margin:1px; font-size:11px; font-weight:700; } .badge-ok { color:#05603a; background:#d1fadf; } .badge-bad { color:#912018; background:#fee4e2; } .badge-warn { color:#93370d; background:#fef0c7; } .badge-muted { color:#475467; background:#eaecf0; } .badge-blue { color:#1849a9; background:#d1e9ff; } .badge-purple { color:#5925dc; background:#ebe9fe; }
.detail { border:1px solid var(--border); border-radius:12px; margin:10px 0; background:white; } .detail summary { cursor:pointer; padding:14px 16px; display:flex; justify-content:space-between; gap:10px; } .detail-body { padding:0 16px 16px; }
dl { display:grid; grid-template-columns:170px 1fr; gap:7px 12px; } dt { color:var(--muted); } dd { margin:0; }
.plots { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:14px; } .plot-card { border:1px solid #edf0f5; border-radius:12px; padding:10px; background:#fbfcfe; } .plot-card h4 { margin:0 0 8px; }
.plot-card img { border-radius:8px; border:1px solid #e5e7eb; } .thumb img { max-height:220px; object-fit:contain; background:white; } .plot-placeholder { color:var(--muted); background:#f2f4f7; border:1px dashed #cbd5e1; padding:28px; border-radius:8px; text-align:center; }
svg { width:100%; height:auto; } .bar { fill:#6096f2; } .svg-label { font-size:11px; fill:#344054; } .svg-val { font-size:11px; fill:#344054; } circle { fill:#175cd3; opacity:.75; } line { stroke:#98a2b3; }
.warnings { color:#7a2e0e; } .best-card { padding:14px; }
@media (max-width:720px){ .hero{display:block}.card{margin:16px 12px}.kpis{padding:16px 12px 0}.detail summary{display:block} }
"""


def _js() -> str:
    return """
function initFilters(){ for (const [id, values] of Object.entries(FILTER_OPTIONS)){ const el=document.getElementById(id); if(!el) continue; for(const value of values){ const opt=document.createElement('option'); opt.value=value; opt.textContent=value; el.appendChild(opt);} } }
function filterTable(){ const search=(document.getElementById('searchBox').value||'').toLowerCase(); const filters=[['versionFilter','version'],['targetFilter','target'],['trackFilter','track'],['modelFilter','model'],['sourceFilter','source'],['opFilter','op'],['stageFilter','stage'],['schemaFilter','schema'],['patchFilter','patch'],['goalFilter','goal']]; for(const row of document.querySelectorAll('#leaderboard tbody tr')){ let show=(row.dataset.name||'').includes(search); for(const [id,key] of filters){ const val=document.getElementById(id).value; if(val && row.dataset[key]!==val) show=false; } row.style.display=show?'':'none'; } }
let SORT_STATE={index:null,asc:true};
function sortTable(index){ const tbody=document.querySelector('#leaderboard tbody'); const rows=Array.from(tbody.querySelectorAll('tr')); const asc=SORT_STATE.index===index ? !SORT_STATE.asc : true; SORT_STATE={index,asc}; rows.sort((a,b)=>{ const av=a.children[index].innerText.trim(); const bv=b.children[index].innerText.trim(); const an=parseFloat(av); const bn=parseFloat(bv); let cmp; if(!Number.isNaN(an) && !Number.isNaN(bn)){ cmp=an-bn; } else { cmp=av.localeCompare(bv); } return asc?cmp:-cmp; }); for(const row of rows){ tbody.appendChild(row); } filterTable(); }
initFilters();
"""
