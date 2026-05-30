from __future__ import annotations

import subprocess
from pathlib import Path


WEB_FILES = [Path("web/index.html"), Path("web/app.js"), Path("web/styles.css")]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _run_node(script: str) -> str:
    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout


def _dom_smoke_script(body: str) -> str:
    return f"""
const assert = require("assert");
const app = require("./web/app.js");

class Element {{
  constructor(tagName) {{
    this.tagName = tagName;
    this.children = [];
    this.className = "";
    this.dataset = {{}};
    this.style = {{
      values: {{}},
      setProperty(name, value) {{ this.values[name] = value; }},
    }};
    this.attributes = {{}};
    this.disabled = false;
    this.hidden = false;
    this.title = "";
    this.type = "";
    this.colSpan = 1;
    this._textContent = "";
    this.listeners = {{}};
  }}
  appendChild(child) {{
    this.children.push(child);
    return child;
  }}
  replaceChildren(...children) {{
    this.children = children;
  }}
  setAttribute(name, value) {{
    this.attributes[name] = String(value);
  }}
  addEventListener(name, listener) {{
    this.listeners[name] = listener;
  }}
  get textContent() {{
    return this._textContent + this.children.map((child) => child.textContent || "").join("");
  }}
  set textContent(value) {{
    this._textContent = String(value);
  }}
  get classList() {{
    const element = this;
    return {{
      add(name) {{
        if (!element.className.split(/\\s+/).includes(name)) {{
          element.className = `${{element.className}} ${{name}}`.trim();
        }}
      }},
      remove(name) {{
        element.className = element.className.split(/\\s+/).filter((part) => part && part !== name).join(" ");
      }},
      contains(name) {{
        return element.className.split(/\\s+/).includes(name);
      }},
    }};
  }}
}}

const elements = {{
  dateSelector: new Element("div"),
  variableSelector: new Element("div"),
  markerLayer: new Element("div"),
  mapEmptyState: new Element("div"),
  forecastTable: new Element("table"),
}};
elements.forecastTable.tbody = new Element("tbody");
global.document = {{
  createElement(tagName) {{ return new Element(tagName); }},
  getElementById(id) {{ return elements[id] || null; }},
  querySelector(selector) {{
    if (selector === "#forecastTable tbody") {{
      return elements.forecastTable.tbody;
    }}
    return null;
  }},
}};

function resetState() {{
  app.state.latest = {{ run: {{}} }};
  app.state.modelStatus = {{}};
  app.state.allPoints = [];
  app.state.selectedStationId = "";
  app.state.selectedHorizon = 1;
  app.state.selectedVariable = "temperature";
  app.state.stationCache = new Map();
  app.state.selectedStationPoints = [];
  app.state.selectedDaily = [];
  app.state.stationDataWarning = "";
  Object.values(elements).forEach((element) => {{
    if (element.replaceChildren) element.replaceChildren();
    element.textContent = "";
    element.hidden = false;
  }});
  elements.forecastTable.tbody.replaceChildren();
}}

{body}
"""


def test_g032_forecast_map_shell_replaces_mvp() -> None:
    index_html = _read(Path("web/index.html"))
    app_js = _read(Path("web/app.js"))
    map_asset = Path("web/assets/korea_simplified.svg")

    for token in [
        "Gisang Forecast",
        "AI 기반 전국 미래 날씨 예측 지도",
        "forecastMap",
        "leafletMap",
        "OpenStreetMap 기반 대한민국 지도",
        "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css",
        "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js",
        "korea_simplified.svg",
        "mapEmptyState",
        "markerLayer",
        "dateSelector",
        "variableSelector",
        "selectedRegionPanel",
        "forecastTable",
        "modelPerformanceSummary",
        "stationWarning",
        "caveatBanner",
        "본 사이트는 AI 기반 기상 예측 모델의 미래 예측값을 지도 형태로 제공하는 연구/베타 서비스입니다.",
        "습도 예측은 beta입니다.",
        "날씨상태는 일부 rule-based 추정을 포함합니다.",
        "강수확률은 AI 모델 검증 전에는 KMA/NWP 직접값 또는 준비 중으로 표시됩니다.",
    ]:
        assert token in index_html

    for token in ["DateSelector", "VariableSelector", "ForecastMap", "RegionForecastPanel", "ForecastTable", "ModelPerformanceSummary", "SourceBadge", "ConfidenceBadge", "CaveatBanner"]:
        assert token in app_js

    assert "MVP" not in index_html
    assert "korea-shape" not in index_html
    assert map_asset.exists()
    assert "대한민국 단순 지도" in _read(map_asset)
    assert "tile.openstreetmap.org" in app_js
    assert "OpenStreetMap" in app_js
    assert "leafletMarkerHtml" in app_js


def test_g032_date_and_variable_selectors_are_present() -> None:
    app_js = _read(Path("web/app.js"))
    index_html = _read(Path("web/index.html"))

    for token in ["내일 D+1", "D+2", "D+3", "D+5"]:
        assert token in app_js

    for token in ["기온", "습도 beta", "강수확률", "바람", "구름", "날씨상태"]:
        assert token in app_js or token in index_html

    for endpoint in [
        "/api/forecast/latest",
        "/api/forecast/station/${encodedStationId}",
        "/api/forecast/station/${encodedStationId}/hourly",
        "/api/forecast/station/${encodedStationId}/daily",
        "/api/model/status",
    ]:
        assert endpoint in app_js


def test_g032_source_badge_static_guardrails() -> None:
    app_js = _read(Path("web/app.js"))

    for field in [
        "temperature_source",
        "temperature_status",
        "temperature_confidence",
        "humidity_source",
        "humidity_status",
        "humidity_confidence",
        "precip_probability_source",
        "precip_probability_status",
        "precip_probability_confidence",
        "wind_source",
        "wind_status",
        "wind_confidence",
        "cloud_source",
        "cloud_status",
        "cloud_confidence",
        "weather_code_source",
        "weather_code_status",
        "weather_code_confidence",
    ]:
        assert field in app_js

    for label in ["AI 예측", "AI beta", "NWP direct", "KMA direct", "규칙 기반 beta", "rule-based beta", "준비 중", "unavailable"]:
        assert label in app_js

    assert "point.temperature_source ||" not in app_js
    assert "point.humidity_source ||" not in app_js
    assert "ai_mos_model_beta" in app_js
    assert "String(status) === \"beta\"" not in app_js
    assert "return \"AI beta\";" not in app_js
    assert "sourceStatusConfidence" in app_js


def test_g032_forbidden_product_ui_and_innerhtml_absent() -> None:
    combined = "\n".join(_read(path) for path in WEB_FILES)
    app_js = _read(Path("web/app.js"))

    assert ".innerHTML" not in app_js
    assert "precip_mm" not in app_js

    for forbidden in [
        "현재 날씨",
        "로그인",
        "회원가입",
        "즐겨찾기",
        "알림",
        "특보",
        "생활지수",
        "admin",
        "leaderboard",
    ]:
        assert forbidden not in combined


def test_g032_fixture_helpers_preserve_source_honesty() -> None:
    script = r'''
const assert = require("assert");
const app = require("./web/app.js");

const unavailablePrecip = {
  station_id: "108",
  valid_time: "2026-01-02T12:00:00Z",
  precip_probability: 87,
  precip_probability_source: "unavailable",
  precip_probability_status: "unavailable",
  precip_probability_confidence: "unavailable",
};
assert.strictEqual(app.valueForVariable(unavailablePrecip, "precip_probability"), "준비 중");
assert.strictEqual(app.isUnavailable(unavailablePrecip, "precip_probability"), true);

const unavailableWeatherCode = {
  weather_code: "sunny",
  weather_label_ko: "맑음",
  weather_code_source: "unavailable",
  weather_code_status: "unavailable",
  weather_code_confidence: "unavailable",
};
assert.strictEqual(app.valueForVariable(unavailableWeatherCode, "weather_code"), "준비 중");

const ruleBased = {
  weather_code: "cloudy",
  weather_label_ko: "흐림",
  weather_code_source: "rule_based_beta",
  weather_code_status: "beta",
  weather_code_confidence: "low",
};
const ruleMeta = app.sourceMetadata(ruleBased, "weather_code");
assert.ok(ruleMeta.label.includes("규칙 기반"));
assert.ok(!ruleMeta.label.includes("AI"));

const missingTempHumiditySource = {
  temperature_c: 21.2,
  humidity_percent: 62,
};
assert.strictEqual(app.sourceMetadata(missingTempHumiditySource, "temperature").label, "준비 중");
assert.strictEqual(app.sourceMetadata(missingTempHumiditySource, "humidity").label, "준비 중");
assert.strictEqual(app.valueForVariable(missingTempHumiditySource, "temperature"), "21.2°C");
assert.strictEqual(app.valueForVariable(missingTempHumiditySource, "humidity"), "62%");

const unavailableTemperature = {
  temperature_c: 30,
  temp_max_c: 33,
  temp_min_c: 21,
  temperature_source: "unavailable",
  temperature_status: "unavailable",
};
assert.strictEqual(app.valueForVariable(unavailableTemperature, "temperature"), "준비 중");
assert.strictEqual(app.colorForValue(unavailableTemperature, "temperature"), "#98a2b3");
assert.strictEqual(app.highLowText(unavailableTemperature), "준비 중");
assert.strictEqual(app.isExplicitUnavailable(unavailableTemperature, "temperature"), true);

const unavailableHumidity = {
  humidity_percent: 88,
  humidity_source: "unavailable",
  humidity_status: "unavailable",
};
assert.strictEqual(app.valueForVariable(unavailableHumidity, "humidity"), "준비 중");
assert.strictEqual(app.colorForValue(unavailableHumidity, "humidity"), "#98a2b3");

const pointLevelDirect = {
  precip_probability: 44,
  precip_probability_source: "kma_forecast_direct",
  precip_probability_status: "direct",
  precip_probability_confidence: "low",
};
assert.strictEqual(app.sourceMetadata(pointLevelDirect, "precip_probability").label, "KMA direct");
assert.strictEqual(app.valueForVariable(pointLevelDirect, "precip_probability"), "44%");
const markerHtml = app.leafletMarkerHtml({ station_name: "<서울&테스트>" }, "<24°C>", "#2563eb");
assert.ok(markerHtml.includes("&lt;서울&amp;테스트&gt;"));
assert.ok(markerHtml.includes("&lt;24°C&gt;"));
assert.ok(!markerHtml.includes("<서울&테스트>"));

console.log("source honesty ok");
'''
    assert "source honesty ok" in _run_node(script)


def test_g032_fixture_horizon_and_variable_helpers() -> None:
    script = r'''
const assert = require("assert");
const app = require("./web/app.js");
const run = { created_at: "2026-01-01T00:00:00Z" };
const points = [
  { station_id: "108", station_name: "서울", valid_time: "2026-01-02T12:00:00Z", temperature_c: 5, humidity_percent: 51, precip_probability: 10, precip_probability_source: "kma_direct", wind_speed_ms: 2, wind_source: "gfs_direct", cloud_cover_percent: 20, cloud_source: "gfs_direct", weather_label_ko: "맑음", weather_code_source: "rule_based_beta" },
  { station_id: "108", station_name: "서울", valid_time: "2026-01-03T12:00:00Z", temperature_c: 9, humidity_percent: 63, precip_probability: 70, precip_probability_source: "kma_direct", wind_speed_ms: 6, wind_source: "gfs_direct", cloud_cover_percent: 80, cloud_source: "gfs_direct", weather_label_ko: "비", weather_code_source: "rule_based_beta" },
  { station_id: "108", station_name: "서울", valid_time: "2026-01-04T12:00:00Z", temperature_c: 8, humidity_percent: 61, precip_probability: 30, precip_probability_source: "kma_direct", wind_speed_ms: 4, wind_source: "gfs_direct", cloud_cover_percent: 60, cloud_source: "gfs_direct", weather_label_ko: "구름많음", weather_code_source: "rule_based_beta" },
  { station_id: "108", station_name: "서울", valid_time: "2026-01-05T12:00:00Z", temperature_c: 7, humidity_percent: 60, precip_probability: 20, precip_probability_source: "kma_direct", wind_speed_ms: 3, wind_source: "gfs_direct", cloud_cover_percent: 50, cloud_source: "gfs_direct", weather_label_ko: "흐림", weather_code_source: "rule_based_beta" },
  { station_id: "108", station_name: "서울", valid_time: "2026-01-06T12:00:00Z", temperature_c: 6, humidity_percent: 58, precip_probability: 15, precip_probability_source: "kma_direct", wind_speed_ms: 2, wind_source: "gfs_direct", cloud_cover_percent: 40, cloud_source: "gfs_direct", weather_label_ko: "맑음", weather_code_source: "rule_based_beta" },
];
const d1 = app.choosePointForHorizon(points, 1, run);
const d2 = app.choosePointForHorizon(points, 2, run);
assert.strictEqual(d1.temperature_c, 5);
assert.strictEqual(d2.temperature_c, 9);
assert.strictEqual(app.valueForVariable(d2, "temperature"), "9°C");
assert.strictEqual(app.valueForVariable(d2, "humidity"), "63%");
assert.strictEqual(app.valueForVariable(d2, "precip_probability"), "70%");
assert.ok(app.valueForVariable(d2, "wind").includes("6m/s"));
assert.strictEqual(app.valueForVariable(d2, "cloud"), "80%");
assert.ok(app.valueForVariable(d2, "weather_code").includes("비"));
assert.strictEqual(app.selectedPointsByStation(points, 2, run)[0].point.temperature_c, 9);
assert.deepStrictEqual(app.deriveDailyRows(points).map((row) => row.day), [1, 2, 3, 5]);
console.log("horizon variable ok");
'''
    assert "horizon variable ok" in _run_node(script)


def test_g032_temperature_map_uses_same_kst_hour_and_daily_high_low() -> None:
    script = r'''
const assert = require("assert");
const app = require("./web/app.js");
const run = { map_reference_hour_kst: 14, lock_map_reference_hour_kst: true };
const points = [
  {
    station_id: "108",
    station_name: "서울",
    forecast_day: "D+1",
    valid_date: "2026-05-31",
    valid_time: "2026-05-31T00:00:00Z",
    temperature_c: 24.8,
    temp_max_c: 30.5,
    temp_min_c: 24.8,
    precip_probability: 0,
    precip_probability_source: "rule_based_beta",
    precip_probability_status: "beta",
  },
  {
    station_id: "108",
    station_name: "서울",
    forecast_day: "D+1",
    valid_date: "2026-05-31",
    valid_time: "2026-05-31T06:00:00Z",
    temperature_c: 30.5,
    temp_max_c: 30.5,
    temp_min_c: 24.8,
    precip_probability: 22,
    precip_probability_source: "rule_based_beta",
    precip_probability_status: "beta",
  },
];

const selected = app.choosePointForHorizon(points, 1, run);
assert.strictEqual(selected.temperature_c, 30.5);
assert.strictEqual(app.dateKeyForPoint(selected), "2026-05-31");
assert.strictEqual(app.highLowText(selected), "30.5°C / 24.8°C");
assert.strictEqual(app.valueForVariable(selected, "precip_probability"), "22%");
assert.strictEqual(app.sourceMetadata(selected, "precip_probability").label, "규칙 기반 beta");
assert.ok(app.displayHourDistance(points[1], run) < app.displayHourDistance(points[0], run));
console.log("same-hour high-low ok");
'''
    assert "same-hour high-low ok" in _run_node(script)


def test_g032_map_density_uses_major_cities_until_zoomed_in() -> None:
    script = r'''
const assert = require("assert");
const app = require("./web/app.js");
const entries = app.EMPTY_STATION_ANCHORS.map((point) => ({ id: point.station_id, point }));
const lowZoom = app.visibleEntriesForZoom(entries, 7, "");
const middleZoom = app.visibleEntriesForZoom(entries, 8, "");
const highZoom = app.visibleEntriesForZoom(entries, 9, "");

assert.ok(entries.length >= 70);
assert.ok(lowZoom.length < middleZoom.length);
assert.ok(middleZoom.length < highZoom.length);
assert.strictEqual(highZoom.length, entries.length);
assert.ok(lowZoom.some((entry) => entry.point.station_name === "서울"));
assert.ok(lowZoom.some((entry) => entry.point.station_name === "부산"));
assert.ok(lowZoom.some((entry) => entry.point.station_name === "제주"));
assert.ok(!lowZoom.some((entry) => entry.point.station_name === "거제"));
assert.ok(highZoom.some((entry) => entry.point.station_name === "거제"));

const withSelectedSecondary = app.visibleEntriesForZoom(entries, 7, "294");
assert.ok(withSelectedSecondary.some((entry) => entry.id === "294"));
console.log("map density ok");
'''
    assert "map density ok" in _run_node(script)


def test_g032_latest_payload_and_disabled_horizon_helpers() -> None:
    script = r'''
const assert = require("assert");
const app = require("./web/app.js");

const payload = {
  forecast_run: { forecast_init_time: "2026-06-01T00:00:00Z" },
  forecast_points: [
    {
      station_id: "108",
      station_name: "서울특별시",
      forecast_day: "D+1",
      valid_date: "2026-06-02",
      temperature_c: 24.1,
      temperature_source: "ai_mos_model",
      humidity_percent: 58,
      humidity_source: "ai_mos_model_beta",
      precip_probability: null,
      precip_probability_source: "unavailable",
      wind_speed_ms: null,
      wind_source: "unavailable",
      cloud_cover_percent: null,
      cloud_source: "unavailable",
      weather_code: "unknown",
      weather_code_source: "rule_based_beta",
      confidence: "medium",
    },
  ],
};
const latest = app.normalizeLatestPayload(payload);
assert.strictEqual(latest.run.forecast_init_time, "2026-06-01T00:00:00Z");
assert.strictEqual(latest.points.length, 1);
assert.strictEqual(app.horizonDayForPoint(latest.points[0], latest.run, latest.points), 1);
assert.deepStrictEqual([...app.availableHorizonDays(latest.points, latest.run)].sort(), [1]);
assert.strictEqual(app.hasHorizonData(latest.points, 1, latest.run), true);
assert.strictEqual(app.hasHorizonData(latest.points, 2, latest.run), false);
assert.strictEqual(app.choosePointForHorizon(latest.points, 2, latest.run), null);
assert.strictEqual(app.valueForVariable(latest.points[0], "precip_probability"), "준비 중");
assert.strictEqual(app.valueForVariable(latest.points[0], "wind"), "준비 중");
assert.strictEqual(app.valueForVariable(latest.points[0], "cloud"), "준비 중");
console.log("payload horizon ok");
'''
    assert "payload horizon ok" in _run_node(script)


def test_g032_default_station_readiness_and_model_status_helpers() -> None:
    script = r'''
const assert = require("assert");
const app = require("./web/app.js");

const run = { forecast_init_time: "2026-06-01T00:00:00Z" };
const points = [
  {
    station_id: "159",
    station_name: "부산광역시",
    forecast_day: "D+1",
    temperature_c: 21,
    temperature_source: "ai_mos_model",
    precip_probability_source: "unavailable",
  },
  {
    station_id: "108",
    station_name: "서울특별시",
    forecast_day: "D+1",
    temperature_c: 24,
    temperature_source: "ai_mos_model",
    humidity_percent: 58,
    humidity_source: "ai_mos_model_beta",
    precip_probability: null,
    precip_probability_source: "unavailable",
    wind_speed_ms: 2.4,
    wind_source: "gfs_direct",
    weather_code: "cloudy",
    weather_code_source: "rule_based_beta",
  },
];

assert.strictEqual(app.chooseDefaultStationId(points, run, 1), "108");
assert.strictEqual(app.variableReadiness("temperature", points, 1, run).label, "AI 예측");
assert.strictEqual(app.variableReadiness("humidity", points, 1, run).label, "AI beta");
assert.strictEqual(app.variableReadiness("precip_probability", points, 1, run).label, "준비 중");
assert.strictEqual(app.variableReadiness("wind", points, 1, run).label, "NWP direct");
assert.strictEqual(app.variableReadiness("weather_code", points, 1, run).label, "규칙 기반 beta");
assert.strictEqual(app.hasModelPerformance({}), false);
assert.strictEqual(app.hasModelPerformance({ warnings: ["모델 성능 요약 파일이 없습니다"] }), false);
assert.strictEqual(app.hasModelPerformance({ temperature_rmse: 1.69, humidity_rmse: 10.4, benchmark_reliability: "strong", site_readiness: "WARN" }), true);
console.log("default readiness ok");
'''
    assert "default readiness ok" in _run_node(script)


def test_g032_static_guards_for_map_and_resilient_fetch() -> None:
    index_html = _read(Path("web/index.html"))
    styles = _read(Path("web/styles.css"))
    app_js = _read(Path("web/app.js"))

    assert "korea-map-asset" in index_html
    assert "korea-map-asset" in styles
    assert "EMPTY_STATION_ANCHORS" in app_js
    for station_name in ["부산", "광주", "목포", "여수", "제주", "서귀포", "거제", "울릉도", "백령도"]:
        assert station_name in app_js
    assert "데이터 없음" in app_js
    assert "예측 데이터가 없습니다. forecast exporter를 실행해 latest forecast_points.json을 생성하세요." in app_js
    assert "모델 성능 정보 없음" in app_js
    assert "return await fetchJson(\"/api/forecast/latest\")" in app_js
    assert "return await fetchJson(\"/api/model/status\")" in app_js
    assert "data/forecasts/latest/forecast_points.json" in app_js
    assert "data/forecasts/latest/forecast_run.json" in app_js
    assert Path("web/data/forecasts/latest/forecast_points.json").exists()
    assert Path("web/data/forecasts/latest/forecast_run.json").exists()
    assert Path("web/data/forecasts/latest/forecast_data.js").exists()
    assert "/data/forecasts/latest/forecast_points.json" in app_js
    assert "/data/artifacts/g030_production_freeze_final/production_model_manifest.json" in app_js
    assert "const latest = await fetchLatestForecast()" in app_js
    assert "const modelStatus = await fetchModelStatus()" in app_js
    assert "const [latest, modelStatus]" not in app_js


def test_g032_embedded_static_data_fallback_is_available() -> None:
    script = r'''
const assert = require("assert");
const app = require("./web/app.js");
global.window = {
  GISANG_FORECAST_DATA: { run: { forecast_run_id: "embedded" }, points: [{ station_id: "108" }] },
  GISANG_MODEL_STATUS_DATA: { temperature_rmse: 1.69, site_readiness: "WARN" },
};

assert.strictEqual(app.embeddedForecastData().run.forecast_run_id, "embedded");
assert.strictEqual(app.embeddedForecastData().points.length, 1);
assert.strictEqual(app.embeddedModelStatusData().site_readiness, "WARN");
console.log("embedded fallback ok");
'''
    assert "embedded fallback ok" in _run_node(script)


def test_g032_dom_smoke_renders_disabled_tabs_markers_and_unavailable_cells() -> None:
    script = _dom_smoke_script(
        r'''
resetState();
const d1Point = {
  station_id: "108",
  station_name: "서울특별시",
  lat: 37.5714,
  lon: 126.9658,
  forecast_day: "D+1",
  valid_date: "2026-06-02",
  temperature_c: 24.1,
  temp_max_c: 27,
  temp_min_c: 18,
  temperature_source: "ai_mos_model",
  humidity_percent: 58,
  humidity_source: "ai_mos_model_beta",
  precip_probability_source: "unavailable",
  wind_source: "unavailable",
  cloud_source: "unavailable",
  weather_code: "cloudy",
  weather_code_source: "rule_based_beta",
};
app.state.latest = { run: { forecast_init_time: "2026-06-01T00:00:00Z" } };
app.state.allPoints = [d1Point];
app.state.selectedStationId = "108";
app.state.selectedStationPoints = [d1Point];
app.DateSelector();
assert.strictEqual(elements.dateSelector.children.length, 4);
assert.strictEqual(elements.dateSelector.children[0].disabled, false);
assert.strictEqual(elements.dateSelector.children[1].disabled, true);
assert.ok(elements.dateSelector.children[1].textContent.includes("데이터 없음"));

app.ForecastMap();
assert.strictEqual(elements.mapEmptyState.hidden, true);
assert.strictEqual(elements.markerLayer.children.length, 1);
assert.ok(elements.markerLayer.children[0].textContent.includes("서울특별시"));
assert.ok(elements.markerLayer.children[0].textContent.includes("24.1°C"));

resetState();
app.ForecastMap();
assert.strictEqual(elements.mapEmptyState.hidden, false);
assert.ok(elements.mapEmptyState.textContent.includes("예측 데이터가 없습니다"));
assert.ok(app.EMPTY_STATION_ANCHORS.length >= 70);
assert.ok(app.EMPTY_STATION_ANCHORS.some((point) => point.station_name === "제주"));
assert.ok(app.EMPTY_STATION_ANCHORS.some((point) => point.station_name === "부산"));
assert.ok(app.EMPTY_STATION_ANCHORS.some((point) => point.station_name === "목포"));
assert.ok(elements.markerLayer.children.length < app.EMPTY_STATION_ANCHORS.length);
assert.ok(elements.markerLayer.children.length <= 20);
assert.ok(elements.markerLayer.children[0].disabled);
assert.ok(elements.markerLayer.children[0].textContent.includes("데이터 없음"));

resetState();
const unavailableTempPoint = {
  station_id: "108",
  station_name: "서울특별시",
  forecast_day: "D+1",
  valid_date: "2026-06-02",
  temperature_c: 24.1,
  temp_max_c: 27,
  temp_min_c: 18,
  temperature_source: "unavailable",
  temperature_status: "unavailable",
  humidity_percent: 58,
  humidity_source: "ai_mos_model_beta",
  precip_probability_source: "unavailable",
  wind_source: "unavailable",
  cloud_source: "unavailable",
  weather_code_source: "rule_based_beta",
};
app.state.latest = { run: { forecast_init_time: "2026-06-01T00:00:00Z" } };
app.state.allPoints = [unavailableTempPoint];
app.state.selectedStationId = "108";
app.state.selectedStationPoints = [unavailableTempPoint];
app.ForecastTable();
const firstRow = elements.forecastTable.tbody.children[0];
assert.strictEqual(firstRow.children[2].textContent, "준비 중");
assert.strictEqual(firstRow.children[3].textContent, "준비 중");
console.log("dom smoke ok");
'''
    )
    assert "dom smoke ok" in _run_node(script)
