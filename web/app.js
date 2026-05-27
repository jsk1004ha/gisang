const API_BASE = window.API_BASE || "";

async function fetchJson(path) {
  const response = await fetch(API_BASE + path);
  if (!response.ok) {
    throw new Error(path);
  }
  return response.json();
}

function appendTextElement(parent, tagName, text, className = "") {
  const element = document.createElement(tagName);
  if (className) {
    element.className = className;
  }
  element.textContent = text;
  parent.appendChild(element);
  return element;
}

function barChart(element, points, key, className = "") {
  element.replaceChildren();
  const values = points.map((point) => Number(point[key])).filter(Number.isFinite);
  const min = Math.min(...values, 0);
  const max = Math.max(...values, 1);

  points.slice(0, 24).forEach((point) => {
    const value = Number(point[key]);
    const bar = document.createElement("div");
    bar.className = `bar ${className}`;
    bar.title = `${point.valid_time}: ${value}`;
    bar.style.height = `${20 + 80 * ((value - min) / (max - min || 1))}px`;
    element.appendChild(bar);
  });
}

function renderForecastCard(point) {
  const card = document.getElementById("forecastCard");
  card.replaceChildren();
  appendTextElement(card, "h2", point.station_name || point.station_id || "관측소");
  appendTextElement(card, "p", `${point.weather_icon || "❔"} ${point.weather_label_ko || "알 수 없음"}`, "weather");
  appendTextElement(card, "p", `기온 ${point.temperature_c ?? "-"}°C · 습도 ${point.humidity_percent ?? "-"}%`);
  appendTextElement(card, "p", `신뢰도: ${point.confidence || "unknown"}`);
}

function renderHourlyTable(points) {
  const tbody = document.querySelector("#hourlyTable tbody");
  tbody.replaceChildren();
  points.slice(0, 24).forEach((point) => {
    const row = document.createElement("tr");
    [
      point.valid_time || "",
      point.temperature_c ?? "-",
      point.humidity_percent ?? "-",
      `${point.weather_icon || ""} ${point.weather_label_ko || ""}`.trim(),
      point.confidence || "",
    ].forEach((value) => {
      appendTextElement(row, "td", String(value));
    });
    tbody.appendChild(row);
  });
}

function render(points) {
  const first = points[0] || {};
  renderForecastCard(first);
  barChart(document.getElementById("tempChart"), points, "temperature_c");
  barChart(document.getElementById("humidityChart"), points, "humidity_percent", "humidity");
  renderHourlyTable(points);
}

async function main() {
  try {
    const latest = await fetchJson("/api/forecast/latest");
    const stationPoints = [...new Map(latest.points.map((point) => [point.station_id, point])).values()];
    const select = document.getElementById("stationSelect");

    stationPoints.forEach((station) => {
      const option = document.createElement("option");
      option.value = station.station_id;
      option.textContent = station.station_name || station.station_id;
      select.appendChild(option);
    });

    const status = await fetchJson("/api/model/status");
    document.getElementById("modelStatus").textContent = `운영 검증: ${
      status.operational_valid ? "완료" : "미완료"
    } · V4-C ${status.v4_c_gate_status}`;
    document.getElementById("warning").style.display = status.operational_valid ? "none" : "block";

    const update = () => {
      render(latest.points.filter((point) => String(point.station_id) === String(select.value)));
    };
    select.onchange = update;
    update();
  } catch (error) {
    const card = document.getElementById("forecastCard");
    card.replaceChildren();
    appendTextElement(card, "h2", "데이터 없음");
    appendTextElement(card, "p", "Forecast API 또는 latest artifact를 먼저 생성하세요.");
  }
}

main();
