from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class WeatherCodeResult:
    weather_code: str
    weather_label_ko: str
    weather_icon: str


WEATHER_LABELS: dict[str, tuple[str, str]] = {
    "clear": ("맑음", "☀️"),
    "mostly_clear": ("대체로 맑음", "🌤️"),
    "partly_cloudy": ("구름 조금", "⛅"),
    "cloudy": ("구름 많음", "☁️"),
    "overcast": ("흐림", "☁️"),
    "rain": ("비", "🌧️"),
    "heavy_rain": ("강한 비", "⛈️"),
    "snow": ("눈", "❄️"),
    "sleet": ("진눈깨비", "🌨️"),
    "fog": ("안개", "🌫️"),
    "strong_wind": ("강풍", "💨"),
    "heat": ("더움", "🔥"),
    "cold": ("추움", "🥶"),
    "unknown": ("알 수 없음", "❔"),
}


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def weather_code_for_values(
    *,
    temperature_c: Any = None,
    humidity_percent: Any = None,
    precip_mm: Any = None,
    precip_probability: Any = None,
    cloud_cover_percent: Any = None,
    wind_speed_ms: Any = None,
    dew_point_c: Any = None,
) -> WeatherCodeResult:
    temp = _float(temperature_c)
    humidity = _float(humidity_percent)
    precip = _float(precip_mm)
    precip_prob = _float(precip_probability)
    cloud = _float(cloud_cover_percent)
    wind = _float(wind_speed_ms)
    dew = _float(dew_point_c)

    code = "unknown"
    if (precip is not None and precip >= 5.0) or (precip_prob is not None and precip_prob >= 0.8):
        code = "snow" if temp is not None and temp <= 0 else "heavy_rain"
    elif (precip is not None and precip > 0.1) or (precip_prob is not None and precip_prob >= 0.5):
        code = "snow" if temp is not None and temp <= 0 else "rain"
    elif temp is not None and dew is not None and humidity is not None and humidity >= 90 and abs(temp - dew) < 1.5:
        code = "fog"
    elif wind is not None and wind >= 14.0:
        code = "strong_wind"
    elif cloud is not None:
        if cloud < 20:
            code = "clear"
        elif cloud < 35:
            code = "mostly_clear"
        elif cloud < 50:
            code = "partly_cloudy"
        elif cloud < 80:
            code = "cloudy"
        else:
            code = "overcast"
    elif temp is not None and temp >= 33:
        code = "heat"
    elif temp is not None and temp <= -10:
        code = "cold"

    label, icon = WEATHER_LABELS[code]
    return WeatherCodeResult(code, label, icon)


def weather_code_for_row(row: dict[str, Any]) -> WeatherCodeResult:
    return weather_code_for_values(
        temperature_c=row.get("temperature_c", row.get("temp", row.get("prediction"))),
        humidity_percent=row.get("humidity_percent", row.get("humidity")),
        precip_mm=row.get("precip_mm", row.get("precipitation", row.get("nwp_tp"))),
        precip_probability=row.get("precip_probability"),
        cloud_cover_percent=row.get("cloud_cover_percent", row.get("cloud_cover", row.get("nwp_cloud_cover"))),
        wind_speed_ms=row.get("wind_speed_ms", row.get("wind_speed", row.get("nwp_wind_speed"))),
        dew_point_c=row.get("dew_point_c", row.get("nwp_dew_point")),
    )
