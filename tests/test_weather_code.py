from weather_korea_forecast.service.weather_code import weather_code_for_values


def test_weather_code_precipitation_and_cloud_rules():
    assert weather_code_for_values(temperature_c=5, precip_mm=6).weather_code == "heavy_rain"
    assert weather_code_for_values(temperature_c=-1, precip_probability=0.6).weather_code == "snow"
    assert weather_code_for_values(cloud_cover_percent=10).weather_code == "clear"
    assert weather_code_for_values(cloud_cover_percent=85).weather_code == "overcast"


def test_weather_code_fog_and_wind_rules():
    assert weather_code_for_values(temperature_c=10, dew_point_c=9, humidity_percent=95).weather_code == "fog"
    assert weather_code_for_values(wind_speed_ms=15, cloud_cover_percent=None).weather_code == "strong_wind"
