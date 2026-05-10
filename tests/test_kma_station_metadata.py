from __future__ import annotations

from weather_korea_forecast.data.download_kma_station_metadata import parse_kma_station_metadata_text


def test_parse_kma_station_metadata_text_filters_and_maps_fields() -> None:
    sample_text = """
header line
90 속초 Sokcho 128.56473 38.25085 17.53 18.73 1.7 10 1.4 11D20401 5182033035
108 서울 Seoul 126.96580 37.57142 85.67 86.67 1.5 10 1.4 11B10101 1111010100
"""
    config = {
        "stations": ["108"],
        "region_class_map": {"108": "inland"},
        "coastal_distance_km_map": {"108": 35.0},
    }

    frame = parse_kma_station_metadata_text(sample_text, config)

    assert frame["station_id"].tolist() == ["108"]
    assert frame.loc[0, "station_name_ko"] == "서울"
    assert frame.loc[0, "station_name_en"] == "Seoul"
    assert float(frame.loc[0, "lat"]) == 37.57142
    assert float(frame.loc[0, "lon"]) == 126.96580
    assert float(frame.loc[0, "elevation"]) == 85.67
    assert frame.loc[0, "region_class"] == "inland"
    assert float(frame.loc[0, "coastal_distance_km"]) == 35.0
