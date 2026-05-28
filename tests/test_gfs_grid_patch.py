from __future__ import annotations

import pandas as pd
import pytest

from weather_korea_forecast.v4.gfs_grid_patch import (
    PATCH_FEATURE_MODE,
    _expand_glob,
    extract_true_gfs_grid_patch_features,
    normalize_gfs_grid_value,
    parse_gfs_grib_filename,
)


def test_parse_gfs_grib_filename_infers_issue_and_horizon() -> None:
    issue, horizon = parse_gfs_grib_filename("gfs_20260504_06_f024_rh2m-t2m-d2m.grib2")

    assert issue.isoformat() == "2026-05-04T06:00:00+00:00"
    assert horizon == 24


def test_expand_glob_supports_absolute_patterns(tmp_path) -> None:
    path = tmp_path / "gfs_20260519_00_f001_korea.grib2"
    path.write_text("dummy", encoding="utf-8")

    assert _expand_glob(str(tmp_path / "*.grib2")) == [path]


def test_true_gfs_grid_patch_features_shape_and_mode() -> None:
    grid = pd.DataFrame(
        [
            {
                "forecast_init_time": "2026-05-04T00:00:00Z",
                "valid_time": "2026-05-04T01:00:00Z",
                "horizon_step": 1,
                "lat": lat,
                "lon": lon,
                "nwp_t2m": lat * 10 + lon,
                "nwp_tp": 1.0 if lat == 37.0 and lon == 127.0 else 0.0,
            }
            for lat in [36.75, 37.0, 37.25]
            for lon in [126.75, 127.0, 127.25]
        ]
    )
    stations = pd.DataFrame([{"station_id": "108", "lat": 37.01, "lon": 127.02}])

    features = extract_true_gfs_grid_patch_features([grid], stations, patch_size=3, variables=["nwp_t2m", "nwp_tp"])

    assert len(features) == 1
    row = features.iloc[0]
    assert row["patch_feature_mode"] == PATCH_FEATURE_MODE
    assert row["patch_size"] == 3
    assert row["nwp_t2m_patch_center"] == pytest.approx(497.0)
    assert row["nwp_t2m_patch_gradient_x"] == pytest.approx(0.5)
    assert row["nwp_tp_patch_sum"] == pytest.approx(1.0)


def test_normalize_gfs_grid_value_converts_units() -> None:
    assert normalize_gfs_grid_value("nwp_t2m", 300.0) == pytest.approx(26.85)
    assert normalize_gfs_grid_value("nwp_sp", 101325.0) == pytest.approx(1013.25)
