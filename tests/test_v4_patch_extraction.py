from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from weather_korea_forecast.v4.patch_extraction import extract_nwp_patches, patches_to_feature_table


def _grid() -> pd.DataFrame:
    rows = []
    for lat in [36.0, 37.0, 38.0]:
        for lon in [126.0, 127.0, 128.0]:
            rows.append(
                {
                    "forecast_init_time": "2024-01-01T00:00:00Z",
                    "valid_time": "2024-01-01T03:00:00Z",
                    "lat": lat,
                    "lon": lon,
                    "nwp_t2m": lat * 10.0 + lon,
                    "nwp_tp": 1.0 if lat == 37.0 and lon >= 127.0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_extract_nwp_patches_returns_centered_long_schema() -> None:
    stations = pd.DataFrame([{"station_id": "108", "lat": 37.1, "lon": 127.2}])

    patches = extract_nwp_patches(_grid(), stations, variables=["nwp_t2m"], patch_size=3, source="gfs_forecast")

    assert len(patches) == 9
    assert list(patches.columns) == [
        "station_id",
        "forecast_init_time",
        "valid_time",
        "horizon_step",
        "source",
        "patch_size",
        "variable",
        "row_offset",
        "col_offset",
        "value",
        "lat",
        "lon",
    ]
    center = patches.loc[(patches["row_offset"] == 0) & (patches["col_offset"] == 0)].iloc[0]
    assert center["station_id"] == "108"
    assert center["horizon_step"] == 3
    assert center["source"] == "gfs_forecast"
    assert center["lat"] == pytest.approx(37.0)
    assert center["lon"] == pytest.approx(127.0)
    assert center["value"] == pytest.approx(497.0)


def test_extract_nwp_patches_pads_edges_to_keep_patch_shape() -> None:
    stations = pd.DataFrame([{"station_id": "edge", "lat": 36.0, "lon": 126.0}])

    patches = extract_nwp_patches(_grid(), stations, variables=["nwp_t2m"], patch_size=3)

    assert len(patches) == 9
    assert patches["value"].isna().sum() == 5
    assert patches.loc[(patches["row_offset"] == -1) & (patches["col_offset"] == -1), "lat"].isna().all()
    center_value = patches.loc[(patches["row_offset"] == 0) & (patches["col_offset"] == 0), "value"].iloc[0]
    assert center_value == pytest.approx(486.0)


def test_patches_to_feature_table_summarizes_tree_features_and_precip_coverage() -> None:
    stations = pd.DataFrame([{"station_id": "108", "lat": 37.0, "lon": 127.0}])
    patches = extract_nwp_patches(_grid(), stations, variables=["nwp_t2m", "nwp_tp"], patch_size=3)

    features = patches_to_feature_table(patches)

    assert len(features) == 1
    row = features.iloc[0]
    assert row["nwp_t2m_patch_center"] == pytest.approx(497.0)
    assert row["nwp_t2m_patch_mean"] == pytest.approx(np.mean([486, 487, 488, 496, 497, 498, 506, 507, 508]))
    assert row["nwp_t2m_patch_min"] == pytest.approx(486.0)
    assert row["nwp_t2m_patch_max"] == pytest.approx(508.0)
    assert row["nwp_t2m_patch_range"] == pytest.approx(22.0)
    assert row["nwp_t2m_patch_gradient_x"] == pytest.approx(2.0)
    assert row["nwp_t2m_patch_gradient_y"] == pytest.approx(20.0)
    assert row["patch_nwp_t2m_mean"] == pytest.approx(row["nwp_t2m_patch_mean"])
    assert row["patch_nwp_t2m_gradient_x"] == pytest.approx(row["nwp_t2m_patch_gradient_x"])
    assert row["nwp_tp_patch_coverage_fraction"] == pytest.approx(2 / 9)
    assert row["patch_nwp_tp_coverage_fraction"] == pytest.approx(2 / 9)


def test_extract_nwp_patches_validates_requested_variables_and_patch_size() -> None:
    stations = pd.DataFrame([{"station_id": "108", "lat": 37.0, "lon": 127.0}])

    with pytest.raises(ValueError, match="positive odd"):
        extract_nwp_patches(_grid(), stations, variables=["nwp_t2m"], patch_size=4)
    with pytest.raises(ValueError, match="missing requested"):
        extract_nwp_patches(_grid(), stations, variables=["missing"], patch_size=3)
