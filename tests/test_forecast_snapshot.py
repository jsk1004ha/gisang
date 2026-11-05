import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pandas as pd
import pytest

from weather_korea_forecast.api import main as api
from weather_korea_forecast.service import forecast_snapshot
from weather_korea_forecast.service.export_forecast import ForecastExportError, export_forecast


@pytest.fixture
def publish(tmp_path):
    def export(run_id=None, *, temperature=1.0, source="gfs"):
        predictions = tmp_path / f"predictions_{run_id or temperature}.csv"
        pd.DataFrame({
            "station_id": ["108"], "valid_time": ["2026-11-05T01:00:00Z"],
            "horizon_step": [1], "temperature_c": [temperature], "cloud_cover_percent": [50],
        }).to_csv(predictions, index=False)
        return export_forecast(
            predictions_path=predictions, station_metadata_path=None,
            output_dir=tmp_path / "forecasts", forecast_run_id=run_id,
            model_version=run_id or "auto", source=source,
        )
    return export


def _assert_snapshot(payload, run_id):
    assert payload["run"]["forecast_run_id"] == run_id
    assert payload["points"]
    assert {point["forecast_run_id"] for point in payload["points"]} == {run_id}


@pytest.mark.parametrize("failure", ["points", "pointer"])
def test_failed_publication_retains_previous_complete_forecast(publish, monkeypatch, failure):
    previous = publish("run_a")
    latest = Path(previous.latest_dir)
    before = {path.name: path.read_bytes() for path in latest.iterdir()}
    if failure == "points":
        original = pd.DataFrame.to_json

        def fail_points(frame, path, *args, **kwargs):
            if Path(path).parent.name == "run_b":
                raise OSError("points write failed")
            return original(frame, path, *args, **kwargs)

        monkeypatch.setattr(pd.DataFrame, "to_json", fail_points)
    else:
        original = forecast_snapshot.os.replace

        def fail_pointer(source, destination):
            if Path(destination).name == "current.json":
                raise OSError("pointer replace failed")
            return original(source, destination)

        monkeypatch.setattr(forecast_snapshot.os, "replace", fail_pointer)
    with pytest.raises(OSError, match="failed"):
        publish("run_b", temperature=10)
    assert {path.name: path.read_bytes() for path in latest.iterdir()} == before
    _assert_snapshot(api.latest_forecast(latest), "run_a")
    assert forecast_snapshot.resolve_forecast_dir(latest) == Path(previous.run_dir)


def test_published_run_ids_cannot_be_overwritten(publish):
    previous = publish("run_a")
    before = (Path(previous.run_dir) / "forecast_points.json").read_bytes()
    with pytest.raises(ForecastExportError, match="cannot be overwritten"):
        publish("run_a", temperature=99)
    assert (Path(previous.run_dir) / "forecast_points.json").read_bytes() == before
    _assert_snapshot(api.latest_forecast(Path(previous.latest_dir)), "run_a")


def test_automatic_run_ids_are_unique_within_the_same_second(publish, monkeypatch):
    exporter = importlib.import_module("weather_korea_forecast.service.export_forecast")
    monkeypatch.setattr(exporter, "_utc_now_id", lambda: "20261105T000000Z")
    first, second = publish(temperature=1), publish(temperature=2)
    assert first.forecast_run_id != second.forecast_run_id
    assert json.loads((Path(first.run_dir) / "forecast_points.json").read_text())[0]["temperature_c"] == 1
    _assert_snapshot(api.latest_forecast(Path(second.latest_dir)), second.forecast_run_id)


@pytest.mark.parametrize("endpoint", ["latest", "station", "status"])
def test_api_pins_one_run_when_current_changes_between_reads(publish, monkeypatch, tmp_path, endpoint):
    previous = publish("run_a")
    original = api._forecast_run
    switched = False

    def switch_after_read(directory):
        nonlocal switched
        run = original(directory)
        if not switched:
            switched = True
            publish("run_b", temperature=20, source="kma")
        return run

    monkeypatch.setattr(api, "_forecast_run", switch_after_read)
    latest = Path(previous.latest_dir)
    if endpoint == "status":
        result = api.model_status(latest, tmp_path / "reports")
        assert result["model_version"] == "run_a"
        assert result["beta_targets"]["cloud"]["source"] == "gfs_direct"
    else:
        result = api.latest_forecast(latest) if endpoint == "latest" else api.station_forecast("108", latest)
        _assert_snapshot(result, "run_a")
        assert result["points"][0]["temperature_c"] == 1
    _assert_snapshot(api.latest_forecast(latest), "run_b")


def test_first_publication_during_a_legacy_read_returns_a_complete_snapshot(publish, monkeypatch):
    previous = publish("run_a")
    latest = Path(previous.latest_dir)
    (latest / "current.json").unlink()
    original = api._forecast_run

    def publish_after_legacy_header(directory):
        run = original(directory)
        if directory == latest:
            publish("run_b", temperature=20)
        return run

    monkeypatch.setattr(api, "_forecast_run", publish_after_legacy_header)
    _assert_snapshot(api.latest_forecast(latest), "run_b")


def test_overlapping_publishers_only_expose_complete_runs(publish, monkeypatch):
    previous = publish("run_a")
    latest = Path(previous.latest_dir)
    ready = Barrier(2)
    original_write = forecast_snapshot._atomic_write
    original_replace = forecast_snapshot.os.replace
    observations = []

    def overlap_at_commit(path, content):
        if path.name == "current.json":
            ready.wait(timeout=5)
        return original_write(path, content)

    def observe_publication(source, destination):
        result = original_replace(source, destination)
        payload = api.latest_forecast(latest)
        _assert_snapshot(payload, payload["run"]["forecast_run_id"])
        observations.append(payload["run"]["forecast_run_id"])
        return result

    monkeypatch.setattr(forecast_snapshot, "_atomic_write", overlap_at_commit)
    monkeypatch.setattr(forecast_snapshot.os, "replace", observe_publication)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(publish, run_id, temperature=value) for run_id, value in (("run_b", 2), ("run_c", 3))]
        for future in futures:
            future.result(timeout=10)
    assert observations
    assert set(observations) <= {"run_b", "run_c"}
    current = api.latest_forecast(latest)
    _assert_snapshot(current, current["run"]["forecast_run_id"])
    for run_id in ("run_a", "run_b", "run_c"):
        points = json.loads((latest.parent / "runs" / run_id / "forecast_points.json").read_text())
        assert {point["forecast_run_id"] for point in points} == {run_id}


def test_legacy_copy_failure_does_not_undo_a_committed_snapshot(publish, monkeypatch):
    previous = publish("run_a")
    latest = Path(previous.latest_dir)
    original = forecast_snapshot.os.replace

    def fail_legacy_copy(source, destination):
        if Path(destination) == latest / "forecast_points.json":
            raise OSError("legacy copy failed")
        return original(source, destination)

    monkeypatch.setattr(forecast_snapshot.os, "replace", fail_legacy_copy)
    with pytest.warns(RuntimeWarning, match="legacy flat files"):
        publish("run_b", temperature=2)
    _assert_snapshot(api.latest_forecast(latest), "run_b")
