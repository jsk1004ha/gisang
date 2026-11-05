"""Publish complete forecast runs and resolve a stable snapshot for each reader."""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import warnings
from pathlib import Path


FORECAST_FILES = ("forecast_run.json", "forecast_points.json", "forecast_points.csv")


def validate_run_id(run_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id) or run_id in {".", ".."}:
        raise ValueError("forecast_run_id must be a portable directory name containing only letters, digits, _, - and .")


def resolve_forecast_dir(forecast_dir: Path) -> Path:
    """Resolve current once; a directory without a pointer is a legacy snapshot."""
    try:
        payload = json.loads((forecast_dir / "current.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return forecast_dir
    run_id = payload.get("forecast_run_id") if isinstance(payload, dict) else None
    if not isinstance(run_id, str):
        raise ValueError("current.json must contain a forecast_run_id")
    validate_run_id(run_id)
    run_dir = forecast_dir.parent / "runs" / run_id
    if not all((run_dir / filename).is_file() for filename in FORECAST_FILES):
        raise ValueError(f"current.json references an incomplete forecast run: {run_id}")
    return run_dir


def _atomic_write(path: Path, content: bytes, *, mode: int) -> None:
    # Replacements keep the destination's permissions. New files inherit the
    # caller's source mode, which already reflects its creation umask.
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        pass
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.chmod(temporary_path, mode)
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def publish_forecast_run(run_dir: Path, latest_dir: Path) -> None:
    """Commit current atomically, then refresh best-effort legacy flat files.

    Readers needing a consistent set of files must resolve current.json once
    and read that immutable run. Flat files remain for older/offline consumers.
    """
    latest_dir.mkdir(parents=True, exist_ok=True)
    pointer = json.dumps({"forecast_run_id": run_dir.name}, ensure_ascii=False).encode("utf-8")
    metadata_mode = stat.S_IMODE((run_dir / "forecast_run.json").stat().st_mode)
    _atomic_write(latest_dir / "current.json", pointer, mode=metadata_mode)
    try:
        # A second publisher may already have committed a newer complete run.
        current_dir = resolve_forecast_dir(latest_dir)
        for filename in FORECAST_FILES:
            source = current_dir / filename
            _atomic_write(latest_dir / filename, source.read_bytes(), mode=stat.S_IMODE(source.stat().st_mode))
    except OSError as error:
        warnings.warn(
            f"Forecast current.json was published; legacy flat files could not all be refreshed: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
