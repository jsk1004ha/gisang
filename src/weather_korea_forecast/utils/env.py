from __future__ import annotations

import os
from pathlib import Path

from weather_korea_forecast.utils.paths import project_root


def load_dotenv(path: str | Path | None = None) -> None:
    env_path = Path(path) if path is not None else project_root() / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())
