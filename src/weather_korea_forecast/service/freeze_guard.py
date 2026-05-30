from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

DEFAULT_ALLOWED_PREFIXES = (
    "web/",
    "docs/",
    "tests/",
    "README.md",
    "CHANGELOG.md",
    "src/weather_korea_forecast/api/",
    "src/weather_korea_forecast/service/",
)
FORBIDDEN_PATTERNS = (
    "src/weather_korea_forecast/training/",
    "src/weather_korea_forecast/models/",
    "src/weather_korea_forecast/v2/train.py",
    "src/weather_korea_forecast/v3/",
    "configs/model/",
    "configs/train/",
    "configs/v2/experiments/",
    "configs/v3/experiments/",
)
FORBIDDEN_FILE_NAMES = {
    "src/weather_korea_forecast/v4/operational_performance.py",
    "src/weather_korea_forecast/v4/operational_ensemble.py",
    "src/weather_korea_forecast/v4/beta_targets.py",
}


class FreezeGuardError(ValueError):
    pass


def load_freeze_record(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FreezeGuardError(f"freeze record must be a JSON object: {path}")
    return payload


def normalize_repo_path(path: str | Path) -> str:
    return str(path).replace("\\", "/").lstrip("./")


def classify_post_freeze_paths(
    paths: Iterable[str | Path],
    freeze_record: dict[str, Any],
    *,
    allowed_prefixes: tuple[str, ...] = DEFAULT_ALLOWED_PREFIXES,
) -> dict[str, list[str]]:
    allowed_scopes = set(freeze_record.get("allowed_post_freeze_change_scopes") or [])
    forbidden_scopes = set(freeze_record.get("forbidden_post_freeze_change_scopes") or [])
    allowed: list[str] = []
    blocked: list[str] = []
    for raw_path in paths:
        path = normalize_repo_path(raw_path)
        if not path:
            continue
        if _is_forbidden_model_path(path, forbidden_scopes):
            blocked.append(path)
            continue
        if path.startswith(allowed_prefixes) or path in allowed_prefixes:
            allowed.append(path)
            continue
        if "docs" in allowed_scopes and path.endswith(".md"):
            allowed.append(path)
            continue
        blocked.append(path)
    return {"allowed": sorted(set(allowed)), "blocked": sorted(set(blocked))}


def assert_post_freeze_paths_allowed(paths: Iterable[str | Path], freeze_record: dict[str, Any]) -> dict[str, list[str]]:
    result = classify_post_freeze_paths(paths, freeze_record)
    if result["blocked"]:
        raise FreezeGuardError("post-freeze changes include forbidden model/training/model-selection paths: " + ", ".join(result["blocked"]))
    return result


def _is_forbidden_model_path(path: str, forbidden_scopes: set[str]) -> bool:
    if path in FORBIDDEN_FILE_NAMES:
        return True
    if any(path.startswith(pattern) for pattern in FORBIDDEN_PATTERNS):
        return True
    if "training" in forbidden_scopes and "/training/" in f"/{path}":
        return True
    if "model_selection" in forbidden_scopes and any(token in path for token in ("operational_performance.py", "operational_ensemble.py", "/models/")):
        return True
    if "experiment_candidates" in forbidden_scopes and "/experiments/" in path:
        return True
    return False


def _read_changed_files(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate post-G030 freeze changes are site/API/schema-consumption only.")
    parser.add_argument("--freeze-record", required=True, type=Path)
    parser.add_argument("--changed-files", required=True, type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    freeze_record = load_freeze_record(args.freeze_record)
    result = assert_post_freeze_paths_allowed(_read_changed_files(args.changed_files), freeze_record)
    if args.json:
        print(json.dumps({"status": "PASS", **result}, ensure_ascii=False, indent=2))
    else:
        print(f"PASS: {len(result['allowed'])} post-freeze paths allowed")


if __name__ == "__main__":
    main()
