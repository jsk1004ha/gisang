from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ForecastPoint:
    station_id: str
    timestamp: str
    prediction: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
