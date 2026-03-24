from __future__ import annotations

import torch


class PersistenceBaseline:
    def __init__(self, seasonal_period: int | None = None) -> None:
        self.seasonal_period = seasonal_period

    def predict_batch(self, batch: dict) -> torch.Tensor:
        encoder = batch["encoder_cont"]
        horizon = batch["target"].shape[1]
        if self.seasonal_period is not None and encoder.shape[1] >= self.seasonal_period:
            base = encoder[:, -self.seasonal_period, 0].unsqueeze(-1)
        else:
            base = encoder[:, -1, 0].unsqueeze(-1)
        return base.unsqueeze(1).repeat(1, horizon, batch["target"].shape[-1])
