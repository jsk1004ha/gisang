from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


class FallbackSeqForecaster(nn.Module):
    def __init__(
        self,
        encoder_length: int,
        encoder_dim: int,
        decoder_length: int,
        decoder_dim: int,
        static_dim: int,
        target_dim: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        input_dim = encoder_length * encoder_dim + decoder_length * decoder_dim + static_dim
        self.decoder_length = decoder_length
        self.target_dim = target_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, decoder_length * target_dim),
        )

    def forward(self, encoder_cont: torch.Tensor, decoder_known: torch.Tensor, static_real: torch.Tensor) -> torch.Tensor:
        batch_size = encoder_cont.shape[0]
        flattened = [
            encoder_cont.reshape(batch_size, -1),
            decoder_known.reshape(batch_size, -1),
        ]
        if static_real.numel() > 0:
            flattened.append(static_real.reshape(batch_size, -1))
        x = torch.cat(flattened, dim=1)
        output = self.network(x)
        return output.reshape(batch_size, self.decoder_length, self.target_dim)


@dataclass
class TrainResult:
    history: list[dict[str, float]]
    best_val_loss: float


class TFTModelWrapper:
    def __init__(self, model: nn.Module, backend: str, config: dict[str, Any], bundle: Any | None = None) -> None:
        self.model = model
        self.backend = backend
        self.config = config
        self.bundle = bundle

    @classmethod
    def from_dataset_bundle(cls, bundle: Any, model_config: dict[str, Any]) -> "TFTModelWrapper":
        backend = model_config["model"].get("backend", "fallback_torch")
        if backend == "pytorch_forecasting":
            return cls(_build_pytorch_forecasting_model(bundle, model_config), backend, model_config, bundle=bundle)

        model = FallbackSeqForecaster(
            encoder_length=bundle.encoder_length,
            encoder_dim=len(bundle.encoder_columns),
            decoder_length=bundle.prediction_length,
            decoder_dim=len(bundle.decoder_columns),
            static_dim=len(bundle.static_columns),
            target_dim=len(bundle.target_columns),
            hidden_size=int(model_config["model"].get("hidden_size", 32)),
            dropout=float(model_config["model"].get("dropout", 0.1)),
        )
        return cls(model=model, backend=backend, config=model_config, bundle=bundle)

    def fit(
        self,
        train_loader: Any,
        val_loader: Any,
        max_epochs: int,
        learning_rate: float,
        device: str = "cpu",
        early_stopping_patience: int = 3,
    ) -> TrainResult:
        if self.backend == "pytorch_forecasting":
            if isinstance(self.model, dict):
                history: list[dict[str, float | str]] = []
                val_losses: list[float] = []
                for target_column, target_model in self.model.items():
                    result = _fit_with_lightning(
                        target_model,
                        train_loader[target_column],
                        val_loader[target_column],
                        max_epochs=max_epochs,
                        device=device,
                        early_stopping_patience=early_stopping_patience,
                    )
                    history.append({"target_name": target_column.replace("target_", "", 1), "best_val_loss": result.best_val_loss})
                    val_losses.append(result.best_val_loss)
                return TrainResult(history=history, best_val_loss=sum(val_losses) / len(val_losses))
            return _fit_with_lightning(
                self.model,
                train_loader,
                val_loader,
                max_epochs=max_epochs,
                device=device,
                early_stopping_patience=early_stopping_patience,
            )

        self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        history: list[dict[str, float]] = []
        best_state: dict[str, torch.Tensor] | None = None
        best_val_loss = float("inf")
        stale_epochs = 0

        for epoch in range(1, max_epochs + 1):
            train_loss = _run_epoch(self.model, train_loader, optimizer, criterion, device, train=True)
            val_loss = _run_epoch(self.model, val_loader, optimizer, criterion, device, train=False)
            history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= early_stopping_patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return TrainResult(history=history, best_val_loss=best_val_loss)

    def predict_loader(self, loader: Any, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, dict[str, list[Any]]]:
        if self.backend == "pytorch_forecasting":
            if isinstance(self.model, dict):
                predictions: list[torch.Tensor] = []
                targets: list[torch.Tensor] = []
                metadata: dict[str, list[Any]] | None = None
                for target_column in self.bundle.target_columns:
                    target_prediction, target_target, target_metadata = _predict_with_lightning(self.model[target_column], loader[target_column], self.bundle)
                    predictions.append(target_prediction)
                    targets.append(target_target)
                    if metadata is None:
                        metadata = target_metadata
                return torch.cat(predictions, dim=-1), torch.cat(targets, dim=-1), metadata or {"station_id": [], "prediction_start": []}
            return _predict_with_lightning(self.model, loader, self.bundle)

        self.model.to(device)
        self.model.eval()
        predictions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        metadata: dict[str, list[Any]] = {"station_id": [], "prediction_start": []}
        with torch.no_grad():
            for batch in loader:
                encoder = batch["encoder_cont"].to(device)
                decoder = batch["decoder_known"].to(device)
                static_real = batch["static_real"].to(device)
                preds = self.model(encoder, decoder, static_real).cpu()
                predictions.append(preds)
                targets.append(batch["target"].cpu())
                metadata["station_id"].extend(batch["station_id"])
                metadata["prediction_start"].extend(batch["prediction_start"])
        return torch.cat(predictions), torch.cat(targets), metadata

    def save(self, path: str | Path, extra_state: dict[str, Any] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend": self.backend,
            "config": self.config,
            "state_dict": _serialize_state_dict(self.model),
            "extra_state": extra_state or {},
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path, bundle: Any) -> "TFTModelWrapper":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        wrapper = cls.from_dataset_bundle(bundle, checkpoint["config"])
        _load_serialized_state_dict(wrapper.model, checkpoint["state_dict"])
        return wrapper

    @classmethod
    def load_for_resume(cls, path: str | Path, bundle: Any, model_config: dict[str, Any] | None = None) -> "TFTModelWrapper":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        wrapper = cls.from_dataset_bundle(bundle, model_config or checkpoint["config"])
        _load_serialized_state_dict(wrapper.model, checkpoint["state_dict"])
        return wrapper


def _run_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    train: bool,
) -> float:
    if len(loader) == 0:
        return float("nan")
    model.train(train)
    losses: list[float] = []
    for batch in loader:
        encoder = batch["encoder_cont"].to(device)
        decoder = batch["decoder_known"].to(device)
        static_real = batch["static_real"].to(device)
        target = batch["target"].to(device)
        if train:
            optimizer.zero_grad()
        prediction = model(encoder, decoder, static_real)
        loss = criterion(prediction, target)
        if train:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return sum(losses) / len(losses)


def _build_pytorch_forecasting_model(bundle: Any, model_config: dict[str, Any]) -> Any:
    try:
        from pytorch_forecasting.models import TemporalFusionTransformer  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pytorch_forecasting is not installed.") from exc

    datasets_by_target = bundle.metadata.get("pf_datasets", {})
    models: dict[str, Any] = {}
    for target_column in bundle.target_columns:
        train_dataset = datasets_by_target.get(target_column, {}).get("train", bundle.train_dataset)
        models[target_column] = TemporalFusionTransformer.from_dataset(
            train_dataset,
            learning_rate=float(model_config["model"].get("learning_rate", 1e-3)),
            hidden_size=int(model_config["model"].get("hidden_size", 32)),
            attention_head_size=int(model_config["model"].get("attention_head_size", 4)),
            dropout=float(model_config["model"].get("dropout", 0.1)),
            hidden_continuous_size=int(model_config["model"].get("hidden_continuous_size", 16)),
        )
    if len(models) == 1:
        return models[bundle.target_columns[0]]
    return models


def _fit_with_lightning(
    model: Any,
    train_loader: Any,
    val_loader: Any,
    max_epochs: int,
    device: str,
    early_stopping_patience: int,
) -> TrainResult:
    try:
        import lightning as L  # type: ignore
        from lightning.pytorch.callbacks import EarlyStopping  # type: ignore
    except ImportError as exc:
        raise RuntimeError("lightning is not installed.") from exc

    callbacks = [EarlyStopping(monitor="val_loss", patience=early_stopping_patience, mode="min")]
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu" if device == "cpu" else "auto",
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=callbacks,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    val_loss = trainer.callback_metrics.get("val_loss")
    best_val_loss = float(val_loss.detach().cpu()) if val_loss is not None else float("nan")
    return TrainResult(history=[], best_val_loss=best_val_loss)


def _predict_with_lightning(model: Any, loader: Any, bundle: Any, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, dict[str, list[Any]]]:
    prediction = model.predict(
        loader,
        return_index=True,
        return_y=True,
        trainer_kwargs={
            "accelerator": "cpu" if device == "cpu" else "auto",
            "devices": 1,
            "logger": False,
            "enable_checkpointing": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        },
    )
    output = prediction.output
    if output.ndim == 2:
        output = output.unsqueeze(-1)

    target = prediction.y[0]
    if target.ndim == 2:
        target = target.unsqueeze(-1)

    index_frame = prediction.index.copy()
    if bundle is None:
        raise RuntimeError("Dataset bundle is required for pytorch_forecasting predictions.")

    time_lookup = (
        bundle.test_frame[["station_id", "time_idx", "datetime"]]
        .drop_duplicates()
        .rename(columns={"time_idx": "prediction_time_idx", "datetime": "prediction_start"})
    )
    index_frame = index_frame.rename(columns={"time_idx": "prediction_time_idx"})
    index_frame = index_frame.merge(time_lookup, on=["station_id", "prediction_time_idx"], how="left")
    metadata = {
        "station_id": index_frame["station_id"].tolist(),
        "prediction_start": index_frame["prediction_start"].astype(str).tolist(),
    }
    return output.cpu(), target.cpu(), metadata


def can_use_pytorch_forecasting() -> bool:
    try:
        import lightning  # noqa: F401
        import pytorch_forecasting  # noqa: F401
    except ImportError:
        return False
    return True


def _serialize_state_dict(model: Any) -> Any:
    if isinstance(model, dict):
        return {name: component.state_dict() for name, component in model.items()}
    return model.state_dict()


def _load_serialized_state_dict(model: Any, state_dict: Any) -> None:
    if isinstance(model, dict):
        for name, component in model.items():
            component.load_state_dict(state_dict[name])
        return
    model.load_state_dict(state_dict)
