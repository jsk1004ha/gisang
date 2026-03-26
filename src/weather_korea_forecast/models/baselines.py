from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch


@dataclass
class BaselineTrainResult:
    history: list[dict[str, float | str]]
    best_val_loss: float


class PersistenceBaseline:
    def __init__(
        self,
        encoder_feature_names: list[str],
        target_columns: list[str],
        seasonal_period: int | None = None,
        target_source_features: list[str] | None = None,
    ) -> None:
        self.seasonal_period = seasonal_period
        self.encoder_feature_names = encoder_feature_names
        self.target_columns = target_columns
        self.target_source_features = target_source_features or _default_target_source_features(target_columns, encoder_feature_names)
        self.target_indices = [_resolve_feature_index(feature_name, encoder_feature_names) for feature_name in self.target_source_features]

    def predict_batch(self, batch: dict) -> torch.Tensor:
        encoder = batch["encoder_cont"]
        horizon = batch["target"].shape[1]
        target_values: list[torch.Tensor] = []
        source_index = -self.seasonal_period if self.seasonal_period is not None and encoder.shape[1] >= self.seasonal_period else -1
        for target_index in self.target_indices:
            base = encoder[:, source_index, target_index].unsqueeze(-1)
            target_values.append(base)
        prediction = torch.cat(target_values, dim=-1)
        return prediction.unsqueeze(1).repeat(1, horizon, 1)

    def save(self, path: str | Path, extra_state: dict[str, Any] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "seasonal_period": self.seasonal_period,
            "encoder_feature_names": self.encoder_feature_names,
            "target_columns": self.target_columns,
            "target_source_features": self.target_source_features,
            "extra_state": extra_state or {},
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path, bundle, model_config: dict | None = None) -> "PersistenceBaseline":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            encoder_feature_names=bundle.encoder_columns,
            target_columns=bundle.target_columns,
            seasonal_period=checkpoint.get("seasonal_period"),
            target_source_features=checkpoint.get("target_source_features"),
        )

    @classmethod
    def load_for_resume(cls, path: str | Path, bundle, model_config: dict | None = None) -> "PersistenceBaseline":
        return cls.load(path, bundle, model_config)


class RidgeRegressionBaseline:
    def __init__(
        self,
        encoder_feature_names: list[str],
        target_columns: list[str],
        prediction_length: int,
        alpha: float = 1.0,
    ) -> None:
        self.encoder_feature_names = encoder_feature_names
        self.target_columns = target_columns
        self.prediction_length = prediction_length
        self.alpha = alpha
        self.weights: torch.Tensor | None = None
        self.bias: torch.Tensor | None = None

    def fit(
        self,
        train_loader,
        val_loader,
        max_epochs: int,
        learning_rate: float,
        device: str = "cpu",
        early_stopping_patience: int = 3,
    ) -> BaselineTrainResult:
        train_features, train_targets = _collect_regression_tensors(train_loader)
        val_features, val_targets = _collect_regression_tensors(val_loader)

        self.weights, self.bias = _solve_ridge_regression(train_features, train_targets, self.alpha)
        train_loss = _mse_loss(self._predict_from_features(train_features), train_targets)
        val_loss = _mse_loss(self._predict_from_features(val_features), val_targets)
        return BaselineTrainResult(
            history=[{"epoch": "closed_form", "train_loss": train_loss, "val_loss": val_loss}],
            best_val_loss=val_loss,
        )

    def predict_batch(self, batch: dict) -> torch.Tensor:
        features = _flatten_batch_features(batch)
        return self._predict_from_features(features)

    def predict_loader(self, loader, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, dict[str, list[Any]]]:
        predictions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        metadata: dict[str, list[Any]] = {"station_id": [], "prediction_start": []}
        for batch in loader:
            predictions.append(self.predict_batch(batch).cpu())
            targets.append(batch["target"].cpu())
            metadata["station_id"].extend(batch["station_id"])
            metadata["prediction_start"].extend(batch["prediction_start"])
        return torch.cat(predictions), torch.cat(targets), metadata

    def save(self, path: str | Path, extra_state: dict[str, Any] | None = None) -> Path:
        if self.weights is None or self.bias is None:
            raise RuntimeError("RidgeRegressionBaseline must be fit before saving.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "weights": self.weights,
            "bias": self.bias,
            "alpha": self.alpha,
            "extra_state": extra_state or {},
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path, bundle, model_config: dict | None = None) -> "RidgeRegressionBaseline":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        alpha = float((model_config or {}).get("model", {}).get("alpha", checkpoint.get("alpha", 1.0)))
        model = cls(
            encoder_feature_names=bundle.encoder_columns,
            target_columns=bundle.target_columns,
            prediction_length=bundle.prediction_length,
            alpha=alpha,
        )
        model.weights = checkpoint["weights"].cpu()
        model.bias = checkpoint["bias"].cpu()
        return model

    @classmethod
    def load_for_resume(cls, path: str | Path, bundle, model_config: dict | None = None) -> "RidgeRegressionBaseline":
        return cls.load(path, bundle, model_config)

    def _predict_from_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.weights is None or self.bias is None:
            raise RuntimeError("RidgeRegressionBaseline must be fit before prediction.")
        outputs = features @ self.weights + self.bias
        return outputs.reshape(features.shape[0], self.prediction_length, len(self.target_columns))

    def feature_importance_frame(self, feature_names: list[str]) -> pd.DataFrame | None:
        if self.weights is None:
            return None
        weights = self.weights.detach().cpu().numpy()
        rows: list[dict[str, float | str | int]] = []
        output_index = 0
        for horizon_step in range(1, self.prediction_length + 1):
            for target_name in self.target_columns:
                for feature_index, feature_name in enumerate(feature_names):
                    rows.append(
                        {
                            "target_column": target_name,
                            "horizon_step": horizon_step,
                            "feature_name": feature_name,
                            "coefficient": float(weights[feature_index, output_index]),
                            "abs_coefficient": abs(float(weights[feature_index, output_index])),
                        }
                    )
                output_index += 1
        return pd.DataFrame(rows).sort_values(["target_column", "horizon_step", "abs_coefficient"], ascending=[True, True, False])


class LightGBMBaseline:
    def __init__(
        self,
        encoder_feature_names: list[str],
        target_columns: list[str],
        prediction_length: int,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.encoder_feature_names = encoder_feature_names
        self.target_columns = target_columns
        self.prediction_length = prediction_length
        self.params = params or {}
        self.model = None

    def fit(
        self,
        train_loader,
        val_loader,
        max_epochs: int,
        learning_rate: float,
        device: str = "cpu",
        early_stopping_patience: int = 3,
    ) -> BaselineTrainResult:
        train_features, train_targets = _collect_regression_tensors(train_loader)
        val_features, val_targets = _collect_regression_tensors(val_loader)
        self.model = _build_lightgbm_regressor(self.params)
        self.model.fit(train_features.numpy(), train_targets.numpy())
        train_prediction = torch.tensor(self.model.predict(train_features.numpy()), dtype=torch.float32)
        val_prediction = torch.tensor(self.model.predict(val_features.numpy()), dtype=torch.float32)
        train_loss = _mse_loss(train_prediction, train_targets)
        val_loss = _mse_loss(val_prediction, val_targets)
        return BaselineTrainResult(
            history=[{"epoch": "lightgbm_fit", "train_loss": train_loss, "val_loss": val_loss}],
            best_val_loss=val_loss,
        )

    def predict_batch(self, batch: dict) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("LightGBMBaseline must be fit before prediction.")
        features = _flatten_batch_features(batch)
        prediction = self.model.predict(features.numpy())
        tensor = torch.tensor(prediction, dtype=torch.float32)
        return tensor.reshape(features.shape[0], self.prediction_length, len(self.target_columns))

    def predict_loader(self, loader, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, dict[str, list[Any]]]:
        predictions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        metadata: dict[str, list[Any]] = {"station_id": [], "prediction_start": []}
        for batch in loader:
            predictions.append(self.predict_batch(batch))
            targets.append(batch["target"].cpu())
            metadata["station_id"].extend(batch["station_id"])
            metadata["prediction_start"].extend(batch["prediction_start"])
        return torch.cat(predictions), torch.cat(targets), metadata

    def save(self, path: str | Path, extra_state: dict[str, Any] | None = None) -> Path:
        if self.model is None:
            raise RuntimeError("LightGBMBaseline must be fit before saving.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "params": self.params,
            "extra_state": extra_state or {},
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path, bundle, model_config: dict | None = None) -> "LightGBMBaseline":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        params = dict((model_config or {}).get("model", {}).get("params", checkpoint.get("params", {})))
        model = cls(
            encoder_feature_names=bundle.encoder_columns,
            target_columns=bundle.target_columns,
            prediction_length=bundle.prediction_length,
            params=params,
        )
        model.model = checkpoint["model"]
        return model

    @classmethod
    def load_for_resume(cls, path: str | Path, bundle, model_config: dict | None = None) -> "LightGBMBaseline":
        return cls.load(path, bundle, model_config)

    def feature_importance_frame(self, feature_names: list[str]) -> pd.DataFrame | None:
        if self.model is None or not hasattr(self.model, "estimators_"):
            return None
        rows: list[dict[str, float | str | int]] = []
        output_index = 0
        for horizon_step in range(1, self.prediction_length + 1):
            for target_name in self.target_columns:
                estimator = self.model.estimators_[output_index]
                if hasattr(estimator, "feature_importances_"):
                    importances = estimator.feature_importances_
                else:
                    continue
                for feature_name, importance in zip(feature_names, importances):
                    rows.append(
                        {
                            "target_column": target_name,
                            "horizon_step": horizon_step,
                            "feature_name": feature_name,
                            "importance": float(importance),
                        }
                    )
                output_index += 1
        return pd.DataFrame(rows).sort_values(["target_column", "horizon_step", "importance"], ascending=[True, True, False])


def _default_target_source_features(target_columns: list[str], encoder_feature_names: list[str]) -> list[str]:
    source_features: list[str] = []
    for target_column in target_columns:
        base_name = target_column.replace("target_", "", 1)
        candidates = [f"obs_{base_name}", target_column, base_name]
        feature_name = next((candidate for candidate in candidates if candidate in encoder_feature_names), None)
        if feature_name is None:
            raise ValueError(f"Could not infer source feature for target column: {target_column}")
        source_features.append(feature_name)
    return source_features


def _resolve_feature_index(feature_name: str, encoder_feature_names: list[str]) -> int:
    if feature_name not in encoder_feature_names:
        raise ValueError(f"Encoder feature '{feature_name}' is not available in encoder columns.")
    return encoder_feature_names.index(feature_name)


def _collect_regression_tensors(loader) -> tuple[torch.Tensor, torch.Tensor]:
    feature_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    for batch in loader:
        feature_rows.append(_flatten_batch_features(batch))
        target_rows.append(batch["target"].reshape(batch["target"].shape[0], -1).cpu())
    return torch.cat(feature_rows, dim=0), torch.cat(target_rows, dim=0)


def _flatten_batch_features(batch: dict) -> torch.Tensor:
    encoder = batch["encoder_cont"].reshape(batch["encoder_cont"].shape[0], -1).cpu()
    decoder = batch["decoder_known"].reshape(batch["decoder_known"].shape[0], -1).cpu()
    static_real = batch["static_real"].reshape(batch["static_real"].shape[0], -1).cpu()
    return torch.cat([encoder, decoder, static_real], dim=1)


def _solve_ridge_regression(features: torch.Tensor, targets: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    ones = torch.ones((features.shape[0], 1), dtype=features.dtype)
    design = torch.cat([features, ones], dim=1)
    eye = torch.eye(design.shape[1], dtype=design.dtype)
    eye[-1, -1] = 0.0
    gram = design.T @ design + alpha * eye
    rhs = design.T @ targets
    coefficients = torch.linalg.solve(gram, rhs)
    return coefficients[:-1], coefficients[-1]


def _mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(torch.square(prediction.reshape(target.shape) - target)).item())


def _build_lightgbm_regressor(params: dict[str, Any]):
    try:
        from lightgbm import LGBMRegressor  # type: ignore
    except ImportError as exc:
        raise RuntimeError("lightgbm is not installed. Install it to use the lightgbm baseline.") from exc
    from sklearn.multioutput import MultiOutputRegressor

    default_params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "n_jobs": 1,
        "random_state": 42,
        "verbosity": -1,
    }
    default_params.update(params)
    return MultiOutputRegressor(LGBMRegressor(**default_params))
