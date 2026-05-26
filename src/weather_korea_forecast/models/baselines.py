from __future__ import annotations

from dataclasses import dataclass
from itertools import product
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


class DecoderFeatureBaseline:
    """Copy configured future-known decoder features into the forecast target.

    This baseline is intentionally simple: it is useful for MOS/backtest
    experiments where a future-valid covariate is already on the target scale
    (for example a forecast-model baseline or an explicitly documented oracle
    feature).  It keeps such experiments config-driven instead of post-editing
    predictions or metrics.
    """

    def __init__(
        self,
        decoder_feature_names: list[str],
        target_columns: list[str],
        target_source_features: list[str],
    ) -> None:
        if len(target_source_features) != len(target_columns):
            raise ValueError("decoder_feature_baseline requires one source feature per target column.")
        self.decoder_feature_names = decoder_feature_names
        self.target_columns = target_columns
        self.target_source_features = target_source_features
        self.target_indices = [_resolve_feature_index(feature_name, decoder_feature_names) for feature_name in target_source_features]

    def fit(
        self,
        train_loader,
        val_loader,
        max_epochs: int,
        learning_rate: float,
        device: str = "cpu",
        early_stopping_patience: int = 3,
    ) -> BaselineTrainResult:
        return BaselineTrainResult(
            history=[
                {
                    "epoch": "decoder_feature_copy",
                    "train_loss": _loader_mse_for_model(self, train_loader),
                    "val_loss": _loader_mse_for_model(self, val_loader),
                }
            ],
            best_val_loss=_loader_mse_for_model(self, val_loader),
        )

    def predict_batch(self, batch: dict) -> torch.Tensor:
        decoder = batch["decoder_known"]
        target_values = [decoder[:, :, feature_index].unsqueeze(-1) for feature_index in self.target_indices]
        return torch.cat(target_values, dim=-1)

    def predict_loader(self, loader, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, dict[str, list[Any]]]:
        predictions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        metadata: dict[str, list[Any]] = {"station_id": [], "prediction_start": [], "region_class": [], "target_context": []}
        for batch in loader:
            predictions.append(self.predict_batch(batch).cpu())
            targets.append(batch["target"].cpu())
            metadata["station_id"].extend(batch["station_id"])
            metadata["prediction_start"].extend(batch["prediction_start"])
            metadata["region_class"].extend(batch.get("region_class", []))
            _extend_target_context_metadata(metadata, batch)
        return torch.cat(predictions), torch.cat(targets), metadata

    def save(self, path: str | Path, extra_state: dict[str, Any] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "decoder_feature_names": self.decoder_feature_names,
            "target_columns": self.target_columns,
            "target_source_features": self.target_source_features,
            "extra_state": extra_state or {},
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path, bundle, model_config: dict | None = None) -> "DecoderFeatureBaseline":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        configured_sources = (model_config or {}).get("model", {}).get("target_source_features")
        return cls(
            decoder_feature_names=bundle.decoder_columns,
            target_columns=bundle.target_columns,
            target_source_features=list(configured_sources or checkpoint["target_source_features"]),
        )

    @classmethod
    def load_for_resume(cls, path: str | Path, bundle, model_config: dict | None = None) -> "DecoderFeatureBaseline":
        return cls.load(path, bundle, model_config)


    def feature_importance_frame(self, feature_names: list[str]) -> pd.DataFrame | None:
        rows = []
        for target_column, source_feature in zip(self.target_columns, self.target_source_features):
            rows.append(
                {
                    "target_column": target_column,
                    "horizon_step": "all",
                    "feature_name": f"decoder_known:{source_feature}",
                    "importance": 1.0,
                }
            )
        return pd.DataFrame(rows)


class RidgeRegressionBaseline:
    def __init__(
        self,
        encoder_feature_names: list[str],
        target_columns: list[str],
        prediction_length: int,
        alpha: float = 1.0,
        alpha_grid: list[float] | None = None,
        alpha_selection: dict[str, Any] | None = None,
    ) -> None:
        self.encoder_feature_names = encoder_feature_names
        self.target_columns = target_columns
        self.prediction_length = prediction_length
        self.alpha = alpha
        self.alpha_grid = alpha_grid
        self.alpha_selection = alpha_selection or {}
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
        val_features, val_targets, val_metadata = _collect_regression_tensors_with_metadata(val_loader)
        selection_metric = str(self.alpha_selection.get("metric", "val_mse"))

        history: list[dict[str, float | str]] = []
        best_candidate: tuple[float, float, float, float, torch.Tensor, torch.Tensor] | None = None
        for candidate_alpha in _candidate_alphas(self.alpha, self.alpha_grid):
            weights, bias = _solve_ridge_regression(train_features, train_targets, candidate_alpha)
            train_loss = _mse_loss(_predict_from_regression_features(train_features, weights, bias), train_targets)
            val_prediction = _predict_from_regression_features(val_features, weights, bias)
            val_loss = _mse_loss(val_prediction, val_targets)
            selection_loss = _ridge_alpha_selection_loss(
                prediction=val_prediction,
                target=val_targets,
                metadata=val_metadata,
                prediction_length=self.prediction_length,
                target_count=len(self.target_columns),
                selection_config=self.alpha_selection,
                default_val_loss=val_loss,
            )
            history_row: dict[str, float | str] = {
                "epoch": "closed_form_alpha_search",
                "alpha": candidate_alpha,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            if selection_metric != "val_mse":
                history_row["selection_metric"] = selection_metric
                history_row["selection_loss"] = selection_loss
            history.append(history_row)
            if best_candidate is None or selection_loss < best_candidate[1]:
                best_candidate = (candidate_alpha, selection_loss, val_loss, train_loss, weights, bias)

        if best_candidate is None:
            raise RuntimeError("No ridge alpha candidates were available.")
        self.alpha, selection_loss, val_loss, train_loss, self.weights, self.bias = best_candidate
        selected_row: dict[str, float | str] = {
            "epoch": "closed_form_selected",
            "alpha": self.alpha,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        if selection_metric != "val_mse":
            selected_row["selection_metric"] = selection_metric
            selected_row["selection_loss"] = selection_loss
        history.append(selected_row)
        train_loss = _mse_loss(self._predict_from_features(train_features), train_targets)
        val_loss = _mse_loss(self._predict_from_features(val_features), val_targets)
        return BaselineTrainResult(
            history=history,
            best_val_loss=val_loss,
        )

    def predict_batch(self, batch: dict) -> torch.Tensor:
        features = _flatten_batch_features(batch)
        return self._predict_from_features(features)

    def predict_loader(self, loader, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, dict[str, list[Any]]]:
        predictions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        metadata: dict[str, list[Any]] = {"station_id": [], "prediction_start": [], "region_class": [], "target_context": []}
        for batch in loader:
            predictions.append(self.predict_batch(batch).cpu())
            targets.append(batch["target"].cpu())
            metadata["station_id"].extend(batch["station_id"])
            metadata["prediction_start"].extend(batch["prediction_start"])
            metadata["region_class"].extend(batch.get("region_class", []))
            _extend_target_context_metadata(metadata, batch)
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
            "alpha_grid": self.alpha_grid,
            "alpha_selection": self.alpha_selection,
            "extra_state": extra_state or {},
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path, bundle, model_config: dict | None = None) -> "RidgeRegressionBaseline":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        alpha = float((model_config or {}).get("model", {}).get("alpha", checkpoint.get("alpha", 1.0)))
        alpha_grid = (model_config or {}).get("model", {}).get("alpha_grid", checkpoint.get("alpha_grid"))
        alpha_selection = (model_config or {}).get("model", {}).get("alpha_selection", checkpoint.get("alpha_selection", {}))
        model = cls(
            encoder_feature_names=bundle.encoder_columns,
            target_columns=bundle.target_columns,
            prediction_length=bundle.prediction_length,
            alpha=alpha,
            alpha_grid=[float(value) for value in alpha_grid] if alpha_grid else None,
            alpha_selection=dict(alpha_selection or {}),
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


class HorizonWiseRidgeRegressionBaseline(RidgeRegressionBaseline):
    """Closed-form ridge baseline with independent alpha/model per horizon output."""

    def __init__(
        self,
        encoder_feature_names: list[str],
        target_columns: list[str],
        prediction_length: int,
        alpha: float = 1.0,
        alpha_grid: list[float] | None = None,
        alpha_selection: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            encoder_feature_names=encoder_feature_names,
            target_columns=target_columns,
            prediction_length=prediction_length,
            alpha=alpha,
            alpha_grid=alpha_grid,
            alpha_selection=alpha_selection,
        )
        self.horizon_alphas: dict[str, float] = {}
        self.horizon_metrics: list[dict[str, float | str | int]] = []

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
        target_count = len(self.target_columns)
        output_count = self.prediction_length * target_count
        self.weights = torch.zeros((train_features.shape[1], output_count), dtype=train_features.dtype)
        self.bias = torch.zeros(output_count, dtype=train_features.dtype)
        history: list[dict[str, float | str]] = []
        self.horizon_alphas = {}
        self.horizon_metrics = []

        for output_index in range(output_count):
            horizon_step = output_index // target_count + 1
            target_column = self.target_columns[output_index % target_count]
            train_target = train_targets[:, output_index : output_index + 1]
            val_target = val_targets[:, output_index : output_index + 1]
            best_candidate: tuple[float, float, float, float, torch.Tensor, torch.Tensor, torch.Tensor] | None = None

            for candidate_alpha in _candidate_alphas(self.alpha, self.alpha_grid):
                weights, bias = _solve_ridge_regression(train_features, train_target, candidate_alpha)
                train_prediction = _predict_from_regression_features(train_features, weights, bias)
                val_prediction = _predict_from_regression_features(val_features, weights, bias)
                train_loss = _mse_loss(train_prediction, train_target)
                val_loss = _mse_loss(val_prediction, val_target)
                history.append(
                    {
                        "epoch": "horizon_alpha_search",
                        "target_column": target_column,
                        "horizon_step": float(horizon_step),
                        "alpha": float(candidate_alpha),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    }
                )
                if best_candidate is None or val_loss < best_candidate[1]:
                    best_candidate = (candidate_alpha, val_loss, val_loss, train_loss, weights, bias, val_prediction)

            if best_candidate is None:
                raise RuntimeError("No horizon-wise ridge alpha candidates were available.")
            candidate_alpha, selection_loss, val_loss, train_loss, weights, bias, val_prediction = best_candidate
            self.weights[:, output_index] = weights[:, 0]
            self.bias[output_index] = bias[0]
            key = f"{target_column}:h{horizon_step}"
            self.horizon_alphas[key] = float(candidate_alpha)
            error = val_prediction.reshape(-1) - val_target.reshape(-1)
            self.horizon_metrics.append(
                {
                    "target_column": target_column,
                    "horizon_step": horizon_step,
                    "alpha": float(candidate_alpha),
                    "val_rmse": float(torch.sqrt(torch.mean(torch.square(error))).item()),
                    "val_mae": float(torch.mean(torch.abs(error)).item()),
                    "val_bias": float(torch.mean(error).item()),
                    "val_loss": float(val_loss),
                    "train_loss": float(train_loss),
                    "selection_loss": float(selection_loss),
                }
            )
            history.append(
                {
                    "epoch": "horizon_alpha_selected",
                    "target_column": target_column,
                    "horizon_step": float(horizon_step),
                    "alpha": float(candidate_alpha),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            )

        val_loss = _mse_loss(self._predict_from_features(val_features), val_targets)
        train_loss = _mse_loss(self._predict_from_features(train_features), train_targets)
        history.append({"epoch": "horizon_wise_closed_form_selected", "train_loss": train_loss, "val_loss": val_loss})
        return BaselineTrainResult(history=history, best_val_loss=val_loss)

    def save(self, path: str | Path, extra_state: dict[str, Any] | None = None) -> Path:
        if self.weights is None or self.bias is None:
            raise RuntimeError("HorizonWiseRidgeRegressionBaseline must be fit before saving.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "weights": self.weights,
            "bias": self.bias,
            "alpha": self.alpha,
            "alpha_grid": self.alpha_grid,
            "alpha_selection": self.alpha_selection,
            "horizon_alphas": self.horizon_alphas,
            "horizon_metrics": self.horizon_metrics,
            "extra_state": extra_state or {},
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path, bundle, model_config: dict | None = None) -> "HorizonWiseRidgeRegressionBaseline":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        alpha = float((model_config or {}).get("model", {}).get("alpha", checkpoint.get("alpha", 1.0)))
        alpha_grid = (model_config or {}).get("model", {}).get("alpha_grid", checkpoint.get("alpha_grid"))
        alpha_selection = (model_config or {}).get("model", {}).get("alpha_selection", checkpoint.get("alpha_selection", {}))
        model = cls(
            encoder_feature_names=bundle.encoder_columns,
            target_columns=bundle.target_columns,
            prediction_length=bundle.prediction_length,
            alpha=alpha,
            alpha_grid=[float(value) for value in alpha_grid] if alpha_grid else None,
            alpha_selection=dict(alpha_selection or {}),
        )
        model.weights = checkpoint["weights"].cpu()
        model.bias = checkpoint["bias"].cpu()
        model.horizon_alphas = {str(key): float(value) for key, value in dict(checkpoint.get("horizon_alphas", {})).items()}
        model.horizon_metrics = [dict(row) for row in checkpoint.get("horizon_metrics", [])]
        return model

    @classmethod
    def load_for_resume(cls, path: str | Path, bundle, model_config: dict | None = None) -> "HorizonWiseRidgeRegressionBaseline":
        return cls.load(path, bundle, model_config)

    def horizon_metrics_frame(self) -> pd.DataFrame | None:
        if not self.horizon_metrics:
            return None
        return pd.DataFrame(self.horizon_metrics).sort_values(["target_column", "horizon_step"])


class LightGBMBaseline:
    def __init__(
        self,
        encoder_feature_names: list[str],
        target_columns: list[str],
        prediction_length: int,
        params: dict[str, Any] | None = None,
        param_grid: dict[str, list[Any]] | None = None,
    ) -> None:
        self.encoder_feature_names = encoder_feature_names
        self.target_columns = target_columns
        self.prediction_length = prediction_length
        self.params = params or {}
        self.param_grid = param_grid or {}
        self.model = None
        self.feature_names: list[str] | None = None
        self.grid_search_results: list[dict[str, Any]] = []
        self.best_params: dict[str, Any] = dict(self.params)

    def fit(
        self,
        train_loader,
        val_loader,
        max_epochs: int,
        learning_rate: float,
        device: str = "cpu",
        early_stopping_patience: int = 3,
    ) -> BaselineTrainResult:
        train_features, train_targets, train_sample_weight = _collect_regression_tensors_with_sample_weight(train_loader)
        val_features, val_targets = _collect_regression_tensors(val_loader)
        self.feature_names = [f"feature_{index}" for index in range(train_features.shape[1])]
        train_frame = pd.DataFrame(train_features.numpy(), columns=self.feature_names)
        val_frame = pd.DataFrame(val_features.numpy(), columns=self.feature_names)
        fit_kwargs = {}
        if train_sample_weight is not None:
            fit_kwargs["sample_weight"] = train_sample_weight.numpy()
        history: list[dict[str, float | str]] = []
        best_model = None
        best_val_loss = float("inf")
        best_train_loss = float("inf")
        best_params = dict(self.params)
        candidate_params = _parameter_grid_candidates(self.params, self.param_grid)
        self.grid_search_results = []
        for trial_index, candidate in enumerate(candidate_params, start=1):
            candidate_model = _build_lightgbm_regressor(candidate)
            candidate_model.fit(train_frame, train_targets.numpy(), **fit_kwargs)
            train_prediction = torch.tensor(candidate_model.predict(train_frame), dtype=torch.float32)
            val_prediction = torch.tensor(candidate_model.predict(val_frame), dtype=torch.float32)
            train_loss = _mse_loss(train_prediction, train_targets)
            val_loss = _mse_loss(val_prediction, val_targets)
            result_row = {
                "trial": trial_index,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_rmse": float(torch.sqrt(torch.tensor(val_loss)).item()),
                **candidate,
            }
            self.grid_search_results.append(result_row)
            history.append({"epoch": "lightgbm_grid_trial", **{k: v for k, v in result_row.items() if isinstance(v, (int, float, str))}})
            if val_loss < best_val_loss:
                best_model = candidate_model
                best_val_loss = val_loss
                best_train_loss = train_loss
                best_params = dict(candidate)
        if best_model is None:
            raise RuntimeError("No LightGBM hyperparameter candidates were available.")
        self.model = best_model
        self.best_params = best_params
        self.params = best_params
        history_row: dict[str, float | str] = {"epoch": "lightgbm_selected", "train_loss": best_train_loss, "val_loss": best_val_loss}
        if train_sample_weight is not None:
            history_row["sample_weight_min"] = float(train_sample_weight.min().item())
            history_row["sample_weight_max"] = float(train_sample_weight.max().item())
        history.append(history_row)
        return BaselineTrainResult(history=history, best_val_loss=best_val_loss)

    def predict_batch(self, batch: dict) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("LightGBMBaseline must be fit before prediction.")
        features = _flatten_batch_features(batch)
        prediction = self.model.predict(_features_to_frame(features, self.feature_names))
        tensor = torch.tensor(prediction, dtype=torch.float32)
        return tensor.reshape(features.shape[0], self.prediction_length, len(self.target_columns))

    def predict_loader(self, loader, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, dict[str, list[Any]]]:
        predictions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        metadata: dict[str, list[Any]] = {"station_id": [], "prediction_start": [], "region_class": [], "target_context": []}
        for batch in loader:
            predictions.append(self.predict_batch(batch))
            targets.append(batch["target"].cpu())
            metadata["station_id"].extend(batch["station_id"])
            metadata["prediction_start"].extend(batch["prediction_start"])
            metadata["region_class"].extend(batch.get("region_class", []))
            _extend_target_context_metadata(metadata, batch)
        return torch.cat(predictions), torch.cat(targets), metadata

    def save(self, path: str | Path, extra_state: dict[str, Any] | None = None) -> Path:
        if self.model is None:
            raise RuntimeError("LightGBMBaseline must be fit before saving.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "params": self.params,
            "param_grid": self.param_grid,
            "grid_search_results": self.grid_search_results,
            "best_params": self.best_params,
            "feature_names": self.feature_names,
            "extra_state": extra_state or {},
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path, bundle, model_config: dict | None = None) -> "LightGBMBaseline":
        checkpoint = _load_torch_checkpoint_with_optional_lightgbm_message(path)
        params = dict((model_config or {}).get("model", {}).get("params", checkpoint.get("params", {})))
        param_grid = dict((model_config or {}).get("model", {}).get("param_grid", checkpoint.get("param_grid", {})))
        model = cls(
            encoder_feature_names=bundle.encoder_columns,
            target_columns=bundle.target_columns,
            prediction_length=bundle.prediction_length,
            params=params,
            param_grid=param_grid,
        )
        model.model = checkpoint["model"]
        model.feature_names = checkpoint.get("feature_names")
        model.grid_search_results = [dict(row) for row in checkpoint.get("grid_search_results", [])]
        model.best_params = dict(checkpoint.get("best_params", params))
        return model

    @classmethod
    def load_for_resume(cls, path: str | Path, bundle, model_config: dict | None = None) -> "LightGBMBaseline":
        return cls.load(path, bundle, model_config)

    def grid_search_results_frame(self) -> pd.DataFrame | None:
        if not self.grid_search_results:
            return None
        return pd.DataFrame(self.grid_search_results).sort_values("val_loss")

    def best_params_dict(self) -> dict[str, Any]:
        return dict(self.best_params)

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


class CatBoostBaseline(LightGBMBaseline):
    """Optional CatBoost multi-output baseline for V3 MOS experiments."""

    def fit(
        self,
        train_loader,
        val_loader,
        max_epochs: int,
        learning_rate: float,
        device: str = "cpu",
        early_stopping_patience: int = 3,
    ) -> BaselineTrainResult:
        train_features, train_targets, train_sample_weight = _collect_regression_tensors_with_sample_weight(train_loader)
        val_features, val_targets = _collect_regression_tensors(val_loader)
        self.feature_names = [f"feature_{index}" for index in range(train_features.shape[1])]
        train_frame = pd.DataFrame(train_features.numpy(), columns=self.feature_names)
        val_frame = pd.DataFrame(val_features.numpy(), columns=self.feature_names)
        self.model = _build_catboost_regressor(self.params)
        fit_kwargs = {}
        if train_sample_weight is not None:
            fit_kwargs["sample_weight"] = train_sample_weight.numpy()
        self.model.fit(train_frame, train_targets.numpy(), **fit_kwargs)
        train_prediction = torch.tensor(self.model.predict(train_frame), dtype=torch.float32)
        val_prediction = torch.tensor(self.model.predict(val_frame), dtype=torch.float32)
        train_loss = _mse_loss(train_prediction, train_targets)
        val_loss = _mse_loss(val_prediction, val_targets)
        return BaselineTrainResult(history=[{"epoch": "catboost_fit", "train_loss": train_loss, "val_loss": val_loss}], best_val_loss=val_loss)

    @classmethod
    def load(cls, path: str | Path, bundle, model_config: dict | None = None) -> "CatBoostBaseline":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        params = dict((model_config or {}).get("model", {}).get("params", checkpoint.get("params", {})))
        model = cls(
            encoder_feature_names=bundle.encoder_columns,
            target_columns=bundle.target_columns,
            prediction_length=bundle.prediction_length,
            params=params,
        )
        model.model = checkpoint["model"]
        model.feature_names = checkpoint.get("feature_names")
        return model


class HorizonWiseLightGBMBaseline(LightGBMBaseline):
    """Explicit horizon-wise LightGBM alias.

    LightGBM uses ``MultiOutputRegressor`` under the hood, so each horizon/target
    output already receives an independent estimator. This class keeps that
    behavior while exposing horizon-wise validation metrics as a first-class
    artifact for V2 experiments.
    """

    def __init__(
        self,
        encoder_feature_names: list[str],
        target_columns: list[str],
        prediction_length: int,
        params: dict[str, Any] | None = None,
        param_grid: dict[str, list[Any]] | None = None,
    ) -> None:
        super().__init__(
            encoder_feature_names=encoder_feature_names,
            target_columns=target_columns,
            prediction_length=prediction_length,
            params=params,
            param_grid=param_grid,
        )
        self.horizon_metrics: list[dict[str, float | str | int]] = []

    def fit(
        self,
        train_loader,
        val_loader,
        max_epochs: int,
        learning_rate: float,
        device: str = "cpu",
        early_stopping_patience: int = 3,
    ) -> BaselineTrainResult:
        result = super().fit(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            device=device,
            early_stopping_patience=early_stopping_patience,
        )
        val_features, val_targets = _collect_regression_tensors(val_loader)
        val_prediction = torch.tensor(
            self.model.predict(_features_to_frame(val_features, self.feature_names)),
            dtype=torch.float32,
        )
        self.horizon_metrics = _horizon_output_metrics(
            prediction=val_prediction,
            target=val_targets,
            target_columns=self.target_columns,
            prediction_length=self.prediction_length,
        )
        return result

    def save(self, path: str | Path, extra_state: dict[str, Any] | None = None) -> Path:
        if self.model is None:
            raise RuntimeError("HorizonWiseLightGBMBaseline must be fit before saving.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "params": self.params,
            "param_grid": self.param_grid,
            "grid_search_results": self.grid_search_results,
            "best_params": self.best_params,
            "feature_names": self.feature_names,
            "horizon_metrics": self.horizon_metrics,
            "extra_state": extra_state or {},
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path, bundle, model_config: dict | None = None) -> "HorizonWiseLightGBMBaseline":
        checkpoint = _load_torch_checkpoint_with_optional_lightgbm_message(path)
        params = dict((model_config or {}).get("model", {}).get("params", checkpoint.get("params", {})))
        param_grid = dict((model_config or {}).get("model", {}).get("param_grid", checkpoint.get("param_grid", {})))
        model = cls(
            encoder_feature_names=bundle.encoder_columns,
            target_columns=bundle.target_columns,
            prediction_length=bundle.prediction_length,
            params=params,
            param_grid=param_grid,
        )
        model.model = checkpoint["model"]
        model.feature_names = checkpoint.get("feature_names")
        model.grid_search_results = [dict(row) for row in checkpoint.get("grid_search_results", [])]
        model.best_params = dict(checkpoint.get("best_params", params))
        model.horizon_metrics = [dict(row) for row in checkpoint.get("horizon_metrics", [])]
        return model

    @classmethod
    def load_for_resume(cls, path: str | Path, bundle, model_config: dict | None = None) -> "HorizonWiseLightGBMBaseline":
        return cls.load(path, bundle, model_config)

    def horizon_metrics_frame(self) -> pd.DataFrame | None:
        if not self.horizon_metrics:
            return None
        return pd.DataFrame(self.horizon_metrics).sort_values(["target_column", "horizon_step"])


class HorizonWiseCatBoostBaseline(CatBoostBaseline):
    """Explicit horizon-wise CatBoost alias with per-output estimators via MultiOutputRegressor."""

    def __init__(
        self,
        encoder_feature_names: list[str],
        target_columns: list[str],
        prediction_length: int,
        params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(encoder_feature_names, target_columns, prediction_length, params=params)
        self.horizon_metrics: list[dict[str, float | str | int]] = []

    def fit(self, train_loader, val_loader, max_epochs: int, learning_rate: float, device: str = "cpu", early_stopping_patience: int = 3) -> BaselineTrainResult:
        result = super().fit(train_loader, val_loader, max_epochs, learning_rate, device, early_stopping_patience)
        val_features, val_targets = _collect_regression_tensors(val_loader)
        val_prediction = torch.tensor(self.model.predict(_features_to_frame(val_features, self.feature_names)), dtype=torch.float32)
        self.horizon_metrics = _horizon_output_metrics(
            prediction=val_prediction,
            target=val_targets,
            target_columns=self.target_columns,
            prediction_length=self.prediction_length,
        )
        return result

    def horizon_metrics_frame(self) -> pd.DataFrame | None:
        if not self.horizon_metrics:
            return None
        return pd.DataFrame(self.horizon_metrics).sort_values(["target_column", "horizon_step"])


class ResidualForecastModel:
    """Two-stage forecast: baseline prediction plus residual model prediction."""

    def __init__(
        self,
        baseline_model: Any,
        residual_model: Any,
        baseline_config: dict[str, Any],
        residual_config: dict[str, Any],
    ) -> None:
        self.baseline_model = baseline_model
        self.residual_model = residual_model
        self.baseline_config = baseline_config
        self.residual_config = residual_config

    def fit(
        self,
        train_loader,
        val_loader,
        max_epochs: int,
        learning_rate: float,
        device: str = "cpu",
        early_stopping_patience: int = 3,
        **kwargs: Any,
    ) -> BaselineTrainResult:
        baseline_result = _fit_component_model(
            self.baseline_model,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            device=device,
            early_stopping_patience=early_stopping_patience,
            **kwargs,
        )
        residual_train_loader = _materialize_residual_batches(train_loader, self.baseline_model)
        residual_val_loader = _materialize_residual_batches(val_loader, self.baseline_model)
        residual_result = _fit_component_model(
            self.residual_model,
            train_loader=residual_train_loader,
            val_loader=residual_val_loader,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            device=device,
            early_stopping_patience=early_stopping_patience,
            **kwargs,
        )
        final_val_loss = _loader_mse_for_model(self, val_loader)
        history = _prefixed_history("baseline", baseline_result.history) + _prefixed_history("residual", residual_result.history)
        history.append({"component": "final", "epoch": "residual_forecast_selected", "val_loss": final_val_loss})
        return BaselineTrainResult(history=history, best_val_loss=final_val_loss)

    def predict_batch(self, batch: dict) -> torch.Tensor:
        return self.baseline_model.predict_batch(batch) + self.residual_model.predict_batch(batch)

    def predict_components_batch(self, batch: dict) -> dict[str, torch.Tensor]:
        baseline = self.baseline_model.predict_batch(batch)
        residual = self.residual_model.predict_batch(batch)
        return {"baseline": baseline, "residual": residual, "final": baseline + residual}

    def predict_loader(self, loader, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, dict[str, list[Any]]]:
        predictions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        metadata: dict[str, list[Any]] = {"station_id": [], "prediction_start": [], "region_class": [], "target_context": []}
        for batch in loader:
            predictions.append(self.predict_batch(batch).cpu())
            targets.append(batch["target"].cpu())
            metadata["station_id"].extend(batch["station_id"])
            metadata["prediction_start"].extend(batch["prediction_start"])
            metadata["region_class"].extend(batch.get("region_class", []))
            _extend_target_context_metadata(metadata, batch)
        return torch.cat(predictions), torch.cat(targets), metadata

    def predict_components_loader(self, loader, device: str = "cpu") -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, list[Any]]]:
        component_predictions: dict[str, list[torch.Tensor]] = {"baseline": [], "residual": [], "final": []}
        targets: list[torch.Tensor] = []
        metadata: dict[str, list[Any]] = {"station_id": [], "prediction_start": [], "region_class": [], "target_context": []}
        for batch in loader:
            components = self.predict_components_batch(batch)
            for name, tensor in components.items():
                component_predictions[name].append(tensor.cpu())
            targets.append(batch["target"].cpu())
            metadata["station_id"].extend(batch["station_id"])
            metadata["prediction_start"].extend(batch["prediction_start"])
            metadata["region_class"].extend(batch.get("region_class", []))
            _extend_target_context_metadata(metadata, batch)
        return {name: torch.cat(tensors) for name, tensors in component_predictions.items()}, torch.cat(targets), metadata

    def save(self, path: str | Path, extra_state: dict[str, Any] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "baseline_model": self.baseline_model,
                "residual_model": self.residual_model,
                "baseline_config": self.baseline_config,
                "residual_config": self.residual_config,
                "extra_state": extra_state or {},
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, bundle, model_config: dict | None = None) -> "ResidualForecastModel":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            baseline_model=checkpoint["baseline_model"],
            residual_model=checkpoint["residual_model"],
            baseline_config=dict(checkpoint.get("baseline_config", {})),
            residual_config=dict(checkpoint.get("residual_config", {})),
        )

    @classmethod
    def load_for_resume(cls, path: str | Path, bundle, model_config: dict | None = None) -> "ResidualForecastModel":
        return cls.load(path, bundle, model_config)

    def feature_importance_frame(self, feature_names: list[str]) -> pd.DataFrame | None:
        frames: list[pd.DataFrame] = []
        for component_name, model in (("baseline", self.baseline_model), ("residual", self.residual_model)):
            if not hasattr(model, "feature_importance_frame"):
                continue
            frame = model.feature_importance_frame(feature_names)
            if frame is None or frame.empty:
                continue
            frame = frame.copy()
            frame.insert(0, "component", component_name)
            frames.append(frame)
        if not frames:
            return None
        return pd.concat(frames, ignore_index=True)

    def horizon_metrics_frame(self) -> pd.DataFrame | None:
        frames: list[pd.DataFrame] = []
        for component_name, model in (("baseline", self.baseline_model), ("residual", self.residual_model)):
            if not hasattr(model, "horizon_metrics_frame"):
                continue
            frame = model.horizon_metrics_frame()
            if frame is None or frame.empty:
                continue
            frame = frame.copy()
            frame.insert(0, "component", component_name)
            frames.append(frame)
        if not frames:
            return None
        return pd.concat(frames, ignore_index=True)


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


def _collect_regression_tensors_with_sample_weight(loader) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    feature_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    weight_rows: list[torch.Tensor] = []
    for batch in loader:
        feature_rows.append(_flatten_batch_features(batch))
        target_rows.append(batch["target"].reshape(batch["target"].shape[0], -1).cpu())
        if "sample_weight" in batch:
            weight_rows.append(batch["sample_weight"].reshape(-1).cpu().float())
    weights = torch.cat(weight_rows, dim=0) if weight_rows else None
    if weights is not None and bool(torch.allclose(weights, torch.ones_like(weights))):
        weights = None
    return torch.cat(feature_rows, dim=0), torch.cat(target_rows, dim=0), weights


def _collect_regression_tensors_with_metadata(loader) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[Any]]]:
    feature_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    metadata: dict[str, list[Any]] = {"station_id": [], "prediction_start": [], "region_class": [], "target_context": []}
    for batch in loader:
        feature_rows.append(_flatten_batch_features(batch))
        target_rows.append(batch["target"].reshape(batch["target"].shape[0], -1).cpu())
        metadata["station_id"].extend([str(value) for value in batch.get("station_id", [])])
        metadata["prediction_start"].extend(batch.get("prediction_start", []))
        metadata["region_class"].extend([str(value) for value in batch.get("region_class", [])])
        _extend_target_context_metadata(metadata, batch)
    return torch.cat(feature_rows, dim=0), torch.cat(target_rows, dim=0), metadata


def _extend_target_context_metadata(metadata: dict[str, list[Any]], batch: dict) -> None:
    context = batch.get("target_context")
    if context is None:
        return
    metadata.setdefault("target_context", [])
    metadata["target_context"].extend(context.cpu())


def _load_torch_checkpoint_with_optional_lightgbm_message(path: str | Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as exc:
        if exc.name == "lightgbm":
            raise RuntimeError(
                "Loading a saved LightGBM baseline requires the optional 'lightgbm' dependency. "
                "Install it with `pip install lightgbm` or `pip install -e .[lgbm]` before running inference."
            ) from exc
        raise


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
    try:
        coefficients = torch.linalg.solve(gram, rhs)
    except RuntimeError:
        jitter = torch.eye(gram.shape[0], dtype=gram.dtype) * max(float(alpha), 1.0) * 1e-8
        jitter[-1, -1] = max(float(alpha), 1.0) * 1e-8
        coefficients = torch.linalg.lstsq(gram + jitter, rhs).solution
    return coefficients[:-1], coefficients[-1]


def _candidate_alphas(alpha: float, alpha_grid: list[float] | None) -> list[float]:
    candidates = [float(value) for value in (alpha_grid or [alpha])]
    if alpha not in candidates:
        candidates.append(float(alpha))
    unique = sorted({value for value in candidates if value >= 0.0})
    if not unique:
        raise ValueError("Ridge alpha candidates must include at least one non-negative value.")
    return unique


def _predict_from_regression_features(features: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return features @ weights + bias


def _mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(torch.square(prediction.reshape(target.shape) - target)).item())


def _ridge_alpha_selection_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    metadata: dict[str, list[Any]],
    prediction_length: int,
    target_count: int,
    selection_config: dict[str, Any],
    default_val_loss: float,
) -> float:
    metric = str(selection_config.get("metric", "val_mse"))
    if metric in {"val_mse", "mse"}:
        return default_val_loss
    if metric != "bias_corrected_holdout_mse":
        raise ValueError(f"Unsupported ridge alpha selection metric: {metric}")

    calibration_fraction = float(selection_config.get("calibration_fraction", 0.7))
    if calibration_fraction <= 0.0 or calibration_fraction > 1.0:
        raise ValueError("model.alpha_selection.calibration_fraction must be in the interval (0, 1].")
    if calibration_fraction >= 1.0 or prediction.shape[0] < 2:
        return default_val_loss

    frame = _ridge_validation_frame(
        prediction=prediction,
        target=target,
        metadata=metadata,
        prediction_length=prediction_length,
        target_count=target_count,
    )
    if frame.empty:
        return default_val_loss
    calibration, holdout = _split_alpha_selection_frame(frame, calibration_fraction)
    if calibration.empty or holdout.empty:
        return default_val_loss

    raw_loss = _frame_mse(holdout["prediction"], holdout["actual"])
    corrected = _apply_validation_mean_bias(
        calibration=calibration,
        holdout=holdout,
        mode=str(selection_config.get("correction_mode", "per_station_horizon")),
    )
    corrected_loss = _frame_mse(corrected["prediction"], corrected["actual"])
    if bool(selection_config.get("apply_when_improves", True)):
        return min(raw_loss, corrected_loss)
    return corrected_loss


def _ridge_validation_frame(
    prediction: torch.Tensor,
    target: torch.Tensor,
    metadata: dict[str, list[Any]],
    prediction_length: int,
    target_count: int,
) -> pd.DataFrame:
    prediction_tensor = prediction.reshape(prediction.shape[0], prediction_length, target_count).detach().cpu()
    target_tensor = target.reshape(target.shape[0], prediction_length, target_count).detach().cpu()
    station_ids = [str(value) for value in metadata.get("station_id", [])]
    prediction_starts = metadata.get("prediction_start", [])
    rows: list[dict[str, object]] = []
    for sample_index in range(prediction_tensor.shape[0]):
        station_id = station_ids[sample_index] if sample_index < len(station_ids) else ""
        prediction_start = _metadata_timestamp(prediction_starts[sample_index]) if sample_index < len(prediction_starts) else pd.NaT
        for horizon_index in range(prediction_length):
            for target_index in range(target_count):
                rows.append(
                    {
                        "station_id": station_id,
                        "prediction_start": prediction_start,
                        "horizon_step": horizon_index + 1,
                        "target_index": target_index,
                        "prediction": float(prediction_tensor[sample_index, horizon_index, target_index].item()),
                        "actual": float(target_tensor[sample_index, horizon_index, target_index].item()),
                    }
                )
    return pd.DataFrame(rows)


def _metadata_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _split_alpha_selection_frame(frame: pd.DataFrame, calibration_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values(["prediction_start", "station_id", "horizon_step", "target_index"]).reset_index(drop=True)
    starts = pd.Series(pd.to_datetime(ordered["prediction_start"], utc=True).dropna().sort_values().unique())
    if len(starts) < 2:
        split_index = max(1, min(len(ordered) - 1, int(round(len(ordered) * calibration_fraction))))
        return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()
    split_start_count = max(1, min(len(starts) - 1, int(round(len(starts) * calibration_fraction))))
    cutoff = starts.iloc[split_start_count - 1]
    row_starts = pd.to_datetime(ordered["prediction_start"], utc=True)
    return ordered.loc[row_starts <= cutoff].copy(), ordered.loc[row_starts > cutoff].copy()


def _apply_validation_mean_bias(calibration: pd.DataFrame, holdout: pd.DataFrame, mode: str) -> pd.DataFrame:
    group_columns = _alpha_selection_group_columns(mode)
    bias = (
        calibration.assign(bias=calibration["prediction"].astype(float) - calibration["actual"].astype(float))
        .groupby(group_columns, dropna=False)["bias"]
        .mean()
        .reset_index()
    )
    corrected = holdout.merge(bias, on=group_columns, how="left")
    corrected["prediction"] = corrected["prediction"].astype(float) - corrected["bias"].fillna(0.0).astype(float)
    return corrected


def _alpha_selection_group_columns(mode: str) -> list[str]:
    if mode == "global":
        return ["target_index"]
    if mode == "per_horizon":
        return ["target_index", "horizon_step"]
    if mode == "per_station_horizon":
        return ["target_index", "station_id", "horizon_step"]
    raise ValueError(f"Unsupported ridge alpha selection correction_mode: {mode}")


def _frame_mse(prediction: pd.Series, actual: pd.Series) -> float:
    prediction_values = prediction.astype(float).to_numpy()
    actual_values = actual.astype(float).to_numpy()
    return float(((prediction_values - actual_values) ** 2).mean())


def _parameter_grid_candidates(base_params: dict[str, Any], param_grid: dict[str, list[Any]] | None) -> list[dict[str, Any]]:
    if not param_grid:
        return [dict(base_params)]
    keys = [str(key) for key in param_grid.keys()]
    values = [list(param_grid[key]) for key in keys]
    candidates = []
    for combination in product(*values):
        params = dict(base_params)
        params.update({key: value for key, value in zip(keys, combination)})
        candidates.append(params)
    return candidates or [dict(base_params)]


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


def _build_catboost_regressor(params: dict[str, Any]):
    try:
        from catboost import CatBoostRegressor  # type: ignore
    except ImportError as exc:
        raise RuntimeError("catboost is not installed. Install it to use the catboost baseline.") from exc
    from sklearn.multioutput import MultiOutputRegressor

    default_params = {
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 6,
        "loss_function": "RMSE",
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
    }
    default_params.update(params)
    return MultiOutputRegressor(CatBoostRegressor(**default_params))


def _features_to_frame(features: torch.Tensor, feature_names: list[str] | None) -> pd.DataFrame | Any:
    feature_array = features.numpy()
    if not feature_names:
        return feature_array
    return pd.DataFrame(feature_array, columns=feature_names)


def _horizon_output_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_columns: list[str],
    prediction_length: int,
) -> list[dict[str, float | str | int]]:
    prediction = prediction.reshape(target.shape)
    target_count = len(target_columns)
    rows: list[dict[str, float | str | int]] = []
    for output_index in range(prediction_length * target_count):
        horizon_step = output_index // target_count + 1
        target_column = target_columns[output_index % target_count]
        error = prediction[:, output_index].reshape(-1) - target[:, output_index].reshape(-1)
        rows.append(
            {
                "target_column": target_column,
                "horizon_step": horizon_step,
                "val_rmse": float(torch.sqrt(torch.mean(torch.square(error))).item()),
                "val_mae": float(torch.mean(torch.abs(error)).item()),
                "val_bias": float(torch.mean(error).item()),
                "val_loss": float(torch.mean(torch.square(error)).item()),
            }
        )
    return rows


def _fit_component_model(
    model: Any,
    train_loader,
    val_loader,
    max_epochs: int,
    learning_rate: float,
    device: str,
    early_stopping_patience: int,
    **kwargs: Any,
) -> BaselineTrainResult:
    if hasattr(model, "fit"):
        return model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            device=device,
            early_stopping_patience=early_stopping_patience,
            **kwargs,
        )
    return BaselineTrainResult(history=[], best_val_loss=_loader_mse_for_model(model, val_loader))


def _materialize_residual_batches(loader, baseline_model: Any) -> list[dict[str, Any]]:
    residual_batches: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            baseline_prediction = baseline_model.predict_batch(batch).detach().cpu()
            residual_batch: dict[str, Any] = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    residual_batch[key] = value.detach().cpu().clone()
                else:
                    residual_batch[key] = value
            residual_batch["target"] = batch["target"].detach().cpu() - baseline_prediction
            residual_batches.append(residual_batch)
    return residual_batches


def _loader_mse_for_model(model: Any, loader) -> float:
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            predictions.append(model.predict_batch(batch).detach().cpu())
            targets.append(batch["target"].detach().cpu())
    if not predictions:
        return float("nan")
    return _mse_loss(torch.cat(predictions), torch.cat(targets))


def _prefixed_history(component: str, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"component": component, **dict(row)} for row in history]
