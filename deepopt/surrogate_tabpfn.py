"""
BoTorch-compatible TabPFN surrogate wrapper.
"""
from typing import Any, Callable, Tuple

import numpy as np
import torch
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal

from deepopt._compat import require_tabpfn_backend
from deepopt.configuration import ConfigSettings
from deepopt.input_scaling import InputScaler, reject_deprecated_original_scale
from deepopt.output_scaling import OutputScaler

_MAX_TOKENS_PER_FORWARD = 3000


class TabPFN(Model):
    """
    BoTorch-compatible wrapper for TabPFN regression.
    """

    def __init__(
        self,
        config: ConfigSettings,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        multi_fidelity: bool = False,
        seed: int = None,
        device: str = "cpu",
        output_scaler: OutputScaler = None,
        input_scaler: InputScaler = None,
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.multi_fidelity = multi_fidelity
        self.input_scaler = input_scaler.to(device) if input_scaler is not None else None
        self.output_scaler = output_scaler.to(device) if output_scaler is not None else None
        self.max_tokens_per_forward = int(config.get_setting("max_tokens_per_forward"))
        if self.max_tokens_per_forward <= 1:
            raise ValueError("max_tokens_per_forward must be greater than 1.")
        self.max_weight = config.get_setting("max_weight")
        self.seed = seed

        X_train = torch.as_tensor(X_train, dtype=torch.float, device=device)
        y_train = torch.as_tensor(y_train, dtype=torch.float, device=device)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        if y_train.shape[-1] != 1:
            raise ValueError("TabPFN currently only supports one scalar output.")

        self.X_train = X_train
        self.y_train = y_train
        self.input_dim = X_train.shape[-1]
        self.output_dim = y_train.shape[-1]
        self._batch_shape = X_train.shape[:-2]
        self.n_train = X_train.shape[-2]
        self.train_inputs = (self.X_train,)

        backend = require_tabpfn_backend(config.get_setting("tabpfn_api"))
        self.backend_api = backend["api"]
        if self.backend_api != "legacy":
            raise RuntimeError("DeepOpt TabPFN optimization currently requires the legacy TabPFN tensor backend.")
        self._init_legacy_backend(backend["load_model_criterion_config"])
        self._prepare_training_context()

    @property
    def batch_shape(self):
        return self._batch_shape

    @batch_shape.setter
    def batch_shape(self, value):
        self._batch_shape = value

    @property
    def num_outputs(self):
        return self.output_dim

    def _init_legacy_backend(self, load_model_criterion_config: Callable[..., Any]) -> None:
        version = self.config.get_setting("tabpfn_version")
        self.tabpfn_model, self.criterion, self.tabpfn_config = load_model_criterion_config(
            model_path=None,
            which="regressor",
            version=version,
            download=True,
            check_bar_distribution_criterion=False,
            cache_trainset_representation=False,
        )
        self.tabpfn_model.to(self.device)
        self.criterion.to(self.device)
        self.tabpfn_model.eval()

    def _prepare_training_context(self) -> None:
        X_flat = self.X_train.moveaxis(-2, 0).reshape(self.n_train, -1)
        y_flat = self.y_train.moveaxis(-2, 0).reshape(self.n_train, -1)
        if len(y_flat) == 0:
            raise ValueError("TabPFN requires at least one training row.")
        hist = torch.histc(y_flat[:, 0].detach().cpu(), bins=10, min=0, max=1)
        bins = torch.linspace(0, 1, steps=11, device=y_flat.device)
        nonzero_bins = torch.nonzero(hist > 0, as_tuple=False).reshape(-1)
        if nonzero_bins.numel() == 0:
            self.X_train_tabpfn = X_flat.to(self.device)
            self.y_train_tabpfn = y_flat.to(self.device)
            self._cap_training_context()
            return
        weights = np.linspace(1, self.max_weight, nonzero_bins.numel())
        max_context = self.max_tokens_per_forward - 1
        context_budget = min(max_context, max(1000, len(y_flat)))
        counts = context_budget * weights / weights.sum()
        counts = np.maximum(1, counts.astype(int))
        x_adjust, y_adjust = [], []
        for bin_index, n_points in zip(nonzero_bins.tolist(), counts):
            left, right = bins[bin_index], bins[bin_index + 1]
            if bin_index == 0:
                locs = y_flat[:, 0] <= right
            elif bin_index == 9:
                locs = y_flat[:, 0] >= left
            else:
                locs = (y_flat[:, 0] >= left) & (y_flat[:, 0] <= right)
            xs, ys = X_flat[locs], y_flat[locs]
            if len(xs) == 0:
                continue
            choice = np.random.choice(len(xs), max(1, n_points), replace=True)
            x_adjust.append(xs[choice])
            y_adjust.append(ys[choice])
        self.X_train_tabpfn = torch.cat(x_adjust, dim=0).to(self.device)
        self.y_train_tabpfn = torch.cat(y_adjust, dim=0).to(self.device)
        self._cap_training_context()

    def _cap_training_context(self) -> None:
        max_context = self.max_tokens_per_forward - 1
        if self.X_train_tabpfn.size(0) > max_context:
            choice = np.random.choice(self.X_train_tabpfn.size(0), max_context, replace=False)
            self.X_train_tabpfn = self.X_train_tabpfn[choice]
            self.y_train_tabpfn = self.y_train_tabpfn[choice]

    def _set_transformed_inputs(self) -> None:
        if hasattr(self, "input_transform") and not self._has_transformed_inputs:
            if hasattr(self, "train_inputs"):
                self._original_train_inputs = self.train_inputs[0]
                with torch.no_grad():
                    X_tf = self.input_transform.preprocess_transform(self.train_inputs[0])
                self.set_train_data(X_tf, strict=False)
                self._has_transformed_inputs = True

    def set_train_data(self, inputs: torch.Tensor = None, targets: torch.Tensor = None, strict: bool = True):
        if inputs is not None:
            if torch.is_tensor(inputs):
                inputs = (inputs,)
            self.train_inputs = inputs
        if targets is not None:
            self.train_targets = targets
        self.prediction_strategy = None

    def posterior(
        self,
        X: torch.Tensor,
        posterior_transform: Callable[[GPyTorchPosterior], GPyTorchPosterior] = None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        X = self.transform_inputs(X)
        mvn = self.forward(X, **kwargs)
        posterior = GPyTorchPosterior(mvn)
        if posterior_transform:
            return posterior_transform(posterior)
        return posterior

    def forward(self, X: torch.Tensor, **kwargs: Any) -> MultivariateNormal:
        means, variances = self.get_prediction_with_uncertainty(
            X,
            original_scale_x=False,
            original_scale_y=False,
            **kwargs,
        )
        covar_diag = variances.squeeze(-1).clamp_min(1e-9) + 1e-6
        means = means.squeeze(-1)
        if covar_diag.ndim == 0:
            covar_diag = covar_diag.reshape(1)
            means = means.reshape(1)
        covars = torch.diag_embed(covar_diag)
        return MultivariateNormal(means, covars)

    def _legacy_predict(self, q_chunk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.tabpfn_model(
            train_x=self.X_train_tabpfn.unsqueeze(-2),
            train_y=self.y_train_tabpfn.unsqueeze(-2),
            test_x=q_chunk.to(self.device).unsqueeze(-2),
            categorical_inds=None,
        )
        return self.criterion.mean(logits), self.criterion.variance(logits)

    def _batched_forward(self, q_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        max_q = int(self.max_tokens_per_forward) - self.X_train_tabpfn.size(0)
        if max_q <= 0:
            raise ValueError("TabPFN context already exceeds max_tokens_per_forward.")
        means_chunks, vars_chunks = [], []
        for start in range(0, q_flat.size(0), max_q):
            q_chunk = q_flat[start : start + max_q]
            mean, variance = self._legacy_predict(q_chunk)
            means_chunks.append(mean.reshape(q_chunk.shape[0], self.output_dim))
            vars_chunks.append(variance.reshape(q_chunk.shape[0], self.output_dim))
        return torch.cat(means_chunks, 0), torch.cat(vars_chunks, 0)

    def get_prediction_with_uncertainty(
        self,
        q: torch.Tensor,
        get_cov: bool = False,
        *args: Any,
        original_scale_x: bool = True,
        original_scale_y: bool = True,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        reject_deprecated_original_scale(args, kwargs)
        q = torch.as_tensor(q, dtype=torch.float, device=self.device)
        if original_scale_x:
            if self.input_scaler is None:
                raise RuntimeError("original_scale_x=True requires this model to have an input_scaler.")
            q = self.input_scaler.transform(q)
        input_shape = q.shape
        if q.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input_dim {self.input_dim}, found tensor of shape {q.shape}.")
        if q.shape[-len(self.batch_shape) - 2 : -2] != self.batch_shape:
            if q.shape[-len(self.batch_shape) - 2 : -2] == torch.Size(len(self.batch_shape) * [1]):
                q = q.expand(*q.shape[: -len(self.batch_shape) - 2], *self.batch_shape, *q.shape[-2:])
            else:
                q = q.expand(*self.batch_shape, *q.shape)
                for _ in range(len(self.batch_shape)):
                    q = q.moveaxis(0, -3)
        q_move = q.moveaxis(-2, 0)
        samples_shape = q_move.shape[: -len(self.batch_shape) - 1]
        q_combine_samples = q_move.reshape(-1, *self.batch_shape, self.input_dim)
        q_flat = q_combine_samples.reshape(q_combine_samples.shape[0], -1)
        mu_flat, var_flat = self._batched_forward(q_flat)
        mu = mu_flat.reshape(*samples_shape, *self.batch_shape, self.output_dim).moveaxis(0, -2)
        var = var_flat.reshape(*samples_shape, *self.batch_shape, self.output_dim).moveaxis(0, -2)
        if mu.shape[:-1] != input_shape[:-1] or var.shape[:-1] != input_shape[:-1]:
            raise RuntimeError("TabPFN prediction reshaping failed.")
        if original_scale_y:
            if self.output_scaler is None:
                raise RuntimeError("original_scale_y=True requires this model to have an output_scaler.")
            mu = self.output_scaler.inverse_transform(mu, q)
            var = self.output_scaler.inverse_variance(var, q)
        if get_cov:
            cov = torch.diag_embed(var.squeeze(-1))
            return mu.squeeze(-1), cov
        return mu, var
