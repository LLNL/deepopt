"""Compatibility helpers for optional and version-sensitive dependencies."""
from typing import Any, Callable, Tuple

import torch


class OptionalDependencyError(RuntimeError):
    """Raised when an optional dependency is required by the selected model path."""


def require_ray_for_deluq() -> Tuple[Any, Any, Any, Any]:
    try:
        import ray
        from ray import tune
        from ray.air.config import RunConfig
        from ray.tune.schedulers import ASHAScheduler
    except Exception as exc:
        raise OptionalDependencyError(
            "delUQ training requires Ray Tune, but Ray could not be imported in this environment. "
            "Non-delUQ DeepOpt models do not require Ray; use model_type='GP' or "
            "model_type='nnEnsemble', or install a Ray build compatible with this platform."
        ) from exc
    return ray, tune, RunConfig, ASHAScheduler


def fit_gpytorch_model(mll: Any, **kwargs: Any) -> Any:
    try:
        from botorch.fit import fit_gpytorch_mll
    except Exception:
        from botorch import fit_gpytorch_model as fit_gpytorch_mll
    return fit_gpytorch_mll(mll, **kwargs)


def _get_sobol_qmc_normal_sampler() -> Callable[..., Any]:
    try:
        from botorch.sampling.normal import SobolQMCNormalSampler
    except Exception:
        from botorch.sampling.samplers import SobolQMCNormalSampler
    return SobolQMCNormalSampler


def make_sobol_qmc_normal_sampler(num_samples: int, seed: int = None) -> Any:
    sampler_cls = _get_sobol_qmc_normal_sampler()
    try:
        return sampler_cls(sample_shape=torch.Size([num_samples]), seed=seed)
    except TypeError:
        return sampler_cls(num_samples, seed=seed)
