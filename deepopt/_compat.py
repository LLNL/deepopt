"""Compatibility helpers for optional and version-sensitive dependencies."""
from typing import Any, Callable, Dict, Tuple

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


def require_tabpfn_backend(api: str = "auto") -> Dict[str, Any]:
    api = (api or "auto").lower()
    if api not in {"auto", "legacy", "modern"}:
        raise ValueError("tabpfn_api must be 'auto', 'legacy', or 'modern'.")

    errors = []
    if api in {"auto", "legacy"}:
        try:
            from tabpfn.base import load_model_criterion_config

            return {"api": "legacy", "load_model_criterion_config": load_model_criterion_config}
        except Exception as exc:
            errors.append(f"legacy tabpfn.base API unavailable: {exc}")
            if api == "legacy":
                raise OptionalDependencyError(
                    "TabPFN support requires the optional tabpfn package with the legacy tensor API. "
                    "Install a compatible tabpfn build or choose tabpfn_api='auto'."
                ) from exc

    if api in {"auto", "modern"}:
        try:
            from tabpfn import TabPFNRegressor
        except Exception as exc:
            errors.append(f"modern TabPFNRegressor API unavailable: {exc}")
            raise OptionalDependencyError(
                "TabPFN support requires the optional tabpfn package, but no supported TabPFN backend "
                "could be imported. Install with `pip install 'tabpfn<2'` or `pip install 'deepopt[tabpfn]'`. "
                f"Backend errors: {'; '.join(errors)}"
            ) from exc
        raise OptionalDependencyError(
            "This installed TabPFN exposes the modern TabPFNRegressor API, but DeepOpt requires "
            "an acquisition-safe predictive uncertainty interface. Use a TabPFN version with "
            "tabpfn.base.load_model_criterion_config or extend the modern backend adapter before optimization."
        )

    raise OptionalDependencyError(
        "TabPFN support requires the optional tabpfn package. Install with `pip install 'tabpfn<2'` "
        "or `pip install 'deepopt[tabpfn]'`."
    )


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
