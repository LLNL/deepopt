import builtins
import subprocess
import sys

import numpy as np
import pytest
import torch

pytest.importorskip("botorch")
pytest.importorskip("gpytorch")


def _block_ray_imports(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ray" or name.startswith("ray."):
            raise RuntimeError("simulated Ray import failure")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def _block_tabpfn_imports(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tabpfn" or name.startswith("tabpfn."):
            raise RuntimeError("simulated TabPFN import failure")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_models_import_when_ray_import_fails():
    script = """
import builtins
real_import = builtins.__import__
def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'ray' or name.startswith('ray.'):
        raise RuntimeError('simulated Ray import failure')
    return real_import(name, globals, locals, fromlist, level)
builtins.__import__ = fake_import
import deepopt.models as models
assert models.GPModel.__name__ == 'GPModel'
assert models.NNEnsembleModel.__name__ == 'NNEnsembleModel'
"""

    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_cli_help_import_when_ray_import_fails():
    script = """
import builtins
from click.testing import CliRunner
real_import = builtins.__import__
def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'ray' or name.startswith('ray.'):
        raise RuntimeError('simulated Ray import failure')
    return real_import(name, globals, locals, fromlist, level)
builtins.__import__ = fake_import
import deepopt.deepopt_cli as cli
result = CliRunner().invoke(cli.deepopt_cli, ['--help'])
assert result.exit_code == 0, result.output
assert 'learn' in result.output
assert 'optimize' in result.output
"""

    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_models_import_when_tabpfn_import_fails():
    script = """
import builtins
real_import = builtins.__import__
def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'tabpfn' or name.startswith('tabpfn.'):
        raise RuntimeError('simulated TabPFN import failure')
    return real_import(name, globals, locals, fromlist, level)
builtins.__import__ = fake_import
import deepopt.models as models
assert models.GPModel.__name__ == 'GPModel'
assert models.TabPFNModel.__name__ == 'TabPFNModel'
"""

    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_cli_help_import_when_tabpfn_import_fails():
    script = """
import builtins
from click.testing import CliRunner
real_import = builtins.__import__
def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'tabpfn' or name.startswith('tabpfn.'):
        raise RuntimeError('simulated TabPFN import failure')
    return real_import(name, globals, locals, fromlist, level)
builtins.__import__ = fake_import
import deepopt.deepopt_cli as cli
result = CliRunner().invoke(cli.deepopt_cli, ['--help'])
assert result.exit_code == 0, result.output
assert 'learn' in result.output
assert 'optimize' in result.output
"""

    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_deluq_train_fails_lazily_when_ray_unavailable(single_fidelity_data_file, minimal_nn_config, monkeypatch, tmp_path):
    from deepopt._compat import OptionalDependencyError
    from deepopt.configuration import ConfigSettings
    from deepopt.models import DelUQModel

    settings = ConfigSettings("delUQ")
    settings.config_settings.update(minimal_nn_config)
    model = DelUQModel(
        config_settings=settings,
        data_file=str(single_fidelity_data_file),
        bounds=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        device="cpu",
    )

    def fail_ray():
        raise OptionalDependencyError("delUQ training requires Ray Tune")

    monkeypatch.setattr("deepopt.models.require_ray_for_deluq", fail_ray)

    with pytest.raises(OptionalDependencyError, match="delUQ.*Ray Tune"):
        model.train(str(tmp_path / "model.ckpt"))


def test_tabpfn_train_fails_lazily_when_tabpfn_unavailable(single_fidelity_data_file, monkeypatch, tmp_path):
    from deepopt._compat import OptionalDependencyError
    from deepopt.configuration import ConfigSettings
    from deepopt.models import TabPFNModel

    settings = ConfigSettings("TabPFN")
    model = TabPFNModel(
        config_settings=settings,
        data_file=str(single_fidelity_data_file),
        bounds=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        device="cpu",
    )

    def fail_tabpfn(api="auto"):
        raise OptionalDependencyError("TabPFN support requires the optional tabpfn package")

    monkeypatch.setattr("deepopt.surrogate_tabpfn.require_tabpfn_backend", fail_tabpfn)

    with pytest.raises(OptionalDependencyError, match="TabPFN.*optional tabpfn"):
        model.train(str(tmp_path / "tabpfn.ckpt"))


def test_tabpfn_fake_backend_saves_checkpoint_and_returns_posterior(single_fidelity_data_file, monkeypatch, tmp_path):
    from deepopt.configuration import ConfigSettings
    from deepopt.models import DEEPOPT_CHECKPOINT_KEY, TabPFNModel, load_deepopt_model

    class FakeTabPFNBackend:
        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def __call__(self, train_x, train_y, test_x, categorical_inds=None):
            return test_x.squeeze(-2).sum(dim=-1, keepdim=True)

    class FakeCriterion:
        def to(self, device):
            self.device = device
            return self

        def mean(self, logits):
            return logits

        def variance(self, logits):
            return torch.ones_like(logits) * 0.05

    def fake_load_model_criterion_config(**kwargs):
        return FakeTabPFNBackend(), FakeCriterion(), {"fake": True}

    monkeypatch.setattr(
        "deepopt.surrogate_tabpfn.require_tabpfn_backend",
        lambda api="auto": {"api": "legacy", "load_model_criterion_config": fake_load_model_criterion_config},
    )
    settings = ConfigSettings("TabPFN")
    wrapper = TabPFNModel(
        config_settings=settings,
        data_file=str(single_fidelity_data_file),
        bounds=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        device="cpu",
    )

    model = wrapper.train(str(tmp_path / "tabpfn.ckpt"))
    checkpoint = torch.load(tmp_path / "tabpfn.ckpt")
    posterior = model.posterior(wrapper.full_train_X[:2].unsqueeze(-2))
    reloaded_model = load_deepopt_model(str(tmp_path / "tabpfn.ckpt"), device="cpu")
    reloaded_posterior = reloaded_model.posterior(wrapper.full_train_X[:2].unsqueeze(-2))

    assert checkpoint[DEEPOPT_CHECKPOINT_KEY]["model_type"] == "TabPFN"
    assert checkpoint["tabpfn_backend"] == {"api": "legacy"}
    assert posterior.mean.shape == torch.Size([2, 1, 1])
    assert reloaded_posterior.mean.shape == torch.Size([2, 1, 1])
    assert torch.isfinite(posterior.variance).all()
    assert torch.isfinite(reloaded_posterior.variance).all()


def test_tabpfn_rejects_fantasize_dependent_acquisition_paths(single_fidelity_data_file):
    from deepopt.configuration import ConfigSettings
    from deepopt.models import TabPFNModel

    wrapper = TabPFNModel(
        config_settings=ConfigSettings("TabPFN"),
        data_file=str(single_fidelity_data_file),
        bounds=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        device="cpu",
    )

    with pytest.raises(NotImplementedError, match="KG acquisition.*TabPFN"):
        wrapper.get_candidates(model=object(), acq_method="KG", q=1)
    with pytest.raises(NotImplementedError, match="MaxValEntropy.*one candidate"):
        wrapper.get_candidates(model=object(), acq_method="MaxValEntropy", q=2)


def test_nnensemble_train_does_not_import_ray(single_fidelity_data_file, minimal_nn_config, monkeypatch, tmp_path):
    _block_ray_imports(monkeypatch)

    from deepopt.configuration import ConfigSettings
    from deepopt.models import NNEnsembleModel

    class FakeNNEnsemble:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self):
            return None

        def save_ckpt(self, directory, name, checkpoint_metadata=None):
            self.saved = (directory, name, checkpoint_metadata)

    monkeypatch.setattr("deepopt.models.NNEnsemble", FakeNNEnsemble)
    settings = ConfigSettings("nnEnsemble")
    settings.config_settings.update(minimal_nn_config)
    model = NNEnsembleModel(
        config_settings=settings,
        data_file=str(single_fidelity_data_file),
        bounds=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        device="cpu",
    )

    trained = model.train(str(tmp_path / "ensemble.ckpt"))

    assert isinstance(trained, FakeNNEnsemble)


def test_make_sobol_qmc_normal_sampler_supports_new_style_constructor(monkeypatch):
    import deepopt._compat as compat

    calls = []

    class NewStyleSampler:
        def __init__(self, sample_shape, seed=None):
            calls.append((sample_shape, seed))

    monkeypatch.setattr(compat, "_get_sobol_qmc_normal_sampler", lambda: NewStyleSampler)

    sampler = compat.make_sobol_qmc_normal_sampler(3, seed=7)

    assert isinstance(sampler, NewStyleSampler)
    assert calls == [(torch.Size([3]), 7)]


def test_make_sobol_qmc_normal_sampler_supports_old_style_constructor(monkeypatch):
    import deepopt._compat as compat

    calls = []

    class OldStyleSampler:
        def __init__(self, *args, **kwargs):
            if "sample_shape" in kwargs:
                raise TypeError("old style sampler")
            calls.append((args, kwargs))

    monkeypatch.setattr(compat, "_get_sobol_qmc_normal_sampler", lambda: OldStyleSampler)

    sampler = compat.make_sobol_qmc_normal_sampler(5, seed=11)

    assert isinstance(sampler, OldStyleSampler)
    assert calls == [((5,), {"seed": 11})]
