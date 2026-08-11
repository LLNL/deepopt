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
