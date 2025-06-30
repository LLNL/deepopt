"""
This module contains default values used throughout the deepopt library.
"""

DELUQ_CONFIG = {
    "ff": True,
    "dist": "uniform",
    "mapping_size": 128,
    "n_layers": 4,
    "hidden_dim": 128,
    "activation": "relu",
    "dropout": True,
    "dropout_prob": 0.2,
    "batchnorm": False,
    "w0": 30,
    "activation_first": True,
    "opt_type": "Adam",
    "learning_rate": 0.001,
    "weight_decay": 0,
    "n_epochs": 1000,
    "batch_size": 1000,
    "variance": 0.001,
}

NNENSEMBLE_CONFIG = {"n_estimators": 100,
                     "ff": True,
                     "dist": "uniform",
                     "mapping_size": 128,
                     "n_layers": 4,
                     "hidden_dim": 128,
                     "activation": "relu",
                     "dropout": True,
                     "droupout_prob": 0.2,
                     "batchnorm": False,
                     "w0": 30,
                     "activation_first": True,
                     "opt_type": "Adam",
                     "learning_rate": 0.001,
                     "weight_decay": 0,
                     "n_epochs": 300,
                     "batch_size": 128,
                     "variance": 0.001}

GP_CONFIG = {}


class Defaults:
    """
    Default values for the DeepOpt library. This must be a class for ray tuning.

    :cvar random_seed: The default random seed. `Default value: 4321`
    :cvar k_folds: The default k-folds value. `Default value: 5`
    :cvar model_type: The default model type. Options here are 'GP', 'delUQ', or 'nnEnsemble'.
        `Default value: 'GP'`
    :cvar multi_fidelity: The default value on whether to run multi-fidelity
        settings or not. `Default value: False`
    :cvar num_candidates: The default number of candidates. `Default value: 2`
    :cvar fidelity_cost: The default fidelity cost range. `Default value: '[1,10]'`
    :cvar num_restarts_low: The default value for the number of restarts to use (low).
        This default is used for the KG acquisition method in multi-fidelity runs. `Default
        value: 5`
    :cvar num_restarts_high: The default value for the number of restarts to use (high).
        This default is used for all acquisition methods in single-fidelity runs and non-KG
        acquisition methods in multi-fidelity runs. `Default value: 5`
    :cvar raw_samples_low: The default value for the number of raw samples to use (low).
        `Default value: 512`
    :cvar raw_samples_high: The default value for the number of raw samples to use (high).
        `Default value: 5000`
    :cvar n_fantasies: The default value for the number of fantasy models to construct. `Default value: 128`
    """

    random_seed: int = 4321
    k_folds: int = 5
    model_type: str = "GP"
    multi_fidelity: bool = False
    num_candidates: int = 2
    fidelity_cost: str = "[1,10]"
    num_restarts_low: int = 5
    num_restarts_high: int = 15
    raw_samples_low: int = 512
    raw_samples_high: int = 5000
    n_fantasies: int = 128
