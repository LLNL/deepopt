# DeepOpt User Guide

## What is DeepOpt?

DeepOpt is a simple and easy-to-use library for performing Bayesian optimization, leveraging the powerful capabilities of [BoTorch](https://botorch.org/). Its key feature is the ability to use neural networks as surrogate functions during the optimization process, allowing Bayesian optimization to work smoothly even on large datasets and in many dimensions. DeepOpt also provides simplified wrappers for BoTorch fitting and optimization routines.

### Key Commands

The DeepOpt library comes equipped with two cornerstone commands:

1. **Learn:** The `learn` command trains a machine learning model on a given set of data. Users can select between a neural network or Gaussian process (GP) model, with support for additional models in the future. Uncertainty quantification (UQ) is available in all models (neural nets currently use the delta-UQ method), allowing for direct use in a Bayesian optmization workflow. The `learn` command supports multi-fidelity modeling with an arbitrary number of fidelities.

2. **Optimize:**  The `optimize` command takes the previously trained model created through the `learn` command and runs a single Bayesian optimization step, proposing a set of candidate points aimed at improving the value of the objective function (output of the learned model). The user can choose between several available acquisition methods for selecting the candidate points. Support for optimization under input uncertainty and risk is available.
