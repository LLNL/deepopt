# Overview

## What is DeepOpt?

Deepopt is a simple and easy-to-use library for performing Bayesian optimization, leveraging the powerful capabilities of BoTorch. Its key feature is the ability to use neural networks as surrogate functions during the optimization process, allowing Bayesian optimization to work smoothly even on large datasets and in many dimensions. Deepopt also provides simplified wrappers for BoTorch fitting and optimization routines.

### Key Commands

The DeepOpt library comes equipped with two cornerstone commands:

1. **Learn:** The `learn` command trains a machine learning model on a given set of data. Users can select between a neural network or Gaussian process (GP) model, with support for additional models in the future. Uncertainty quantification is available in all models (neural nets currently use the delta-UQ method), allowing for direct use in a Bayesian optmization workflow. The `learn` command supports multi-fidelity modeling with an arbitrary number of fidelities.

2. **Optimize:**  The `optimize` command takes the previously trained model created through the `learn` command and runs a single Bayesian optimization step, proposing a set of candidate points aimed at improving the value of the objective function (output of the learned model). The user can choose between several available acquisition methods for selecting the candidate points. Support for optimization under input uncertainty and risk is available.

## Getting Started with DeepOpt

Choose an objective function and "simulate" some initial data

```python
import torch
import numpy as np

input_dim = 5
num_points = 10

X = torch.rand(num_points, input_dim)
y = -(X**2).sum(axis=1)

np.savez('sims.npz', X=X, y=y) # Save data to file 'sims.npz'
```

Load the deepopt class

```python
from deepopt.deepopt_cli import DeepoptConfigure

bounds = torch.FloatTensor(input_dim*[[0,1]]).T # Learning and optimizing will take place within these input bounds
dc = DeepoptConfigure(data_file='sims.npz', bounds=bounds)
```

Train a GP surrogate on the data

```python
dc.learn(outfile='learner_GP.ckpt', model_type='GP')
```

Propose new points using Expected Improvement (EI)

```python
dc.optimize(outfile='suggested_inputs.npy', learner_file='learner_GP.ckpt', acq_method='EI', model_type='GP')
```

The array of new points is recorded in `suggested_inputs.npy`
