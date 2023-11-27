# Tutorial: Using neural network surrogates

One of the powerful features of deepopt is the ability to use neural network surrogates in place of Gaussian process surrogates during optimization. In this tutorial, we'll repeat the "Getting Started" example, but using neural networks in place of Gaussian process. One key difference is the need for a neural network configuration file that specifies the network architecture, activations functions, and a few other parameters. We'll describe these in detail as we move along.

Neural networks in DeepOpt use the "delta-UQ" method for uncertainty quantification (ref. to delUQ paper). The naming conventions reflect this, so to use neural networks, we set the model_type to 'delUQ' and the model class is called DelUQModel.

## Create the initial data
Just as in "Getting Started", we'll put some initial data in a 'sims.npz' file

```python
import torch
import numpy as np

input_dim = 5
num_points = 10

X = torch.rand(num_points, input_dim)
y = -(X**2).sum(axis=1)

np.savez('sims.npz', X=X, y=y) # Save data to file 'sims.npz'
```

## Default neural network

Using ConfigSettings without passing a configuration file name will use the default neural net configuration:

```python
from deepopt.configuration import ConfigSettings
from deepopt.models import DelUQModel

cs = ConfigSettings(model_type='delUQ')
bounds = torch.FloatTensor(input_dim*[[0,1]]).T # Learning and optimizing will take place within these input bounds
model = DelUQModel(data_file='sims.npz', bounds=bounds)
```

Training and optimizing are done as in "Getting Started", with the array of new points being recorded in 'suggested_inputs.npy':

```python
model.learn(outfile='learner_GP.ckpt')
model.optimize(outfile='suggested_inputs.npy', learner_file='learner_GP.ckpt', acq_method='EI')
```

## Changing the neural network configuration

The following parameters are available to specify in the neural network configuration file:

### ff
To use "Fourier features" set this to True (otherwise False). When using Fourier features, a Fourier transform with learnable frequencies is implemented prior to the neural network layer. The number of such frequencies is set by the "mapping_size" parameter in the configuration file. Using Fourier features can help the network better learn small-scale features in the data without smearing them out.

### mapping_size
The number of Fourier frequencies to learn when using Fourier features

### dist
The initial distribution of Fourier frequencies. Choices are "uniform", "gaussian", and "laplace".

### variance
The scale of the frequency distribution ("dist") when using Fourier features. A "uniform" distribution is constant between +/- scale, a "gaussian" uses scale as the standard deviation, and the "laplace" distribution uses scale as the exponential decay factor.

This parameter is optimized during hyperparameter tuning, so it is not necessary to set precisely.

### n_layers
The total number of layers in the neural network. This includes the first and last layer, so n_layers=4 will have 2 hidden layers.

### hidden_dim
The number of neurons in each hidden layer (width of the network).

### activation
The activation function to use. Currently supported activations are "relu", "tanh", "identity", and "siren". The "identity" activation will remove any non-linearity in the network, reducing the surrogate to linear regression. The "siren" activaton uses a sine function and initializes the layer weights differently than usual. For more details see the SIREN paper.

### w0
The "w0" parameter to use for initializing weights in a SIREN network. The weight matrix in each layers is w0*W, where W is initalized uniformly on -1/input_dim to 1/input_dim in the first layer and uniformly on -sqrt(6/layer_dim)/w0 to sqrt(6/layer_dim)/w0 in all other layers.

### dropout
Whether to use dropout regularization (True) or not (False)

### dropout_prob
When using dropout, this sets the probability of dropping a neuron.

### activation_first
If True, the activation function is applied first followed by batchnorm and dropout. Otherwise, the order is dropout-activation-batchnorm.

### learning_rate
The learning rate to use in the optimizer. This is optimized during hyperparameter tuning, so it is not necessary to set precisely.

### n_epochs
The number of epochs to train for. We recommend keeping a large number (>=1000) when using smaller datasets.

### batch_size
The batch size for
