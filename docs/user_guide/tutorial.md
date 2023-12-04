# Tutorial: Using neural network surrogates

One of the powerful features of deepopt is the ability to use neural network surrogates in place of Gaussian process surrogates during optimization. In this tutorial, we'll repeat the [Getting Started](./index.md#getting-started-with-deepopt) example, but using neural networks in place of Gaussian process. One key difference is the need for a neural network configuration file that specifies the network architecture, activations functions, and a few other parameters. We'll describe these in detail as we move along.

Neural networks in DeepOpt use the ["delta-UQ"](https://arxiv.org/abs/2110.02197) method for uncertainty quantification. The naming conventions reflect this, so to use neural networks, we set the model_type to "delUQ" and the model class is called `DelUQModel`.

## Create the initial data
Just as in [Getting Started](./index.md#getting-started-with-deepopt), we'll put some initial data in a 'sims.npz' file

```{.py title="generate_simulation_inputs.py" linenums="1"}
import torch
import numpy as np

input_dim = 5
num_points = 10

X = torch.rand(num_points, input_dim)
y = -(X**2).sum(axis=1)

np.savez('sims.npz', X=X, y=y) # (1)
```
1. Save data to file 'sims.npz'

We can now generate the `sims.npz` file with:

```bash
python generate_simulation_inputs.py
```

## Default neural network

From here we can either use the DeepOpt API or we can use the DeepOpt CLI.

If you're using the DeepOpt API, you'll first need to load the `ConfigSettings` class:

```{.py linenums="1" title="run_deepopt.py"}
from deepopt.configuration import ConfigSettings
from deepopt.deepopt_cli import get_deepopt_model

model_type = 'delUQ' # (1)
model_class = get_deepopt_model(model_type=model_type) # (2)
cs = ConfigSettings(model_type=model_type) #(3)
bounds = torch.FloatTensor(input_dim*[[0,1]]).T  # (4)
model = model_class(data_file='sims.npz', bounds=bounds, config_settings=cs)  # (5)
```

1. Set the model type to use throughout the script.
2. Set the model class associated with the selected model type (in this case `DelUQModel`)
3. This sets up the neural network configuration (more generally the model configuration). Since we don't pass a configuration file, the default configuration will be used.
4. Learning and optimizing will take place within these input bounds
5. Model is loaded the same way as with GP, but now we are using `DelUQModel`

Training and optimizing are done as in [Getting Started](./index.md#getting-started-with-deepopt), with the array of new points being recorded in 'suggested_inputs.npy':

=== "DeepOpt API"
```{.py title="run_deepopt.py" linenums=9}
model.learn(outfile=f'learner_{model_type}.ckpt') # (1)
```
1. Train the neural network and save its state to a checkpoint file.

=== "DeepOpt CLI"
```bash
input_dim = 5
bounds = ""
for i in {1..input_dim-1}; do bounds+="[0,1],"; done
bounds+="[0,1]"
deepopt learn -i sims.npz -o learner_delUQ.ckpt -m delUQ -b $bounds
```

The checkpoint files saved by DeepOpt use `torch.save` under the hood. They are python dictionaries and can be viewed using `torch.load`:
```{.py title="view_ckpt.py" linenums=1}
import torch
ckpt = torch.load(f'learner_{model_type}.ckpt')
print(ckpt.keys())
```
The delUQ model has 4 entries in the checkpoint dictionary: `epoch` is the number of epochs the NN was trained for, `state_dict` is a dictionary containing all of the values of the NN weights, biases, and other layer parameters, `B` is the initial transformation to frequency space when using Fourier features, and `opt_state_dict` contains the optimizer parameters.

Now that we saved the trained model, we can use it to propose new candidate points:
=== "DeepOpt API"
```{.py title="run_deepopt.py" linenums=10}
model.optimize(outfile='suggested_inputs.npy', learner_file=f'learner_{model_type}.ckpt', acq_method='EI') # (1)
```
1. Use Expected Improvement to acquire new points based on the model saved in learner_file and save those points as a numpy array in outfile.

=== "DeepOpt CLI"
```bash
deepopt optimize -i sims.npz -o suggested_inputs.npy -l learner_delUQ.ckpt -m delUQ -b $bounds -a EI
```

The saved file `suggested_inputs.npy` is a `numpy` save file containing the array of new points with dimension Nxd (N= # of new points, d = input dimensions). We can view the file using `numpy.load`:
```bash
python -c "import numpy as np; print(np.load('suggested_inputs.npy'))"
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
