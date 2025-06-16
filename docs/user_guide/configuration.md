# Configuration Settings

!!! note

    Currently only delUQ models can be configured; GP models will be run as-is.

The DeepOpt library allows users to define custom configurations for training your model and conducting Bayesian optimization. To ensure a flexible and user-friendly experience, DeepOpt supports configuration through YAML and JSON files. This guide is designed to walk you through the available options and best practices for setting up your configuration files.

## The Base Configuration Options

There are several configuration options that are standard in a configuration file for delUQ models. Below is a table containing each of these options, a description of what they do, and their default values:

| Option           | Description | Default   |
| ------------     | ----------- | -------   |
| ff               | To use "Fourier features" set this to True (otherwise False). When using Fourier features, a Fourier transform with learnable frequencies is implemented prior to the neural network layer. The number of such frequencies is set by the `mapping_size` parameter in the configuration file. Using Fourier features can help the network better learn small-scale features in the data without smearing them out.        | True      |
| activation       | The activation function to use. Currently supported activations are "relu", "tanh", "identity", and "siren". The "identity" activation will remove any non-linearity in the network, reducing the surrogate to linear regression. The "siren" activaton uses a sine function and initializes the layer weights differently than usual. For more details see [the SIREN paper.](https://arxiv.org/abs/2006.09661)        | relu      |
| w0 | The "w0" parameter to use for initializing weights in a SIREN network. The weight matrix in each layers is w0*W, where W is initalized uniformly on -1/input_dim to 1/input_dim in the first layer and uniformly on -sqrt(6/layer_dim)/w0 to sqrt(6/layer_dim)/w0 in all other layers. | 30 |
| n_layers         | The total number of layers in the neural network. This includes the first and last layer, so `n_layers=4` will have 2 hidden layers.        | 4         |
| hidden_dim       | The number of neurons in each hidden layer (width of the network).        | 128       |
| mapping_size     | The number of Fourier frequencies to learn when using Fourier features        | 128       |
| dropout          | Whether to use dropout regularization (True) or not (False)        | True      |
| dropout_prob     | When using `dropout`, this sets the probability of dropping a neuron.        | 0.2       |
| activation_first | When `dropout` is `True`, this sets whether neurons are dropped before or after applying the activation function (it has no effect of `dropout` is `False`). If `True`, the activation function is applied first followed by batchnorm (if `batchnorm` is `True`) and dropout regularization. Otherwise, the dropout regularization is performed first followed by the activation function and batchnorm (if `batchnorm` is `True`) regularization.        | True      |
| learning_rate    | The learning rate to use in the optimizer. This is optimized during hyperparameter tuning, so it is not necessary to set precisely.        | 0.001     |
| n_epochs         | The number of epochs to train for. We recommend keeping a large number (>=1000) when using smaller datasets.        | 1000      |
| batch_size       | The batch size to use during training. If larger than dataset size, the entire dataset will be used as a single batch during each epoch of training.        | 1000      |
| dist             | The initial distribution of Fourier frequencies. Choices are "uniform", "gaussian", and "laplace".        | uniform   |
| opt_type         | Optimizer to use. Choices are "Adam" for the Adam optimizer and "SGD" for stochastic gradient descent.        | Adam      |
| variance         | The scale of the frequency distribution ("dist") when using Fourier features. A "uniform" distribution is constant between +/- scale, a "gaussian" uses scale as the standard deviation, and the "laplace" distribution uses scale as the exponential decay factor.</br></br> This parameter is optimized during hyperparameter tuning, so it is not necessary to set precisely.        | 0.0015625 |
| batchnorm        | Whether to use batchnorm regularization (True) or not (False)        | False     |
| weight_decay     | Strength of weight decay (L2 penalty) to use during optimization        | 0     |