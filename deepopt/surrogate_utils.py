"""
This module contains the neural network modules used throughout
DeepOpt. This includes MLP and SIREN neural networks.
"""
from math import cos, pi
from typing import Any, Dict, Type, Union

import numpy as np
import torch
from torch import nn
from torch.optim import SGD, Adam


class MLPLayer(nn.Module):
    """
    A class representation for a layer of an MLP neural network.
    """

    def __init__(
        self,
        activation: str,
        input_dim: int,
        output_dim: int,
        do: bool = True,
        dop: float = 0.3,
        bn: bool = True,
        w0: int = 30,
        activation_first: bool = True,
        is_first: bool = False,
        is_last: bool = False,
    ):
        """
        Create a layer of the MLP neural network.

        :param activation: The type of activation function to apply to this layer
        :param input_dim: The size of the input sample
        :param output_dim: The size of the output sample
        :param do: If True, apply a dropout technique to this layer. Otherwise, don't.
        :param dop: The probability of an element to be dropped out. This will only be applied
            if `do=True`.
        :param bn: If True, apply a batch normalization over the input. Otherwise, don't.
        :param w0: Weight upscaling for first layer of SIREN network (only used if activation='siren')
        :param activation_first: Whether to apply activation before (if True) or after (if False) dropout
        :param is_first: If True, this is the first layer in our MLP network. Weights in a SIREN network are initialized differently in first layer.
        :param is_last: If True, this is the last layer in our MLP network so don't apply any dropout,
            batch normalization, or activation.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.do = do
        self.bn = bn
        self.is_first = is_first
        self.is_last = is_last
        
        self.linear = nn.Linear(input_dim, output_dim)

        self.dropout = nn.Dropout(dop)
        self.batchnorm = nn.BatchNorm1d(output_dim)
        
        self.activation = activation
        self.activation_first = activation_first

        if activation == "relu":
            self.activation_fn = nn.ReLU()
        elif activation == "tanh":
            self.activation_fn = nn.Tanh()
        elif activation == "identity":
            self.activation_fn = nn.Identity()
        elif activation == 'siren':
            self.w0 = w0
            self.activation_fn = lambda x: torch.sin(self.w0*x)
            self.init_weights()
        else:
            raise NotImplementedError("Only 'relu', 'tanh', 'siren', and 'identity' activations are supported")
        
    def init_weights(self):
        """
        Initialize the weights for this layer
        """
        b = 1 / \
            self.input_dim if self.is_first else np.sqrt(6 / self.input_dim) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass computation for this layer.

        :param x: The input tensor for this layer

        :returns: The output tensor for this layer
        """
        x = self.linear(x)
        if self.is_last:
            return self.w0*x if self.activation=='siren' else x
        else:
            if self.do and not self.activation_first:
                x = self.dropout(x)
            x = self.activation_fn(x)
            if self.bn:
                x = self.batchnorm(x)
            if self.do and self.activation_first:
                x = self.dropout(x)
            return x

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) neural network module.
    This uses a nonlinear activation function to train a model.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        unc_type: str,
        input_dim: int,
        output_dim: int,
        device: str = "cpu",
    ):
        """
        Create an MLP neural network

        :param config: Configuration settings provided by the user
        :param unc_type: Type of encoding. Options: "deltaenc" or anything else
        :param input_dim: The size of the input sample
        :param output_dim: The size of the output sample
        :param device: Which device to run the neural network on.
            Options: 'cpu' or 'gpu'.
        """
        super().__init__()
        self.config = config
        self.unc_type = unc_type
        self.device = device

        if self.config["ff"]:
            scale = np.sqrt(self.config["variance"])  # /(input_dim-1)
            if self.config["dist"] == "uniform":
                mn = -scale
                mx = scale
                self.B = torch.rand((self.config["mapping_size"], input_dim)) * (mx - mn) + mn
            elif self.config["dist"] == "gaussian":
                self.B = torch.randn((self.config["mapping_size"], input_dim)) * scale
            elif self.config["dist"] == "laplace":
                rp = np.random.laplace(loc=0.0, scale=scale, size=(self.config["mapping_size"], input_dim))
                self.B = torch.from_numpy(rp).float()
            self.B = self.B.to(device)
            if self.unc_type == "deltaenc":
                first_layer_dim = self.config["mapping_size"] * 4
            else:
                first_layer_dim = self.config["mapping_size"] * 2
        else:
            self.B = None
            if self.unc_type == 'deltaenc':
                first_layer_dim = 2*input_dim                
                
        layers = []
        for i in range(self.config["n_layers"]):
            is_first = (i == 0)
            is_last = (i == (self.config["n_layers"] - 1))
            input_dim_lyr = first_layer_dim if is_first else self.config["hidden_dim"]
            output_dim_lyr = output_dim if is_last else self.config["hidden_dim"]
            layers.append(
                MLPLayer(
                    activation=self.config['activation'],
                    input_dim=input_dim_lyr,
                    output_dim=output_dim_lyr,
                    do=self.config['dropout'],
                    dop=self.config['dropout_prob'],
                    bn=self.config['batchnorm'],
                    w0=self.config['w0'],
                    activation_first=self.config['activation_first'],
                    is_first=is_first,
                    is_last=is_last,
                )
            )


        self.mlp = nn.Sequential(*layers).to(device)

    def input_mapping(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform the input data into a format that can be processed by the MLP
        neural network.

        :param x: The tensor of input data to transform

        :returns: The tensor of transformed input data
        """
        if self.B is None:
            return x.to(self.device)

        x_proj = (2.0 * np.pi * x).float().to(self.device) @ self.B.t()
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass computation for the MLP neural network

        :param x: The input tensor to the neural network

        :returns: The output tensor computed from the forward pass
        """
        if self.unc_type == "deltaenc":
            out = self.mlp(x.to(self.device))
        else:
            h = self.input_mapping(x.to(self.device))
            out = self.mlp(h)
        return out

def create_optimizer(network: Type[nn.Module], config: Dict[str, Any]) -> Union[Adam, SGD]:
    """
    This function instantiates and returns optimizer objects of the input neural network

    :param network: The input neural network
    :param config: The configuration options provided by the user
    """
    if config["opt_type"] == "Adam":
        optimizer = Adam(
            network.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay_factor"] if config["weight_decay"] else 0.0,
        )

    elif config["opt_type"] == "SGD":
        optimizer = SGD(
            network.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay_factor"] if config["weight_decay"] else 0.0,
        )

    else:
        raise NotImplementedError("Only Adam and SGD optimizers supported as of now")

    return optimizer


def proposed_lr(config, epoch, epoch_per_cycle):
    # Cosine Annealing Learning Rate Update
    # https://github.com/moskomule/pytorch.snapshot.ensembles/blob/master/se.py
    iteration = int(epoch % epoch_per_cycle)
    return config["learning_rate"] * (cos(pi * iteration / epoch_per_cycle) + 1) / 2


def prepare_cut_mix_batch(config, input, target):
    # Generate Mixed Sample
    # https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py
    lam = np.random.beta(config["beta"], config["beta"])
    rand_index = torch.randperm(input.size()[0])
    target_a = target
    target_b = target[rand_index]

    num_dim_mixed = np.random.randint(input.size()[1] // 2)
    mix_dim = torch.LongTensor(np.random.choice(range(input.size()[1]), num_dim_mixed))

    input[:, mix_dim] = input[(rand_index), :][:, (mix_dim)]
    return input, target_a, target_b, lam
