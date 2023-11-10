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
        :param is_first: If True, this is the first layer in our MLP network so don't apply any dropout
            or batch normalization.
        :param is_last: If True, this is the last layer in our MLP network so don't apply any dropout
            or batch normalization.
        """
        super().__init__()

        self.do = do
        self.bn = bn
        self.is_first = is_first
        self.is_last = is_last

        if activation == "relu":
            self.activation_fn = nn.ReLU()
        elif activation == "tanh":
            self.activation_fn = nn.Tanh()
        elif activation == "identity":
            self.activation_fn = nn.Identity()
        else:
            raise NotImplementedError("Only 'relu', 'tanh' and 'identity' activations are supported")

        self.linear = nn.Linear(input_dim, output_dim)

        self.dropout = nn.Dropout(dop)
        self.batchnorm = nn.BatchNorm1d(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass computation for this layer.

        :param x: The input tensor for this layer

        :returns: The output tensor for this layer
        """
        x = self.activation_fn(self.linear(x))
        if self.is_first or self.is_last:
            return x

        if self.bn:
            x = self.batchnorm(x)
        if self.do:
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
            if self.unc_type == "deltaenc":
                first_layer_dim = 2 * input_dim

        layers = [
            MLPLayer(
                self.config["activation"],
                first_layer_dim,
                self.config["hidden_dim"],
                do=False,
                dop=0.0,
                bn=False,
                is_first=True,
                is_last=False,
            )
        ]

        for _ in range(1, self.config["n_layers"] - 1):
            layers.append(
                MLPLayer(
                    self.config["activation"],
                    self.config["hidden_dim"],
                    self.config["hidden_dim"],
                    do=self.config["dropout"],
                    dop=self.config["dropout_prob"],
                    bn=self.config["batchnorm"],
                    is_first=False,
                    is_last=False,
                )
            )
        layers.append(
            MLPLayer(
                "identity",
                self.config["hidden_dim"],
                output_dim,
                do=False,
                dop=0.0,
                bn=False,
                is_first=False,
                is_last=True,
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


class SirenLayer(nn.Module):
    """
    A class representation for a layer of a SIREN neural network.
    """

    def __init__(
        self,
        in_f: int,
        out_f: int,
        do: bool = True,
        dop: float = 0.3,
        bn: bool = True,
        w0: int = 30,
        is_first: bool = False,
        is_last: bool = False,
    ):
        """
        Create a layer of the SIREN neural network.

        :param in_f: The size of the input
        :param out_f: The size of the output
        :param do: If True, apply a dropout technique to this layer. Otherwise, don't.
        :param dop: The probability of an element to be dropped out. This will only be applied
            if `do=True`.
        :param bn: If True, apply a batch normalization over the input. Otherwise, don't.
        :param w0: A s
        :param is_first: If True, this is the first layer in our SIREN network so initialize the weights
            differently.
        :param is_last: If True, this is the last layer in our MLP network so don't apply any dropout
            or batch normalization.
        """
        super().__init__()
        self.do = do
        self.bn = bn
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        if self.do:
            self.dropout = nn.Dropout(dop)
        if self.bn:
            self.batchnorm = nn.BatchNorm1d(out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights for this layer
        """
        b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass computation for this layer.

        :param x: The input tensor for this layer

        :returns: The output tensor for this layer
        """
        # x = self.linear(x)
        # return x if self.is_last else torch.sin(self.w0 * x)
        x = self.linear(x)
        if self.is_last:
            return x

        x = torch.sin(self.w0 * x)
        if self.do:
            x = self.dropout(x)
        if self.bn:
            x = self.batchnorm(x)
        return x


class SIREN(nn.Module):
    """
    Sinusoidal Representation Networks (SIREN) neural network module.
    This uses a sinusoidal activation function to train a model.
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
        Create a SIREN neural network

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

        if self.config["siren_ff"]:
            scale = np.sqrt(self.config["variance"])
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
                input_dim = self.config["mapping_size"] * 4
            else:
                input_dim = self.config["mapping_size"] * 2
        else:
            self.B = None
            if self.unc_type == "deltaenc":
                input_dim *= 2

        layers = [
            SirenLayer(
                input_dim,
                self.config["hidden_dim"],
                do=self.config["dropout"],
                dop=self.config["dropout_prob"],
                bn=self.config["batchnorm"],
                is_first=True,
            )
        ]
        for _ in range(1, self.config["n_layers"] - 1):
            layers.append(
                SirenLayer(
                    self.config["hidden_dim"],
                    self.config["hidden_dim"],
                    do=self.config["dropout"],
                    dop=self.config["dropout_prob"],
                    bn=self.config["batchnorm"],
                )
            )
        layers.append(
            SirenLayer(
                self.config["hidden_dim"],
                output_dim,
                do=self.config["dropout"],
                dop=self.config["dropout_prob"],
                bn=self.config["batchnorm"],
                is_last=True,
            )
        )
        self.siren = nn.Sequential(*layers)

    def input_mapping(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform the input data into a format that can be processed by the SIREN
        neural network.

        :param x: The tensor of input data to transform

        :returns: The tensor of transformed input data
        """
        if self.B is None:
            return x

        x_proj = (2.0 * np.pi * x).float() @ self.B.t()
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass computation for the SIREN neural network

        :param x: The input tensor to the neural network

        :returns: The output tensor computed from the forward pass
        """
        if self.unc_type == "deltaenc":
            out = self.siren(x)
        else:
            h = self.input_mapping(x)
            out = self.siren(h)
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
