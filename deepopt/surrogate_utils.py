# Surrogate Utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from collections import OrderedDict
from math import pi
from math import cos
from torch.optim import Adam, SGD


device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu' #


class MLPLayer(nn.Module):
    def __init__(self, activation, input_dim, output_dim, do=True, dop=0.3, bn=True, is_first=False, is_last=False):
        super().__init__()

        self.do = do
        self.bn = bn
        self.is_first = is_first
        self.is_last = is_last

        if activation == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation == 'tanh':
            self.activation_fn = nn.Tanh()
        elif activation == 'identity':
            self.activation_fn = nn.Identity()
        else:
            raise NotImplementedError("Only 'relu', 'tanh' and 'identity' activations are supported")

        self.linear = nn.Linear(input_dim, output_dim)

        self.dropout = nn.Dropout(dop)
        self.batchnorm = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.activation_fn(self.linear(x))
        if self.is_first or self.is_last:
            return x
        else:
            if self.bn:
                x = self.batchnorm(x)
            if self.do:
                x = self.dropout(x)
            return x

class MLP(nn.Module):
    def __init__(self, config, unc_type, input_dim, output_dim, device='cpu'):
        super(MLP, self).__init__()
        self.config = config
        self.unc_type = unc_type

        if self.config['ff']:
            scale = np.sqrt(self.config['variance'])#/(input_dim-1)
            if self.config['dist'] == 'uniform':
                mn = -scale
                mx = scale
                self.B = torch.rand((self.config['mapping_size'], input_dim)) * (mx - mn) + mn
            elif self.config['dist'] == 'gaussian':
                self.B = torch.randn((self.config['mapping_size'], input_dim)) * scale
            elif self.config['dist'] == 'laplace':
                rp = np.random.laplace(loc=0., scale =scale, size = (self.config['mapping_size'], input_dim))
                self.B = torch.from_numpy(rp).float()
            self.B = self.B.to(device)
            if self.unc_type == 'deltaenc':
                input_dim = self.config['mapping_size']*4
            else:
                input_dim = self.config['mapping_size']*2
        else:
            self.B = None
            if self.unc_type == 'deltaenc':
                input_dim*=2
                
        layers = [MLPLayer(self.config['activation'], input_dim, self.config['hidden_dim'], do=False, dop=0.0, bn=False, is_first=True, is_last=False)]

        for i in range(1, self.config['n_layers'] - 1):
            layers.append(MLPLayer(self.config['activation'], self.config['hidden_dim'], self.config['hidden_dim'], do=self.config['dropout'], dop=self.config['dropout_prob'], bn=self.config['batchnorm'], is_first=False, is_last=False))
        layers.append(MLPLayer('identity', self.config['hidden_dim'], output_dim, do=False, dop=0.0, bn=False, is_first=False, is_last=True))

        self.mlp = nn.Sequential(*layers)

    def input_mapping(self,x):
        if self.B is None:
            return x
        else:
            x_proj = (2. * np.pi * x).float() @ self.B.t() #torch.matmul(2. * np.pi * x, self.B.t())#
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def forward(self,x):
        if self.unc_type == 'deltaenc':
            out = self.mlp(x)
        else:
            h = self.input_mapping(x)
            out = self.mlp(h)
        #print(self.mlp[8].linear.weight)
        return out

class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, do=True, dop=0.3, bn=True, w0=30, is_first=False, is_last=False):
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
        b = 1 / \
            self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        #x = self.linear(x)
        #return x if self.is_last else torch.sin(self.w0 * x)
        x = self.linear(x)
        if self.is_last:
            return x
        else:
            x = torch.sin(self.w0 * x)
            if self.do:
                x = self.dropout(x)
            if self.bn:
                x = self.batchnorm(x)
            return x

class SIREN(nn.Module):
    def __init__(self, config, unc_type, input_dim, output_dim,device='cpu'):
        super(SIREN, self).__init__()

        self.config = config
        self.unc_type = unc_type

        if self.config['siren_ff']:
            scale = np.sqrt(self.config['variance'])
            if self.config['dist'] == 'uniform':
                mn = -scale
                mx = scale
                self.B = torch.rand((self.config['mapping_size'], input_dim)) * (mx - mn) + mn
            elif self.config['dist'] == 'gaussian':
                self.B = torch.randn((self.config['mapping_size'], input_dim)) * scale
            elif self.config['dist'] == 'laplace':
                rp = np.random.laplace(loc=0., scale =scale, size = (self.config['mapping_size'], input_dim))
                self.B = torch.from_numpy(rp).float()
            self.B = self.B.to(device)
            if self.unc_type == 'deltaenc':
                input_dim = self.config['mapping_size']*4
            else:
                input_dim = self.config['mapping_size']*2
        else:
            self.B = None
            if self.unc_type == 'deltaenc':
                input_dim*=2

        layers = [SirenLayer(input_dim, self.config['hidden_dim'], do=self.config['dropout'], dop=self.config['dropout_prob'], bn=self.config['batchnorm'], is_first=True)]
        for i in range(1, self.config['n_layers'] - 1):
            layers.append(SirenLayer(self.config['hidden_dim'], self.config['hidden_dim'], do=self.config['dropout'], dop=self.config['dropout_prob'], bn=self.config['batchnorm']))
        layers.append(SirenLayer(self.config['hidden_dim'], output_dim, do=self.config['dropout'], dop=self.config['dropout_prob'], bn=self.config['batchnorm'], is_last=True))
        self.siren = nn.Sequential(*layers)

    def input_mapping(self,x):
        if self.B is None:
            return x
        else:
            x_proj = (2. * np.pi * x).float() @ self.B.t()
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def forward(self,x):
        if self.unc_type == 'deltaenc':
            out = self.siren(x)
        else:
            h = self.input_mapping(x)
            out = self.siren(h)
        return out

def create_optimizer(network, config):
    """
    This function instantiates and returns optimizer objects of the input neural network
    """
    if config['opt_type'] == 'Adam':
        optimizer = Adam(network.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay_factor'] if config['weight_decay'] else 0.0)

    elif config['opt_type'] == 'SGD':
        optimizer = SGD(network.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay_factor'] if config['weight_decay'] else 0.0)

    else:
        raise NotImplementedError("Only Adam and SGD optimizers supported as of now")

    return optimizer


def proposed_lr(config, epoch, epoch_per_cycle):
    # Cosine Annealing Learning Rate Update
    # https://github.com/moskomule/pytorch.snapshot.ensembles/blob/master/se.py
    iteration = int(epoch % epoch_per_cycle)
    return config['learning_rate'] * (cos(pi * iteration / epoch_per_cycle) + 1) / 2

def prepare_cut_mix_batch(config, input, target):
    # Generate Mixed Sample
    # https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py
    lam = np.random.beta(config['beta'], config['beta'])
    rand_index = torch.randperm(input.size()[0])
    target_a = target
    target_b = target[rand_index]

    num_dim_mixed = np.random.randint(input.size()[1]//2)
    mix_dim = torch.LongTensor(np.random.choice(range(input.size()[1]), num_dim_mixed))

    input[:, mix_dim] = input[(rand_index),:][:,(mix_dim)]
    return input, target_a, target_b, lam
