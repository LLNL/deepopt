"""
This module contains the single-fidelity and multi-fidelity GP models.
"""
import os
import warnings
from copy import copy,deepcopy
from typing import Any, Callable, Tuple, Type, Union, List

import numpy as np
import torch
from botorch import settings
from botorch.models.model import Model
from botorch.models.gp_regression_fidelity import SingleTaskGP, SingleTaskMultiFidelityGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.samplers import MCSampler
from gpytorch.distributions import MultivariateNormal
from torch import nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, TensorDataset

from deepopt.configuration import ConfigSettings
from deepopt.surrogate_utils import MLP as Arch
from deepopt.surrogate_utils import create_optimizer

device = "cuda" if torch.cuda.is_available() else "cpu"


class GP:
    """
    The `GP` class is designed as a wrapper around BoTorch's GP models to provide them with 
    functionanility expected in a DeepOpt surrogate.
    """
    
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        bounds: np.ndarray,
        multi_fidelity: bool = False,
        **kwargs,
    ):
        """
        Process parameters to pass to BoTorch GP model.
        
        :param X_train: Input data to train on
        :param y_train: Output data to train on
        :param bounds: Scaling paramaters for X_train. The scaled X_train will be 
        (X_train-bounds[0])/(bounds[1]-bounds[0]) except for the fidelity dimension 
        (which is left untouched).
        :param multi_fidelity: Whether the model is multi-fidelity. If it is, 
        the different fidelities have outputs independently scaled to [0,1] 
        and a different kernel is used for the fidelity dimension.
        """
        # super().__init__()
        self.X_train = X_train.float()
        self.y_train = y_train.float()
        self.bounds = bounds.float()
        self.multi_fidelity = multi_fidelity
        
        self.input_dim = self.X_train.shape[-1]
        self.output_dim = self.y_train.shape[-1]
        
        if self.multi_fidelity:
            self.y_max_dict = {int(i):self.y_train[self.X_train[...,-1]==i].max() for i in self.X_train[...,-1].unique()}
            self.y_min_dict = {int(i):self.y_train[self.X_train[...,-1]==i].min() for i in self.X_train[...,-1].unique()}

        X_gp = self.in_scaler(self.X_train)
        y_gp = self.out_scaler(self.y_train,fidelity_array=self.X_train[...,-1])
        # super_init_dict = {'data_fidelity':self.input_dim-1} if self.multi_fidelity else {}
        # super().__init__(X_gp,y_gp,**super_init_dict,**kwargs)
        self.gp = SingleTaskMultiFidelityGP(X_gp,y_gp,data_fidelity=self.input_dim-1
                                            ) if self.multi_fidelity else SingleTaskGP(X_gp,y_gp)
        
        match_attr_list = ['X_train',
                           'y_train',
                           'bounds',
                           'multi_fidelity',
                           'input_dim',
                           'output_dim',
                           'y_max_dict',
                           'y_min_dict',
                           'in_scaler',
                           'in_scaler_inv',
                           'out_scaler',
                           'out_scaler_inv',
                           'get_prediction_with_uncertainty']
        for attr in match_attr_list:
            if hasattr(self,attr):
                setattr(self.gp,attr,getattr(self,attr))

    def in_scaler(self,X):
        X_scl = (X - self.bounds[0])/(self.bounds[1] - self.bounds[0])
        if self.multi_fidelity:
            X_scl[:,-1] = X[:,-1].round()
        return X_scl
            
    def in_scaler_inv(self,X_scl):
        X = X_scl*(self.bounds[1] - self.bounds[0]) + self.bounds[0]
        if self.multi_fidelity:
            X[:,-1] = X_scl[:,-1].round()
        return X
            
    def out_scaler(self,Y,fidelity_array=None):
        if self.multi_fidelity:
            assert fidelity_array is not None, "Must provide fidelities to scale outputs in multi-fidelity model."
            assert (Y.squeeze(-1).shape==fidelity_array.squeeze(-1).shape
                    ), f"Fidelity array must be same shape as output to scale. Got output shape {Y.shape} and fidelity array shape {fidelity_array.shape}"
            y_maxs = fidelity_array.clone()
            y_mins = fidelity_array.clone()
            for i in fidelity_array.unique():
                y_maxs[fidelity_array==i] = self.y_max_dict[int(i)]
                y_mins[fidelity_array==i] = self.y_min_dict[int(i)]            
            return (Y - y_mins)/(y_maxs - y_mins)
        
        else:
            return (Y - self.y_train.min())/(self.y_train.max() - self.y_train.min())
        
    def out_scaler_inv(self,Y_scl,fidelity_array=None):
        if self.multi_fidelity:
            assert fidelity_array is not None, "Must provide fidelities to unscale outputs in multi-fidelity model."
            assert (Y_scl.squeeze(-1).shape==fidelity_array.squeeze(-1).shape
                    ), f"Fidelity array must be same shape as output to scale. Got output shape {Y_scl.shape} and fidelity array shape {fidelity_array.shape}"
            y_maxs = fidelity_array.clone()
            y_mins = fidelity_array.clone()
            for i in fidelity_array.unique():
                y_maxs[fidelity_array==i] = self.y_max_dict[int(i)]
                y_mins[fidelity_array==i] = self.y_min_dict[int(i)]            
            return Y_scl*(y_maxs - y_mins) + y_mins
        
        else:
            return Y_scl*(self.y_train.max() - self.y_train.min()) + self.y_train.min()
        
    def fit(self):
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_model(mll)
        
    def get_prediction_with_uncertainty(
        self,
        q: torch.Tensor,
        get_cov: bool = False,
        original_scale: bool = True,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a tensor calculate the prediction with uncertainty.

        :param q: A tensor with data we'll calculate prediction with uncertainty for
        :param get_cov: If True, get the covariance. Otherwise, get the variance.
        :param original_scale: If True, apply an inverse scaling transformation to get the scaled
            predictions back to the original scale. Otherwise, don't re-scale the predictions.

        :returns: A tuple containing the mean tensor and the variance (or covariance if
            `get_cov=True`) tensor
        """
        if (original_scale is True) or (original_scale=='input') or (original_scale=='both'):
            q_scl = self.in_scaler(q)
        else:
            q_scl = q
            
        unscl_y = True if (original_scale is True) or (original_scale=='output') or (original_scale=='both') else False
            
        # post = self.posterior(q_scl)
        if hasattr(self,'posterior'):
            post = self.posterior(q_scl)
        elif hasattr(self,'gp'):
            post = self.gp.posterior(q_scl)
        else:
            print('Could not find posterior method to evaluate.')
            return
        if unscl_y:
            mu_scl = post.mean
            mu = self.out_scaler_inv(mu_scl,fidelity_array=q[...,-1]) if self.multi_fidelity else self.out_scaler_inv(mu_scl)
            if self.multi_fidelity:
                y_mins = self.out_scaler_inv(torch.zeros_like(q[...,-1]),fidelity_array=q[...,-1])
                y_maxs = self.out_scaler_inv(torch.ones_like(q[...,-1]),fidelity_array=q[...,-1])
                scl = torch.einsum('...i,...j-->...ij',y_maxs-y_mins,y_maxs-y_mins)
                if get_cov:
                    cov_scl = post.mvn.covariance_matrix
                    return mu, cov_scl*scl
                else:
                    var_scl = post.variance
                    return mu, var_scl*torch.diagonal(scl,dim1=-2,dim2=-1)
            else:
                out_unc = post.mvn.covariance_matrix if get_cov else post.variance
                return mu, out_unc*(self.y_train.max()-self.y_train.min())**2
        else:
            if get_cov:
                return post.mean, post.mvn.covariance_matrix
            else:
                return post.mean, post.variance

# class GP_sf(DeepOptGPMixin,SingleTaskGP):
#     def __init__(
#         self,
#         X_train: np.ndarray,
#         y_train: np.ndarray,
#         bounds: np.ndarray,
#         **kwargs
#     ):
#         super().__init__(X_train=X_train,y_train=y_train,bounds=bounds,multi_fidelity=False,**kwargs)
        
# class GP_mf(DeepOptGPMixin,SingleTaskMultiFidelityGP):
#     def __init__(
#         self,
#         X_train: np.ndarray,
#         y_train: np.ndarray,
#         bounds: np.ndarray,
#         **kwargs
#     ):
#         super().__init__(X_train=X_train,y_train=y_train,bounds=bounds,multi_fidelity=True,**kwargs)
