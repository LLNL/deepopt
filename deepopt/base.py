import os
import numpy as np
import torch
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal

class BaseModel(Model):
    """
    Base class for all models implemented in the repo
    """

    def post_forward(self, means, variances):
        # TODO: maybe the two cases can be merged into one with torch.diag_embed
        assert means.ndim == variances.ndim
        if means.ndim == 2 or means.ndim == 1:
            means_squeeze,variances_squeeze = means.squeeze(), variances.squeeze()
            if means_squeeze.ndim==0:
                means_squeeze = torch.Tensor([means_squeeze])
            if variances_squeeze.ndim==0:
                variances_squeeze = torch.Tensor([variances_squeeze])
            mvn = MultivariateNormal(means_squeeze, torch.diag(variances_squeeze + 1e-6))
        else:
            covar_diag = variances.squeeze(-1) + 1e-6
            covars = torch.zeros(*covar_diag.shape,covar_diag.shape[-1])
            for i in range(covar_diag.shape[-1]):
                covars[...,i,i] = covar_diag[...,i]
            mvn = MultivariateNormal(means.squeeze(-1),covars)
        return mvn

    @property
    def num_outputs(self):
        return self.output_dim

    def get_prediction_with_uncertainty(self, X, **kwargs):
        if X.ndim == 3:
            assert len(kwargs) == 0, "no kwargs can be given if X.ndim == 3"
            preds = self.get_prediction_with_uncertainty(X.view(X.size(0) * X.size(1), X.size(2)))
            return preds[0].view(X.size(0), X.size(1), 1), preds[1].view(X.size(0), X.size(1), 1)
        elif X.ndim ==4:
            assert len(kwargs) == 0, "no kwargs can be given if X.ndim == 4"
            preds = self.get_prediction_with_uncertainty(X.view(X.size(0) * X.size(1) * X.size(2), X.size(3)))
            return preds[0].view(X.size(0), X.size(1), X.size(2), 1), preds[1].view(X.size(0), X.size(1), X.size(2), 1)


    def posterior(self, X, **kwargs):
        # Transformations are applied at evaluation time.
        # An acquisiton's objective funtion will call
        # the model's posterior.
        X = self.transform_inputs(X)
        mvn = self.forward(X, **kwargs)
        return GPyTorchPosterior(mvn)

    def forward(self, X, posterior_transform=None, observation_noise=False, **kwargs):
        use_variances = kwargs.get('use_variances')
        if any([use_variances is None,use_variances is False]):
            means, covs = self.get_prediction_with_uncertainty(X,get_cov=True,original_scale=False,**kwargs)
            try:
                return MultivariateNormal(means,covs+1e-6*torch.eye(covs.shape[-1]))
            except Exception as e:
                print(e)
                print('Trying with stronger regularization (1e-5)')
                try:
                    return MultivariateNormal(means,covs+1e-5*torch.eye(covs.shape[-1]))
                except Exception as e:
                    print(e)
                    print('Trying with even stronger regularization (1e-4)')
                    try:
                        return MultivariateNormal(means,covs+1e-4*torch.eye(covs.shape[-1]))
                    except Exception as e:
                        print(e)
                        print('Trying with yet stronger regularization (1e-3)')
                        return MultivariateNormal(means,covs+1e-3*torch.eye(covs.shape[-1]))
                
        else:
            means, variances = self.get_prediction_with_uncertainty(X, **kwargs)
            return self.post_forward(means, variances)
