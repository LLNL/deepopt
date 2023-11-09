"""
The `base` module contains the base class for all models used throughout the DeepOpt
codebase.
"""
import os
import numpy as np
from typing import Any, Callable, Tuple

import torch
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal

class BaseModel(Model):
    """
    Base class for all models implemented in the repo
    """

    def post_forward(self, means: torch.Tensor, variances: torch.Tensor) -> MultivariateNormal:
        """
        Post processing of the mean and variance tensors returned by the `forward` method.
        Here we use the mean and variance tensors to create a multivariate normal object.

        :param means: A tensor of means from our predictions
        :param variances: A tensor of variances from our predictions

        :returns: A multivariate normal object calculated from `means` and `variances`
        """
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

        # elif means.ndim == 3:
        #     assert means.size(-1) == variances.size(-1) == 1
        #     try:
        #         mvn = MultivariateNormal(means.squeeze(-1), torch.diag_embed(variances.squeeze(-1) + 1e-6))
        #     except RuntimeError:
        #         print('RuntimeError')
        #         print(torch.diag_embed(variances.squeeze(-1)) + 1e-6)
        #     else:
        #         mvn = MultivariateNormal(means.squeeze(-1), torch.diag_embed(variances.squeeze(-1) + 1e-6))
                
        # elif means.ndim > 3:
        #     assert means.size(-1) == variances.size(-1) == 1
        #     try:
        #         covar_diag = variances.squeeze(-1) + 1e-6
        #         covars = torch.zeros(*covar_diag.shape,covar_diag.shape[-1])
        #         for i in range(covar_diag.shape[-1]):
        #             covars[...,i,i] = covar_diag[...,i]
        #         mvn = MultivariateNormal(means.squeeze(-1),covars)
        #     except RuntimeError:
        #         print('RuntimeError')
        #         print(covar_diag)
        #     else:
        #         mvn = MultivariateNormal(means.squeeze(-1), torch.diag_embed(variances.squeeze(-1) + 1e-6))

        # else:
        #     raise NotImplementedError("Something is wrong, just cmd+f this error message and you can start debugging.")

            
#        elif means.ndim == 3:
#            assert means.size(-1) == variances.size(-1) == 1
#            try:
#                mvn = MultivariateNormal(means.squeeze(-1), torch.diag_embed(variances.squeeze(-1) + 1e-6))
#            except RuntimeError:
#                print('RuntimeError')
#                print(torch.diag_embed(variances.squeeze(-1)) + 1e-6)
#
#        else:
#            raise NotImplementedError("Something is wrong, just cmd+f this error message and you can start debugging.")

        return mvn

    @property
    def num_outputs(self):
        return self.output_dim

    def get_prediction_with_uncertainty(self, X: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a tensor calculate the prediction with uncertainty.

        :param X: A tensor with data we'll calculate prediction with uncertainty for

        :returns: A tuple of tensors where the first tensor represents the mean of the predictions
            and the second represents either the variance or co-variance of the predictions
        """
        if X.ndim == 3:
            assert len(kwargs) == 0, "no kwargs can be given if X.ndim == 3"
            preds = self.get_prediction_with_uncertainty(X.view(X.size(0) * X.size(1), X.size(2)))
            return preds[0].view(X.size(0), X.size(1), 1), preds[1].view(X.size(0), X.size(1), 1)
        elif X.ndim == 4:
            assert len(kwargs) == 0, "no kwargs can be given if X.ndim == 4"
            preds = self.get_prediction_with_uncertainty(X.view(X.size(0) * X.size(1) * X.size(2), X.size(3)))
            return preds[0].view(X.size(0), X.size(1), X.size(2), 1), preds[1].view(X.size(0), X.size(1), X.size(2), 1)


    def posterior(
        self,
        X: torch.Tensor,
        posterior_transform: Callable[[GPyTorchPosterior], GPyTorchPosterior] = None,
        # observation_noise: bool = False,
        **kwargs
    ) -> GPyTorchPosterior:
        """
        Computes a posterior based on GPyTorch's multi-variate Normal distributions.
        Posterior transformation is done if a `posterior_transform` function is provided.

        :param X: A batch_shape x q x d-dim Tensor, where d is the dimension of the feature
            space and q is the number of points considered jointly.
        :param posterior_transform: An optional function to transform the computed posterior
            before returning
        
        :returns: A GPyTorchPosterior object with information on the posterior we calculated
        """
        # Transformations are applied at evaluation time.
        # An acquisiton's objective funtion will call
        # the model's posterior.
        X = self.transform_inputs(X)
        mvn = self.forward(X, **kwargs)
        if posterior_transform:
            return posterior_transform(GPyTorchPosterior(mvn))
        else:
            return GPyTorchPosterior(mvn)

    def forward(self, X: torch.Tensor, **kwargs) -> MultivariateNormal:
        """
        Compute the model output at X with uncertainties, then use that to
        compute a multivariate normal.

        :param X: A batch_shape x q x d-dim Tensor, where d is the dimension of the feature
            space and q is the number of points considered jointly.

        :returns: A multivariate normal object computed using `X`
        """
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
