"""
This module establishes the entrypoint to the DeepOpt library and handles the
functionality for learning and optimizing.
"""
import click
import json
import torch
import yaml
import os
import psutil
import numpy as np
import random
import ray
import warnings
from botorch import fit_gpytorch_model
from botorch.acquisition import PosteriorMean, qExpectedImprovement, qNoisyExpectedImprovement
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient, qKnowledgeGradient
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
from botorch.acquisition.risk_measures import CVaR, VaR, RiskMeasureMCObjective
from botorch.acquisition.objective import ExpectationPosteriorTransform
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.models.cost import AffineFidelityCostModel
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP, SingleTaskGP
from botorch.models.model import Model
from botorch.models.transforms.input import InputPerturbation
from botorch.models.transforms.outcome import Standardize
from botorch.models.deterministic import DeterministicModel
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed
from botorch.sampling.qmc import MultivariateNormalQMCEngine
from botorch.sampling.samplers import SobolQMCNormalSampler
from dataclasses import dataclass
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from numpy import ndarray
from os import getcwd
from os.path import join, basename, dirname
from ray import tune
from ray.air.config import RunConfig
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import KFold
from torch import save, from_numpy, manual_seed
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from types import SimpleNamespace
from typing import Dict, Any, Union, Type, List, Mapping, Optional, Tuple


DEVELOP = False
DEEPOPT_PATH = ""


def set_deepopt_path():
    """
    Set the deepopt path if necessary. This will set the deepopt path
    provided with the `--develop` flag as your path to the deepopt library.
    """
    if DEVELOP:
        import sys
        sys.path.insert(0, DEEPOPT_PATH)

        import deepopt
        print(f"Sourcing deepopt from {deepopt.__file__}.")


class FidelityCostModel(DeterministicModel):
    """
    The cost model for multi-fidelity runs
    """

    def __init__(self, fidelity_weights: ndarray):
        """
        Initialize the fidelity cost model with the weights for different fidelities.
        
        :param fidelity_weights: An ndarray of weight values for different fidelities
        """
        super().__init__()
        self._num_outputs = 1
        self.fidelity_weights = torch.Tensor(fidelity_weights)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute the fidelity cost based on the provided input tensor.

        Given an input tensor X, extract the last element along the last dimension, round it to the nearest integer,
        and use this value as an index to retrieve the corresponding fidelity weight from the pre-defined
        fidelity weights tensor. Return the retrieved weight as a tensor with an additional dimension.

        :param X: The input tensor representing the data for the computation.

        :returns: A tensor containing the fidelity weight for the provided input, expanded to have an additional dimension.
        """
        return self.fidelity_weights[X[...,-1].round().long()].unsqueeze(-1)


class ConditionalOption(click.Option):
    """
    A custom click option to represent a conditional option.
    Conditional options are click options that depend on independent option(s)
    and therefore cannot be used without said independent option(s).
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the conditional option by saving which independent option(s)
        this depends on. In some cases we may also need to save what the independent
        option(s) must be equal to so we'll grab that value too if necessary.
        """
        self.depends_on = kwargs.pop("depends_on")
        self.equal_to = kwargs.pop("equal_to", None)
        kwargs['help'] = f"[NOTE: This argument is only used if {self.depends_on} is used.] " + kwargs.get('help','')
        super().__init__(*args, **kwargs)


    def handle_parse_result(
        self, ctx: click.Context, opts: Mapping[str, Any], args: List[str]
    ) -> Tuple[Any, List[str]]:
        """
        After click parses the result, validate the result and check if our conditions
        are satisfied to use this option.

        :param ctx: A click context object storing the params passed in by the user
        :param opts: A mapping of options provided for this option
        :param args: A list of args provided for this option

        :returns: A tuple containing the value of this option that was given by the
            user and the args that the user provided
        """
        value, args = super().handle_parse_result(ctx, opts, args)
        is_conditional_opt_used = self.name in opts
        is_dependency_used = False if not ctx.params.get(self.depends_on, None) else True
        if is_dependency_used and self.equal_to is not None:
            is_dependency_used = ctx.params.get(self.depends_on) == self.equal_to
        if is_conditional_opt_used and not is_dependency_used:
            if self.equal_to is not None:
                click.echo(f"Option {self.name} will not be used and is set to None. Used only when {self.depends_on} is {self.equal_to}.")
            else:
                click.echo(f"Option {self.name} will not be used and is set to None. Used only when {self.depends_on} is not None.")
            value = None
            ctx.params[self.name] = value
        return value, args
    
class Defaults:
    """
    Default values for the DeepOpt library

    :cvar random_seed: The default random seed. `Default value: 4321`
    :cvar k_folds: The default k-folds value. `Default value: 5`
    :cvar model_type: The default model type. Options here are 'GP' or 'delUQ'.
        `Default value: 'GP'`
    :cvar multi_fidelity: The default value on whether to run multi-fidelity
        settings or not. `Default value: False`
    :cvar num_candidates: The default number of candidates. `Default value: 2`
    :cvar fidelity_cost: The default fidelity cost range. `Default value: '[1,10]'`
    :cvar num_restarts_low: The default value for the number of restarts to use (low).
        This default is used for the KG acquisition method in multi-fidelity runs. `Default
        value: 5`
    :cvar num_restarts_high: The default value for the number of restarts to use (high).
        This default is used for all acquisition methods in single-fidelity runs and non-KG
        acquisition methods in multi-fidelity runs. `Default value: 5`
    :cvar raw_samples_low: The default value for the number of raw samples to use (low).
        `Default value: 512`
    :cvar raw_samples_high: The default value for the number of raw samples to use (high).
        `Default value: 5000`
    """
    random_seed: int = 4321
    k_folds: int = 5
    model_type: str = 'GP'
    multi_fidelity: bool = False
    num_candidates: int = 2
    fidelity_cost: str = '[1,10]'
    num_restarts_low: int = 5
    num_restarts_high: int = 15
    raw_samples_low: int = 512
    raw_samples_high: int = 5000
    
@dataclass
class DeepoptConfigure:
    """
    The heart of the DeepOpt library.

    This class handles training the dataset, loading the model, and
    obtaining the candidates. Both the `deepopt-c learn` and the
    `deepopt-c optimize` calls will go through this class to handle
    their processing.

    :cvar config_file: A YAML file with configuration values to use throughout the
        learn/optimize processes.
    :cvar data_file: A .npz or .npy file containing the data to use as input
    :cvar random_seed: The random seed to use when training and optimizing
    :cvar bounds: Reasonable limits on where to do your optimization search
    :cvar multi_fidelity: True if we're doing a multi-fidelity run, False otherwise
    :cvar num_fidelities: The number of fidelities to use if we're doing a
        multi-fidelity run. `Default: None`
    :cvar kfolds: The number of kfolds to use when training a delUQ surrogate.
        `Default: None`
    :cvar full_train_X: The full input dataset. This is read in from `data_file`.
        `Default: None`
    :cvar full_train_Y: The full output dataset. This is read in from `data_file`.
        `Default: None`
    :cvar input_dim: The dimensions of `full_train_X`. `Default: None`
    :cvar output_dim: The dimensions of `full_train_Y`. `Default: None`
    :cvar config: The configuration options read in from `config_file`. `Default: None`
    :cvar device: The device to run on. This option is read in from `config_file`.
        Options for this configuration are `cpu` and `gpu`. `Default: None`
    :cvar target: Whether to fit the neural network with the y that pairs with the x or
        to the difference y-Y. This option is read in from `config_file`. Options for this
        configuration are `y`, `dy`, and `None`. `Default: None`
    :cvar target_fidelities: Explicitly states our target (highest) fidelity. This is
        saved in a dict format since it's necessary for BoTorch. `Default: None`
    """

    config_file: str
    data_file: str
    bounds: ndarray
    random_seed: int = Defaults.random_seed
    multi_fidelity: bool = Defaults.multi_fidelity
    num_fidelities: int = None
    k_folds: int = Defaults.k_folds
    full_train_X: ndarray = None
    full_train_Y: ndarray = None
    input_dim: int = None
    output_dim: int = None
    config: Dict[str, Any] = None
    device: str = 'cpu'
    target: str = 'dy'
    target_fidelities: Dict[int, float] = None

    def __post_init__(self) -> None:

        input_data = np.load(self.data_file)
        self.X_orig = from_numpy(input_data["X"]).float()
        self.Y_orig = from_numpy(input_data["y"]).float()
        if len(self.Y_orig.shape)==1:
            self.Y_orig = self.Y_orig.reshape(-1,1)
        self.full_train_X = (self.X_orig-self.bounds[0])/(self.bounds[1]-self.bounds[0])
        if self.multi_fidelity:
            self.full_train_X[:,-1] = self.X_orig[:,-1].round()
            self.num_fidelities = int(self.bounds[1,-1]) + 1
        else:
            self.num_fidelities = 1
            
        self.full_train_X = self.full_train_X.to(self.device)
        self.full_train_Y = self.Y_orig.clone().to(self.device)
        
        self.input_dim = self.full_train_X.size(-1)
        self.output_dim = self.full_train_Y.shape[-1]
        assert self.output_dim==1, "Multi-output models not currently supported."
        self.target_fidelities = {self.input_dim-1: self.num_fidelities-1}
    
        with open(self.config_file, "r") as file:
            if ".yaml" in self.config_file:
                self.config = yaml.safe_load(file)
            else:
                self.config = json.load(file)
            # TODO: when running single fidelity with deluq, should n_epochs be set to 1000?
        manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        

    def _delta_enc(self, q: torch.Tensor, a: torch.Tensor, y_q: int) -> torch.Tensor:  # a is the anchor and q is the query
        """
        Encodes the input tensors `q` and `a` by computing the residual between them.

        :param q: The query tensor
        :param a: The anchor tensor

        :returns: Encoded tensor obtained by concatenating the residual and the anchor along axis 1
        """
        residual = q - a
        inp = torch.cat([residual, a], axis=1)
        out = y_q
        return inp, out


    def _deluq_experiment(self, ray_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Training experiment used by ray tuning.

        :param ray_config: Configurations for the tuning, i.e. hyperparmeters to tune.

        :returns: A dictionary representing the score.
        """
        set_deepopt_path()
        from deepopt.surrogate_utils import MLP as Arch
        from deepopt.surrogate_utils import create_optimizer
        from deepopt.deltaenc import DeltaEnc
        

        self.config['variance'] = ray_config["variance"]  # (2**-3)**2
        self.config['learning_rate'] = ray_config["learning_rate"]  # 0.01
        
        seed = ray_config["seed"]
        manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        dataset = TensorDataset(self.full_train_X, self.full_train_Y)
        
        if self.k_folds>len(self.full_train_X):
            kfold = KFold(n_splits=len(self.full_train_X), shuffle=True)
        else:
            kfold = KFold(n_splits=self.k_folds, shuffle=True)

        cv_loss_fun = torch.nn.MSELoss()
        cv_score = 0
        for _, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            train_subsampler = SubsetRandomSampler(train_ids)
            test_subsampler = SubsetRandomSampler(test_ids)

            train_loader = DataLoader(dataset,
                                    batch_size=len(train_ids),
                                    sampler=train_subsampler)
            test_loader = DataLoader(dataset,
                                    batch_size=len(test_ids),
                                    sampler=test_subsampler)

            net = Arch(config=self.config, unc_type='deltaenc', input_dim=self.input_dim, output_dim=self.output_dim, device=self.device)
            opt = create_optimizer(net, self.config)

            for _, (X_train, y_train) in enumerate(train_loader):
                model = DeltaEnc(network=net, config=self.config, optimizer=opt, 
                                    X_train=X_train, y_train=y_train, target=self.target, multi_fidelity=self.multi_fidelity)
                model.train()
                model.fit()

            model.eval()
            with torch.no_grad(): #TODO: is this needed?
                for _, (X_test, y_test) in enumerate(test_loader):
                    if self.multi_fidelity:
                        test_fid_locs = [X_test[...,-1]==i for i in model.X_train[...,-1].unique()]
                        y_test_scaled = y_test.clone()
                        for fid_loc,y_min,y_max in zip(test_fid_locs,model.y_min,model.y_max):
                            y_test_scaled[fid_loc] = model.out_scaler(y_test_scaled[fid_loc],y_min,y_max)
                    else:
                        y_test_scaled = model.out_scaler(y_test,model.y_min,model.y_max)
                    y_pred,_ = model.get_prediction_with_uncertainty(X_test,original_scale=False)
                    cv_score += cv_loss_fun(y_test_scaled, y_pred)

        return {"score": cv_score.item()}


    def _train_gp(self, out_file: str) -> Union[SingleTaskGP, SingleTaskMultiFidelityGP]:
        """
        Train the GP surrogate and save the model produced.

        :param out_file: The name of the output file to save the model to

        :returns: The model produced by training the GP surrogate. This will be a `SingleTaskGP`
             model from BoTorch if we're doing a single-fidelity run or a `SingleTaskMultiFidelityGP`
             model from BoTorch if we're doing a multi-fidelity run.
        """

        print("Training GP Surrogate.")
        model: Union[SingleTaskGP, SingleTaskMultiFidelityGP] = None
        mll: ExactMarginalLogLikelihood = None

        if self.multi_fidelity:
            model = SingleTaskMultiFidelityGP(
                self.full_train_X, 
                self.full_train_Y,
                outcome_transform=Standardize(m=1),
                data_fidelity=self.input_dim-1)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
        else:
            model = SingleTaskGP(
                self.full_train_X,
                self.full_train_Y,
                outcome_transform=Standardize(m=1)
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)

        fit_gpytorch_model(mll)

        state = {
            "state_dict": model.state_dict()
        }
        save(state, join(getcwd(), dirname(out_file), basename(out_file)))
        return model


    def _train_deluq(self, out_file: str) -> Type[Model]:
        """
        Train the delUQ surrogate and save the model produced. We use ray to
        train the surrogate here.

        :param out_file: The name of the output file to save the model to

        :returns: The DeltaEnc model produced by training the delUQ surrogate. 
        """

        print("Training DelUQ Surrogate.")
        set_deepopt_path()
        from deepopt.surrogate_utils import MLP as Arch
        from deepopt.surrogate_utils import create_optimizer
        from deepopt.deltaenc import DeltaEnc


        warnings.filterwarnings("ignore", category=UserWarning)
        cpu_count = max(4,psutil.cpu_count(logical=False)-3)
        # cpu_count = 4 if os.cpu_count() == 0 else (os.cpu_count()-3)
        gpu_count = torch.cuda.device_count() # outputs warning when gpu not found
        warnings.resetwarnings()
    
        ray.init(num_cpus=cpu_count,num_gpus=gpu_count)
        num_samples = 20
        search_space = {
            "variance": tune.loguniform((2 ** -3) ** 2, 5e-1),  # (2 ** -3) ** 2,
            "learning_rate": tune.loguniform(2e-4, 5e-1),
            "seed": tune.randint(0,10000)
        }
        trainable_with_resources = tune.with_resources(
            trainable=self._deluq_experiment,
            resources={
                "cpu": 1 if cpu_count < num_samples else 2,
            }
        )
        tuner = tune.Tuner(
            trainable=trainable_with_resources,
            run_config=RunConfig(
                verbose=0,
            ),
            tune_config=tune.TuneConfig(
                num_samples=num_samples,
                scheduler=ASHAScheduler(
                    metric="score",
                    mode="min",
                ),
            ),
            param_space=search_space,
        )
        result = tuner.fit()
        best_result = result.get_best_result(metric="score", mode="min")
        print(best_result)

        for key, val in best_result.config.items():
            print(f"{key} {val}")
            if key in self.config:
                self.config[key] = val
        net = Arch(config=self.config, unc_type='deltaenc', input_dim=self.input_dim, output_dim=self.output_dim, device=self.device)
        opt = create_optimizer(net, self.config)

        model = DeltaEnc(network=net, config=self.config, optimizer=opt, 
                            X_train=self.full_train_X, y_train=self.full_train_Y, target=self.target, multi_fidelity=self.multi_fidelity)

        model.fit()
        if basename(out_file).split('.')[-1]=='ckpt':
            fname = basename(out_file)[:-5]
        else:
            fname = basename(out_file)
        model.save_ckpt(join(getcwd(), dirname(out_file)), fname)
        ray.shutdown()
        return model
    
    def train(self, model_type: str, out_file: str) -> Type[Model]:
        """
        Train the surrogate model (either GP or delUQ) and save the resulting
        model to a checkpoint file.

        :param model_type: The type of model to train (GP or delUQ)
        :param out_file: The name of the checkpoint file where we will save
            the model trained on the dataset
        
        :returns: The model produced by training. If `model_type` is GP this will
            be a `SingleTaskGP` model from BoTorch if we're doing a single-fidelity run or a
            `SingleTaskMultiFidelityGP` model from BoTorch if we're doing a multi-fidelity run.
            If `model_type` is delUQ this will be a `DeltaEnc` model.
        """
        
        model: Type[Model] = None

        if model_type == "GP":
            model = self._train_gp(out_file=out_file)
        elif model_type == "delUQ":
            model =  self._train_deluq(out_file=out_file)

        return model
    

    def _load_gp(self, learner_file: str)-> Union[SingleTaskGP, SingleTaskMultiFidelityGP]:
        """
        Load in the GP model from the learner file.

        :param learner_file: The learner file that has the model we want to load
        
        :returns: Either a `SingleTaskGP` model or a `SingleTaskMultiFidelityGP` model
            depending on if we're doing a single-fidelity run or a multi-fidelity run
        """

        model: Union[SingleTaskGP, SingleTaskMultiFidelityGP] = None

        if self.multi_fidelity:
            model = SingleTaskMultiFidelityGP(
                self.full_train_X, self.full_train_Y,
                outcome_transform=Standardize(m=1),
                data_fidelity=self.input_dim-1)
        else:
            model = SingleTaskGP(
                self.full_train_X, self.full_train_Y,
                outcome_transform=Standardize(m=1),
            )
        state_dict = torch.load(learner_file)
        model.load_state_dict(state_dict["state_dict"])
        return model
    

    def _load_deluq(self, learner_file: str) -> Type[Model]:
        """
        Load in the delUQ model from the learner file.

        :param learner_file: The learner file that has the model we want to load
        
        :returns: A 'DeltaEnc' model.
        """

        set_deepopt_path()
        from deepopt.surrogate_utils import MLP as Arch
        from deepopt.surrogate_utils import create_optimizer
        from deepopt.deltaenc import DeltaEnc
    
        net = Arch(config=self.config, unc_type='deltaenc', input_dim=self.input_dim, output_dim=self.output_dim, device=self.device)
        opt = create_optimizer(net, self.config)

        model = DeltaEnc(network=net, config=self.config, optimizer=opt, 
                            X_train=self.full_train_X, y_train=self.full_train_Y, target=self.target, multi_fidelity=self.multi_fidelity)

        # DeltaEnc model requries the parent path and file name to be separated.
        # Extension of file is also removed and assumed to be ".ckpt".
        if basename(learner_file).split('.')[-1]=='ckpt':
            file_name = basename(learner_file)[:-5]
        else:
            file_name = basename(learner_file)
        # file_name = basename(learner_file).split(".")[0]
        dir_name = dirname(learner_file)
        model.load_ckpt(dir_name, file_name)
        return model

    def load_model(self, model_type: str, learner_file: str) -> Type[Model]:
        """
        Load the surrogate model (either GP or delUQ) from the learner file.

        :param model_type: The type of model to load (GP or delUQ)
        :param learner_file: The name of the checkpoint file where we will load
            the model in from
        
        :returns: The model we loaded in. If `model_type` is GP this will
            be a `SingleTaskGP` model from BoTorch if we're doing a single-fidelity run or a
            `SingleTaskMultiFidelityGP` model from BoTorch if we're doing a multi-fidelity run.
            If `model_type` is delUQ this will be a `DeltaEnc` model.
        """

        model: Type[Model] = None

        if model_type == "GP":
            model = self._load_gp(learner_file=learner_file)
        elif model_type == "delUQ":
            model = self._load_deluq(learner_file=learner_file)

        return model
    
    def _project(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project X onto the target set of fidelities.

        This function assumes that the set of feasible fidelities is a box, so projecting here
        just means setting each fidelity parameter to its target value.

        (This docstring was copy/pasted from Botorch at:
        https://botorch.org/api/acquisition.html#botorch.acquisition.utils.project_to_target_fidelity)

        :param X: A batch_shape x q x d-dim Tensor of with q d-dim design points for each t-batch

        :returns: A batch_shape x q x d-dim Tensor X_proj with fidelity parameters projected to
            the `target_fidelity` values.
        """
        return project_to_target_fidelity(X=X, target_fidelities=self.target_fidelities)

    def _get_candidates_mf(
        self, 
        model: Type[Model], 
        acq_method: str, 
        q: int, 
        fidelity_cost: ndarray, 
        risk_objective: Optional[Type[RiskMeasureMCObjective]] = None,
        risk_n_deltas: Optional[int] = None, 
        n_fantasies: Optional[int] = 128
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the candidates for a multi-fidelity run.

        The bounds will be set and the fidelity cost model will be applied here first.
        Then whatever acquisition method requested with `acq_method` will be applied.

        :param model: The model loaded in by `load_model`. This will be a `SingleTaskMultiFidelityGP`
            model if we used GP to train the model or a `DeltaEnc` model if we used delUQ.
        :param acq_method: The acquisition method. Either 'GIBBON', 'MaxValEntropy', or 'KG'
        :param q: The number of candidates provided by the user (or the default value assigned
            in Default)
        :param fidelity_cost: A list of how expensive each fidelity should be seen as
        :param risk_objective: Either a `VaR` or a `CVaR` risk objective object from BoTorch. This will
            be determined by the `risk_measure` argument given by the user to the `deepopt-c optimize`
            command.
        :param risk_n_deltas: The number of input perturbations to sample for X's uncertainty
        :param n_fantasies: Number of fantasies to generate. The higher this number the more accurate
            the model (at the expense of model complexity and performance).

        :returns: A two element tuple containing a q x d-dim tensor of generated candidates
            and an associated acquisition value.
        """

        set_deepopt_path()
        from deepopt.acquisition import qMultiFidelityMaxValueEntropy, qMultiFidelityLowerBoundMaxValueEntropy
        
        bounds = torch.FloatTensor(self.input_dim*[[0,1]]).T
        bounds[1,-1] = self.num_fidelities-1
        
        cost_model = FidelityCostModel(fidelity_weights=fidelity_cost)
        cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

        if acq_method == "GIBBON" or acq_method == "MaxValEntropy":
            n_candidates = 2000*self.num_fidelities
            candidate_set = torch.rand(n_candidates, self.input_dim)
            candidate_set[:,-1]*=self.num_fidelities-1
            candidate_set[:,-1] = candidate_set[:,-1].round()
            if acq_method == "MaxValEntropy":
                q_acq = qMultiFidelityMaxValueEntropy(
                    model,
                    num_fantasies=n_fantasies,
                    cost_aware_utility=cost_aware_utility,
                    project=self._project,
                    candidate_set=candidate_set,
                    seed = self.random_seed
                )
            else:
                q_acq = qMultiFidelityLowerBoundMaxValueEntropy(
                    model,
                    posterior_transform=ExpectationPosteriorTransform(n_w=risk_n_deltas) if risk_objective else None,
                    num_fantasies=n_fantasies,
                    cost_aware_utility=cost_aware_utility,
                    project=self._project,
                    candidate_set=candidate_set,
                    seed = self.random_seed
                )
            candidates, acq_value = optimize_acqf_mixed(
                q_acq,
                bounds=bounds,
                fixed_features_list=[{self.input_dim-1:i} for i in range(self.num_fidelities)],
                q=q,
                num_restarts=Defaults.num_restarts_high,
                raw_samples=Defaults.raw_samples_high,
                options={"seed": self.random_seed}
            )
        elif acq_method == "KG":
            curr_val_acqf = FixedFeatureAcquisitionFunction(
                acq_function=PosteriorMean(model,posterior_transform=ExpectationPosteriorTransform(n_w=risk_n_deltas) if risk_objective else None),
                d=self.input_dim,
                columns=[self.input_dim-1],
                values=[self.num_fidelities-1],
            )

            _, current_value = optimize_acqf(
                acq_function=curr_val_acqf,
                bounds=bounds[:, :-1],
                q=1,
                num_restarts=Defaults.num_restarts_high,
                raw_samples=Defaults.raw_samples_high,
                options={"batch_limit": 10, "maxiter": 200, "seed": self.random_seed},
            )

            mfkg_acqf = qMultiFidelityKnowledgeGradient(
                model=model,
                num_fantasies=n_fantasies,
                sampler = SobolQMCNormalSampler(n_fantasies, seed=self.random_seed),
                inner_sampler = SobolQMCNormalSampler(n_fantasies, seed=self.random_seed),
                current_value=current_value,
                cost_aware_utility=cost_aware_utility,
                project=self._project,
                objective=risk_objective
            )
            candidates, acq_value = optimize_acqf_mixed(
                acq_function=mfkg_acqf,
                bounds=bounds,
                fixed_features_list=[{self.input_dim-1:i} for i in range(self.num_fidelities)],
                q=q,
                num_restarts=Defaults.num_restarts_low,
                raw_samples=Defaults.raw_samples_low,
                options={"batch_limit": 5, "maxiter": 200, "seed": self.random_seed},
            )

        print(f"{acq_value = }")
        return candidates, acq_value

    def _get_candidates_sf(
        self,
        model: Type[Model], 
        acq_method: str, 
        q: int,
        risk_objective: Optional[Type[RiskMeasureMCObjective]] = None,
        risk_n_deltas: Optional[int] = None,
        n_fantasies: Optional[int] = 128
    ) -> Tuple[Any, Any]:
        """
        Get the candidates for a single-fidelity run.

        The bounds will be set first, then whatever acquisition method requested with `acq_method`
        will be applied.

        :param model: The model loaded in by `load_model`. This will be a `SingleTaskGP`
            model if we used GP to train the model or a `DeltaEnc` if we used delUQ.
        :param acq_method: The acquisition method. Either 'EI', 'NEI', 'MaxValEntropy', or 'KG'
        :param q: The number of candidates provided by the user (or the default value assigned
            in Default)
        :param risk_objective: Either a `VaR` or a `CVaR` risk objective object from BoTorch. This will
            be determined by the `risk_measure` argument given by the user to the `deepopt-c optimize`
            command.
        :param risk_n_deltas: The number of input perturbations to sample for X's uncertainty
        :param n_fantasies: Number of fantasies to generate. The higher this number the more accurate
            the model (at the expense of model complexity and performance).

        :returns: A two element tuple containing a q x d-dim tensor of generated candidates
            and an associated acquisition value.
        """
        
        set_deepopt_path()
        from deepopt.acquisition import qMaxValueEntropy, qLowerBoundMaxValueEntropy
        
        bounds = torch.FloatTensor(self.input_dim*[[0,1]]).T
        
        if acq_method == "EI":
            q_acq = qExpectedImprovement(model, self.full_train_Y.max().item(), objective=risk_objective)
        elif acq_method == "NEI":
            q_acq = qNoisyExpectedImprovement(model, self.full_train_X, objective=risk_objective, prune_baseline=True)
            #TODO: Verify call syntax for qNoisyExpectedImprovement (why does it need inputs?)
        elif acq_method == "MaxValEntropy":
            n_candidates = 1000
            candidate_set = torch.rand(n_candidates, self.input_dim)
            q_acq = qMaxValueEntropy(
                model,
                posterior_transform=ExpectationPosteriorTransform(n_w=risk_n_deltas) if risk_objective else None,
                candidate_set=candidate_set,
                num_fantasies = n_fantasies,
                seed=self.random_seed
            )
        elif acq_method == 'KG':
            argmax_pmean, max_pmean = optimize_acqf(
                acq_function=PosteriorMean(model,posterior_transform=ExpectationPosteriorTransform(n_w=risk_n_deltas) if risk_objective else None),
                bounds=bounds,
                q=1,
                num_restarts=Defaults.num_restarts_high,
                raw_samples=Defaults.raw_samples_high,
            )
            q_acq = qKnowledgeGradient(
                model=model,
                num_fantasies=n_fantasies,
                sampler=SobolQMCNormalSampler(n_fantasies,seed=self.random_seed),
                inner_sampler=SobolQMCNormalSampler(n_fantasies,seed=self.random_seed),
                current_value=max_pmean,
                objective=risk_objective
                )
        candidates, acq_value = optimize_acqf(
                q_acq, 
                bounds=bounds,
                q=q,
                num_restarts=Defaults.num_restarts_high,
                raw_samples=Defaults.raw_samples_low if acq_method in ['MaxValEntropy','KG'] else Defaults.raw_samples_high,
                sequential=True if acq_method=='MaxValEntropy' else False,
                options={'seed':self.random_seed}
                )
        print(f"{acq_value=}")
        return candidates, q_acq

    def get_candidates(
        self, 
        model: Type[Model], 
        acq_method: str, 
        q: int,
        risk_objective: Optional[Type[RiskMeasureMCObjective]] = None,
        risk_n_deltas: Optional[int] = None,
        fidelity_cost: Optional[ndarray] = None
        ) -> Tuple[Any, Any]:
        """
        Get the candidates using the model loaded in with `load_model` and the acquisition method
        requested by the user.

        :param model: The model loaded in by `load_model`.
        :param acq_method: The acquisition method. Either 'EI', 'NEI', 'GIBBON', 'MaxValEntropy', or 'KG'
        :param q: The number of candidates provided by the user (or the default value assigned
            in Default)
        :param risk_objective: Either a `VaR` or a `CVaR` risk objective object from BoTorch. This will
            be determined by the `risk_measure` argument given by the user to the `deepopt-c optimize`
            command.
        :param risk_n_deltas: The number of input perturbations to sample for X's uncertainty
        :param fidelity_cost: A list of how expensive each fidelity should be seen as

        :returns: A two element tuple containing a q x d-dim tensor of generated candidates
            and an associated acquisition value.
        """
        
        print(f"Number of simulations: {len(self.full_train_X)}. Current max: {self.full_train_Y.max().item():.5f}")

        if self.multi_fidelity:
            candidates, acq_value = self._get_candidates_mf(model=model, acq_method=acq_method, q=q, fidelity_cost=fidelity_cost, risk_objective=risk_objective,risk_n_deltas=risk_n_deltas)
        else:
            candidates, acq_value = self._get_candidates_sf(model=model, acq_method=acq_method, q=q, risk_objective=risk_objective,risk_n_deltas=risk_n_deltas)
        return candidates, acq_value

    def get_risk_measure_objective(self, risk_measure: str, **kwargs) -> Type[RiskMeasureMCObjective]:
        """
        Given a risk measure, return the associated BoTorch risk measure object.

        :param risk_measure: The risk measure to use. Options are 'CVaR' (Conditional Value-at-Risk)
            and 'VaR' (Value-at-Risk).

        :returns: Either a `CVaR` or `VaR` risk measure object from BoTorch
        """
        if risk_measure == "CVaR":
            return CVaR(**kwargs)
        elif risk_measure == "VaR":
            return VaR(**kwargs)
        else:
            return None
        
    def _multiv_normal_samples(self, n: int, std_devs: ndarray) -> torch.Tensor:
        """
        Create a multivariate normal and draw `n` quasi-Monte Carlo (qMC) samples from the
        multivariate normal.

        :param n: The number of qMC samples to draw from the multivariate normal we'll
            obtain from `std_devs`
        :param std_devs: The tensor we'll draw qMC samples from

        :returns: A n x d tensor of samples where d is the dimension of the samples
        """
        mean = torch.zeros_like(std_devs)
        cov = torch.diag(std_devs)
        engine = MultivariateNormalQMCEngine(mean, cov,seed=self.random_seed)
        samples = engine.draw(n)
        return samples
    
    def get_input_perturbation(self, risk_n_deltas: int, bounds: ndarray, X_stddev: ndarray) -> InputPerturbation:
        """
        Get the input perturbation.

        :param risk_n_deltas: The number of input perturbations to sample for X's uncertainty
        :param bounds: Scaled bounds for each input dimension
        :param X_stddev: Scaled uncertainity in X (stddev) in each dimension

        :returns: A transform that adds the set of perturbations to the given input
        """
        assert (len(X_stddev) == len(bounds.T)), f"Expected {len(bounds.T)} values for X_stddev but recieved {len(X_stddev)}."
        input_pertubation = InputPerturbation(perturbation_set=self._multiv_normal_samples(risk_n_deltas, X_stddev), bounds=bounds).eval()
        return input_pertubation
    
    def learn(self, outfile: str, model_type: str = Defaults.model_type) -> None:
        """
        The method to process the `deepopt-c learn` command.

        Here we'll train a model on our dataset and save the model to a checkpoint file.

        :param outfile: The name of the checkpoint file where we will save the model
            trained on the dataset
        :param model_type: The type of surrogate to use. Options: 'GP' or 'delUQ'
        """
        print(f"""
        Infile: {self.data_file}
        Outfile: {outfile}
        Config File: {self.config_file}
        Random Seed: {self.random_seed}
        K-Folds: {self.k_folds}
        Bounds: {self.bounds}
        Model Type: {model_type}
        Multi-Fidelity: {self.multi_fidelity}
        """)
        self.train(model_type=model_type,out_file=outfile)
        
    def optimize(
        self,
        outfile: str,
        learner_file: str,
        acq_method: str,
        model_type: str = Defaults.model_type,
        num_candidates: int = Defaults.num_candidates,
        fidelity_cost: str = Defaults.fidelity_cost,
        risk_measure: str = None,
        risk_level: float = None,
        risk_n_deltas: int = None,
        x_stddev: str = None
    ) -> None:
        """
        The function to process the `deepopt-c optimize` command.

        Here we'll use the model created by `learn` to produce new simulation points.

        :param outfile: The name of the file to save the proposed candidates in
        :param learner_file: The name of the checkpoint file produced by `learn`
        :param acq_method: The acquisiton function. Single-fidelity options:
            'KG', 'MaxValEntropy', 'EI', or 'NEI'. Multi-fidelity options: 'KG' or
            'MaxValEntropy'
        :param model_type: The type of surrogate to use. Options: 'GP' or 'delUQ'
        :param num_candidates: The number of candidates
        :param fidelity_cost: List of costs for each fidelity
        :param risk_measure: The risk measure to use. Options: 'CVaR' (Conditional Value-at-Risk)
                or 'VaR' (Value-at-Risk).
        :param risk_level: The risk level (a float between 0 and 1)
        :param risk_n_deltas: The number of input perturbations to sample for X's uncertainty
        :param x_stddev: Uncertainity in X (stddev) in each dimension
        """
        print(f"""
        Infile: {self.data_file}
        Outfile: {outfile}
        Config File: {self.config_file}
        Learner File: {learner_file}
        Random Seed: {self.random_seed}
        Bounds: {self.bounds}
        Acq Method: {acq_method}
        Model Type: {model_type}
        Multi-Fidelity: {self.multi_fidelity}
        Fidelity Cost: {fidelity_cost}
        """)
        
        model = self.load_model(model_type=model_type,learner_file=learner_file)
                
        if risk_measure:
            assert acq_method!="MaxValEntropy", 'Risk measure not yet supported for MaxValueEntropy acquisition'
            x_stddev_scaled = x_stddev/(self.bounds[1]-self.bounds[0])
            bounds_scaled = torch.FloatTensor(self.input_dim*[[0,1]]).T
            if self.multi_fidelity:
                x_stddev_scaled[-1] = 0
            risk_objective = self.get_risk_measure_objective(risk_measure=risk_measure, alpha=risk_level, n_w=risk_n_deltas)
            input_pertubation = self.get_input_perturbation(risk_n_deltas=risk_n_deltas, bounds=bounds_scaled, X_stddev=x_stddev_scaled)
            model.input_transform = input_pertubation
        else:
            risk_objective = None
        model.eval()
        
        candidates, _ = self.get_candidates(
            model=model, 
            acq_method=acq_method, 
            q=num_candidates, 
            risk_objective=risk_objective,
            risk_n_deltas=risk_n_deltas,
            fidelity_cost=fidelity_cost,
        )
        if self.multi_fidelity:
            candidates[:,:-1] = candidates[:,:-1]*(self.bounds[1,:-1]-self.bounds[0,:-1] + self.bounds[0,:-1])
            candidates[:,-1] = candidates[:,-1].round()
        else:
            candidates = candidates*(self.bounds[1]-self.bounds[0]) + self.bounds[0]
        candidates_npy = candidates.cpu().detach().numpy()
        np.save(outfile, candidates_npy)

@click.group()
@click.option("--develop", help="If developing package, pass in path to your development repo. [example: --develop /usr/workspace/tran67/deepopt]", type=click.Path(exists=True), required=False)
def deepopt_cli(develop):
    """
    The entrypoint to the DeepOpt library.
    """
    if develop:
        global DEEPOPT_PATH, DEVELOP
        DEEPOPT_PATH = develop
        DEVELOP = True
        set_deepopt_path()
    
@deepopt_cli.command()
@click.option("-i", "--infile", help="Input data to train from.", type=click.Path(exists=True), required=True)
@click.option("-o", "--outfile", help="Outfile to save model checkpoint.", type=click.STRING, required=True)
@click.option("-c", "--config-file", help="Config file containing hyper parameters.", type=click.Path(exists=True), required=True)
@click.option("-b","--bounds", help="Bounds for each input dimension.", type=click.STRING, required=True)
@click.option("-r", "--random-seed", help="Random seed.", default=Defaults.random_seed, show_default=True, type=click.INT)
@click.option("-k", "--k-folds", help="Number of k-folds.", default=Defaults.k_folds, show_default=True, type=click.INT)
@click.option("-m", "--model-type", help="What kind of surrogate are you using?", default=Defaults.model_type, show_default=True, type=click.Choice(["GP", "delUQ"]))
@click.option("--multi-fidelity", help="Single or mult-fidelity?", is_flag=True, default=Defaults.multi_fidelity, type=click.BOOL, show_default=True)
def learn(infile, outfile, config_file, bounds, random_seed, k_folds, model_type, multi_fidelity) -> None:
    """
    Train a model on a dataset and save that model to an output file.
    """
    bounds = torch.FloatTensor(json.loads(bounds)).T
    dc = DeepoptConfigure(config_file=config_file, data_file=infile, multi_fidelity=multi_fidelity, random_seed=random_seed,bounds=bounds,k_folds=k_folds)
    dc.learn(outfile=outfile,model_type=model_type)


@deepopt_cli.command()
@click.option("-i", "--infile", help="Training data path.", type=click.Path(exists=True), required=True)
@click.option("-o", "--outfile", help="Where to place the suggested candidates.", type=click.STRING, required=True)
@click.option("-c", "--config-file", help="Config file containing hyper parameters.", type=click.Path(exists=True), required=True)
@click.option("-l", "--learner-file", help="Learner path. Ex: /learners/my_learner.ckpt", type=click.Path(exists=True), required=True)
@click.option("-b", "--bounds", help="Bounds for each input dimension.", type=click.STRING, required=True)
@click.option("-a", "--acq-method", 
              help="""
              \b
              The acquisiton function. 
              [NOTE: Some acquistion functions only work with a specific fidelity.]
              \b
              Single    - [KG|MaxValEntropy|EI|NEI]
              Multi     - [KG|MaxValEntropy] 
              """, 
              type=click.Choice(["EI", "NEI", "KG", "MaxValEntropy"]), required=True)
@click.option("-r", "--random-seed", help="Random seed.", default=Defaults.random_seed, show_default=True, type=click.INT)
@click.option("-m", "--model-type", help="What kind of surrogate are you using?", show_default=True, default=Defaults.model_type, type=click.Choice(["GP", "delUQ"]))
@click.option("-q", "--num-candidates", help="The number of candidates.", default=Defaults.num_candidates, type=click.INT, show_default=True)
@click.option("--multi-fidelity", help="Single or mult-fidelity?", is_flag=True, default=Defaults.multi_fidelity, show_default=True, type=click.BOOL)
@click.option("--risk-measure", help="The risk measure to apply.", type=click.Choice(["VaR", "CVaR"]))
@click.option("--risk-level", help="The risk level.", type=click.FloatRange(0, 1, min_open=True), cls=ConditionalOption, depends_on="risk_measure")
@click.option("--risk-n-deltas", help="The number of input perturbations to sample for X's uncertainty. [example: --risk-n-deltas 10].", type=click.INT, cls=ConditionalOption, depends_on="risk_measure")
@click.option("--X-stddev", help="Uncertainity in X (stddev) in each dimension. [example: --X-stddev [0.00005]].", type=click.STRING, cls=ConditionalOption, depends_on="risk_measure")
@click.option("--fidelity-cost", help="List of costs for each fidelity.", type=click.STRING, default=Defaults.fidelity_cost, show_default=True, cls=ConditionalOption, depends_on="multi_fidelity", equal_to=True)
def optimize(
    infile, 
    outfile, 
    config_file,
    learner_file, 
    bounds, 
    acq_method, 
    random_seed, 
    model_type, 
    num_candidates, 
    multi_fidelity, 
    risk_measure, 
    risk_level, 
    risk_n_deltas, 
    x_stddev, 
    fidelity_cost,
    ) -> None:
    """
    Load in the model created by `learn` and use it to propose new simulation points.
    """
    bounds = torch.FloatTensor(json.loads(bounds)).T
    dc = DeepoptConfigure(config_file=config_file, data_file=infile, multi_fidelity=multi_fidelity, random_seed=random_seed,bounds=bounds)
    
    risk_measure = None if risk_measure=='None' else risk_measure
    if risk_measure:
        x_stddev = torch.FloatTensor(json.loads(x_stddev))
    if multi_fidelity:
        fidelity_cost = torch.FloatTensor(json.loads(fidelity_cost))
    dc.optimize(outfile=outfile,learner_file=learner_file,acq_method=acq_method,model_type=model_type,
                num_candidates=num_candidates,fidelity_cost=fidelity_cost,
                risk_measure=risk_measure,risk_level=risk_level,risk_n_deltas=risk_n_deltas,x_stddev=x_stddev)

def main():
    deepopt_cli(max_content_width=800)


if __name__ == "__main__":
    main()

