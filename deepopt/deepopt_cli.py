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
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
from botorch.acquisition.risk_measures import CVaR, VaR, RiskMeasureMCObjective
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
from typing import Dict, Any, Union, Type, List, Mapping, Tuple


DEVELOP = False
DEEPOPT_PATH = ""


def set_deepopt_path():
    if DEVELOP:
        import sys
        sys.path.insert(0, DEEPOPT_PATH)

        import deepopt
        print(f"Sourcing deepopt from {deepopt.__file__}.")
        
class FidelityCostModel(DeterministicModel):
    def __init__(self,fidelity_weights):
        super().__init__()
        self._num_outputs = 1
        self.fidelity_weights = torch.Tensor(fidelity_weights)
        
    def forward(self, X):
        return self.fidelity_weights[X[...,-1].round().long()].unsqueeze(-1)


class ConditionalOption(click.Option):
    def __init__(self, *args, **kwargs):
        self.depends_on = kwargs.pop("depends_on")
        self.equal_to = kwargs.pop("equal_to", None)
        kwargs['help'] = f"[NOTE: This argument is only used if {self.depends_on} is used.] " + kwargs.get('help','')
        super().__init__(*args, **kwargs)


    def handle_parse_result(
        self, ctx: click.Context, opts: Mapping[str, Any], args: List[str]
    ) -> Tuple[Any, List[str]]:
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


@dataclass
class DeepoptConfigure:

    config_file: str
    data_file: str
    random_seed: int
    bounds: ndarray
    multi_fidelity: bool
    num_fidelities: int = None
    kfolds: int = None
    full_train_X: ndarray = None
    full_train_Y: ndarray = None
    input_dim: int = None
    output_dim: int = None
    config: Dict[str, Any] = None
    device: str = None
    surrogate_type: str = None
    encoding: str = None
    target: str = None
    target_fidelities: Dict[int, float] = None

    def __post_init__(self) -> None:

        input_data = np.load(self.data_file)
        self.X_orig = from_numpy(input_data["X"])
        self.Y_orig = from_numpy(input_data["y"])
        self.full_train_X = (self.X_orig-self.bounds[0])/(self.bounds[1]-self.bounds[0])
        if self.multi_fidelity:
            self.full_train_X[:,-1] = self.X_orig[:,-1].round()
            self.num_fidelities = int(self.bounds[1,-1]) + 1
        else:
            self.num_fidelities = 1
        self.full_train_Y = from_numpy(input_data["y"])
        self.input_dim = self.full_train_X.size(-1)
        self.output_dim = 1
        self.target_fidelities = {self.input_dim-1: self.num_fidelities-1}
    
        with open(self.config_file, "r") as file:
            if ".yaml" in self.config_file:
                self.config = yaml.safe_load(file)
            else:
                self.config = json.load(file)
            self.device = self.config.get("device", "cpu")
            self.surrogate_type = self.config.get("surrogate_type", "FF")
            self.encoding = self.config.get("encoding", "default")
            self.target = self.config.get("target", "dy")
            # TODO: when running single fidelity with deluq, should n_epochs be set to 1000?

    def _delta_enc(self, q, a, y_q):  # a is the anchor and q is the query
        residual = q - a
        inp = torch.cat([residual, a], axis=1)
        out = y_q
        return inp, out


    def _deluq_experiment(self, ray_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Training experiment used by ray tuning.

        Args:
            ray_config (Dict[str, Any]): Configurations for the tuning, i.e. hyperparmeters to tune.

        Returns:
            Dict[str, Any]: A dictionary representing the score.
        """
        set_deepopt_path()
        from deepopt.surrogate_utils import MLP as Arch
        from deepopt.surrogate_utils import create_optimizer
        from deepopt.deltaenc import DeltaEnc, DeltaEncMF
        

        self.config['variance'] = ray_config["variance"]  # (2**-3)**2
        self.config['learning_rate'] = ray_config["learning_rate"]  # 0.01
        
        seed = ray_config["seed"]
        manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        dataset = TensorDataset(self.full_train_X, self.full_train_Y)

        kfold = KFold(n_splits=self.kfolds, shuffle=True)

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

            net = Arch(self.config, 'deltaenc', self.input_dim, self.output_dim).to(self.device)
            opt = create_optimizer(net, self.config)

            for _, (X_train, y_train) in enumerate(train_loader):
                if self.multi_fidelity:
                    model = DeltaEncMF(net, self.config, opt, X_train, y_train,
                                    self.surrogate_type, 'default', 'dy', self.device)
                else:
                    model = DeltaEnc(net, self.config, opt, X_train, y_train,
                                    self.surrogate_type, 'default', 'dy', self.device)
                model.train()
                model.fit()

            model.eval()
            with torch.no_grad():
                for _, (X_test, y_test) in enumerate(test_loader):
                    y_pred = []
                    h = net.input_mapping(X_test)
                    for i in range(len(X_test)):
                        x_denc = []
                        # using the entire possible xi xj matchings
                        for j in range(len(X_test)):
                            xd, _ = self._delta_enc(
                                h[j].unsqueeze(0), h[i].unsqueeze(0), 0)
                            x_denc.append(xd)

                        x_denc = torch.cat(x_denc)
                        y_denc = net(x_denc).detach().ravel()
                        y_pred.append(torch.mean(y_denc))

                    y_pred = torch.stack(y_pred).view(-1, 1)
                    cv_score += cv_loss_fun(y_test, y_pred)

        return {"score": cv_score.item()}


    def _train_gp(self, out_file: str) -> SingleTaskMultiFidelityGP:

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

        print("Training DelUQ Surrogate.")
        set_deepopt_path()
        from deepopt.surrogate_utils import MLP as Arch
        from deepopt.surrogate_utils import create_optimizer
        from deepopt.deltaenc import DeltaEnc, DeltaEncMF


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
        net = Arch(self.config, 'deltaenc', self.input_dim, self.output_dim).to(self.device)
        opt = create_optimizer(net, self.config)
        
        if self.multi_fidelity:
            model = DeltaEncMF(
                net, self.config, opt, self.full_train_X, self.full_train_Y,
                self.surrogate_type, 'default', 'dy', self.device
                )
        else:
            model = DeltaEnc(
                net, self.config, opt, self.full_train_X, self.full_train_Y,
                self.surrogate_type, 'default', 'dy', self.device
                )

        model.fit()
        if basename(out_file).split('.')[-1]=='ckpt':
            fname = basename(out_file)[:-5]
        else:
            fname = basename(out_file)
        model.save_ckpt(join(getcwd(), dirname(out_file)), fname)
        ray.shutdown()
        return model
    
    def train(self, model_type: str, out_file: str) -> Type[Model]:
        
        model: Type[Model] = None

        if model_type == "GP":
            model = self._train_gp(out_file=out_file)
        elif model_type == "delUQ":
            model =  self._train_deluq(out_file=out_file)

        return model
    

    def _load_gp(self, learner_file: str)-> Union[SingleTaskGP, SingleTaskMultiFidelityGP]:

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

        set_deepopt_path()
        from deepopt.surrogate_utils import MLP as Arch
        from deepopt.surrogate_utils import create_optimizer
        from deepopt.deltaenc import DeltaEnc, DeltaEncMF
    
        model = Arch(self.config, 'deltaenc', self.input_dim, self.output_dim).to(self.device)
        opt = create_optimizer(model, self.config)

        kwargs = SimpleNamespace(
            network=model,
            config=self.config,
            optimizer=opt,
            X_train=self.full_train_X,
            y_train=self.full_train_Y,
            surrogate_type=self.surrogate_type,
            encoding=self.encoding,
            target=self.target,
            device=self.device,
        )

        if self.multi_fidelity:
            model = DeltaEncMF(**kwargs.__dict__)
        else:
            model = DeltaEnc(**kwargs.__dict__)

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

        model: Type[Model] = None

        if model_type == "GP":
            model = self._load_gp(learner_file=learner_file)
        elif model_type == "delUQ":
            model = self._load_deluq(learner_file=learner_file)

        return model
    
    def _project(self, X):
        return project_to_target_fidelity(X=X, target_fidelities=self.target_fidelities)

    def _get_candidates_mf(
        self, 
        model: Type[Model], 
        acq_method: str, 
        q: int, 
        fidelity_cost: ndarray, 
        risk_objective: Type[RiskMeasureMCObjective] = None, 
        n_fantasies: int = 128):

        set_deepopt_path()
        from deepopt.acquisition import qMultiFidelityMaxValueEntropy, qMultiFidelityLowerBoundMaxValueEntropy
        
        bounds = torch.FloatTensor(self.input_dim*[[0,1]]).T
        bounds[1,-1] = self.num_fidelities-1
        
        cost_model = FidelityCostModel(fidelity_weights=fidelity_cost)
        cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

        if acq_method == "GIBBON" or acq_method == "MaxValEntropy":
            n_candidates = 5000
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
                num_restarts=15,
                raw_samples=5000,
                options={"seed": self.random_seed}
            )
        elif acq_method == "KG":
            curr_val_acqf = FixedFeatureAcquisitionFunction(
                acq_function=PosteriorMean(model),
                d=self.input_dim,
                columns=[self.input_dim-1],
                values=[self.num_fidelities-1],
            )

            _, current_value = optimize_acqf(
                acq_function=curr_val_acqf,
                bounds=bounds[:, :-1],
                q=1,
                num_restarts=10,
                raw_samples=1024,
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
            NUM_RESTARTS = 5
            RAW_SAMPLES = 128
            candidates, acq_value = optimize_acqf_mixed(
                acq_function=mfkg_acqf,
                bounds=bounds,
                fixed_features_list=[{self.input_dim-1:i} for i in range(self.num_fidelities)],
                q=q,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
                options={"batch_limit": 5, "maxiter": 200, "seed": self.random_seed},
            )

        print(f"{acq_value = }")
        return candidates, acq_value

    def _get_candidates_sf(
        self,
        model: Type[Model], 
        acq_method: str, 
        q: int,
        risk_objective: Type[RiskMeasureMCObjective] = None,
        n_fantasies: int = 128
    ) -> Tuple[Any, Any]:
        
        set_deepopt_path()
        from deepopt.acquisition import qMaxValueEntropy, qLowerBoundMaxValueEntropy
        
        bounds = torch.FloatTensor(self.input_dim*[[0,1]]).T
        
        if acq_method == "EI":
            q_acq = qExpectedImprovement(model, self.full_train_Y.max().item(), objective=risk_objective)
        elif acq_method == "NEI":
            q_acq = qNoisyExpectedImprovement(model, self.full_train_X, objective=risk_objective, prune_baseline=True)
        elif acq_method == "MaxValEntropy":
            n_candidates = 5000
            candidate_set = torch.rand(n_candidates, self.input_dim)
            q_acq = qMaxValueEntropy(
                model,
                candidate_set=candidate_set,
                num_fantasies = n_fantasies,
                seed=self.random_seed
            )

        candidates, acq_value = optimize_acqf(
                q_acq, 
                bounds=bounds,
                q=q,
                num_restarts=15,
                raw_samples=5000,
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
        risk_objective: Type[RiskMeasureMCObjective] = None,
        fidelity_cost: ndarray = None
        ) -> Tuple[Any, Any]:
        
        print(f"Number of simulations: {len(self.full_train_X)}. Current max: {self.full_train_Y.max().item():.5f}")

        if self.multi_fidelity:
            candidates, acq_value = self._get_candidates_mf(model=model, acq_method=acq_method, q=q, fidelity_cost=fidelity_cost, risk_objective=risk_objective)
        else:
            candidates, acq_value = self._get_candidates_sf(model=model, acq_method=acq_method, q=q, risk_objective=risk_objective)
        return candidates, acq_value

    def get_risk_measure_objective(self, risk_measure, **kwargs) -> Type[RiskMeasureMCObjective]:
        if risk_measure == "CVaR":
            return CVaR(**kwargs)
        elif risk_measure == "VaR":
            return VaR(**kwargs)
        else:
            return None
        
    def _multiv_normal_samples(self, n, std_devs):
        mean = torch.zeros_like(std_devs)
        cov = torch.diag(std_devs)
        engine = MultivariateNormalQMCEngine(mean, cov,seed=self.random_seed)
        samples = engine.draw(n)
        return samples
    
    def get_input_pertubation(self, risk_n_deltas: int, bounds: ndarray, X_stddev: ndarray):
        assert (len(X_stddev) == len(bounds.T)), f"Expected {len(bounds.T)} values for X_stddev but recieved {len(X_stddev)}."
        input_pertubation = InputPerturbation(perturbation_set=self._multiv_normal_samples(risk_n_deltas, X_stddev), bounds=bounds).eval()
        return input_pertubation


@click.group()
@click.option("--develop", help="If developing package, pass in path to your development repo. [example: --develop /usr/workspace/tran67/deepopt]", type=click.Path(exists=True))
def deepopt_cli(develop):
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
@click.option("-r", "--random-seed", help="Random seed.", default=4321, show_default=True, type=click.INT)
@click.option("-k", "--k-folds", help="Number of k-folds.", default=5, show_default=True, type=click.INT)
@click.option("-m", "--model-type", help="What kind of surrogate are you using?", default="GP", show_default=True, type=click.Choice(["GP", "delUQ"]))
@click.option("--multi-fidelity", help="Single or mult-fidelity?", is_flag=True, default=False, type=click.BOOL, show_default=True)
def learn(infile, outfile, config_file, bounds, random_seed, k_folds, model_type, multi_fidelity) -> None:

    print(f"""
    Infile: {infile}
    Outfile: {outfile}
    Config File: {config_file}
    Random Seed: {random_seed}
    K-Folds: {k_folds}
    Bounds: {bounds}
    Model Type: {model_type}
    Multi-Fidelity: {multi_fidelity}
    """)
    manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    bounds = torch.FloatTensor(json.loads(bounds)).T
    dc = DeepoptConfigure(config_file=config_file, data_file=infile, multi_fidelity=multi_fidelity, random_seed=random_seed,bounds=bounds)
    dc.kfolds = k_folds
    dc.train(model_type=model_type, out_file=outfile)


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
              Single    - [MaxValEntropy|EI|NEI]
              Multi     - [KG|MaxValEntropy] 
              """, 
              type=click.Choice(["EI", "NEI", "KG", "MaxValEntropy"]),
              required=True)
@click.option("-r", "--random-seed", help="Random seed.", default=4321, show_default=True, type=click.INT)
@click.option("-m", "--model-type", help="What kind of surrogate are you using?", show_default=True, type=click.Choice(["GP", "delUQ"]))
@click.option("-q", "--num-candidates", help="The number of candidates.", default=2, type=click.INT, show_default=True)
@click.option("--multi-fidelity", help="Single or mult-fidelity?", is_flag=True, default=False, show_default=True, type=click.BOOL)
@click.option("--risk-measure", help="The risk measure to apply.", type=click.Choice(["VaR", "CVaR"]), cls=ConditionalOption, depends_on="multi_fidelity", equal_to=False)
@click.option("--risk-level", help="The risk level.", type=click.FloatRange(0, 1, min_open=True), cls=ConditionalOption, depends_on="risk_measure")
@click.option("--risk-n-deltas", help="The number of input pertubations to sample for X's uncertainty. [example: --risk-n-deltas 10].", type=click.INT, cls=ConditionalOption, depends_on="risk_measure")
@click.option("--X-stddev", help="Uncertainity in X (stddev) in each dimension. [example: --X-stddev [0.00005]].", type=click.STRING, cls=ConditionalOption, depends_on="risk_measure")
@click.option("--fidelity-cost", help="List of costs for each fidelity.", type=click.STRING, default='[1,10]', show_default=True, cls=ConditionalOption, depends_on="multi_fidelity", equal_to=True)
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

    print(f"""
    Infile: {infile}
    Outfile: {outfile}
    Config File: {config_file}
    Learner File: {learner_file}
    Bounds: {bounds}
    Acq Method: {acq_method}
    Random Seed: {random_seed}
    Model Type: {model_type}
    Multi-Fidelity: {multi_fidelity}
    Fidelity Cost: {fidelity_cost}
    """)

    manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    bounds = torch.FloatTensor(json.loads(bounds)).T
    dc = DeepoptConfigure(config_file=config_file, data_file=infile, multi_fidelity=multi_fidelity, random_seed=random_seed,bounds=bounds)
    model = dc.load_model(model_type=model_type, learner_file=learner_file)
    if risk_measure:
        x_stddev = torch.FloatTensor(json.loads(x_stddev)).T
        x_stddev_scaled = x_stddev/(bounds[1]-bounds[0])
        bounds_scaled = torch.FloatTensor(dc.input_dim*[[0,1]]).T
        if dc.multi_fidelity:
            x_stddev_scaled[-1] = 0
        risk_objective = dc.get_risk_measure_objective(risk_measure=risk_measure, alpha=risk_level, n_w=risk_n_deltas)
        input_pertubation = dc.get_input_pertubation(risk_n_deltas=risk_n_deltas, bounds=bounds_scaled, X_stddev=x_stddev_scaled)
        model.input_transform = input_pertubation
    else:
        risk_objective = None
    model.eval()
    
    if dc.multi_fidelity:
        fidelity_cost = torch.FloatTensor(json.loads(fidelity_cost))

    candidates, _ = dc.get_candidates(
        model=model, 
        acq_method=acq_method, 
        q=num_candidates, 
        risk_objective=risk_objective,
        fidelity_cost=fidelity_cost,
    )
    candidates = candidates*(bounds[1]-bounds[0]) + bounds[0]
    candidates_npy = candidates.cpu().detach().numpy()
    np.save(outfile, candidates_npy)


def main():
    deepopt_cli(max_content_width=800)


if __name__ == "__main__":
    main()

