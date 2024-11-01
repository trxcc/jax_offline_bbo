import warnings
from typing import Callable, Any, Optional

import numpy as np 
import jax 
import jax.numpy as jnp 
import time 
import torch 
from botorch.models import FixedNoiseGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim import optimize_acqf
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement 
from botorch.acquisition import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning


from src.data.datamodule import JAXDataModule
from src.task.base_task import OfflineBBOExperimenter
from src.search.base_searcher import Searcher
from src.utils.logger import RankedLogger
from src._typing import PRNGKeyArray as KeyArray

warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

log = RankedLogger(__name__, rank_zero_only=True)

class BOqEISearcher(Searcher):
    
    def __init__(
        self,
        key: KeyArray,
        score_fn: Callable[[jnp.ndarray], jnp.ndarray],
        datamodule: JAXDataModule,
        task: OfflineBBOExperimenter,
        num_solutions: int,
        noise_se: float,
        batch_size: int,
        num_restarts: int,
        raw_samples: int,
        batch_limit: int,
        maxiter: int,
        iterations: int,
        mc_samples: int,
        gp_samples: int,
    ) -> None:
        super().__init__(key, score_fn, datamodule, task, num_solutions)
        self.noise_se = noise_se 
        self.batch_size = batch_size
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.batch_limit = batch_limit
        self.maxiter = maxiter
        self.iterations = iterations
        self.mc_samples = mc_samples
        self.gp_samples = gp_samples

    def run(self) -> jnp.ndarray:
        tkwargs = {
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "dtype": torch.float64
        }
        
        x, y = self.datamodule.x, self.datamodule.y 
        gp_indices = jnp.argsort(y.squeeze())[-self.gp_samples:]
        initial_x, initial_y = x[gp_indices], y[gp_indices]
        initial_x = torch.from_numpy(np.array(initial_x)).to(**tkwargs)
        initial_y = torch.from_numpy(np.array(initial_y)).to(**tkwargs)
        input_size = x.shape[1]
        
        NOISE_SE = self.noise_se
        train_yvar = torch.tensor(NOISE_SE ** 2).to(**tkwargs)
        
        bounds = torch.tensor(
            [jnp.min(x, axis=0).reshape([input_size]).tolist(),
            jnp.max(x, axis=0).reshape([input_size]).tolist()]
        ).to(**tkwargs)
        
        standard_bounds = torch.zeros_like(bounds)
        standard_bounds[1] = 1
        
        def initialize_model(train_x, train_obj, state_dict=None):
            train_x = (train_x - bounds[0]) / (bounds[1] - bounds[0])
            train_obj = (train_obj - train_obj.mean()) / train_obj.std()
            # define models for objective
            model_obj = FixedNoiseGP(train_x, train_obj,
                                    train_yvar.expand_as(train_obj)).to(train_x)
            # combine into a multi-output GP model
            model = ModelListGP(model_obj)
            mll = SumMarginalLogLikelihood(model.likelihood, model)
            # load state dict if it is passed
            if state_dict is not None:
                model.load_state_dict(state_dict)
            return mll, model

        def obj_callable(Z: torch.Tensor, X: Optional[torch.Tensor] = None):
            return Z[..., 0]

        # define a feasibility-weighted objective for optimization
        obj = GenericMCObjective(obj_callable)

        BATCH_SIZE = self.batch_size
        
        @jax.jit
        def objective(x):
            # print(x.shape, self.score_fn(x).shape)
            return self.score_fn(x)
        
        def optimize_acqf_and_get_observation(acq_func):
            """Optimizes the acquisition function, and returns
            a new candidate and a noisy observation."""
            # optimize
            try:
                candidates, _ = optimize_acqf(
                    acq_function=acq_func,
                    bounds=standard_bounds,
                    q=BATCH_SIZE,
                    num_restarts=self.num_restarts,
                    raw_samples=self.raw_samples,  # used for intialization heuristic
                    options={"batch_limit": self.batch_limit,
                            "maxiter": self.maxiter})
            except RuntimeError:
                return
            # observe new values
            new_x = candidates.detach()
            # exact_obj = self.score_fn(candidates)
            exact_obj = torch.from_numpy(
                np.array(objective(jnp.array(candidates.detach().cpu().numpy())))
            ).to(**tkwargs)
            
            new_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
            return new_x, new_obj

        N_BATCH = self.iterations
        MC_SAMPLES = self.mc_samples

        best_observed_ei = []

        # call helper functions to generate initial training data and initialize model
        train_x_ei = initial_x.reshape([initial_x.shape[0], input_size])
        train_x_ei = torch.tensor(train_x_ei).to(**tkwargs)

        train_obj_ei = initial_y.reshape([initial_y.shape[0], 1])
        train_obj_ei = torch.tensor(train_obj_ei).to(**tkwargs)

        best_observed_value_ei = train_obj_ei.max().item()
        mll_ei, model_ei = initialize_model(train_x_ei, train_obj_ei)
        best_observed_ei.append(best_observed_value_ei)

        # run N_BATCH rounds of BayesOpt after the initial random batch
        for iteration in range(1, N_BATCH + 1):

            t0 = time.time()

            # fit the models
            fit_gpytorch_model(mll_ei)

            # define the qEI acquisition module using a QMC sampler
            qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

            # for best_f, we use the best observed noisy values as an approximation
            tmp_train_obj = (train_obj_ei - train_obj_ei.mean()) / train_obj_ei.std()
            qLogEI = qLogExpectedImprovement(
                model=model_ei, best_f=tmp_train_obj.max(),
                sampler=qmc_sampler, objective=obj)

            # optimize and get new observation
            result = optimize_acqf_and_get_observation(qLogEI)
            if result is None:
                print("RuntimeError was encountered, most likely a "
                    "'symeig_cpu: the algorithm failed to converge'")
                break
            new_x_ei, new_obj_ei = result

            # update training points
            train_x_ei = torch.cat([train_x_ei, new_x_ei])
            train_obj_ei = torch.cat([train_obj_ei, new_obj_ei])

            # update progress
            best_value_ei = obj(train_x_ei).max().item()
            best_observed_ei.append(best_value_ei)

            # reinitialize the models so they are ready for fitting on next iteration
            # use the current state dict to speed up fitting
            mll_ei, model_ei = initialize_model(
                train_x_ei, train_obj_ei, model_ei.state_dict())

            t1 = time.time()
            log.info(f"Batch {iteration:>2}: best_value = "
                f"({best_value_ei:>4.2f}), "
                f"time = {t1 - t0:>4.2f}.")

        x_sol = jnp.array(train_x_ei.detach().cpu().numpy())
        y_sol = jnp.array(train_obj_ei.detach().cpu().numpy())
        
        solution = x_sol[jnp.argsort(y_sol.squeeze())[-self.num_solutions:]]
        return solution

        
        