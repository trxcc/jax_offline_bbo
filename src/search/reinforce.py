from typing import Callable, Dict

import jax 
import jax.numpy as jnp 
import optax
from omegaconf import DictConfig

from src.data.datamodule import JAXDataModule
from src.task.base_task import OfflineBBOExperimenter
from src.search.base_searcher import Searcher
from src.utils.logger import RankedLogger
from src._typing import PRNGKeyArray as KeyArray
from src.search.utils.reinforce_marginal import DiscreteMarginal, ContinuousMarginal

log = RankedLogger(__name__, rank_zero_only=True)

class ReinforceSearcher(Searcher):
    
    def __init__(
        self,
        key: KeyArray,
        score_fn: Callable[[jnp.ndarray], jnp.ndarray],
        datamodule: JAXDataModule,
        task: OfflineBBOExperimenter,
        num_solutions: int,
        learning_rate: DictConfig,
        exploration_std: float,
        iterations: int,
        batch_size: DictConfig,
    ) -> None:
        super().__init__(key, score_fn, datamodule, task, num_solutions)
        self.exploration_std = exploration_std
        self.iterations = iterations 
        if isinstance(learning_rate, float):
            self.learning_rate = learning_rate
        else:
            if task.is_discrete:
                self.learning_rate = learning_rate["discrete"]
            else:
                self.learning_rate = learning_rate["continuous"]
        
        if isinstance(batch_size, int):
            self.batch_size = batch_size
        else:
            if task.is_discrete:
                self.batch_size = batch_size["discrete"]
            else:
                self.batch_size = batch_size["continuous"]

    def run(self) -> jnp.ndarray:
        x = jnp.array(self.task.x_np)
        y = jnp.array(self.task.y_np)
        initial_x = x[jnp.argsort(y.squeeze())[-self.num_solutions:]]
        
        if self.task.is_discrete:
            logits = jnp.pad(self.task.to_logits(initial_x), [[0, 0], [0, 0], [1, 0]])
            probs = jax.nn.softmax(logits / 1e-5)
            logits = jnp.log(jnp.mean(probs, axis=0))
            self.sampler = DiscreteMarginal(logits)
        
        else:
            mean = jnp.mean(initial_x, axis=0)
            logstd = jnp.log(jnp.ones_like(mean) * self.exploration_std)
            self.sampler = ContinuousMarginal(mean, logstd)
            
        rl_opt = optax.adam(learning_rate=self.learning_rate)
        self.key, param_key = jax.random.split(self.key)
        params = self.sampler.init(param_key)
        opt_state = rl_opt.init(params)
        
        @jax.jit
        def objective(x):
            return self.score_fn(x)
        
        @jax.jit
        def compute_loss(params, key):
            td = self.sampler.apply(params)
            tx = td.sample(seed=key, sample_shape=(self.batch_size,))
            
            ty = objective(tx)
            
            mean_y = jnp.mean(ty)
            standard_dev_y = jnp.std(ty - mean_y)
            
            log_probs = td.log_prob(jax.lax.stop_gradient(tx))
            normalized_rewards = jax.lax.stop_gradient((ty - mean_y) / standard_dev_y)
            loss = -jnp.mean(log_probs[:, jnp.newaxis] * normalized_rewards)
            
            return loss, (ty, mean_y)

        @jax.jit
        def update_step(params, key, opt_state):
            loss_grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
            (loss, (ty, mean_y)), grads = loss_grad_fn(
                params, key)
            
            updates, new_opt_state = rl_opt.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            
            return new_params, new_opt_state, loss, ty, mean_y
        
        for iteration in range(self.iterations):
            self.key, iteration_key = jax.random.split(self.key)
            params, opt_state, loss, ty, mean_y = update_step(
                params, iteration_key, opt_state
            )
            
            log.info(f"[Iteration {iteration}] Average Prediction = {mean_y}")
            
        self.key, final_key = jax.random.split(self.key)
        td = self.sampler.apply(params)
        solution = td.sample(seed=final_key, sample_shape=(self.num_solutions,))
        
        return solution
            
        