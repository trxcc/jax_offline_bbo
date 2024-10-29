from typing import Callable, Any

import numpy as np 
import jax 
import jax.numpy as jnp
import optax 
from omegaconf import DictConfig
from tqdm import tqdm

from src.data.datamodule import JAXDataModule
from src.task.base_task import OfflineBBOExperimenter
from src.search.base_searcher import Searcher

class GradSearcher(Searcher):
    def __init__(
        self, 
        score_fn: Callable[[Any, jnp.ndarray], jnp.ndarray], 
        datamodule: JAXDataModule, 
        task: OfflineBBOExperimenter, 
        num_solutions: int,
        learning_rate: DictConfig,
        search_steps: DictConfig,
        scale_lr: bool = False,
    ) -> None:
        super().__init__(score_fn, datamodule, task, num_solutions)
        if task.is_discrete:
            self.learning_rate = learning_rate["discrete"]
            self.search_steps = search_steps["discrete"]
        else:
            self.learning_rate = learning_rate["continuous"]
            self.search_steps = search_steps["continuous"]
            
        self.scale_lr = scale_lr    
        
         
    def run(self) -> jnp.ndarray:
        x_init = self.get_initial_designs(self.datamodule.x, self.datamodule.y, self.num_solutions)

        if self.scale_lr:
            self.learning_rate = self.learning_rate * \
                jnp.sqrt(jnp.prod(jnp.array(x_init.shape[1:])))

        @jax.jit
        def optimization_step_grad(x):
            def objective(x):
                return self.score_fn(x).sum()  
            
            grad_fn = jax.grad(objective)
            grads = grad_fn(x)
            x = x + self.learning_rate * grads
            return x

        x_opt = x_init
        for _ in range(self.search_steps):
            x_opt = optimization_step_grad(x_opt)

        return x_opt

class AdamSearcher(GradSearcher):
    
    def run(self) -> jnp.ndarray:
        x_init = self.get_initial_designs(self.datamodule.x, self.datamodule.y, self.num_solutions)

        if self.scale_lr:
            self.learning_rate = self.learning_rate * \
                jnp.sqrt(jnp.prod(jnp.array(x_init.shape[1:])))
                
        self.optimizer = optax.adam(self.learning_rate)
        
        opt_state = self.optimizer.init(x_init)

        @jax.jit
        def optimization_step_adam(opt_state, x):
            def objective(x):
                return -self.score_fn(x).sum() 
            
            grad_fn = jax.grad(objective)
            grads = grad_fn(x)
            updates, opt_state = self.optimizer.update(grads, opt_state, x)
            x = optax.apply_updates(x, updates)
            return x, opt_state

        x_opt = x_init
        for _ in range(self.search_steps):
            x_opt, opt_state = optimization_step_adam(opt_state, x_opt)

        return x_opt