from typing import Callable

import abc 
import numpy as np 
import jax
import jax.numpy as jnp 

from src.task.base_task import OfflineBBOExperimenter
from src.data.datamodule import JAXDataModule

class Searcher:
    def __init__(
        self, 
        score_fn: Callable,
        datamodule: JAXDataModule,
        task: OfflineBBOExperimenter,
        num_solutions: int,
    ):
        self.score_fn = score_fn
        self.datamodule = datamodule
        self.task = task 
        
        self.num_solutions = num_solutions
    
    @staticmethod
    # @jax.jit
    def get_initial_designs(x: jnp.ndarray, y:jnp.ndarray, k: int = 128) -> jnp.ndarray:
        indices = jnp.argsort(y.flatten())[-k:] 
        return x[indices]
    
    @abc.abstractmethod
    def run(self) -> np.ndarray:
        pass 
