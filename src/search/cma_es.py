from typing import Callable, Any

import cma
import jax 
import jax.numpy as jnp 

from src.data.datamodule import JAXDataModule
from src.task.base_task import OfflineBBOExperimenter
from src.search.base_searcher import Searcher
from src.utils.logger import RankedLogger
from src._typing import PRNGKeyArray as KeyArray

log = RankedLogger(__name__, rank_zero_only=True)

class CMAESSearcher(Searcher):
    
    def __init__(
        self, 
        key: KeyArray,
        score_fn: Callable[[Any, jnp.ndarray], jnp.ndarray],
        datamodule: JAXDataModule,
        task: OfflineBBOExperimenter,
        num_solutions: int,
        sigma: float, 
        max_iterations: int
    ) -> None:
        super().__init__(key, score_fn, datamodule, task, num_solutions)
        self.sigma = sigma 
        self.max_iterations = max_iterations
        
    def run(self) -> jnp.ndarray:
        x_init = self.get_initial_designs(self.datamodule.x, self.datamodule.y, self.num_solutions)
        
        # @jax.jit
        def objective(x):
            return self.score_fn(x.reshape(1, -1))
        
        x = x_init
        result = [] 
        for i in range(self.num_solutions):
            xi = x[i].flatten().tolist()
            es = cma.CMAEvolutionStrategy(xi, self.sigma)
            step = 0
            while not es.stop() and step < self.max_iterations:
                solutions = es.ask()
                es.tell(
                    solutions,
                    [objective(jnp.array(xj)).flatten().item() for xj in solutions]
                )
                step += 1
            
            result.append(es.result.xbest)
            log.info(f"CMA: {i + 1} / {self.num_solutions}")
            
        return jnp.array(result)
        