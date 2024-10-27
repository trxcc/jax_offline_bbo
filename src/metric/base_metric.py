import jax.numpy as jnp 

class Metric:
    def __init__(self, maximize: bool = False) -> None:
        self.maximize = maximize
        self.reset()
    
    def update(self, preds: jnp.ndarray, targets: jnp.ndarray) -> None:
        raise NotImplementedError
    
    def compute(self) -> float:
        raise NotImplementedError
    
    def reset(self) -> None:
        raise NotImplementedError