import jax
import jax.numpy as jnp

from src.metric.base_metric import Metric

class MSE(Metric):
    """MSE by JAX"""
    def __init__(self) -> None:
        super().__init__(maximize=False)
        self.total_error = 0.0
        self.count = 0
    
    @staticmethod
    @jax.jit
    def _compute_mse(preds: jnp.ndarray, targets: jnp.ndarray) -> float:
        return jnp.mean((preds - targets) ** 2)
    
    def update(self, preds: jnp.ndarray, targets: jnp.ndarray) -> None:
        error = self._compute_mse(preds, targets)
        self.total_error += float(error)
        self.count += 1
    
    def compute(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_error / self.count
    
    def reset(self) -> None:
        self.total_error = 0.0
        self.count = 0