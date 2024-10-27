from typing import List
import jax
import jax.numpy as jnp

from src.metric.base_metric import Metric
from src._typing import PRNGKeyArray as KeyArray

class AUPRC(Metric):
    """Area Under the Precision-Recall Curve implemented in JAX"""
    def __init__(self, max_samples: int = 10000):
        super().__init__(maximize=True)
        self.max_samples = max_samples
        self.reset()

    @staticmethod
    @jax.jit
    def _calculate_percentage_overlap(indices1: jnp.ndarray, indices2: jnp.ndarray) -> jnp.ndarray:
        n = indices1.shape[0]
        
        match_matrix = jnp.equal(
            indices1[:, jnp.newaxis], 
            indices2[jnp.newaxis, :]
        )
        
        cumsum_matches = jnp.cumsum(
            jnp.cumsum(match_matrix, axis=0), 
            axis=1
        )
        
        diagonal_matches = jnp.diagonal(cumsum_matches)
        
        indices = jnp.arange(1, n + 1)
        percentages = diagonal_matches / indices
        
        return percentages

    @staticmethod
    @jax.jit
    def _trapz(y: jnp.ndarray, dx: float = 1.0) -> float:
        y_avg = (y[:-1] + y[1:]) / 2
        return jnp.sum(y_avg) * dx

    @staticmethod
    @jax.jit
    def _compute_auprc_internal(y1: jnp.ndarray, y2: jnp.ndarray) -> float:
        """Compute AUPRC for already sampled data"""
        indices_1 = jnp.argsort(y1)
        indices_2 = jnp.argsort(y2)
        
        data = AUPRC._calculate_percentage_overlap(indices_1, indices_2)
        area = AUPRC._trapz(data, dx=1/y1.shape[0])
        
        return area

    @staticmethod
    def _compute_auprc(y1: jnp.ndarray, y2: jnp.ndarray, rng: KeyArray, max_samples: int) -> float:
        """Compute AUPRC with sampling"""
        y1 = y1.reshape(-1)
        y2 = y2.reshape(-1)
        
        if len(y1) > max_samples:
            indices = jax.random.permutation(rng, len(y1))[:max_samples]
            y1 = y1[indices]
            y2 = y2[indices]
        
        return AUPRC._compute_auprc_internal(y1, y2)

    def update(self, preds: jnp.ndarray, targets: jnp.ndarray) -> None:
        self.preds.append(preds.reshape(-1))
        self.targets.append(targets.reshape(-1))

    def compute(self) -> float:
        if not self.preds:
            return 0.0
        
        all_preds = jnp.concatenate(self.preds)
        all_targets = jnp.concatenate(self.targets)
        
        rng = jax.random.PRNGKey(0)
        
        return float(self._compute_auprc(
            all_preds, 
            all_targets, 
            rng, 
            self.max_samples
        ))

    def reset(self) -> None:
        self.preds = []
        self.targets = []