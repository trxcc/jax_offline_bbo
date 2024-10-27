import jax
import jax.numpy as jnp

from src.metric.base_metric import Metric

class SpearmanCorrelation(Metric):
    """Spearman correlation coefficient by JAX"""
    def __init__(self) -> None:
        super().__init__(maximize=True)
        self.preds = []
        self.targets = []
    
    def update(self, preds: jnp.ndarray, targets: jnp.ndarray) -> None:
        self.preds.append(preds.reshape(-1))
        self.targets.append(targets.reshape(-1))
    
    @staticmethod
    @jax.jit
    def _compute_rank(x: jnp.ndarray) -> jnp.ndarray:
        """Compute rank and deal with ties"""
        sorted_idx = jnp.argsort(x)
        ranks = jnp.zeros_like(x)
        ranks = ranks.at[sorted_idx].set(jnp.arange(len(x)))
        
        diff = jnp.concatenate([jnp.array([1.]), jnp.diff(x[sorted_idx])])
        ties = diff == 0
        
        # Remove the conditional and use pure array operations
        tie_groups = jnp.concatenate([jnp.array([True]), ties[:-1]])
        tie_sizes = jnp.cumsum(tie_groups)
        tie_groups = jnp.concatenate([ties[1:], jnp.array([True])])
        tie_sizes = tie_sizes * tie_groups - tie_sizes * ~tie_groups
        
        # Use where instead of conditional assignment
        ranks = jnp.where(
            jnp.any(ties),
            ranks - tie_sizes + (tie_sizes + 1) / 2,
            ranks
        )
            
        return ranks
    

    @staticmethod
    @jax.jit
    def _spearman_correlation(x: jnp.ndarray, y: jnp.ndarray) -> float:
        """Calculate Spearman correlation of two array"""
        rank_x = SpearmanCorrelation._compute_rank(x)
        rank_y = SpearmanCorrelation._compute_rank(y)
        
        n = len(x)
        mean_x = jnp.mean(rank_x)
        mean_y = jnp.mean(rank_y)
        
        numerator = jnp.sum((rank_x - mean_x) * (rank_y - mean_y))
        denominator = jnp.sqrt(jnp.sum((rank_x - mean_x) ** 2) * jnp.sum((rank_y - mean_y) ** 2))
        
        correlation = jnp.where(denominator == 0, 0.0, numerator / denominator)
        
        return correlation

    def compute(self) -> float:
        if not self.preds:
            return 0.0
        
        all_preds = jnp.concatenate(self.preds)
        all_targets = jnp.concatenate(self.targets)
        
        return float(self._spearman_correlation(all_preds, all_targets))
    
    def reset(self) -> None:
        self.preds = []
        self.targets = []