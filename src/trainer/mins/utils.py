from typing import Optional

import jax 
import jax.numpy as jnp 
import numpy as np 
from jax.random import gumbel, normal
from src._typing import PRNGKeyArray as KeyArray

@jax.jit
def disc_noise(
    x: jnp.ndarray, 
    key: KeyArray, 
    keep: float = 0.9, 
    temp: float = 5.0
) -> jnp.ndarray:
    p = jnp.ones_like(x)
    p = p / jnp.sum(p, axis=-1, keepdims=True)
    p = keep * x + (1.0 - keep) * p
    gumbel_noise = gumbel(key, p.shape)
    return jax.nn.softmax((jnp.log(p) + gumbel_noise) / temp, axis=-1)

@jax.jit 
def cont_noise(
    x: jnp.ndarray, 
    key: KeyArray, 
    noise_std: float
) -> jnp.ndarray:
    return x + noise_std * normal(key, x.shape)

@jax.jit
def adaptive_temp_v2(scores: jnp.ndarray) -> jnp.ndarray:
    """Calculate an adaptive temperature value based on the
    statistics of the scores array
    """
    inverse_arr = scores
    max_score = jnp.max(inverse_arr)
    scores_new = inverse_arr - max_score
    quantile_ninety = jnp.quantile(scores_new, q=0.9)
    return jnp.maximum(jnp.abs(quantile_ninety), 0.001)


@jax.jit
def softmax(arr: jnp.ndarray,
            temp: float = 1.0) -> jnp.ndarray:
    """Calculate the softmax using JAX by normalizing a vector
    to have entries that sum to one
    """
    max_arr = jnp.max(arr)
    arr_new = arr - max_arr
    exp_arr = jnp.exp(arr_new / temp)
    return exp_arr / jnp.sum(exp_arr)


@jax.jit
def _compute_weights(hist: jnp.ndarray,
                    bin_edges: jnp.ndarray,
                    scores_np: jnp.ndarray,
                    base_temp: jnp.ndarray) -> jnp.ndarray:
    """JIT-able portion of weight computation"""
    hist = hist / jnp.sum(hist)
    softmin_prob = softmax(bin_edges[1:], temp=base_temp)

    provable_dist = softmin_prob * (hist / (hist + 1e-3))
    provable_dist = provable_dist / (jnp.sum(provable_dist) + 1e-7)

    bin_indices = jnp.searchsorted(bin_edges[1:], scores_np)
    bin_indices = jnp.clip(bin_indices, 0, 19)
    hist_prob = hist[bin_indices]

    weights = provable_dist[bin_indices] / (hist_prob + 1e-7)
    weights = jnp.clip(weights, a_min=0.0, a_max=5.0)
    return weights[:, jnp.newaxis].astype(jnp.float32)


def get_weights(scores: jnp.ndarray, base_temp: Optional[float] = None) -> jnp.ndarray:
    """Calculate weights used for training a model inversion
    network with a per-sample reweighted objective
    """
    scores_np = scores[:, 0]
    
    # Convert to numpy for histogram calculation
    scores_np_host = jax.device_get(scores_np)
    hist, bin_edges = np.histogram(scores_np_host, bins=20)
    
    # Convert back to JAX arrays
    hist = jnp.array(hist, dtype=jnp.float32)
    bin_edges = jnp.array(bin_edges, dtype=jnp.float32)
    
    if base_temp is None:
        base_temp = adaptive_temp_v2(scores_np)
        
    return _compute_weights(hist, bin_edges, scores_np, base_temp)