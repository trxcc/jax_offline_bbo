import jax 
import jax.numpy as jnp 
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