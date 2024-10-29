import jax 
import jax.numpy as jnp 

from functools import partial

class NLL:
    
    @partial(jax.jit, static_argnames=['self'])
    def __call__(
        self, 
        mean: jnp.ndarray, 
        std: jnp.ndarray, 
        y: jnp.ndarray
    ) -> jnp.ndarray:
        
        nll = 0.5 * jnp.log(2 * jnp.pi * std**2) + 0.5 * ((y - mean)**2) / (std**2)
        return nll.mean()