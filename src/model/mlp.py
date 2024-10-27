from typing import Any, Callable, Optional, Tuple

import flax.linen as nn 
import jax 
import jax.numpy as jnp 

class MLP(nn.Module):
    """MLP module."""
    
    input_size: int
    hidden_sizes: Tuple[int]
    output_size: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    kernel_init: Callable[..., Any] = jax.nn.initializers.lecun_uniform()
    bias: bool = True
    final_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
    final_kernel_init: Optional[Callable[..., Any]] = None
    
    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        hidden = obs
        
        # Hidden layers
        for hidden_size in self.hidden_sizes:
            hidden = nn.Dense(
                hidden_size,
                kernel_init=self.kernel_init,
                use_bias=self.bias
            )(hidden)
            hidden = self.activation(hidden)
        
        # Output layer
        kernel_init = self.final_kernel_init if self.final_kernel_init is not None else self.kernel_init
        hidden = nn.Dense(
            self.output_size,
            kernel_init=kernel_init,
            use_bias=self.bias
        )(hidden)
        
        if self.final_activation is not None:
            hidden = self.final_activation(hidden)

        return hidden