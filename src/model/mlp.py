from typing import Any, Callable, Optional, Tuple, Sequence

import flax.linen as nn 
import jax 
import jax.numpy as jnp 

from src.task.base_task import OfflineBBOExperimenter

class MLP(nn.Module):
    """MLP module."""
    
    task: OfflineBBOExperimenter
    hidden_sizes: Sequence[int]
    output_size: int
    if_embedding: bool = False 
    embedding_size: int = 50
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    kernel_init: Callable[..., Any] = jax.nn.initializers.lecun_uniform()
    bias: bool = True
    final_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
    final_kernel_init: Optional[Callable[..., Any]] = None
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.task.is_discrete:
            x = nn.Embed(
                num_embeddings=self.task.num_classes,
                features=self.embedding_size,
            )(x)
        
        x = x.reshape((x.shape[0], -1))
        
        # Shared layers
        hidden = x
        
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
    
class DualHeadMLP(nn.Module):
    
    task: OfflineBBOExperimenter
    hidden_sizes: Sequence[int]
    output_size: int = 1
    if_embedding: bool = False 
    embedding_size: int = 50
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    kernel_init: Callable[..., Any] = jax.nn.initializers.lecun_uniform()
    bias: bool = True
    initial_max_std: float = 0.2
    initial_min_std: float = 0.1
    final_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
    final_kernel_init: Optional[Callable[..., Any]] = None
    std_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self.task.is_discrete:
            x = nn.Embed(
                num_embeddings=self.task.num_classes,
                features=self.embedding_size,
            )(x)
        
        x = x.reshape((x.shape[0], -1))
        
        # Shared layers
        hidden = x
        for hidden_size in self.hidden_sizes:
            hidden = nn.Dense(
                hidden_size,
                kernel_init=self.kernel_init,
                use_bias=self.bias
            )(hidden)
            hidden = self.activation(hidden)

        # Use final_kernel_init if specified, otherwise use default kernel_init
        kernel_init = self.final_kernel_init if self.final_kernel_init is not None else self.kernel_init

        # Mean head
        mean = nn.Dense(
            self.output_size,
            kernel_init=kernel_init,
            use_bias=self.bias,
            name='mean_layer'
        )(hidden)

        if self.final_activation is not None:
            mean = self.final_activation(mean)

        # Logstd head
        logstd = nn.Dense(
            self.output_size,
            kernel_init=kernel_init,
            use_bias=self.bias,
            name='logstd_layer'
        )(hidden)

        # Parameters for logstd clamping
        max_logstd = self.param(
            'max_logstd',
            lambda _: jnp.full((1, 1), jnp.log(self.initial_max_std))
        )
        min_logstd = self.param(
            'min_logstd',
            lambda _: jnp.full((1, 1), jnp.log(self.initial_min_std))
        )

        # Clamp logstd using softplus
        logstd = max_logstd - jax.nn.softplus(max_logstd - logstd)
        logstd = min_logstd + jax.nn.softplus(logstd - min_logstd)

        if self.std_activation is not None:
            logstd = self.std_activation(logstd)

        return mean, jnp.exp(logstd)

    def get_distribution(self, obs: jnp.ndarray, params: Any) -> Any:
        """
        Returns a Normal distribution using the forward pass results.
        Note: This requires installing `distrax` for JAX distributions.
        """
        import distrax
        mean, std = self.apply(params, obs)
        return distrax.Normal(mean, std)