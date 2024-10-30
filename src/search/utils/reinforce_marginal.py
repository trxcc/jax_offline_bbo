import jax
import jax.numpy as jnp
import distrax
from flax import linen as nn

class ContinuousMarginal(nn.Module):
    initial_mean: jnp.ndarray
    initial_logstd: jnp.ndarray
    
    def setup(self):
        """Initialize the parameters of the module"""
        self.mean = self.param('mean', 
                             lambda _: self.initial_mean)
        self.logstd = self.initial_logstd
        
    def get_params(self):
        """Return distribution parameters"""
        return {
            "loc": self.mean,
            "scale_diag": jnp.exp(self.logstd)
        }
    
    def get_distribution(self):
        """Return a multivariate normal distribution"""
        params = self.get_params()
        return distrax.MultivariateNormalDiag(
            loc=params["loc"],
            scale_diag=params["scale_diag"]
        )
    
    def __call__(self):
        """Forward pass - returns the distribution"""
        return self.get_distribution()
    
class DiscreteMarginal(nn.Module):
    initial_logits: jnp.ndarray
    
    def setup(self):
        """Initialize the parameters of the module"""
        self.logits = self.param('logits',
                                lambda _: self.initial_logits)
    
    def get_params(self):
        """Return distribution parameters"""
        return {
            "logits": jax.nn.log_softmax(self.logits)
        }
    
    def get_distribution(self):
        """Return a categorical distribution"""
        params = self.get_params()
        return distrax.Categorical(logits=params["logits"])
    
    def __call__(self):
        """Forward pass - returns the distribution"""
        return self.get_distribution()