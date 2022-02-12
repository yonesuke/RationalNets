import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence

class Rational(nn.Module):
    """
    Rational activation function
    ref: Nicolas Boullé, Yuji Nakatsukasa, and Alex Townsend,
        Rational neural networks,
        arXiv preprint arXiv:2004.01902 (2020).
    
    """
    
    alpha_init = lambda *args: jnp.array([1.1915, 1.5957, 0.5, 0.0218])
    beta_init = lambda *args: jnp.array([2.383, 0.0, 1.0])
    
    @nn.compact
    def __call__(self, x):
        alpha = self.param("alpha", self.alpha_init)
        beta = self.param("beta", self.beta_init)
        return jnp.polyval(alpha, x)/jnp.polyval(beta, x)
    
class RationalMLP(nn.Module):
    """
    Multi layer perceptron with rational activation function.
    """
    features: Sequence[int]
    
    @nn.compact
    def __call__(self, x):
        for feature in self.features[:-1]:
            x = nn.Dense(feature)(x)
            x = Rational()(x)
        x = nn.Dense(self.features[-1])(x)
        return x
