# RationalNets
JAX/Flax implementation of rational neural nets.

Original
- paper: Nicolas Boullé, Yuji Nakatsukasa, and Alex Townsend, [Rational neural networks](https://arxiv.org/abs/2004.01902), arXiv preprint arXiv:2004.01902 (2020).
- github: https://github.com/NBoulle/RationalNets


## Installation
RationalNets can be installed with pip directly from GitHub, with the following command:
```
pip install git+https://github.com/yonesuke/RationalNets.git
```

## QuickStart
```python
import jax.numpy as jnp
from jax import random
from rationalnets import RationalMLP

model = RationalMLP([12, 8, 4])
batch = jnp.ones((32, 10))
variables = model.init(random.PRNGKey(0), batch)
output = model.apply(variables, batch)
```
