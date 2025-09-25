from functools import partial

import jax
from jax.experimental import pallas as pl
import ray
import jax.numpy as jnp
import numpy as np

def add_vectors_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]

@jax.jit
def add_vectors(x: jax.Array, y: jax.Array):
    return pl.pallas_call(
        add_vectors_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype), # same shape and dtype
    )(x, y)

@ray.remote(resources={"TPU": 1})
def run_on_tpu():
    print(add_vectors(jnp.arange(8), jnp.arange(8)))

if __name__ == "__main__":
    ray.get(run_on_tpu.remote())