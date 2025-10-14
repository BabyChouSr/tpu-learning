"""
RAY_ADDRESS=http://localhost:8266 ray job submit \
  --runtime-env-json='{"working_dir": "./", "pip": ["einops"]}' \
  -- python kernels/matmul.py
"""

import jax
import jax.numpy as jnp
import math
import numpy as np
import ray
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# where is the padding?
def matmul_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] @ y_ref[...]

def matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bn: int = 128,
):
    # let x = (a, b) and y = (b, c)
    # grid should be divided into [a // bm, b // bk], [b // bk, c // bn]
    a, b = x.shape
    b, c = y.shape

    grid_shape = (math.ceil(a / bm), math.ceil(c / bn))
    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((a, c), dtype=x.dtype),
        grid=grid_shape,
        in_specs=[
            pl.BlockSpec((bm, b), lambda i, j: (i, 0)),
            pl.BlockSpec((b, bn), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
    )(x, y)

def matmul_kernel_3d(x_ref, y_ref, o_ref):
    @pl.when(pl.program_id(2) == 0)
    def _():
        o_ref[...] = jnp.zeros_like(o_ref)

    o_ref[...] += x_ref[...] @ y_ref[...]

def matmul_grid_3d(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bn: int = 128,
    bk: int = 128,
):
    m, k = x.shape
    _, n = y.shape

    # Pad contraction dim (k) for both x (last axis) and y (first axis),
    # and pad y's last axis (n) so it's divisible by bn. Use zero padding.
    # This pads to the block size which is 128
    pad_k = (-k) % bk
    pad_n = (-n) % bn

    if pad_k:
        x = jnp.pad(x, ((0, 0), (0, pad_k)))
        y = jnp.pad(y, ((0, pad_k), (0, 0)))
    if pad_n:
        y = jnp.pad(y, ((0, 0), (0, pad_n)))

    k_padded = k + pad_k
    n_padded = n + pad_n

    grid_shape = (
        math.ceil(m / bm),
        math.ceil(n_padded / bn),
        k_padded // bk,
    )
    out_padded = pl.pallas_call(
        matmul_kernel_3d,
        out_shape=jax.ShapeDtypeStruct((m, n_padded), dtype=x.dtype),
        grid=grid_shape,
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
            pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
    )(x, y)
    return out_padded[:, :n]


@ray.remote(resources={"TPU": 1})
def run_on_tpu():
    k1, k2 = jax.random.split(jax.random.key(0))
    x = jax.random.normal(k1, (1024, 1024))
    y = jax.random.normal(k2, (1024, 1024))
    z = matmul(x, y)
    np.testing.assert_allclose(z, x @ y)

    k1, k2 = jax.random.split(jax.random.key(1))
    x = jax.random.normal(k1, (1025, 1025))
    y = jax.random.normal(k2, (1025, 1025))
    z = matmul(x, y)
    np.testing.assert_allclose(z, x @ y)

    k1, k2 = jax.random.split(jax.random.key(0))
    x = jax.random.normal(k1, (1024, 1024))
    y = jax.random.normal(k2, (1024, 1024))
    z = matmul_grid_3d(x, y)
    np.testing.assert_allclose(z, x @ y)

    k1, k2 = jax.random.split(jax.random.key(1))
    x = jax.random.normal(k1, (1025, 1025))
    y = jax.random.normal(k2, (1025, 1025))
    z = matmul_grid_3d(x, y)
    np.testing.assert_allclose(z, x @ y)

if __name__ == "__main__":
    ray.get(run_on_tpu.remote())