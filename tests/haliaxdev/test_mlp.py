import pytest
import jax
import jax.numpy as jnp

import equinox as eqx

import haliax as hax
import haliax.nn as hnn

from haliaxdev.layers.mlp import MLP


def _to_array(x):
    # Convert possible Haliax NamedArray or nnx Param to raw jnp.ndarray
    if hasattr(x, "array"):
        return jnp.asarray(x.array)
    if hasattr(x, "value"):
        return jnp.asarray(x.value)
    return jnp.asarray(x)


def _extract_linear_params(hax_linear):
    # Try common attribute names
    weight = None
    for name in ("weight", "w", "kernel", "W"):
        if hasattr(hax_linear, name):
            weight = getattr(hax_linear, name)
            break
    assert weight is not None, "Could not find weight param on haliax.nn.Linear"
    bias = getattr(hax_linear, "bias", None)
    w = _to_array(weight)
    b = None if bias is None else _to_array(bias)
    return w, b


def _set_nnx_linear_params(nnx_linear, w, b, in_dim, out_dim):
    # Flax nnx.Linear expects kernel shape (in_dim, out_dim)
    if w.shape == (out_dim, in_dim):
        kernel = w.T
    else:
        kernel = w
    # Assign
    if hasattr(nnx_linear, "kernel") and hasattr(nnx_linear.kernel, "value"):
        nnx_linear.kernel.value = kernel
    elif hasattr(nnx_linear, "kernel"):
        nnx_linear.kernel = kernel
    # Bias handling
    if hasattr(nnx_linear, "bias"):
        if b is None:
            b_val = jnp.zeros((out_dim,), dtype=kernel.dtype)
        else:
            if b.shape != (out_dim,):
                b_val = b.reshape((out_dim,))
            else:
                b_val = b
        if hasattr(nnx_linear.bias, "value"):
            nnx_linear.bias.value = b_val
        else:
            nnx_linear.bias = b_val


@pytest.mark.skipif(pytest.importorskip("flax", reason="Flax not installed") is None, reason="Flax not installed")
def test_swiglu_mlp_matches_flax_nnx():
    from flax import nnx

    key = jax.random.PRNGKey(0)

    # Define axes (sizes must match the model under test)
    Hidden = hax.Axis("hidden", 64)
    GateUp = hax.Axis("gate_up", 128)

    # Initialize Haliax MLP and random input
    mlp = MLP.init(Hidden, GateUp, key=key)

    x_key, _ = jax.random.split(key)
    try:
        x = hax.random.normal(x_key, Hidden)
    except Exception:
        # Fallback: build via jax then wrap if needed
        x = jax.random.normal(x_key, (Hidden.size,))
        if hasattr(hax, "named"):
            try:
                x = hax.named(x, axes=(Hidden,))
            except Exception:
                pass

    y_hax = mlp(x)
    print(mlp)
    y_hax_arr = _to_array(y_hax)

    # Build equivalent Flax NNX model
    class NnxSwiGLU(nnx.Module):
        def __init__(self, in_dim: int, up_dim: int, *, key):
            # k1, k2, k3 = jax.random.split(key, 3)
            rngs = nnx.Rngs(0)
            self.w1 = nnx.Linear(in_dim, up_dim, rngs=rngs)
            self.w2 = nnx.Linear(in_dim, up_dim, rngs=rngs)
            self.w3 = nnx.Linear(up_dim, in_dim, rngs=rngs)

        def __call__(self, x):
            return self.w3(nnx.silu(self.w1(x)) * self.w2(x))

    nnx_mlp = NnxSwiGLU(Hidden.size, GateUp.size, key=key)

    # Copy parameters from Haliax to NNX
    w1, b1 = _extract_linear_params(mlp.w1)
    w2, b2 = _extract_linear_params(mlp.w2)
    w3, b3 = _extract_linear_params(mlp.w3)

    _set_nnx_linear_params(nnx_mlp.w1, w1, b1, Hidden.size, GateUp.size)
    _set_nnx_linear_params(nnx_mlp.w2, w2, b2, Hidden.size, GateUp.size)
    _set_nnx_linear_params(nnx_mlp.w3, w3, b3, GateUp.size, Hidden.size)

    # Prepare input for NNX
    x_arr = _to_array(x)
    y_nnx = nnx_mlp(x_arr)

    # Compare
    assert y_nnx.shape == y_hax_arr.shape
    assert jnp.allclose(y_nnx, y_hax_arr, atol=1e-5, rtol=1e-5)


