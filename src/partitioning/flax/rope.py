import jax.numpy as jnp
from flax import nnx

def build_rope_cache(seq_len, dim, freq=10000):
    # [1, D // 2]
    inv_freq = 1 / freq ** (jnp.arange(0, dim, 2) / dim)[None, :]

    # [S, 1]
    pos = jnp.arange(0, seq_len)[:, None]

    # [S, D//2]
    m_theta = jnp.einsum("i,j->ij", pos, inv_freq)

    # [S, D]
    cos, sin = jnp.cos(m_theta), jnp.sin(m_theta)
    return cos, sin

def apply_rope(x, cos, sin):
    bsz, seq_len, dim = x.shape
    # For broadcasting the batch dimension
    cos = cos[None, :, :]
    sin = sin[None, :, :]

    x_odd = x[..., 1::2]
    x_even = x[..., 0::2]

    x_rot_odd = cos * x_odd + sin * x_even
    x_rot_even = cos * x_even - sin * x_odd

    x_rot = jnp.empty_like(x)
    x_rot = x_rot.at[..., 0::2].set(x_rot_even)
    x_rot = x_rot.at[..., 1::2].set(x_rot_odd)

    return x_rot

class RotaryEmbedding(nnx.Module):
    def __init__(self, freq, seq_len, dim):
        cos, sin = build_rope_cache(seq_len, dim, freq)

        self.cos = nnx.Variable(collection="constants", value=cos)
        self.sin = nnx.Variable(collection="constants", value=sin)
    
    def __call__(self, x):
        bsz, seq_len, dim = x.shape
        cos = self.cos.value
        sin = self.sin.value

        cos = cos[:seq_len, ...]
        sin = sin[:seq_len, ...]

        x = apply_rope(x, cos, sin)
        return x

