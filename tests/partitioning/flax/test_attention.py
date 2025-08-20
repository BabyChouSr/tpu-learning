import jax
import jax.numpy as jnp
from flax import nnx
from einops import rearrange

from partitioning.flax.attention import MegatronAttention


class BaselineAttention(nnx.Module):
    def __init__(self, hidden_dim: int, head_dim: int, num_heads: int, rngs: nnx.Rngs):
        self.wqkv = nnx.Linear(hidden_dim, head_dim * num_heads * 3, use_bias=False, kernel_init=nnx.initializers.xavier_uniform(), rngs=rngs)
        self.wo = nnx.Linear(head_dim * num_heads, hidden_dim, use_bias=False, kernel_init=nnx.initializers.xavier_uniform(), rngs=rngs)
        self.num_heads = num_heads
        self.head_dim = head_dim

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        qkv = self.wqkv(x)
        qkv = rearrange(qkv, 'b s (a h d_head) -> b s a h d_head', a=3, h=self.num_heads)
        q, k, v = jnp.split(qkv, 3, axis=2)
        q, k, v = q.squeeze(2), k.squeeze(2), v.squeeze(2)
        q = rearrange(q, 'b s h d -> b h s d')
        k = rearrange(k, 'b s h d -> b h d s')
        v = rearrange(v, 'b s h d -> b h s d')
        scores = jnp.einsum('bhqd,bhdk->bhqk', q, k) / jnp.sqrt(self.head_dim)
        scores = jax.nn.softmax(scores, axis=-1)
        values = jnp.einsum('bhqk,bhkd->bhkd', scores, v)
        values = rearrange(values, 'b h s d_head -> b s (h d_head)', h=self.num_heads)
        return self.wo(values)


def test_attention_matches_non_tensor_parallel():
    hidden_dim = 64
    num_heads = 8
    head_dim = hidden_dim // num_heads

    # Same seeds for deterministic parameter init and inputs
    rngs_tp = nnx.Rngs(0)
    rngs_ref = nnx.Rngs(0)
    key = jax.random.PRNGKey(0)

    attn_tp = MegatronAttention(hidden_dim, head_dim, num_heads, rngs_tp)
    attn_ref = BaselineAttention(hidden_dim, head_dim, num_heads, rngs_ref)

    x = jax.random.normal(key, (2, 16, hidden_dim))

    y_tp = attn_tp(x)
    y_ref = attn_ref(x)

    assert jnp.allclose(y_tp, y_ref, atol=1e-5, rtol=1e-5), (
        f"Tensor-parallel attention output differs from baseline.\n"
        f"max|diff|={jnp.max(jnp.abs(y_tp - y_ref))}"
    )
