import ray
import numpy as np
import jax.numpy as jnp
import jax
from flax import nnx
from einops import rearrange
from jax.profiler import TraceAnnotation
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import pjit

from partitioning.flax.mlp import ColumnParallelLinear, RowParallelLinear

# Megatron-style Multi-head Attention
class MegatronAttention(nnx.Module):
    def __init__(self, hidden_dim, head_dim, num_heads, rngs):
        self.wqkv = ColumnParallelLinear(hidden_dim, head_dim * num_heads * 3, rngs)
        self.wo = RowParallelLinear(head_dim * num_heads, hidden_dim, rngs)
        self.num_heads = num_heads
        self.head_dim = head_dim

    def __call__(self, x):
        # [B, S, D] -> [B, S, 3D_X]
        with TraceAnnotation("qkv chunk"):
            qkv = self.wqkv(x)
        # a is just axis for q, k, v
        qkv = rearrange(qkv, 'b s (a h d_head) -> b s a h d_head', a=3, h=self.num_heads)
        q, k, v = jnp.split(qkv, 3, axis=2)

        # [b, s, 1, H_X, D] -> [b, s, H_X, D]
        q, k, v = q.squeeze(2), k.squeeze(2), v.squeeze(2)
        q = rearrange(q, "b s h d->b h s d")
        k = rearrange(k, "b s h d ->b h d s") # kT
        v = rearrange(v, "b s h d ->b h s d")

        # Use q, k as a placeholder for the position of query and key
        with TraceAnnotation("qk matmul"):
            scores = jnp.einsum('bhqd,bhdk->bhqk', q, k) / jnp.sqrt(self.head_dim)
        
        with TraceAnnotation("softmax"):
            scores = jax.nn.softmax(scores, axis=-1)
        values = jnp.einsum('bhqk,bhkd->bhkd', scores, v)
        values = rearrange(values, "b h s d_head -> b s (h d_head)", h=self.num_heads)

        with TraceAnnotation("out proj"):
            outputs = self.wo(values)

        return outputs

class GroupedQueryAttention(nnx.Module):
    def __init__(self, hidden_dim, head_dim, num_heads, num_kv_heads, rngs):
        self.wq = ColumnParallelLinear(hidden_dim, hidden_dim, rngs)

        kv_size = hidden_dim * num_kv_heads // num_heads
        self.wk = ColumnParallelLinear(hidden_dim, kv_size, rngs)
        self.wv = ColumnParallelLinear(hidden_dim, kv_size, rngs)
        self.wo = RowParallelLinear(kv_size, hidden_dim, rngs)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_head_groups = num_heads // num_kv_heads
        self.head_dim = head_dim
    
    def __call__(self, x):
        # [b, s, d] -> [b, s,]
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        
        # Difference now is that the number of kv heads is different between q and k, v
        q = rearrange(q, "b s (h d_head)->b h s d_head", h=self.num_heads)
        k = rearrange(k, "b s (h d_head)->b h d_head s", h=self.num_kv_heads)
        v = rearrange(v, "b s (h d_head)->b h s d_head", h=self.num_kv_heads)

        # Rearrange into groups of heads
        # h in this case will become the num_kv_heads and g is the number of groups
        q = rearrange(q, "b (h g) s d_head -> b g h s d_head", g=self.num_head_groups)

        # Omit the g in the output will sum across that dimension. This means that for each kv head
        # we sum across the group of query heads.
        scores = jnp.einsum("b g h q d,b h d k->b h q k", q, k) / jnp.sqrt(self.head_dim)
        scores = jax.nn.softmax(scores, axis=-1)
        values = jnp.einsum("bhqk,bhkd->bhqd", scores, v)
        values = rearrange(values, "b h q d->b q (h d)", h=self.num_kv_heads)
        outputs = self.wo(values)
        return outputs
    
@nnx.jit(static_argnames=("hidden_dim", "head_dim", "num_heads", "attention_type"))
def create_sharded_attention(hidden_dim, head_dim, num_heads, attention_type):
    if attention_type == "megatron":
        attention = MegatronAttention(hidden_dim, head_dim, num_heads, nnx.Rngs(0))
    elif attention_type == "gqa":
        # 4 query heads per 1 key and value head
        attention = GroupedQueryAttention(hidden_dim, head_dim, num_heads, num_heads // 4, nnx.Rngs(0))

    state = nnx.state(attention)
    pspec = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspec)
    nnx.update(attention, sharded_state)
    return attention


def megatron_attention_parallelism():
    mesh = Mesh(np.array(jax.devices()).reshape(-1,), ("model", ))

    hidden_dim = 1024
    num_heads = 8
    head_dim = hidden_dim // num_heads
    with jax.profiler.trace('/tmp/jax-trace', create_perfetto_link=True, create_perfetto_trace=True):
        with mesh:
            attention = create_sharded_attention(hidden_dim, head_dim, num_heads, "gqa")
            x = jax.random.normal(jax.random.PRNGKey(0), (1, 16, hidden_dim))
            # Compile the forward with pjit and print compiler IR text
            def forward(inp):
                return attention(inp)

            forward_pjit = pjit.pjit(
                forward,
                in_shardings=NamedSharding(mesh, P(None, None, None)),
                out_shardings=NamedSharding(mesh, P(None, None, None)),
            )

            # lowered = forward_pjit.lower(x)
            print(forward_pjit.lower(x).compile().as_text())

            # NOTE(chris): Actually, using the forward_pjit does not lead to 
            # a very informational trace. Seems like need to use the eager version of it
            # We do see the all-reduce at the end of the out_proj trace which is what we want
            # Execute once after compilation
            # y = forward_pjit(x)
            y = attention(x)
            # print(y.shape)
            y.block_until_ready()

        # non_sharded_y = non_sharded_attention(x)
        # assert jnp.allclose(y, non_sharded_y), f"Not close, tensor parallel: {y}, normal: {non_sharded_y}"

@ray.remote(resources={"TPU-v4-8-head": 1})
def run_func_on_cluster():
    megatron_attention_parallelism()

if __name__ == "__main__":
    ray.get(run_func_on_cluster.remote())
    # megatron_attention_parallelism()