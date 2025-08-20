from flax import nnx
import numpy as np
import jax.numpy as jnp
import jax
import ray
from jax.sharding import NamedSharding, PartitionSpec as P, Mesh

from partitioning.flax.mlp import MLP

# all_outputs: [E, B, S, D]
# expert_indices: [B, S, K]
def gather_topk(all_outputs, expert_indices):
    # Move batch axes first for easier vmapping: [B, S, E, D]
    # token_outputs = jnp.transpose(all_outputs, (1, 2, 0, 3))

    # Gather [K, D] for a single (b, s)
    def gather_for_s(e_d, idx_k):
        return jnp.take(e_d, idx_k, axis=0)  # [K, D]

    # Map over S then B
    gather_for_b = jax.vmap(gather_for_s, in_axes=(0, 0), out_axes=0)     # ([S,E,D], [S,K]) -> [S,K,D]
    gather_all  = jax.vmap(gather_for_b, in_axes=(0, 0), out_axes=0)      # ([B,S,E,D], [B,S,K]) -> [B,S,K,D]

    return gather_all(all_outputs, expert_indices)                   # [B, S, K, D]

class DeepseekMoE(nnx.Module):
    def __init__(self, hidden_dim, intermediate_dim, num_experts, top_k, rngs):
        self.gate = nnx.Linear(hidden_dim, num_experts, rngs=rngs)
        self.shared_experts = MLP(hidden_dim, intermediate_dim, rngs)
        self.experts = [MLP(hidden_dim, intermediate_dim, rngs) for _ in range(num_experts)]
        self.top_k = top_k

    def __call__(self, x):
        # gate = jax.nn.sigmoid()
        # [b, s, d] -> [b, s, e]
        router_logits = jax.nn.sigmoid(self.gate(x))

        # [b, s, k]
        expert_values, expert_indices = jax.lax.top_k(router_logits, k=self.top_k)

        # [b, s, 1]
        expert_normalization = jnp.sum(expert_values, axis=-1, keepdims=True)

        # [b, s, k]
        expert_values /= expert_normalization

        # [e, b, s, d]
        # NOTE(chris): Certainly this can be more optimized with the usage of a kernel
        # so that we can select the exact experts we want here.
        all_outputs = jnp.stack([expert(x) for expert in self.experts], axis=2)
        print(f"all outputs shape: {all_outputs.shape}")

        # [b, s, k, d]
        selected_outputs = gather_topk(all_outputs, expert_indices)
        print(f"selected outputs shape: {selected_outputs.shape}")

        # [b, s, k, 1] * [b, s, k, d]
        expert_weighted_outputs = (expert_values[..., None] * selected_outputs).sum(axis=2)

        return x + self.shared_experts(x) + expert_weighted_outputs
    
@nnx.jit
def create_sharded_model():
    rngs = nnx.Rngs(0)
    hidden_dim = 16
    num_experts = 16
    top_k = 4
    moe = DeepseekMoE(hidden_dim, hidden_dim * 4, num_experts, top_k, rngs)
    state = nnx.state(moe)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(moe, sharded_state)
    return moe

def run_flax_tensor_parallelism(mesh):
    key = jax.random.PRNGKey(0)
    hidden_dim = 16
    seq_len = 16
    with mesh:
        mlp = create_sharded_model()

    x = jax.random.normal(key, (1, seq_len, hidden_dim))
    x = jax.device_put(x, NamedSharding(mesh, P(None)))  # replicate across model

    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    with mesh: # NOTE(chris): Active mesh context required to run SPMD
        # print("input sharding")
        # jax.debug.visualize_array_sharding(x)
        print(x.shape)
        y = mlp(x)

        print("output sharding")
        # jax.debug.visualize_array_sharding(y)
        print(y.shape)
        # y.block_until_ready()

@ray.remote(resources={"TPU-v4-8-head": 1})
def run_func_on_cluster():
    mesh = Mesh(np.array(jax.devices()).reshape(-1,), ("model", ))
    run_flax_tensor_parallelism(mesh)

if __name__ == "__main__":
    ray.get(run_func_on_cluster.remote())