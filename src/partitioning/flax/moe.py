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

# If we were to just multiply using all of the experts
class GShardMoE(nnx.Module):
	def __init__(self, hidden_dim, intermediate_dim, num_experts, top_k, rngs):
		self.gate = nnx.Linear(hidden_dim, num_experts, rngs=rngs)
		self.shared_experts = MLP(hidden_dim, intermediate_dim, rngs)
		self.experts = [MLP(hidden_dim, intermediate_dim, rngs) for _ in range(num_experts)]
		self.top_k = top_k
		self.num_experts = num_experts
		
	def __call__(self, x):
		# x: [B, S, D]
		B, S, D = x.shape
		# Router: probabilities per expert
		router_probs = jax.nn.softmax(self.gate(x), axis=-1)  # [B, S, E]
		# Top-k per token
		expert_values, expert_indices = jax.lax.top_k(router_probs, k=self.top_k)  # [B, S, K]
		expert_values = expert_values / (jnp.sum(expert_values, axis=-1, keepdims=True) + 1e-9)
		# Compute all experts on all tokens (no capacity limit): [B, S, E, D]
		all_outputs = jnp.stack([expert(x) for expert in self.experts], axis=2)
		# Build per-expert weights and combine
		one_hot = jax.nn.one_hot(expert_indices, self.num_experts)  # [B, S, K, E]
		weights_E = (one_hot * expert_values[..., None]).sum(axis=2)  # [B, S, E]
		moe_out = jnp.einsum('b s e, b s e d -> b s d', weights_E, all_outputs)  # [B, S, D]
		return x + self.shared_experts(x) + moe_out


class DeepseekMoE(nnx.Module):
	def __init__(self, hidden_dim, intermediate_dim, num_experts, top_k, rngs):
		self.gate = nnx.Linear(hidden_dim, num_experts, rngs=rngs)
		self.shared_experts = MLP(hidden_dim, intermediate_dim, rngs)
		self.experts = [MLP(hidden_dim, intermediate_dim, rngs) for _ in range(num_experts)]
		self.top_k = top_k
		self.capacity_factor = 1.25

	def __call__(self, x):
		# x: [B, S, D]
		B, S, D = x.shape
		num_experts = len(self.experts)
		K = self.top_k
		N = B * S  # tokens

		# Router and top-k selection per token
		router_logits = self.gate(x)  # [B, S, E]
		# Prefer softmax probabilities for stability before top_k
		router_probs = jax.nn.softmax(router_logits, axis=-1)
		expert_values, expert_indices = jax.lax.top_k(router_probs, k=K)  # [B, S, K]
		expert_values = expert_values / (jnp.sum(expert_values, axis=-1, keepdims=True) + 1e-9)

		# Flatten tokens
		x_tokens = x.reshape(N, D)                   # [N, D]
		expert_indices_flat = expert_indices.reshape(N * K)  # [N*K]
        # expert_values_flat is how much to weigh each expert
		expert_values_flat = expert_values.reshape(N * K)    # [N*K]

		# Capacity per expert: ceil(capacity_factor * average assignments per expert)
		# average assignments per expert = (N*K)/E
		C = int(jnp.ceil(self.capacity_factor * (N * K) / num_experts))

		# Compute per-pair position in expert via cumulative count over flattened pairs
		oh_e = jax.nn.one_hot(expert_indices_flat, num_experts, dtype=x.dtype)  # [N*K, E]
		cumsum_per_expert = jnp.cumsum(oh_e, axis=0)                           # [N*K, E]
        
        # Position of the token in the Expert's batch
		positions_flat = (cumsum_per_expert * oh_e).sum(axis=1) - 1            # [N*K], 0-based
          
        # Check that the number of tokens is less than the capacity
		within_capacity = positions_flat < C                                    # [N*K]

		# Build dispatch tensor: [N, E, C]
		oh_c = jax.nn.one_hot(jnp.clip(positions_flat, 0, C - 1), C, dtype=x.dtype)  # [N*K, C]

        # For each token / number of experts, for each expert which capacity it takes up
        # if more than capacity, gets masked by within_capacity tensor
		pair_dispatch = (within_capacity.astype(x.dtype)[:, None, None] * oh_e[:, :, None] * oh_c[:, None, :])  # [N*K, E, C]
		dispatch = pair_dispatch.reshape(N, K, num_experts, C).sum(axis=1)  # [N, E, C]

		# Expert inputs: [E, C, D]
		expert_inputs = jnp.einsum('n e c, n d -> e c d', dispatch, x_tokens)

		# Run experts batched per expert
		expert_outputs = jnp.stack([self.experts[e](expert_inputs[e]) for e in range(num_experts)], axis=0)  # [E, C, D]

		# Combine back to tokens with gate weights
        # Gives us the weighted value of each token contribution
		pair_combine = pair_dispatch * expert_values_flat[:, None, None]                 # [N*K, E, C]
          
        # Token contribution
		combine = pair_combine.reshape(N, K, num_experts, C).sum(axis=1)                 # [N, E, C]
          
        # Expert-weighted token outputs
		output_tokens = jnp.einsum('n e c, e c d -> n d', combine, expert_outputs)       # [N, D]

		# Residual/shared path and reshape back
		return x + self.shared_experts(x) + output_tokens.reshape(B, S, D)
    
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