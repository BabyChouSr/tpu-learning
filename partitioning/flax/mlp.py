import ray
import numpy as np
from flax import nnx
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

class ColumnParallelLinear(nnx.Module):
    def __init__(self, input_dim, output_dim, rngs):
        column_partitioning = nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            # NamedSharding(mesh, P(None, "model"))
            (None, "model"),
        )
        self.linear = nnx.Linear(input_dim, output_dim, use_bias=False, kernel_init=column_partitioning, rngs=rngs)
    
    def __call__(self, x):
        return self.linear(x)

class RowParallelLinear(nnx.Module):
    def __init__(self, input_dim, output_dim, rngs):
        row_partitioning = nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            # NamedSharding(mesh, P("model", None))
            ("model", None),
        )
        self.linear = nnx.Linear(input_dim, output_dim, use_bias=False, kernel_init=row_partitioning, rngs=rngs)
    
    def __call__(self, x):
        return self.linear(x)

class MLP(nnx.Module): # Megatron-style partitioning
    def __init__(self, hidden_dim, intermediate_dim, rngs):
        self.w1 = ColumnParallelLinear(hidden_dim, intermediate_dim, rngs)
        self.act = nnx.gelu
        self.w2 = RowParallelLinear(intermediate_dim, hidden_dim, rngs)
        # self.mesh = mesh

    def __call__(self, x):
        # Let X be the axis in which we are sharding.
        # [B, H] * [H, D_X] => [B, D_X]
        x = self.w1(x)

        # This actually isn't needed since there is no communication happening within this op.
        # x = jax.lax.with_sharding_constraint(x, P(None, "model")) # Column parallel sharded

        # [B, D_X] => [B, D_X]
        x = self.act(x)

        # [B, D_X] * [D_X, H] => [B, H]_X => all-reduce [B, H]
        x = self.w2(x)
        # NOTE(chris): I noticed that writing just jax.lax_with_sharding_constraint P(None) was not sufficient. Why?
        # Probably because I wasn't creating the model with a sharded state.
        # "model" ends up being an unbound axis, but this actually isn't needed.
        # To fix, maybe we need to use nnx.shard_map or jax.pmap with the in_shardings and out_sharding specified?
        # x = jax.lax.psum(x, "model") # All-reduce row parallel
        return x

# NOTE(chris): This is needed to get the sharding of the model to work.
@nnx.jit
def create_sharded_model():
    rngs = nnx.Rngs(0)
    hidden_dim = 16
    mlp = MLP(hidden_dim, 4 * hidden_dim, rngs)
    state = nnx.state(mlp)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(mlp, sharded_state)
    return mlp

def run_flax_tensor_parallelism(mesh):

    key = jax.random.PRNGKey(0)
    hidden_dim = 16
    with mesh:
        mlp = create_sharded_model()

    x = jax.random.normal(key, (1, hidden_dim))
    x = jax.device_put(x, NamedSharding(mesh, P(None)))  # replicate across model

    with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        with mesh: # NOTE(chris): Active mesh context required to run SPMD
            print("input sharding")
            jax.debug.visualize_array_sharding(x)
            y = mlp(x)

            print("output sharding")
            jax.debug.visualize_array_sharding(y)
            y.block_until_ready()

@ray.remote(resources={"TPU-v6e-8-head": 1})
def run_func_on_cluster():
    mesh = Mesh(np.array(jax.devices()).reshape(-1,), ("model", ))
    run_flax_tensor_parallelism(mesh)

if __name__ == "__main__":
    ray.get(run_func_on_cluster.remote())