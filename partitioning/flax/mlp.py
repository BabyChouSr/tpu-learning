import ray
import numpy as np
from flax import nnx
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

class ColumnParallelLinear(nnx.Module):
    def __init__(self, input_dim, output_dim, mesh, rngs):
        column_partitioning = nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            NamedSharding(mesh, P(None, "model"))
        )
        self.linear = nnx.Linear(input_dim, output_dim, use_bias=False, kernel_init=column_partitioning, rngs=rngs)
    
    def __call__(self, x):
        return self.linear(x)

class RowParallelLinear(nnx.Module):
    def __init__(self, input_dim, output_dim, mesh, rngs):
        row_partitioning = nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            NamedSharding(mesh, P("model", None))
        )
        self.linear = nnx.Linear(input_dim, output_dim, use_bias=False, kernel_init=row_partitioning, rngs=rngs)
    
    def __call__(self, x):
        return self.linear(x)

class MLP(nnx.Module): # Megatron-style partitioning
    def __init__(self, hidden_dim, intermediate_dim, mesh, rngs):
        self.w1 = ColumnParallelLinear(hidden_dim, intermediate_dim, mesh, rngs)
        self.act = nnx.gelu
        self.w2 = RowParallelLinear(intermediate_dim, hidden_dim, mesh, rngs)
        self.mesh = mesh

    def __call__(self, x):
        x = self.w1(x)
        x = jax.lax.with_sharding_constraint(x, NamedSharding(self.mesh, P(None, "model"))) # Column parallel sharded
        x = self.act(x)
        x = self.w2(x)
        # NOTE(chris): I noticed that writing just jax.lax_with_sharding_constraint P(None) was not sufficient. Why?
        x = jax.lax.psum(x, "model") # All-reduce row parallel
        return x
    
# def mlp_forward(model, x):
#     return model(x)

def run_flax_tensor_parallelism(mesh):
    rngs = nnx.Rngs(0)
    hidden_dim = 16
    mlp = MLP(hidden_dim, 4 * hidden_dim, mesh, rngs)

    key = jax.random.PRNGKey(0)
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

@ray.remote(resources={"TPU-v5p-8-head": 1})
def run_func_on_cluster():
    mesh = Mesh(np.array(jax.devices()).reshape(4,), ("model", ))
    run_flax_tensor_parallelism(mesh)

if __name__ == "__main__":
    ray.get(run_func_on_cluster.remote())