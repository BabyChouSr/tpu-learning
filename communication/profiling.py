import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map
import numpy as np
import time

mesh = Mesh(np.array(jax.devices()).reshape(1, -1), ("data", "model"))

# Define shard_map functions for collective operations
def all_gather_fn(x):
    return shard_map(
        lambda x: jax.lax.all_gather(x, "model"),
        mesh=mesh,
        in_specs=P(None, "model"),
        out_specs=P(None, None),
        check_rep=False
    )(x)

def psum_fn(x):
    return shard_map(
        lambda x: jax.lax.psum(x, "model"),
        mesh=mesh,
        in_specs=P(None, "model"),
        out_specs=P(None, "model"),
        check_rep=False
    )(x)

jax.profiler.start_trace("/tmp/profile-data")
x = jax.random.normal(jax.random.PRNGKey(0), (1, 1024))
x = jax.device_put(x, NamedSharding(mesh, P(None, "model")))
print(jax.debug.visualize_array_sharding(x))

start = time.time()
all_gathered_x = all_gather_fn(x)
all_gathered_x.block_until_ready()
end = time.time()
print(f"Time taken to all gather: {end - start}")

x = jax.random.normal(jax.random.PRNGKey(1), (1, 1024))
x = jax.device_put(x, NamedSharding(mesh, P(None, "model")))

start = time.time()
z = psum_fn(x)
z.block_until_ready()
end = time.time()
print(f"Time taken for psum (all-reduce): {end - start}")

jax.profiler.stop_trace()
