import ray
import jax
import numpy as np
import equinox as eqx
import haliax as hax
import haliax.nn as hnn
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

class MLP(eqx.Module): # SwiGLU MLP
    w1: hnn.Linear # [Hidden] -> [GateUp]
    w2: hnn.Linear # [GateUp] -> [Hidden]

    @staticmethod
    def init(Hidden, GateUp, *, key):
        k1, k2 = jax.random.split(key, 2)
        w1 = hnn.Linear.init(In=Hidden, Out=GateUp, key=k1)
        w2 = hnn.Linear.init(In=GateUp, Out=Hidden, key=k2)
        return MLP(w1, w2)

    def __call__(self, x):
        return self.w2(hnn.activations.gelu(self.w1(x)))
    
Batch = hax.Axis("batch", 1)
Hidden = hax.Axis("hidden", 32)
GateUp = hax.Axis("gate_up", 128)
mp_degree = 4

def haliax_tensor_parallelism(mesh):
    key = jax.random.PRNGKey(0)
    mlp = MLP.init(Hidden, GateUp, key=key)

    dp_axis_mapping = {"batch": "data"}
    tp_axis_mapping = {"gate_up": "model"}
    compute_axis_mapping = dp_axis_mapping | tp_axis_mapping
    # NOTE(chris): Some issues here after the shard mapping, loses w1 and w2 as attributes
    mlp = hax.shard_with_axis_mapping(mlp, compute_axis_mapping, mesh)
    print(mlp.w1)
    # print(type(mlp.w2.weight))
    # print(getattr(mlp.w2.weight, "array", None))
    # print(mlp.w2.weight.axes, mlp.w2.weight.shape if hasattr(mlp.w2.weight, "shape") else None)

    x = hax.random.normal(jax.random.PRNGKey(1), (Batch, Hidden))
    x = hax.shard_with_axis_mapping(x, compute_axis_mapping, mesh)

    print("input array sharding")
    jax.debug.visualize_array_sharding(x.array)

    y = mlp(x)
    print(mlp)
    print("output array sharding")
    jax.debug.visualize_array_sharding(y.array)

@ray.remote(resources={"TPU-v5p-8-head": 1})
def run_func_on_cluster():
    mesh = Mesh(np.array(jax.devices()).reshape(1, mp_degree), ("data", "model"))
    haliax_tensor_parallelism(mesh)

if __name__ == "__main__":
    ray.get(run_func_on_cluster.remote())
