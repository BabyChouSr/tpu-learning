import equinox as eqx
import jax
import jax.numpy as jnp
import haliax as hax
import haliax.nn as hnn

class MLP(eqx.Module): # SwiGLU MLP
    w1: hnn.Linear # [Hidden] -> [GateUp]
    w2: hnn.Linear # [Hidden] -> [GateUp]
    w3: hnn.Linear # [GateUp] -> [Hidden]

    @staticmethod
    def init(Hidden, GateUp, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        w1 = hnn.Linear.init(In=Hidden, Out=GateUp, key=k1)
        w2 = hnn.Linear.init(In=Hidden, Out=GateUp, key=k2)
        w3 = hnn.Linear.init(In=GateUp, Out=Hidden, key=k3)
        return MLP(w1, w2, w3)

    def __call__(self, x):
        hidden_state = hnn.activations.silu(self.w1(x)) * self.w2(x)
        hidden_state = self.w3(hidden_state)
        return hidden_state