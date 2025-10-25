import jax
import haliax as hax

from haliaxdev.layers.attention import Attention
from haliaxdev.layers.mlp import MLP
from .utils import time_forward


def benchmark_attention():
    key = jax.random.PRNGKey(0)
    Key = hax.Axis("head_dim", 64)
    Embed = hax.Axis("embed", 512)
    Head = hax.Axis("head", 8)
    Pos = hax.Axis("position", 128)

    model = Attention.init(Embed, Head, Key, key=key)
    x_key, _ = jax.random.split(key)
    x = hax.random.normal(x_key, (Pos, Embed))

    def forward(x):
        return model(x, is_causal=False)

    return time_forward(forward, x)


def benchmark_mlp():
    key = jax.random.PRNGKey(0)
    Hidden = hax.Axis("hidden", 512)
    GateUp = hax.Axis("gate_up", 2048)

    model = MLP.init(Hidden, GateUp, key=key)
    x_key, _ = jax.random.split(key)
    x = hax.random.normal(x_key, Hidden)

    def forward(x):
        return model(x)

    return time_forward(forward, x)

