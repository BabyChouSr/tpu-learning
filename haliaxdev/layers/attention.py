import equinox as eqx
import jax
import jax.numpy as jnp
import haliax as hax
import haliax.nn as hnn

def dot_product_attention(Key, QPos, KPos, q, k, v, *, is_causal):
    # We pass in the string "head_dim", "position", and "key_position"
    # We need to then resolve it so that we can find the size of it.
    QPos = q.resolve_axis(QPos)
    KPos = k.resolve_axis(KPos)
    Key = q.resolve_axis(Key)

    scores = hax.dot(q, k, axis=Key) / jnp.sqrt(Key.size) # sqrt(d_k)

    if is_causal:
        causal_mask = hax.arange(QPos).broadcast_axis(KPos) >= hax.arange(KPos)
        scores = scores - 1e9 * (1.0 - causal_mask)
    
    scores = hnn.softmax(scores, KPos)
    values = hax.dot(scores, v, axis=KPos)
    return values

class Attention(eqx.Module):
    wq: hnn.Linear
    wk: hnn.Linear
    wv: hnn.Linear
    wo: hnn.Linear

    @staticmethod
    def init(Embed, Head, Key, *, key):
        wq = hnn.Linear.init(In=Embed, Out=(Head, Key), key=key)
        wk = hnn.Linear.init(In=Embed, Out=(Head, Key), key=key)
        wv = hnn.Linear.init(In=Embed, Out=(Head, Key), key=key)
        wo = hnn.Linear.init(In=(Head, Key), Out=Embed, key=key)
        return Attention(wq, wk, wv, wo)

    def __call__(self, x, is_causal):
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        # We rename because when we do qk^t, q has shape
        # [bsz, head, position, key_dim]
        # k has SAME shape
        # [bsz, head, position, key_dim]. This causes
        # a name clash which is not allowed to Haliax.
        # We rename with an alias which allows us to do the
        # matmul still.
        k = k.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})

        values = dot_product_attention("head_dim", "position", "key_position", q, k, v, is_causal=is_causal)
        outputs = self.wo(values)
        return outputs




