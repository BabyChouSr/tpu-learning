import pytest
import numpy as np
import jax
import jax.numpy as jnp

import haliax as hax
from haliaxdev.layers.attention import Attention, dot_product_attention


def _to_array(x):
    # Convert possible Haliax NamedArray or parameter wrapper to raw jnp.ndarray
    if hasattr(x, "array"):
        return jnp.asarray(x.array)
    if hasattr(x, "value"):
        return jnp.asarray(x.value)
    return jnp.asarray(x)


@pytest.mark.skipif(pytest.importorskip("torch", reason="PyTorch not installed") is None, reason="PyTorch not installed")
@pytest.mark.parametrize("is_causal", [False, True])
def test_attention_matches_pytorch_ground_truth(is_causal):
    import torch

    # Define axes
    Key = hax.Axis("head_dim", 16)
    Embed = hax.Axis("embed", 128)
    Head = hax.Axis("head", 8)
    QPos = hax.Axis("position", 32)
    KPos = QPos.alias("key_position")

    # Initialize module and random input
    key = jax.random.PRNGKey(0)
    attention = Attention.init(Embed, Head, Key, key=key)

    x_key, _ = jax.random.split(key)
    x = hax.random.normal(x_key, (QPos, Embed))

    # Compute Q, K, V via Haliax linears to ensure identical projection params
    q = attention.wq(x)
    k = attention.wk(x)
    v = attention.wv(x)

    # Rename for Haliax attention and compute reference values
    k_named = k.rename({"position": "key_position"})
    v_named = v.rename({"position": "key_position"})
    values_hax = dot_product_attention("head_dim", "position", "key_position", q, k_named, v_named, is_causal=is_causal)

    # Convert to NumPy then Torch, asserting expected shapes
    q_np = np.asarray(_to_array(q))  # (q_pos, head, key)
    k_np = np.asarray(_to_array(k_named))  # (k_pos, head, key)
    v_np = np.asarray(_to_array(v_named))  # (k_pos, head, key)
    assert q_np.shape == (QPos.size, Head.size, Key.size)
    assert k_np.shape == (KPos.size, Head.size, Key.size)
    assert v_np.shape == (KPos.size, Head.size, Key.size)

    q_t = torch.from_numpy(q_np).to(dtype=torch.float32)  # (q_pos, head, key)
    k_t = torch.from_numpy(k_np).to(dtype=torch.float32)  # (k_pos, head, key)
    v_t = torch.from_numpy(v_np).to(dtype=torch.float32)  # (k_pos, head, key)

    # PyTorch attention: [head, q_pos, key] x [head, key, k_pos] -> [head, q_pos, k_pos]
    q_th = q_t.permute(1, 0, 2)  # (head, q_pos, key)
    k_th = k_t.permute(1, 2, 0)  # (head, key, k_pos)
    v_th = v_t.permute(1, 0, 2)  # (head, k_pos, key)

    scale = float(Key.size) ** 0.5
    scores = torch.matmul(q_th, k_th) / scale  # (head, q_pos, k_pos)

    if is_causal:
        q_len = q_th.shape[1]
        k_len = v_th.shape[1]
        causal_mask = (torch.arange(q_len).unsqueeze(1) >= torch.arange(k_len).unsqueeze(0))  # (q_pos, k_pos)
        scores = scores.masked_fill(~causal_mask.to(dtype=torch.bool).unsqueeze(0), -1e9)

    attn = torch.softmax(scores, dim=-1)
    values_t = torch.matmul(attn, v_th)  # (head, q_pos, key)
    values_t = values_t.permute(1, 0, 2).contiguous()  # (q_pos, head, key)

    # Compare against Haliax attention values
    values_hax_np = np.asarray(_to_array(values_hax))
    assert values_hax_np.shape == (QPos.size, Head.size, Key.size)
    np.testing.assert_allclose(values_t.numpy(), values_hax_np, atol=1e-5, rtol=1e-5)
