import math
import jax
import jax.numpy as jnp
from partitioning.flax.paged_attention import ragged_paged_attention
from jax.experimental.pallas.ops.tpu.ragged_paged_attention import ref_ragged_paged_attention

# queries, # [num_tokens, num_q_heads, head_dim]
# kv_pages, # [num_pages, num_slots, num_kv_heads, head_dim]
# page_indices, # [num_seqs, num_pages_per_seq]
# kv_lens, # [num_seqs]
# cu_q_lens, # [num_seqs + 1]
# num_seqs, # int
# *,
# sm_scale=1.0,
# mask_value=-1e7,

_MAX_NUM_SEQS = 6
_NUM_PAGES = 4
_NUM_SLOTS = 4
_NUM_PAGES_PER_SEQ = 2
_NUM_QUERY_HEADS = 4
_NUM_KV_HEADS = 2
_HEAD_DIM = 64
_SEQ_LEN = 4
_NUM_SEQS = 3
_NUM_QUERY_TOKENS = 3

def test_ragged_paged_attention():
    key = jax.random.key(0)

    # One query token per sequence: queries shape [num_tokens, num_q_heads, head_dim]
    key, q_key = jax.random.split(key, 2)
    queries = jax.random.normal(q_key, (_NUM_SEQS, _NUM_QUERY_HEADS, _HEAD_DIM))

    # kv_pages shape [num_pages, num_slots, combined_kv_heads, head_dim]
    # combined_kv_heads = 2 * num_kv_heads (interleaved K,V along head axis)
    key, kv_key = jax.random.split(key, 2)
    kv_pages = jax.random.normal(kv_key, (_NUM_PAGES, _NUM_SLOTS, 2 * _NUM_KV_HEADS, _HEAD_DIM))

    # Page mapping per sequence: seq0 -> [0,1], seq1 -> [2,0], seq2 -> [3,0]
    page_indices = jnp.zeros((_MAX_NUM_SEQS, _NUM_PAGES_PER_SEQ), dtype=jnp.int32)
    page_indices = page_indices.at[0].set(jnp.array([0, 1], dtype=jnp.int32))
    page_indices = page_indices.at[1].set(jnp.array([2, 0], dtype=jnp.int32))
    page_indices = page_indices.at[2].set(jnp.array([3, 0], dtype=jnp.int32))
    print(page_indices.shape)

    # KV lengths per sequence
    kv_lens = jnp.zeros((_MAX_NUM_SEQS,), dtype=jnp.int32)
    kv_lens = kv_lens.at[0].set(5)
    kv_lens = kv_lens.at[1].set(1)
    kv_lens = kv_lens.at[2].set(2)

    # Cumulative query lengths (one query per sequence)
    cu_q_lens = jnp.arange(_NUM_SEQS, dtype=jnp.int32)
    
    cu_q_lens = jnp.arange(_MAX_NUM_SEQS + 1, dtype=jnp.int32)

    attn_output = ragged_paged_attention(
        queries,
        kv_pages,
        kv_lens,
        page_indices,
        cu_q_lens,
        _NUM_SEQS,
    )

    breakpoint()

    num_seqs_array = jnp.array(_NUM_SEQS, dtype=jnp.int32).reshape(1,) # needed to conform to api
    ref_attn_output = ref_ragged_paged_attention(
        queries,
        kv_pages,
        kv_lens,
        page_indices,
        cu_q_lens,
        num_seqs_array,
    )

    # Check shapes match and values are close
    assert attn_output.shape == ref_attn_output.shape
    assert jnp.allclose(attn_output, ref_attn_output, rtol=1e-4, atol=1e-4)


