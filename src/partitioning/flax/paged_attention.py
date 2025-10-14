import jax
import jax.numpy as jnp

def ragged_paged_attention(
    queries, # [num_tokens, num_q_heads, head_dim]
    kv_pages, # [num_pages, num_slots, num_kv_heads, head_dim]
    kv_lens, # [num_seqs]
    page_indices, # [num_seqs, num_pages_per_seq]
    cu_q_lens, # [num_seqs + 1]
    num_seqs, # int
    *,
    sm_scale=1.0,
    mask_value=-1e7,
):
    _, _, num_combined_kv_heads, head_dim = kv_pages.shape
    num_kv_heads = num_combined_kv_heads // 2

    _, num_q_heads, _ = queries.shape
    num_queries_per_kv = num_q_heads // num_kv_heads

    outputs = []
    for seq_idx in range(num_seqs):
        q_start = cu_q_lens[seq_idx]
        q_end = cu_q_lens[seq_idx + 1]
        q = queries[q_start: q_end]

        page_indices_for_seq = page_indices[seq_idx] # gives us indices to index into kv_pages
        k = kv_pages[page_indices_for_seq, :, 0::2, :].reshape(-1, num_kv_heads, head_dim)
        v = kv_pages[page_indices_for_seq, :, 1::2, :].reshape(-1, num_kv_heads, head_dim)

        k = jnp.repeat(k, num_queries_per_kv, axis=1)
        v = jnp.repeat(v, num_queries_per_kv, axis=1)

        attn = jnp.einsum("qhd,khd->hqk", q, k)
        attn *= sm_scale

        kv_len = kv_lens[seq_idx]
        q_len = q_end - q_start
        # query should attend to everything <= these tokens
        q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(jnp.int32, attn.shape, 1)
        kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
        mask = q_span < kv_span

        attn += jnp.where(mask, mask_value, 0.0)
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)

    return jnp.concatenate(outputs, axis=0)

        