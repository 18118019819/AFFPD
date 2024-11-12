# from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

# flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False,
#                           window_size=(-1, -1), alibi_slopes=None, deterministic=False):
"""dropout_p should be set to 0.0 during evaluation
If Q, K, V are already stacked into 1 tensor, this function will be faster than
calling flash_attn_func on Q, K, V since the backward pass avoids explicit concatenation
of the gradients of Q, K, V.
If window_size != (-1, -1), implements sliding window local attention. Query at position i
will only attend to keys between [i - window_size[0], i + window_size[1]] inclusive.
Arguments:
    qkv: (batch_size, seqlen, 3, nheads, headdim)
    dropout_p: float. Dropout probability.
    softmax_scale: float. The scaling of QK^T before applying softmax.
        Default to 1 / sqrt(headdim).
    causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
    window_size: (left, right). If not (-1, -1), implements sliding window local attention.
    alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of (-alibi_slope * |i - j|) is added to
        the attention score of query i and key j.
    deterministic: bool. Whether to use the deterministic implementation of the backward pass,
        which is slightly slower and uses more memory. The forward pass is always deterministic.
Return:
    out: (batch_size, seqlen, nheads, headdim).
"""

flash_attn_qkvpacked_func()

# 开发者：ppy
# 时间：2024/4/4 14:31
