"""
Scaled Dot-Product Attention (NumPy only)

Function:
    scaled_dot_product_attention(Q, K, V, mask=None)

Inputs:
    Q: numpy array of shape (batch, seq_len_q, d_k)
    K: numpy array of shape (batch, seq_len_k, d_k)
    V: numpy array of shape (batch, seq_len_v, d_v)  # seq_len_k == seq_len_v typically
    mask: (optional) numpy array broadcastable to (batch, seq_len_q, seq_len_k)
          mask positions with value 0 will be masked out (set to large negative score)

Returns:
    output: numpy array of shape (batch, seq_len_q, d_v)
    attn_weights: numpy array of shape (batch, seq_len_q, seq_len_k)
"""
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    # Q, K, V: (batch, seq_len, d_k or d_v)
    # 1) compute raw scores = Q @ K^T
    #    For batched inputs, we use np.matmul which handles batch dims.
    d_k = Q.shape[-1]
    # scores shape: (batch, seq_len_q, seq_len_k)
    scores = np.matmul(Q, K.transpose(0, 2, 1))  # (batch, seq_q, seq_k)

    # 2) scale scores by sqrt(d_k)
    scores = scores / np.sqrt(d_k)

    # 3) apply mask (if provided) by setting masked positions to a very large negative value
    if mask is not None:
        # mask is expected to have 1s for valid positions and 0s for masked positions,
        # but function supports any array broadcastable to scores shape.
        # We map mask == 0 -> -1e9; mask != 0 leaves scores unchanged.
        scores = np.where(mask, scores, -1e9)

    # 4) numerically stable softmax along last axis (seq_len_k)
    # subtract max for numerical stability
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
    attn_weights = exp_scores / (sum_exp + 1e-9)  # (batch, seq_q, seq_k)

    # 5) attention output = attn_weights @ V
    output = np.matmul(attn_weights, V)  # (batch, seq_q, d_v)

    return output, attn_weights
