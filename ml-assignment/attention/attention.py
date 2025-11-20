import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Implements Scaled Dot-Product Attention.
    
    Args:
        Q: Query matrix of shape      (batch, heads, seq_len, d_k)
        K: Key matrix of shape        (batch, heads, seq_len, d_k)
        V: Value matrix of shape      (batch, heads, seq_len, d_v)
        mask: Optional mask matrix    (batch, heads, seq_len, seq_len)
    
    Returns:
        output: Attention output
        attention_weights: Softmax attention weights
    """

    # Compute QK^T
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2))

    # Scale by sqrt(d_k)
    d_k = Q.shape[-1]
    scores = scores / np.sqrt(d_k)

    # Apply mask if present
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # Stable softmax
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Weighted sum with values
    output = np.matmul(attention_weights, V)

    return output, attention_weights
