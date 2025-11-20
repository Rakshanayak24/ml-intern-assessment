import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Implements Scaled Dot-Product Attention.
    
    Args:
        Q: Query matrix      (batch, heads, seq_len, d_k)
        K: Key matrix        (batch, heads, seq_len, d_k)
        V: Value matrix      (batch, heads, seq_len, d_v)
        mask: Optional mask  (batch, heads, seq_len, seq_len)

    Returns:
        output: Attention output
        attention_weights: Softmax attention weights
    """

    # Step 1: Compute raw attention scores: QK^T
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2))

    # Step 2: Scale scores
    d_k = K.shape[-1]
    scores = scores / np.sqrt(d_k)

    # Step 3: Apply mask (optional)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # Step 4: Softmax over last dimension
    # subtract max for numerical stability
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Step 5: Multiply attention weights with V
    output = np.matmul(attention_weights, V)

    return output, attention_weights

