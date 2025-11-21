import numpy as np
from typing import Optional, Tuple


def __call__(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
             mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply multi-head attention.

    Args:
        Q, K, V: (batch, seq_len, d_model)
        mask: optional; broadcastable to (batch, seq_q, seq_k)
              or (batch, 1, 1, seq_k)

    Returns:
        output: (batch, seq_q, d_model)
        attn_weights: (batch, num_heads, seq_q, seq_k)
    """

    batch_size = Q.shape[0]

    # Linear projections
    Q_lin = Q @ self.Wq
    K_lin = K @ self.Wk
    V_lin = V @ self.Wv

    # Split into heads
    Q_heads = self._split_heads(Q_lin)
    K_heads = self._split_heads(K_lin)
    V_heads = self._split_heads(V_lin)

    # Expand 3D mask → 4D
    if mask is not None and mask.ndim == 3:
        mask = mask[:, None, :, :]

    # Merge heads with batch dim
    b, h, sq, d = Q_heads.shape
    Q_resh = Q_heads.reshape(b * h, sq, d)
    K_resh = K_heads.reshape(b * h, K_heads.shape[2], d)
    V_resh = V_heads.reshape(b * h, V_heads.shape[2], d)

    # Prepare mask for vectorized attention
    if mask is not None:
        mask_resh = np.repeat(mask, self.num_heads, axis=1)
        mask_resh = mask_resh.reshape(b * h, mask.shape[-2], mask.shape[-1])
    else:
        mask_resh = None

    # Apply scaled dot-product attention
    output_resh, attn_resh = scaled_dot_product_attention(
        Q_resh, K_resh, V_resh, mask=mask_resh
    )

    # Reshape outputs back to (batch, heads, seq, depth)
    output_heads = output_resh.reshape(b, h, sq, d)
    attn_heads = attn_resh.reshape(b, h, sq, K_heads.shape[2])

    # Combine heads
    combined = self._combine_heads(output_heads)

    # Final projection
    out = combined @ self.Wo

    return out, attn_heads
