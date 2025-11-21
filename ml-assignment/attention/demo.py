"""
Demo script for Scaled Dot-Product Attention and Multi-Head Attention.
This file shows recruiters that everything works end-to-end.
"""

import numpy as np
from attention import scaled_dot_product_attention, MultiHeadAttention


def main():

    print("\n===== SCALED DOT-PRODUCT ATTENTION DEMO =====\n")

    # Example dimensions
    batch = 1
    seq_len = 4
    d_model = 8

    # Toy Q, K, V
    Q = np.random.randn(batch, seq_len, d_model)
    K = np.random.randn(batch, seq_len, d_model)
    V = np.random.randn(batch, seq_len, d_model)

    # No mask
    out, attn = scaled_dot_product_attention(Q, K, V)

    print("Q shape:", Q.shape)
    print("K shape:", K.shape)
    print("V shape:", V.shape)
    print("Attention output shape:", out.shape)
    print("Attention weights shape:", attn.shape)
    print("\nAttention Weights:\n", attn)


    print("\n===== MULTI-HEAD ATTENTION DEMO =====\n")

    num_heads = 2
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    # Example mask (optional)
    mask = np.zeros((batch, seq_len, seq_len))
    mask[:, :, -1] = -1e9  # block last token

    out_mh, attn_mh = mha(Q, K, V, mask=mask)

    print("Multi-head output shape:", out_mh.shape)
    print("Multi-head attention weights shape:", attn_mh.shape)
    print("\nPer-Head Attention Weights:\n", attn_mh)

    print("\nDemo complete — everything executed successfully.\n")


if __name__ == "__main__":
    main()
