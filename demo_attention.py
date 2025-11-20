"""
Demo for scaled_dot_product_attention.

Shows a small example with batch size 1 and prints the attention weights and output.
Run:
    python task2_attention/demo_attention.py
"""
import numpy as np
from task2_attention.attention import scaled_dot_product_attention

def demo():
    # Simple example
    # batch = 1, seq_len = 2, d_k = d_v = 3
    Q = np.array([[[1.0, 0.0, 1.0],
                   [0.0, 1.0, 0.0]]])  # shape (1,2,3)

    K = np.array([[[1.0, 0.0, 1.0],
                   [0.0, 1.0, 0.0]]])  # shape (1,2,3)

    V = np.array([[[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]]])  # shape (1,2,3)

    # No mask
    output, attn_weights = scaled_dot_product_attention(Q, K, V, mask=None)

    print("Queries (Q):\n", Q)
    print("\nKeys (K):\n", K)
    print("\nValues (V):\n", V)
    print("\nAttention weights (batch, seq_q, seq_k):\n", attn_weights)
    print("\nOutput (batch, seq_q, d_v):\n", output)

    # Example with mask (mask out second key for the second query)
    # mask shape (batch, seq_q, seq_k)
    mask = np.array([[[1, 1],
                      [1, 0]]])  # second query should not attend to second key

    output_masked, attn_weights_masked = scaled_dot_product_attention(Q, K, V, mask=mask)
    print("\n--- With mask applied ---")
    print("\nMask:\n", mask)
    print("\nAttention weights (masked):\n", attn_weights_masked)
    print("\nOutput (masked):\n", output_masked)

if __name__ == "__main__":
    demo()



