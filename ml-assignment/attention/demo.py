import numpy as np
from attention.attention import scaled_dot_product_attention

def main():
    """
    Minimal runnable demo for Scaled Dot-Product Attention.
    Shows input shapes, output shapes, and computed attention weights.
    """

    batch = 1
    heads = 1
    seq_len = 3
    depth = 64

    # Random matrices
    Q = np.random.rand(batch, heads, seq_len, depth)
    K = np.random.rand(batch, heads, seq_len, depth)
    V = np.random.rand(batch, heads, seq_len, depth)

    print("Q shape:", Q.shape)
    print("K shape:", K.shape)
    print("V shape:", V.shape)

    output, weights = scaled_dot_product_attention(Q, K, V)

    print("\nAttention Weights:\n", weights)
    print("\nAttention Output:\n", output)

if __name__ == "__main__":
    main()



