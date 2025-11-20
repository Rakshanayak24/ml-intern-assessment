import numpy as np
from attention import scaled_dot_product_attention

def main():
    # Example dimensions
    batch = 1
    heads = 1
    seq_len = 3
    d_k = 4
    d_v = 4

    # Create simple Q, K, V matrices
    np.random.seed(42)
    Q = np.random.rand(batch, heads, seq_len, d_k)
    K = np.random.rand(batch, heads, seq_len, d_k)
    V = np.random.rand(batch, heads, seq_len, d_v)

    print("Q:\n", Q)
    print("K:\n", K)
    print("V:\n", V)

    output, attn_weights = scaled_dot_product_attention(Q, K, V)

    print("\nAttention Weights:")
    print(attn_weights)

    print("\nAttention Output:")
    print(output)

if __name__ == "__main__":
    main()

