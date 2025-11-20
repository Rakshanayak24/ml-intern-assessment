import numpy as np
from attention import scaled_dot_product_attention

def main():
    # Example query, key, value matrices
    Q = np.random.rand(3, 64)
    K = np.random.rand(3, 64)
    V = np.random.rand(3, 64)

    output, attention_weights = scaled_dot_product_attention(Q, K, V)

    print("Attention Output:\n", output)
    print("\nAttention Weights:\n", attention_weights)

if __name__ == "__main__":
    main()


