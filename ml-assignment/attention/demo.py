import numpy as np
from attention.attention import ScaledDotProductAttention, MultiHeadAttention


def demo_scaled_attention():
    print("\n=== Scaled Dot-Product Attention Demo ===")

    Q = np.random.rand(1, 3, 4)
    K = np.random.rand(1, 3, 4)
    V = np.random.rand(1, 3, 4)

    attention = ScaledDotProductAttention()
    output, weights = attention(Q, K, V)

    print("Output:\n", output)
    print("Attention Weights:\n", weights)


def demo_multihead():
    print("\n=== Multi-Head Attention Demo ===")

    d_model = 8
    num_heads = 2

    Q = np.random.rand(1, 3, d_model)
    K = np.random.rand(1, 3, d_model)
    V = np.random.rand(1, 3, d_model)

    mha = MultiHeadAttention(d_model, num_heads)
    output, weights = mha(Q, K, V)

    print("MHA Output:\n", output)
    print("\nHead Weights Shapes:")
    for i, w in enumerate(weights):
        print(f"Head {i+1}:", w.shape)


if __name__ == "__main__":
    demo_scaled_attention()
    demo_multihead()

