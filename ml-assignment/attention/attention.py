import numpy as np


class ScaledDotProductAttention:
    def __call__(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask=None):
        """
        Compute scaled dot-product attention.
        """
        d_k = Q.shape[-1]

        # QK^T
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)

        # Optional masking
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        # Softmax
        attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)

        # Weighted sum
        output = np.matmul(attention_weights, V)

        return output, attention_weights


class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Initialize weights
        self.W_Q = np.random.randn(num_heads, d_model, self.d_k)
        self.W_K = np.random.randn(num_heads, d_model, self.d_k)
        self.W_V = np.random.randn(num_heads, d_model, self.d_k)
        self.W_O = np.random.randn(num_heads * self.d_k, d_model)

        self.attention = ScaledDotProductAttention()

    def split_heads(self, X):
        """
        Split (batch, seq_len, d_model) into (batch, num_heads, seq_len, d_k)
        """
        batch, seq_len, d_model = X.shape
        X_split = X.reshape(batch, seq_len, self.num_heads, self.d_k)
        return X_split.transpose(0, 2, 1, 3)

    def combine_heads(self, X):
        """
        Combine (batch, num_heads, seq_len, d_k) to (batch, seq_len, d_model)
        """
        batch, num_heads, seq_len, d_k = X.shape
        X_transposed = X.transpose(0, 2, 1, 3)
        return X_transposed.reshape(batch, seq_len, num_heads * d_k)

    def __call__(self, Q, K, V):
        """
        Forward pass
        """
        batch = Q.shape[0]

        Q_heads = np.matmul(Q, self.W_Q)
        K_heads = np.matmul(K, self.W_K)
        V_heads = np.matmul(V, self.W_V)

        outputs = []
        weights = []

        for i in range(self.num_heads):
            out, att = self.attention(Q_heads[:, :, i, :],
                                      K_heads[:, :, i, :],
                                      V_heads[:, :, i, :])
            outputs.append(out)
            weights.append(att)

        outputs = np.stack(outputs, axis=1)  # (batch, heads, seq, d_k)
        combined = self.combine_heads(outputs)
        final_output = np.matmul(combined, self.W_O)

        return final_output, weights
