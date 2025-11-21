import numpy as np

class ScaledDotProductAttention:
    def __call__(self, Q, K, V):
        d_k = Q.shape[-1]

        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
        weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        output = np.matmul(weights, V)

        return output, weights


class MultiHeadAttention:
    def __init__(self, d_model=4, num_heads=2):
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        # Simple linear projections (random for demo)
        self.Wq = np.random.rand(d_model, d_model)
        self.Wk = np.random.rand(d_model, d_model)
        self.Wv = np.random.rand(d_model, d_model)

        self.attention = ScaledDotProductAttention()

    def split_heads(self, x):
        """
        x shape: (batch, seq_len, d_model)
        return: (batch, num_heads, seq_len, depth)
        """
        batch, seq_len, d_model = x.shape
        x = x.reshape(batch, seq_len, self.num_heads, self.depth)
        return x.transpose(0, 2, 1, 3)

    def __call__(self, Q, K, V):
        # Linear projections
        Q = np.matmul(Q, self.Wq)
        K = np.matmul(K, self.Wk)
        V = np.matmul(V, self.Wv)

        # Split heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Apply attention per head
        outputs = []
        weights_list = []

        for h in range(self.num_heads):
            out, att = self.attention(Q[:, h], K[:, h], V[:, h])
            outputs.append(out)
            weights_list.append(att)

        # Stack & reshape back to (batch, seq_len, d_model)
        outputs = np.stack(outputs, axis=1)       # (batch, heads, seq_len, depth)
        outputs = outputs.transpose(0, 2, 1, 3)   # (batch, seq_len, heads, depth)
        batch, seq_len, heads, depth = outputs.shape
        outputs = outputs.reshape(batch, seq_len, heads * depth)

        weights_list = np.stack(weights_list, axis=1)  # (batch, heads, seq_len, seq_len)

        return outputs, weights_list

