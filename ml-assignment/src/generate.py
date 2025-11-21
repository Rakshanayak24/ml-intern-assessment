import random
import numpy as np

def generate(self, max_length=200):
    """Generate text using probabilistic trigram sampling."""
    if not self.model:
        return ""

    # Start with padding tokens
    w1, w2 = "<s>", "<s>"
    generated = []

    for _ in range(max_length):
        key = (w1, w2)
        if key not in self.model:
            break

        next_words = self.model[key]
        words = list(next_words.keys())
        counts = np.array(list(next_words.values()), dtype=float)

        # Convert counts → probability distribution
        probs = counts / counts.sum()

        # Sample probabilistically instead of greedy max
        next_word = np.random.choice(words, p=probs)

        if next_word == "</s>":
            break

        generated.append(next_word)

        # Shift context
        w1, w2 = w2, next_word

    return " ".join(generated)


