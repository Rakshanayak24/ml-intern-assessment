import random
import re
from collections import defaultdict, Counter

class TrigramModel:
    def __init__(self):
        self.trigram_counts = defaultdict(Counter)
        self.vocab = set()

    def fit(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        tokens = text.split()
        tokens = ["<s>", "<s>"] + tokens + ["</s>"]

        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
            self.trigram_counts[(w1, w2)][w3] += 1
            self.vocab.add(w3)

    def _next_word(self, w1, w2):
        counts = self.trigram_counts.get((w1, w2))
        if not counts:
            return random.choice(list(self.vocab))
        total = sum(counts.values())
        r = random.randint(1, total)
        s = 0
        for word, c in counts.items():
            s += c
            if r <= s:
                return word

    def generate(self, max_length=50):
        w1, w2 = "<s>", "<s>"
        result = []

        for _ in range(max_length):
            w3 = self._next_word(w1, w2)
            if w3 == "</s>":
                break
            result.append(w3)
            w1, w2 = w2, w3

        return " ".join(result)
