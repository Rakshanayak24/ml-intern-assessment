import re
import random
from collections import defaultdict, Counter


class TrigramModel:
    def __init__(self):
        self.trigrams = defaultdict(Counter)
        self.vocab = set()
        self.UNK = "<UNK>"

    def _clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _tokenize(self, text):
        return text.split()

    def fit(self, text):
        if not text.strip():
            self.trigrams = {}
            return

        text = self._clean_text(text)
        tokens = self._tokenize(text)

        # Build vocabulary (keep ALL words, no low-frequency filtering)
        self.vocab = set(tokens)

        # Padding
        tokens = ["<s>", "<s>"] + tokens + ["</s>"]

        # Build trigram counts
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
            self.trigrams[(w1, w2)][w3] += 1

    def _choose_next(self, dist):
        words, counts = zip(*dist.items())
        total = sum(counts)
        probs = [c / total for c in counts]
        return random.choices(words, probs)[0]

    def generate(self, max_length=50):
        if not self.trigrams:
            return ""

        w1, w2 = "<s>", "<s>"
        result = []

        for _ in range(max_length):
            dist = self.trigrams.get((w1, w2), None)
            if not dist:
                break

            next_word = self._choose_next(dist)

            if next_word == "</s>":
                break

            result.append(next_word)
            w1, w2 = w2, next_word

        return " ".join(result)
