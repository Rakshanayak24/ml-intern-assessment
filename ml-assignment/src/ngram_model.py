import re
import random
from collections import defaultdict

class TrigramModel:
    def __init__(self):
        self.trigram_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.context_counts = defaultdict(lambda: defaultdict(int))
        self.vocab = set()
        self.START = "<s>"
        self.END = "</s>"

    def _clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text

    def _tokenize(self, text):
        if not text.strip():
            return []
        tokens = text.split()
        for t in tokens:
            self.vocab.add(t)
        return tokens

    def fit(self, text):
        cleaned = self._clean_text(text)
        tokens = self._tokenize(cleaned)

        if len(tokens) == 0:
            return

        tokens = [self.START, self.START] + tokens + [self.END]

        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
            self.trigram_counts[w1][w2][w3] += 1
            self.context_counts[w1][w2] += 1

    def _sample_next(self, w1, w2):
        if w1 not in self.trigram_counts or w2 not in self.trigram_counts[w1]:
            return self.END

        choices = self.trigram_counts[w1][w2]
        total = sum(choices.values())
        r = random.uniform(0, total)
        cumulative = 0

        for word, count in choices.items():
            cumulative += count
            if cumulative >= r:
                return word

        return self.END

    def generate(self, max_length=50):
        if len(self.vocab) == 0:
            return ""

        w1, w2 = self.START, self.START
        output = []

        for _ in range(max_length):
            nxt = self._sample_next(w1, w2)
            if nxt == self.END:
                break
            output.append(nxt)
            w1, w2 = w2, nxt

        return " ".join(output)
