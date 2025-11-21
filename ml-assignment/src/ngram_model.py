import re
import random
from collections import defaultdict, Counter

class TrigramModel:
    def __init__(self):
        """
        Initializes data structures for storing trigram counts.
        counts[(w1, w2)][w3] = count
        """
        self.trigram_counts = defaultdict(Counter)
        self.vocab = set()
        self.UNK = "<UNK>"
        self.START = "<s>"
        self.END = "</s>"

    def _clean_text(self, text):
        """Lowercase, remove punctuation except sentence boundaries."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9.?! ]+", " ", text)
        return text

    def _tokenize(self, text):
        """Split text into sentences and token words."""
        sentences = re.split(r"[.?!]", text)
        tokenized = []

        for sent in sentences:
            words = sent.strip().split()
            if len(words) > 0:
                tokenized.append(words)

        return tokenized

    def fit(self, text):
        """
        Train the trigram model on given text.
        """
        if not text.strip():
            # Empty text → nothing to train
            self.trigram_counts = defaultdict(Counter)
            self.vocab = set()
            return

        cleaned = self._clean_text(text)
        sentences = self._tokenize(cleaned)

        # Build vocabulary
        freq = Counter(w for s in sentences for w in s)
        self.vocab = {w for w, c in freq.items() if c >= 2}  # unknown word handling
        self.vocab.update([self.START, self.END, self.UNK])

        for sent in sentences:
            # Replace rare words with UNK
            sent = [w if w in self.vocab else self.UNK for w in sent]

            # Add padding
            padded = [self.START, self.START] + sent + [self.END]

            # Count trigrams
            for i in range(len(padded) - 2):
                w1, w2, w3 = padded[i], padded[i+1], padded[i+2]
                self.trigram_counts[(w1, w2)][w3] += 1

    def _sample_next(self, context):
        """Sample next word using trigram probability distribution."""
        counts = self.trigram_counts.get(context, None)
        if not counts:
            return self.END

        words = list(counts.keys())
        freqs = list(counts.values())
        total = sum(freqs)

        # Convert to probabilities
        probs = [f / total for f in freqs]

        # Random weighted sampling
        return random.choices(words, weights=probs, k=1)[0]

    def generate(self, max_length=50):
        """
        Generate text up to max_length words.
        """
        if len(self.trigram_counts) == 0:
            return ""

        w1, w2 = self.START, self.START
        output_words = []

        for _ in range(max_length):
            w3 = self._sample_next((w1, w2))
            if w3 == self.END:
                break
            output_words.append(w3)
            w1, w2 = w2, w3

        return " ".join(output_words)
