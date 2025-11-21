import random
from collections import defaultdict
from typing import List, Tuple

class TrigramModel:
    def __init__(self):
        self.trigrams = defaultdict(list)
        self.starts = []

    def train(self, tokens: List[str]):
        """Train the trigram model from token list."""
        if len(tokens) < 3:
            return

        # Collect starts
        for i in range(len(tokens) - 2):
            if tokens[i][0].isalpha():  
                self.starts.append((tokens[i], tokens[i+1]))
                break

        # Build trigram pairs
        for w1, w2, w3 in zip(tokens, tokens[1:], tokens[2:]):
            self.trigrams[(w1, w2)].append(w3)

    def generate_text(self, max_len=30) -> List[str]:
        """Generate raw sequence of tokens."""
        if not self.starts:
            return ["no", "data"]

        current = random.choice(self.starts)
        w1, w2 = current
        output = [w1, w2]

        for _ in range(max_len - 2):
            next_words = self.trigrams.get((w1, w2), None)
            if not next_words:
                break
            w3 = random.choice(next_words)
            output.append(w3)
            w1, w2 = w2, w3

        return output

    # -------------------------------
    # NEW: Sentence & Paragraph Tools
    # -------------------------------
    def generate_sentence(self, max_len=40):
        """Generate a clean sentence."""
        tokens = self.generate_text(max_len)
        sentence = " ".join(tokens)
        return sentence.capitalize() + "."

    def generate_paragraph(self, num_sentences=4):
        """Generate multiple sentences."""
        sentences = [self.generate_sentence() for _ in range(num_sentences)]
        return " ".join(sentences)


