import os

def load_corpus():
    """Load and tokenize the example corpus."""
    filepath = os.path.join(os.path.dirname(__file__), "..", "data", "example_corpus.txt")

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read().lower()

    tokens = text.split()
    return tokens


