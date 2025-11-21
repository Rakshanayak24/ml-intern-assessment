from src.ngram_model import TrigramModel
from src.utils import load_corpus

if __name__ == "__main__":
    # Load tokens from corpus
    tokens = load_corpus()

    # Train trigram model
    model = TrigramModel()
    model.train(tokens)

    print("\nGenerated Text:\n")
    text = model.generate_paragraph(num_sentences=4)
    print(text)
