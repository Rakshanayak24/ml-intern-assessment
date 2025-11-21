from src.ngram_model import TrigramModel

def main():
    model = TrigramModel()

    with open("data/example_corpus.txt", "r") as f:
        text = f.read()

    model.fit(text)

    generated = model.generate()
    print("\nGenerated Text:\n")
    print(generated)

if __name__ == "__main__":
    main()




