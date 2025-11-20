import os
from ngram_model import TrigramModel

def main():
    # Create Trigram Model
    model = TrigramModel()

    # Build absolute path to example_corpus.txt
    base_dir = os.path.dirname(os.path.dirname(__file__))  # go to project root
    data_path = os.path.join(base_dir, "data", "example_corpus.txt")

    # Read training text
    if not os.path.exists(data_path):
        print(f"Error: Cannot find file at {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Train model
    model.fit(text)

    # Generate text
    generated_text = model.generate()
    print("\nGenerated Text:\n")
    print(generated_text)

if __name__ == "__main__":
    main()
