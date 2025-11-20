import os
from src.ngram_model import TrigramModel

def main():
    model = TrigramModel()

    # Locate example_corpus.txt regardless of current working directory
    base_dir = os.path.dirname(os.path.abspath(__file__))  # folder containing generate.py
    project_root = os.path.dirname(base_dir)               # go to project root
    data_path = os.path.join(project_root, "data", "example_corpus.txt")

    if not os.path.exists(data_path):
        print(f"Error: Cannot find example_corpus.txt at: {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    model.fit(text)
    generated_text = model.generate()

    print("\nGenerated Text:\n")
    print(generated_text)

if __name__ == "__main__":
    main()

