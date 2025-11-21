from ngram_model import TrigramModel  # adjust import according to your repo

# 1. Create model instance
model = TrigramModel()

# 2. Load or train the model
model.train("../data/example_corpus.txt")  # or however training is done

# 3. Generate text
output_text = model.generate(max_length=800)

# 4. Print or save
print(output_text)

with open("generated_text.txt", "w", encoding="utf-8") as f:
    f.write(output_text)



