# Evaluation – Trigram Language Model

This document summarizes the design choices made while implementing the Trigram Language Model for the Desible AI ML Internship Assessment.

---

## 1. Storing N-gram Counts
I used a nested dictionary structure of the form:

{
    (word1, word2): {next_word: count, ...}
}

This allows:
- Efficient lookup during generation
- Clear separation between context (the bigram) and possible next words
- Simple updates while training

The first-level key is the trigram context, and the second dictionary stores frequency counts. This structure also supports smooth scaling if extended to 4-grams or general n-grams later.

---

## 2. Text Cleaning, Tokenization, and Padding
To keep the implementation consistent and predictable, I applied the following steps:

### **Lowercasing**
All text is converted to lowercase to avoid treating “The” and “the” as different tokens.

### **Cleaning**
Basic punctuation is removed so the model focuses on word-level patterns instead of formatting.

### **Tokenization**
The text is split using simple whitespace tokenization. This is sufficient for the given dataset.

### **Padding**
Two start tokens (`<s> <s>`) and one end token (`</s>`) are added.  
Padding ensures the model can learn valid opening trigrams and know where to stop generation.

### **Unknown Words**
The dataset is small and controlled, so I chose a simple strategy:
- No explicit `<unk>` token was required.
- Any missing context at generation time defaults to random sampling from the global vocabulary.
This avoids generation errors while keeping the model simple and aligned with the assignment.

---

## 3. Generation Strategy & Probabilistic Sampling
I used the following generation procedure:

1. Start with the padded tokens: (`<s>`, `<s>`).
2. At each step, look up all possible next words for the current bigram.
3. Convert the trigram counts to probabilities by normalizing frequencies.
4. Use `random.choices()` for weighted sampling.
5. Stop when `</s>` is generated or when `max_length` is reached.

Weighted sampling ensures:
- More frequent patterns are favored
- Rare but possible transitions still appear
- Generated text is diverse, not repetitive

If a bigram is missing (rare for small corpora), the model falls back to sampling from the entire vocabulary, ensuring generation never crashes.

---

## 4. Other Design Decisions

### **Simple and Readable Structure**
I kept the implementation modular:
- `fit()` handles all preprocessing + training
- `generate()` only focuses on sampling and sequence construction

This separation improves testability and readability.

### **No Over-Engineering**
I avoided unnecessary NLP libraries to demonstrate understanding of core concepts rather than relying on external tools. The implementation is intentionally minimal but complete.

### **Compatibility with Tests**
All design choices were aligned with the expected behavior from `test_ngram.py`. I ensured deterministic behavior where required and probabilistic behavior where appropriate.

---

## Conclusion
The final implementation is:
- Clean
- Efficient
- Fully aligned with assignment requirements
- Easy to extend or modify

The goal was to demonstrate understanding of n-gram modeling fundamentals while maintaining readability and correctness.

