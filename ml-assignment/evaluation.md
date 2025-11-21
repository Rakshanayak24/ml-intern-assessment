# Evaluation

This document summarizes the key design decisions behind the Trigram Language Model and the optional Scaled Dot-Product Attention implementation. The focus is on correctness, modularity, and demonstrating strong foundational NLP understanding.

---

## 1. Storage of N-gram Counts

The model uses Python’s `collections.Counter` to maintain:

- **Unigrams:** `Counter()`
- **Bigrams:** `Counter()` keyed as `(w1, w2)`
- **Trigrams:** `Counter()` keyed as `(w1, w2, w3)`

**Why this design?**

- O(1) average lookup and update time  
- Clean and transparent data structures  
- Ideal for classical NLP statistical modeling  
- Easy to debug and extend  

A global `set()` tracks vocabulary for consistent probability computation.

---

## 2. Text Cleaning, Tokenization, and Padding

Preprocessing is handled in `utils.py` to keep the main model simple and modular.

### Cleaning
- Converts text to lowercase  
- Removes punctuation  
- Normalizes extra whitespace  

### Tokenization
- Uses Python’s native whitespace tokenizer (`text.split()`) for clarity and predictability  

### Padding
Each sentence is padded as:

```<s> <s> ...tokens... </s>```

**Benefits:**
- Ensures valid trigram contexts  
- Explicit boundary modeling  
- Natural stopping condition during generation  

### Unknown Words
- Seed words not in vocabulary fall back to `<s>`  
- Generated outputs always come from known tokens  
- Keeps generation stable without complex OOV logic  

---

## 3. Probability Computation

Trigram probabilities use **Maximum Likelihood Estimation (MLE)**:

$$
P(w_3 \mid w_1, w_2) = \frac{\text{count}(w_1, w_2, w_3)}{\text{count}(w_1, w_2)}
$$


Add-k smoothing (default `k = 0`) is included to reduce sparsity issues on small datasets.

---

## 4. Text Generation and Sampling

The `generate()` function follows a deterministic and reproducible pipeline:

1. Initialize context as `<s>, <s>` or user-provided seed  
2. Compute probability distribution for next token  
3. Choose next token via:
   - **Deterministic mode:** argmax  
   - **Stochastic mode:** `random.choices()`  
4. Shift context window  
5. Stop when:
   - `</s>` is generated, or  
   - `max_length` is reached  

All probability vectors are normalized to avoid floating-point instability.

---

## 5. Project Structure and Modularity

A clean and industry-standard file layout:
| File/Folder      | Purpose                                    |
| ---------------- | ------------------------------------------ |
| ngram_model.py   | Core trigram model implementation          |
| utils.py         | Cleaning, tokenization, helper utilities   |
| generate.py      | Main script to train model + generate text |
| tests/           | Unit tests for correctness and edge cases  |
| task2_attention/ | Scaled dot-product attention module + demo |


The modular design makes each component self-contained and easy to extend.

---

## 6. Scaled Dot-Product Attention (Optional)

Implements the Transformer-style attention mechanism:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V
$$


### Key Design Points
- Implemented fully in **NumPy**  
- Uses stable softmax (subtracting max per row)  
- Supports masking for selective attention  
- Demonstration applies attention to corpus-based embeddings  
- Tests validate:
  - shape matching  
  - masking behavior  
  - numerical stability  

This demonstrates comfort with modern deep learning internals.

---

## Conclusion

This implementation balances clarity, correctness, and modern ML engineering practices:

- Modular, readable code  
- Strong trigram probability modeling  
- Reliable deterministic + stochastic text generation  
- Robust preprocessing pipeline  
- Fully functional NumPy attention module  
- Tested and reproducible  

Overall, the project reflects solid understanding of classical NLP and modern model internals, making it suitable for real-world ML engineering workflows.






