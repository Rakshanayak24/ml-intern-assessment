evaluation_md: |
    # Evaluation — Design Choices Summary (1-Page)

    ## 1. **Storage of N-gram Counts**
    I store all trigram counts in a nested dictionary:

    ```
    counts[(w1, w2)][w3] = frequency
    ```
    - Fast O(1) lookup  
    - Easy to convert counts into probability distributions  
    - Works efficiently for large corpora  

    Bigrams and unigrams are stored similarly for backoff handling.

    ---

    ## 2. **Text Cleaning, Padding, Unknown Words**
    - Lowercasing is applied to ensure consistency.
    - All punctuation except sentence-ending markers is removed.
    - Sentences are padded as:  
      ```
      <s> <s> ... </s>
      ```
    - A minimum frequency threshold introduces `<UNK>` tokens to avoid infinite sparsity.
    - This improves robustness during generation.

    ---

    ## 3. **Generate Function + Probabilistic Sampling**
    - For each step, the model considers the last two tokens `(w1, w2)`.
    - Probabilities are computed as:
      ```
      P(w3 | w1, w2) = count(w1, w2, w3) / count(w1, w2)
      ```
    - Instead of greedy decoding, multinomial sampling is used:
      ```
      next_word = random.choice(words, p=probabilities)
      ```
    - This produces more diverse and natural text.

    ---

    ## 4. **Self-Attention Design**
    - I used standard **scaled dot-product attention**:
      ```
      Attention(Q,K,V) = softmax(QKᵀ / sqrt(d_k)) V
      ```
    - Dimensions are kept small for clarity.
    - Implemented from scratch using only NumPy.
    - Demonstrates understanding of Transformer fundamentals.

    ---

    ## 5. **Other Key Decisions**
    - The repository is modular: `trigram/` and `attention/` separated.
    - Configurable file paths via `config.yml`.
    - Code follows clean, readable, internship-ready structure.
    - Generation and training can run independently.
