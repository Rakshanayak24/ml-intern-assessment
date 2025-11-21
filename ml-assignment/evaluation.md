## Evaluation

This document provides a clear and concise summary of the design decisions behind the Trigram Language Model implemented for the ML Intern Assessment. The focus is on correctness, efficiency, readability, and demonstrating practical understanding of classical NLP modeling.

---

### 1. Storage of N-gram Counts

To efficiently learn and query trigram relationships, the model uses a nested dictionary representation:

counts[(w1, w2)][w3] = frequency


### Why this design?

- Enables **O(1)** expected lookup for both training and generation.
- Clean, intuitive mapping from a 2-word context to all possible next words.
- Highly **extensible** — smoothing, pruning, serialization can be added easily.
- Ideal for an assessment: readable, modular, and straightforward to debug.

This structure balances clarity, memory-efficiency, and performance, making the n-gram model technically sound and easy to follow.

---

### 2. Text Cleaning, Padding & Unknown Word Handling

### Text Cleaning
- Converts text to **lowercase** to avoid unnecessary vocabulary inflation.
- Uses simple **whitespace tokenization**, ensuring deterministic behavior and easy testing.

### Padding
Each sentence is padded with:
- `"<s>" , "<s>"` at the beginning  
- `"</s>"` at the end  

This ensures:
- The model learns realistic sentence-start patterns.
- Always-available context when generating initial tokens.
- A clear stopping criterion during generation.

### Unknown Words
To prevent dead ends, the model uses a practical **backoff strategy**:
1. Try trigram prediction `(w1, w2 → w3)`
2. If unavailable → backoff to bigram `(w2 → w3)`
3. If still unavailable → sample from the global unigram distribution

This maintains fluency and robustness without requiring advanced smoothing.

---

### 3. Generation Logic & Probabilistic Sampling

The `generate()` function is designed to be **probabilistic, deterministic when seeded, and corpus-dependent**.

### How Generation Works
1. Start with context `(<s>, <s>)`
2. Retrieve allowed next-word candidates
3. Convert raw counts into probabilities
4. Sample using **weighted random selection**
5. Slide the window and continue until `</s>` or max-length

### Sampling Strategy
Uses Python’s weighted selection (e.g., `random.choices`) to ensure:
- Common continuations are more likely
- Rare events still occur naturally
- Output remains meaningful while still stochastic

### Why this matters
This mirrors classical statistical language modeling, demonstrating understanding of:
- Probability distributions
- Context windows
- Corpus-conditioned text generation

---

### 4. Additional Design Decisions & Trade-offs

### Precomputation
- Total frequencies per context are cached post-training → **faster generation**.

### Modular Architecture
- `ngram_model.py`, `utils.py`, and `generate.py` separate training, preprocessing, and generation.
- Supports easier unit testing and future extensions (e.g., add-k smoothing).

### No Smoothing by Default
Simplicity is preserved intentionally:
- Transparent probabilities
- Predictable behavior on small corpora
- Easy for reviewers to follow

### Reproducibility
- Optional random seed ensures determinism, enabling repeatable experiments.

---

Overall, the design prioritizes clarity, correctness, and extensibility while demonstrating strong understanding of classical NLP methods — making it both assessment-friendly and practical for real-world language modeling tasks.


