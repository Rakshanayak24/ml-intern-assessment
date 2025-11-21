## Evaluation

This document provides a clear and concise summary of the design decisions behind the Trigram Language Model implemented for the ML Intern Assessment. The focus is on correctness, efficiency, readability, and demonstrating practical understanding of classical NLP modeling.

----

### 1. Storage of N-gram Counts

To efficiently learn and query trigram relationships, the model uses a nested dictionary structure:

```counts[(w1, w2)][w3] = frequency```

## Why this design?

* Provides O(1) expected lookup for both training and generation.

* Cleanly represents the mapping from a 2-word context to all possible next words.

* Easier to debug and extend (e.g., smoothing, serialization, integer indexing).

* Ideal for an academic/assessment setting where readability matters as much as performance.

This decision balances clarity, speed, and extensibility, making the model simple to follow yet efficient in practice.

---

### 2. Text Cleaning, Padding & Unknown Word Handling
## Text Cleaning

* Converts all text to lowercase for consistency and reduced vocabulary size.

* Uses whitespace-based tokenization to keep parsing predictable and easy to test.

## Padding

Each line/sentence receives:

* Two start tokens:``` <s>, <s>```

* One end token:``` </s>```

This ensures:

* The model correctly learns how sentences start.

* Valid contexts always exist during generation.

* The model has a natural stopping point.

## Unknown Words

Instead of dropping unseen tokens, the model uses a tiered fallback strategy:

1. Trigram context → if available

2. Bigram backoff (based on last word)

3. Unigram distribution from entire corpus

This avoids dead ends during generation and keeps sampling robust without requiring heavy smoothing.

---

### 3. Generation Logic & Probabilistic Sampling

The ```generate()``` function is designed to be predictable, probabilistic, and interpretable.

### How Generation Works

1. Start with context:

```(<s>, <s>)```


2. Look up all possible next tokens for this context.

3. Convert their counts into probabilities.

4. Sample the next word using weighted random sampling.

5. Slide the context window forward and continue.

## Sampling Strategy

Uses ```random.choices(population, weights=...) ```(or cumulative-sum walk) to ensure:

* Higher-frequency continuations are more likely.

* Rare but valid continuations still occur naturally.

* Stops when the model emits ```</s> ```or reaches a length limit.

## Why this matters

This approach produces human-like, corpus-dependent text while staying faithful to statistical language modeling principles. It also demonstrates understanding of randomness, distributions, and generative modeling.

---

## 4. Additional Design Decisions & Trade-offs
## Precomputation

* Stores total count per context after training → improves runtime performance during generation.

## Modular Code Design

* Training logic (fit), generation logic, utilities, and demo execution are kept separate.

## Makes the code easier to maintain, test, and extend.

## No Smoothing by Default

Smoothing methods (Add-K, Kneser–Ney) are intentionally omitted to maintain:

* Transparent probability distributions

* Predictable behavior for small corpora

* Simplicity for assessment review

The implementation can easily accommodate smoothing later.

## Reproducibility

* Supports user-defined seeds to generate the same sequence consistently, which is important for experiments and evaluation.


