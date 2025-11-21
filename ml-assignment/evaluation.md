Evaluation – Design Choices for Trigram Language Model

This document summarizes the key design decisions made while implementing the Trigram Language Model and attention components in this assignment.

1. N-gram Count Storage

I used nested Python dictionaries to store unigram, bigram, and trigram counts:

Unigrams: count[w1]

Bigrams: count[(w1, w2)]

Trigrams: count[(w1, w2, w3)]

These structures provide:

O(1) average-time lookup

Simple iteration for probability computation

Clean organization when computing conditional probabilities

Using tuples as keys also avoids string concatenation overhead and keeps the model implementation simple and efficient.

2. Text Cleaning & Preprocessing
Tokenization

Converted text to lowercase for normalization.

Split on whitespace to keep tokenization simple and deterministic.

Padding

Added start tokens to preserve the structure of sentence boundaries:

<START> <START> w1 w2 w3 ... <END>


This ensures the first two trigrams have valid context and allows the model to learn proper sentence openings.

Unknown Words

To avoid sparsity, I replaced rare/unseen words with:

<UNK>


This makes generation stable by preventing zero-probability states.

3. Probability Computation & Smoothing

For each trigram (w1, w2 → w3):

P(w3 | w1, w2) = count(w1, w2, w3) / count(w1, w2)


If a condition never appeared, I used a backoff strategy:

Backoff to bigram

Backoff to unigram

If all fail → sample <UNK>

This prevents dead ends during generation while keeping the model simple and interpretable without full Kneser–Ney smoothing.

4. Generate Function & Sampling Strategy

The generate_sentence() function:

Starts with:

context = ["<START>", "<START>"]


Looks up all valid trigrams for this context.

Builds a probability distribution.

Samples the next token using multinomial sampling (np.random.choice).

Shifts context:

context = [context[1], next_word]


Stops at <END> or when max length is reached.

This ensures:

Diversity in generated text

Non-deterministic behavior

Smooth continuation even with sparse data

The paragraph generator simply calls the sentence generator repeatedly and joins the results.

5. Additional Design Considerations
Efficient Counting

Using a single pass over the corpus to build all n-grams improved speed and memory locality.

Corpus Flexibility

I kept the corpus loading logic minimal so users can replace example_corpus.txt with large datasets without modifying code.

Modular Architecture

Part 1 and Part 2 (attention) are cleanly separated:

src/ → Trigram model

attention/ → Scaled Dot-Product + Multi-Head Attention

Each has its own demo runner and can be evaluated independently.

6. Attention Module (Part 2)

Although not required for the trigram evaluation, the attention module was implemented using:

Pure NumPy (no PyTorch/TensorFlow)

A vectorized Scaled Dot-Product Attention

A Multi-Head Attention layer with:

Head splitting

Per-head attention computation

Head concatenation

Final linear projection

This demonstrates practical understanding of core transformer components.


