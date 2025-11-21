evaluation: "Design Choices for Trigram Language Model"

ngram_storage:
  approach: "Nested dictionaries with tuple keys"
  examples:
    unigram: "count[w1]"
    bigram: "count[(w1, w2)]"
    trigram: "count[(w1, w2, w3)]"
  reasons:
    - "O(1) average lookup time"
    - "Cleaner probability computation"
    - "Avoids string concatenation overhead"
    - "Efficient for iteration and sampling"

text_cleaning_preprocessing:
  normalization:
    - "Converted text to lowercase"
    - "Whitespace-based tokenization"
  padding:
    start_tokens: ["<s>", "<s>"]
    end_token: "</s>"
    reason: "Preserves sentence boundaries and enables early trigram formation"
  unknown_words:
    token: "<unk>"
    reason: "Prevents zero-probability issues caused by unseen tokens"

probability_computation:
  formula: "P(w3 | w1, w2) = count(w1, w2, w3) / count(w1, w2)"
  smoothing_strategy:
    type: "Backoff"
    steps:
      - "Try trigram probability"
      - "Fallback to bigram"
      - "Fallback to unigram"
      - "If all fail → return <unk>"
    reason:
      - "Avoids dead ends during generation"
      - "Simpler than full Kneser–Ney but effective for this task"

generation_strategy:
  process:
    - "Start with context = [<s>, <s>]"
    - "Retrieve candidate next words based on trigram context"
    - "Build probability distribution from counts"
    - "Sample next token using numpy multinomial sampling"
    - "Slide context window forward"
    - "Stop at </s> or max length"
  benefits:
    - "Produces non-deterministic, varied text"
    - "Works even with sparse datasets"
  paragraph_generation:
    method: "Generate multiple sentences and join them"
    reason: "Ensures coherent multi-sentence output"

additional_design_choices:
  efficient_counting:
    description: "Single pass over corpus to build unigram, bigram, and trigram counts"
  corpus_flexibility:
    description: "User can replace example_corpus.txt with any dataset without modifying code"
  modular_structure:
    layout:
      - "src/ → trigram implementation"
      - "attention/ → attention mechanisms"
    benefit: "Clean separation of tasks and easy evaluation"

attention_module_summary:
  components:
    - "Scaled Dot-Product Attention (NumPy)"
    - "Multi-Head Attention with head splitting and combining"
  features:
    - "Vectorized matrix operations"
    - "Linear projections using NumPy arrays"
    - "Per-head attention weight output"
  reason: "Demonstrates core transformer mechanisms without external ML frameworks"

conclusion:
  highlights:
    - "Simple, robust, and efficient trigram model"
    - "Handles rare words and sparse data gracefully"
    - "Clear modularity for both assignment parts"
    - "Stable and extendable design suitable for larger corpora"


