
  🧠 AI/ML Assignment — Trigram Language Model + Scaled Dot-Product Attention

  This repository contains two core components demonstrating foundations of classical NLP and modern deep learning:

  **Task 1 — Trigram Language Model (N=3)**
  Implemented fully from scratch using Python, including text preprocessing, n-gram counting, probability computation, and sampling-based text generation.

  **Task 2 — Scaled Dot-Product Attention (Optional)**
  A NumPy-only implementation of the core operation behind Transformer architectures (BERT, GPT, etc.), including a demo script.
# Trigram Language Model + Self-Attention Module  
    A clean and fully functional implementation of a **Trigram Language Model** along with a lightweight **Self-Attention module**.  
    This repository demonstrates classic n-gram modeling as well as modern attention-based token representation — making it ideal for ML/NLP internship evaluations.

    ---

    ## 📌 Project Structure

    ```
    .
    ├── data/
    │   └── input.txt
    ├── trigram/
    │   ├── model.py
    │   ├── utils.py
    │   └── __init__.py
    ├── attention/
    │   ├── attention.py
    │   └── __init__.py
    ├── README.md
    ├── evaluation.md
    └── config.yml
    ```

    ---

    # 🚀 How to Run (Both Models)

    ## 1️⃣ **Run Trigram Language Model**

    ### **Step 1 — Install requirements**
    ```bash
    pip install -r requirements.txt
    ```
    (Only uses standard Python libraries; no heavy dependencies.)

    ### **Step 2 — Train the model**
    ```bash
    python trigram/model.py --train data/input.txt --save model.pkl
    ```

    ### **Step 3 — Generate text**
    ```bash
    python trigram/model.py --generate model.pkl --seed "the world"
    ```

    Output sample:
    ```
    the world is full of amazing discoveries waiting to be explored ...
    ```

    ---

    ## 2️⃣ **Run the Attention Module**

    ### **Step 1 — Simply import and run**
    ```bash
    python attention/attention.py
    ```

    ### **What it does**
    - Builds token embeddings  
    - Computes Query–Key–Value  
    - Applies scaled dot-product attention  
    - Returns attention-weighted representations  

    ### **Example output**
    ```
    Attention weights:
    [[0.21 0.54 0.25]
     [0.33 0.18 0.49]
     [0.40 0.12 0.48]]

    Context vectors:
    [[...token 1...]
     [...token 2...]
     [...token 3...]]
    ```

    ---

    # 🧠 Summary of What This Repo Demonstrates
    ✔ Understanding of classical NLP modeling (Trigrams)  
    ✔ Ability to implement sampling-based text generation  
    ✔ Working knowledge of attention (core foundation of Transformers)  
    ✔ Clean code and reproducible execution  
    ✔ Real-world ML workflow (training → saving → loading → inference)  

    ---

    # 📄 Evaluation  
    Please see **evaluation.md** for the required 1-page design summary.

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

  config_yml: |
    paths:
      train_data: "data/input.txt"
      trigram_model: "model.pkl"

    model:
      min_word_frequency: 2
      unk_token: "<UNK>"
      padding: true

    attention:
      embedding_dim: 32
      num_tokens: 3

  metadata:
    author: "Raksha Nayak"
    purpose: "Submission-ready ML/NLP project for internship selection"
    last_updated: "2025-11-21"


