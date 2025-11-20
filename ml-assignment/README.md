 # Trigram Language Model and Scaled Dot-Product Attention
    Desible AI/ML Internship Assessment — Implementation by Raksha Nayak

    This repository contains my implementations for both tasks in the Desible AI/ML Internship Assessment:

    1. Trigram (N=3) Language Model built from scratch
    2. Scaled Dot-Product Attention implemented using NumPy

    Both parts are modular, easy to run, and include demo scripts.

    ------------------------------------------------------------

    ## Running in Google Colab

    Run the full project inside Google Colab:

    ```bash
    !git clone https://github.com/Rakshanayak24/ml-intern-assessment.git
    %cd ml-intern-assessment/ml-assignment
    !pip install -r requirements.txt
    ```

    ------------------------------------------------------------

    ## Project Structure

    ```
    ml-assignment/
    │
    ├── data/
    │   └── example_corpus.txt
    │
    ├── src/
    │   ├── ngram_model.py        # Trigram model implementation
    │   ├── utils.py
    │   └── generate.py           # Train + text generation pipeline
    │
    ├── attention/
    │   ├── attention.py          # NumPy Scaled Dot-Product Attention
    │   ├── demo.py               # Demo script
    │   └── __init__.py
    │
    ├── tests/
    │   └── test_ngram.py         # Unit tests
    │
    ├── README.md
    └── evaluation.md
    ```

    ------------------------------------------------------------

    ## Task 1 — Trigram Language Model

    ### Train & Generate

    ```bash
    python src/generate.py
    ```

    This script:
    - preprocesses the corpus  
    - builds trigram counts  
    - converts counts to probabilities  
    - generates new text using weighted sampling  

    ### Run Unit Tests

    ```bash
    pytest tests/test_ngram.py
    ```

    ------------------------------------------------------------

    ## Task 2 — Scaled Dot-Product Attention (NumPy)

    Implementation located in:

    ```
    attention/attention.py
    ```

    ### Run Demo

    ```bash
    python attention/demo.py
    ```

    Example Output:

    ```
    Q = [[1 0]]
    K = [[1 1]]
    V = [[0.5 2. ]]

    Attention Weights:
    [[1.]]

    Attention Output:
    [[0.5 2. ]]
    ```

    ------------------------------------------------------------
