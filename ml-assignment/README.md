
  🧠 AI/ML Assignment — Trigram Language Model + Scaled Dot-Product Attention

  This repository contains two core components demonstrating foundations of classical NLP and modern deep learning:

  **Task 1 — Trigram Language Model (N=3)**
  Implemented fully from scratch using Python, including text preprocessing, n-gram counting, probability computation, and sampling-based text generation.

  **Task 2 — Scaled Dot-Product Attention (Optional)**
  A NumPy-only implementation of the core operation behind Transformer architectures (BERT, GPT, etc.), including a demo script.
# Trigram Language Model + Self-Attention Module  

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

  
    author: "Raksha Nayak"
    purpose: "Submission-ready ML/NLP project for internship selection"
    last_updated: "2025-11-21"


