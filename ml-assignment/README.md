🧠 AI/ML Assignment — Trigram Language Model + Scaled Dot-Product Attention

This repository contains two core components demonstrating foundations of classical NLP and modern deep learning:

Task 1 — Trigram Language Model (N=3)
Implemented fully from scratch using Python, including text preprocessing, n-gram counting, probability computation, and sampling-based text generation.

Task 2 — Scaled Dot-Product Attention (Optional)
A NumPy-only implementation of the core operation behind Transformer architectures (BERT, GPT, etc.), including a demo script.

This assignment showcases clean code design, probabilistic modeling, understanding of linear algebra, and modular project structure.

📂 Project Structure
ml-assignment/
│
├── data/
│   └── example_corpus.txt
│
├── src/
│   ├── ngram_model.py
│   ├── utils.py
│   └── generate.py
│
├── attention/
│   ├── attention.py
│   └── demo_attention.py
│
├── tests/
│   └── test_ngram.py
│
├── requirements.txt
├── evaluation.md
└── README.md   ← (this file)

🚀 How to Run the Project
1️⃣ Create and Activate Virtual Environment
python3 -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

2️⃣ Install Dependencies
pip install -r requirements.txt

🟦 TASK 1 — TRIGRAM LANGUAGE MODEL
▶️ How to Run the Trigram Generator
python -m src.generate


This will:

Read corpus from data/example_corpus.txt

Clean and tokenize text

Train a trigram language model

Print generated text using probabilistic sampling

🔧 How It Works (Short Explanation)

Text is cleaned → lowercased, punctuation removed

<s> and <e> tokens mark sentence boundaries

Trigrams (w1, w2, w3) are counted in a nested dictionary

Probabilities are computed as:

P(w3 | w1, w2) = count(w1, w2, w3) / sum(count(w1, w2, *))


Text generation starts with <s>, <s> and samples the next words based on these probabilities

Continues until <e> or max_length reached

A full 1-page explanation goes in evaluation.md.

🧪 Run Pytests
pytest -v


Pytests validate:

model training

text generation

empty text handling

short text behavior

🟧 TASK 2 — SCALED DOT-PRODUCT ATTENTION (Optional)

This task implements the mathematical attention mechanism from Transformers using only NumPy.

Formula:

Attention(Q, K, V) = softmax( QKᵀ / √d_k ) · V


Where:

Q → Queries

K → Keys

V → Values

d_k → Key dimensionality

▶️ How to Run the Attention Demo

Navigate to the attention/ directory:

cd attention


Run:

python demo_attention.py


The script will:

Create random Q, K, V matrices

Call scaled_dot_product_attention()

Print:

Attention Output

Attention Weights (Softmax matrix)

🧪 Manual Testing (Optional)
import numpy as np
from attention import scaled_dot_product_attention

Q = np.random.rand(1, 3, 4)
K = np.random.rand(1, 3, 4)
V = np.random.rand(1, 3, 4)

output, weights = scaled_dot_product_attention(Q, K, V)
print(output)
print(weights)
