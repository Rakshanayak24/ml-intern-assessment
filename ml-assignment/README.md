🌟 Trigram Language Model + Scaled Dot-Product Attention
Desible AI/ML Internship Assessment — Completed by Raksha Nayak

This repository contains my end-to-end implementation of both tasks from the Desible AI / ML Internship Assessment:

✔ Task 1 — Trigram (N=3) Language Model (from scratch)
✔ Task 2 — Scaled Dot-Product Attention using NumPy only (optional, completed)

The project is modular, clean, unit-tested, and includes runnable demos.

📥 Run in Google Colab (Recommended)

You can run the full project inside Colab using:

!git clone https://github.com/Rakshanayak24/ml-intern-assessment.git
%cd ml-intern-assessment/ml-assignment
!pip install -r requirements.txt


Now you're ready to run the model or attention demo.

🏗️ Project Structure
ml-assignment/
│
├── data/
│   └── example_corpus.txt
│
├── src/
│   ├── ngram_model.py        # Trigram model implementation
│   ├── utils.py
│   └── generate.py           # Train + generate text
│
├── attention/
│   ├── attention.py          # Scaled Dot-Product Attention (NumPy-only)
│   ├── demo.py               # Demo script
│   └── __init__.py
│
├── tests/
│   └── test_ngram.py         # Unit tests
│
├── README.md                 # Documentation
└── evaluation.md             # 1-page design choices summary

🚀 Task 1 — Trigram Language Model
📌 Install Dependencies
pip install -r requirements.txt

📌 Train & Generate Text
python src/generate.py


This will:

Load & clean the corpus

Build trigram counts

Compute probabilities

Generate new text using probabilistic sampling

🧪 Run Unit Tests
pytest tests/test_ngram.py


All tests should pass with the final implementation.

🧠 Task 2 — Scaled Dot-Product Attention (NumPy Only)

Implementation located in:

attention/attention.py


Run the demo:

python attention/demo.py

Example Output:
Q = [[1 0]]
K = [[1 1]]
V = [[0.5 2. ]]

Attention Weights:
[[1.]]

Attention Output:
[[0.5 2. ]]
