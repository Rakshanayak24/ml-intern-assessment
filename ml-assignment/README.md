## Trigram Language Model

This directory contains the implementation of a Trigram Language Model for the Desible AI ML Internship Assessment. The model learns trigram probabilities from text and generates new sequences based on those learned patterns.

---

## How to Run

### 1. Install Dependencies
Run the following:

pip install -r requirements.txt

---

### 2. Train and Generate Text
To train the model on the example corpus and generate sample text:

python ml-assignment/src/generate.py

This script loads the corpus, trains the TrigramModel, and prints generated output.

---

### 3. Run Tests
To verify your implementation:

pytest ml-assignment/tests/test_ngram.py

All tests should pass when the TrigramModel is implemented correctly.

---

## Project Structure

ml-assignment/
│
├── data/
│   └── example_corpus.txt
│
├── src/
│   ├── ngram_model.py       # TrigramModel implementation
│   ├── utils.py
│   └── generate.py
│
├── tests/
│   └── test_ngram.py
│
├── README.md
└── evaluation.md

---

## Design Choices

A full explanation of design decisions is provided in evaluation.md as required by the assignment.
