**Trigram Language Model**

This directory contains the core assignment files for the Trigram Language Model & Attention Mechanisms.

**How to Run**

This README explains how to run the text-suite and the example generator included in the assignment.

**Prerequisites**

Python 3.8+

**Install dependencies:**
git clone https://github.com/Rakshanayak24/ml-intern-assessment.git
cd ml-intern-assessment/ml-assignment
python -m pip install -r requirements.txt

**Create Virtual Environment (Optional)**

**Linux / Mac:**
python -m venv venv
source venv/bin/activate


**Windows PowerShell:**
python -m venv venv
venv\Scripts\activate


**Project Structure**
ml-assignment/
│
├── src/
│   ├── generate.py
│   ├── ngram_model.py
│   ├── utils.py
│   └── __init__.py
│
├── attention/
│   ├── attention.py
│   ├── demo.py
│   └── __init__.py
│
├── data/
│   └── example_corpus.txt
│
├── tests/
├── evaluation.md
├── README.md
└── requirements.txt


**Part 1 — Run the Trigram Language Model**

This command trains the trigram model using example_corpus.txt and generates text.

python -m src.generate

**Custom Dataset**

To train on your own corpus, replace the file:

data/example_corpus.txt

Then run again:

python -m src.generate

**Part 2 — Run Scaled Dot-Product & Multi-Head Attention**

This runs the NumPy-based attention demonstration.

python -m attention.demo

**Expected Output:**

* Running the attention module produces:

* Scaled Dot-Product Attention output matrix

* Attention weight matrix

* Multi-head attention output

* Shapes of per-head attention weights





  
  


