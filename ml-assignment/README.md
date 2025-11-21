project: "ML Intern Assessment – Trigram Language Model & Attention Mechanisms"

steps:

  - step: "Clone the Repository"
    commands:
      - git clone https://github.com/Rakshanayak24/ml-intern-assessment.git
      - cd ml-intern-assessment/ml-assignment

  - step: "Create and Activate Virtual Environment"
    commands:
      - python -m venv venv
      - source venv/bin/activate        # Linux / Mac
      - venv\Scripts\activate           # Windows PowerShell

  - step: "Install Dependencies"
    commands:
      - pip install -r requirements.txt

  - step: "Project Structure"
    tree:
      - src/
          - generate.py
          - ngram_model.py
          - utils.py
          - __init__.py
      - attention/
          - attention.py
          - demo.py
          - __init__.py
      - data/
          - example_corpus.txt
      - tests/
      - evaluation.md
      - README.md

  - step: "Run Part 1 – Trigram Language Model"
    description: "Trains the model on example_corpus.txt and generates text."
    commands:
      - python -m src.generate

  - step: "Customize Training Corpus"
    description: "Replace example_corpus.txt with your own larger dataset."
    file_to_edit: "data/example_corpus.txt"
    commands:
      - python -m src.generate

  - step: "Run Part 2 – Scaled Dot-Product & Multi-Head Attention"
    description: "Runs NumPy-based attention mechanism demo."
    commands:
      - python -m attention.demo

  - step: "Expected Output"
    outputs:
      - "Scaled Dot-Product Attention Output Matrix"
      - "Attention Weights Matrix"
      - "Multi-Head Attention Output"
      - "Per-head Attention Weight Shapes"




  
  


