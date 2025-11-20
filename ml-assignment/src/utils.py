# This file is optional.
# You can add any utility functions you need for your implementation here.
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

def tokenize(text):
    if not text.strip():
        return []
    return text.split()

