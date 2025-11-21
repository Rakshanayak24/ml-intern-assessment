# This file is optional.
# You can add any utility functions you need for your implementation here.
"""
Utility functions for the Trigram Language Model.

This file is optional according to the assignment, but including these helpers
shows good software engineering practice and keeps the main model clean.
"""

import re
import random
from collections import Counter


def clean_text(text):
    """
    Lowercase text, remove punctuation (except .?!), and normalize spaces.

    Args:
        text (str): Raw text.

    Returns:
        str: Cleaned text.
    """
    text = text.lower()
    # keep alphanumeric + sentence punctuation
    text = re.sub(r"[^a-z0-9.?! ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sentence_tokenize(text):
    """
    Split text into sentences, where each sentence is a list of tokens.

    Args:
        text (str): Cleaned text.

    Returns:
        list[list[str]]: Tokenized sentences.
    """
    sentences = re.split(r"[.?!]", text)
    tokenized = []

    for s in sentences:
        words = s.strip().split()
        if words:
            tokenized.append(words)

    return tokenized


def replace_rare_words(sentences, min_freq=2, unk_token="<UNK>"):
    """
    Replace low-frequency words with <UNK>.

    Args:
        sentences (list[list[str]]): Tokenized sentences.
        min_freq (int): Minimum frequency to keep a word.

    Returns:
        (list[list[str]], set): Updated sentences and final vocabulary set.
    """
    counter = Counter(w for sent in sentences for w in sent)

    vocab = {w for w, c in counter.items() if c >= min_freq}
    vocab.add(unk_token)

    updated = [
        [w if w in vocab else unk_token for w in sent]
        for sent in sentences
    ]

    return updated, vocab


def weighted_sample(counter_obj):
    """
    Sample a key based on frequencies using a probabilistic distribution.

    Args:
        counter_obj (Counter): Maps words → counts.

    Returns:
        str | None: Sampled word or None if empty.
    """
    if not counter_obj:
        return None

    words = list(counter_obj.keys())
    counts = list(counter_obj.values())
    total = sum(counts)

    if total == 0:
        return None

    probabilities = [c / total for c in counts]

    return random.choices(words, weights=probabilities, k=1)[0]


