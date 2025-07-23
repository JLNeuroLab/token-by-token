import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from collections import defaultdict

# Add BPE module path (parent directory)
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, module_path)

from BPE.bytepair_encoding import load_and_normalize, save_item

# Path to current working directory (e.g., for local files)
current_dir = os.path.dirname(os.path.abspath(__file__))

def create_Ngrams(tokens: str, n: int):
    ngrams = []
    tokens = tokens.split("_")
    for i in range(len(tokens)):
        if i < len(tokens) - n + 1:
            ngram = tuple(tokens[i:i+n])
            ngrams.append(ngram)
        else:
            break
    return ngrams

def compute_freq(ngrams: list):
    ngram_freq = defaultdict(int)
    for ngram in ngrams:
        if ngram in ngram_freq:
            ngram_freq[ngram] += 1
        else:
            ngram_freq[ngram] = 1
    return dict(ngram_freq)

def predict_next_word(words: list, freq_dict, n):
    ngram_pred = defaultdict(int)
    if len(words) == n - 1:
        for ngram, frequency in freq_dict.items():
            if list(ngram[:-1]) == words:
                ngram_pred[ngram] = frequency
        if ngram_pred:
            best_ngram = max(ngram_pred, key=ngram_pred.get)
            prediction = best_ngram[-1]
        else:
            best_ngram = max(freq_dict, key=freq_dict.get)
            prediction = best_ngram[-1]
            print("No ngram found to match the word, fallback to most frequent ngram's final word")
    else:
        raise ValueError("The word provided does not match the ngram length")
    return prediction

def laplace_smoothed_probs(ngram_freq, context_freq, vocab_size):
    """ 
    Compute conditional probabilities with Laplace (Add-one) smoothing

    Args:
        ngram_freq = dictionary with ngrams and corresponding counts
        context_freq = dictionary with (n-1)-grams counts
        vocab_size = total number of unique tokens
    """
    laplace_probs = {}
    for ngram, freq in ngram_freq:
        context = ngram[:-1]

        context_count = context_freq.get(context, 0)

        smoothed_prob = (freq + 1) / context_count + vocab_size

        laplace_probs[ngram] = smoothed_prob

    return laplace_probs
