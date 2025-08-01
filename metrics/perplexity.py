
import numpy as np
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from ngrams.ngram_model import build_all_ngram_freqs, interpolate_probs_with_laplace, create_ngrams

def compute_perplexity(test_tokens: list, ngrams_probs: dict, n: int):
    """
        Calculates the perplexity score of the ngram model on a text.

        Args:
            test_tokens (list): tokens resulting from word tokenization of the text (BPE)
            ngrams_probs (dict): mapping ngrams to interpolated probabilities
            n (int): ngram order
        Returns:
            Perplexity score
    """
    ngrams = create_ngrams(test_tokens, n)
    
    log_probs_sum = 0.0
    counts = 0

    for ngram in ngrams:
        prob = ngrams_probs.get(ngram, 1e-8)
        log_probs_sum += np.log(prob)
        counts += 1

    avg_log_prob = log_probs_sum / counts if counts > 0 else float('-inf')
    perplexity = np.exp(-avg_log_prob)

    return perplexity

if __name__ == "__main__":
    train_tokens = ["the", "cat", "sat", "on", "the", "mat"]

    n = 3
    ngram_freq = build_all_ngram_freqs(train_tokens, n)
    vocab_size = len(set(train_tokens))
    lambdas = {"example": [0.1, 0.3, 0.6]}

    interpolated_probs = interpolate_probs_with_laplace(ngram_freq, lambdas, vocab_size)

    test_tokens = ["the", "cat", "sat", "on", "the", "bed"]

    perplexity = compute_perplexity(test_tokens, interpolated_probs["example"], n)
    print(f"Perplexity: {perplexity:.4f}") 