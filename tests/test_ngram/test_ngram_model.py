import sys
import os
import numpy as np

from llm_project.models.ngrams.model import NGram

def main():
    print("===== NGram Test Script =====")

    # -------------------------------
    # Setup
    # -------------------------------
    tokens = list("abracadabra")
    n = 3
    model = NGram(tokens, n, enable_debug=True)
    lambdas = np.array([0.2, 0.3, 0.5])

    print("\n--- Testing create_ngrams ---")
    trigrams = list(model.create_ngrams())
    print("Generated trigrams:", trigrams)

    print("\n--- Testing build_all_ngram_freqs ---")
    freqs = model.build_all_ngram_freqs(max_n=3)
    print("All ngram frequencies:", freqs)
    print("ngram_dict (sample):", dict(list(model.ngram_dict.items())[:5]))

    print("\n--- Testing build_context_counts_cache ---")
    model.build_context_counts_cache()
    print("Context counts cache (sample):")
    for length, context_map in model.context_counts_cache.items():
        print(f"Length {length}: {dict(list(context_map.items())[:5])}")

    print("\n--- Testing get_interpolated_prob_single ---")
    prob_single = model.get_interpolated_prob_single(trigrams[0], lambdas)
    print(f"Interpolated probability (single) for {trigrams[0]}: {prob_single}")

    print("\n--- Testing get_interpolated_prob_batch ---")
    prob_batch = model.get_interpolated_prob_batch(trigrams, lambdas)
    print("Interpolated probabilities (batch):", prob_batch)

    print("\n===== All tests completed =====")

if __name__ == "__main__":
    main()
