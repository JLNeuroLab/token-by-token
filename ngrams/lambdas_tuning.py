import os
import sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # noqa
sys.path.insert(0, root)  # noqa

from ngrams.ngram_model import train_ngram_model  # noqa
from bpe.bytepair_encoding import load_and_normalize, BPE_encoder, BPE_segmenter  # noqa
from metrics.perplexity import compute_perplexity  # noqa


def find_best_lambdas():
    """
    Finds the optimal lambda weights for an n-gram model by minimizing
    perplexity on a validation set.
    """
    # --- Data preparation
    print("--- Preparing Data ---")

    # Define paths and parameters
    datapath_train = os.path.join(root, "data", "Shakespeare_clean_train.txt")
    datapath_valid = os.path.join(root, "data", "Shakespeare_clean_valid.txt")
    max_k = 2000  # BPE merges
    n = 3  # N-gram order (trigram model)

    # Load and tokenize tra/val data
    train_text = load_and_normalize(datapath_train)
    validation_text = load_and_normalize(datapath_valid)

    print(f"Training set size: {len(train_text)} chars")
    print(f"Validation set size: {len(validation_text)} chars")

    # Train BPE on the tdata set to get tokens and merges
    train_tokens, _, _, merges = BPE_encoder(train_text, max_k)

    # Apply the same tokenization to the validation set
    validation_tokens = BPE_segmenter(validation_text, merges)

    # --- Define lambda
    print("\n--- Defining Lambda Candidates ---")

    # Weights are for [unigram, bigram, trigram]
    lambda_candidates = {
        "set1 (more weight on trigram)": [0.1, 0.3, 0.6],
        "set2 (balanced)": [0.33, 0.33, 0.34],
        "set3 (more weight on bigram)": [0.1, 0.6, 0.3],
        "set4 (more weight on unigram)": [0.6, 0.3, 0.1],
    }
    print(f"Testing {len(lambda_candidates)} sets of lambdas.")

    # --- optimization loop
    print("\n--- Starting Hyperparameter Tuning ---")

    best_lambdas = None
    lowest_perplexity = float("inf")

    for label, current_lambdas in lambda_candidates.items():
        print(f"\nTesting {label}: {current_lambdas}...")

        # Train the model with the current set of lambdas on the t_toks
        # Wrap the lambdas list in another dictionary
        model_probs = train_ngram_model(train_tokens, n, {"model": current_lambdas})

        # Evaluate the model on the validation tokens
        perplexity = compute_perplexity(validation_tokens, model_probs["model"], n)
        print(f"Perplexity on validation set: {perplexity:.4f}")

        # Check if this is the best result so far
        if perplexity < lowest_perplexity:
            lowest_perplexity = perplexity
            best_lambdas = current_lambdas
            print(f"!!! New best perplexity found: {lowest_perplexity:.4f} !!!")

    # --- 4. REPORT THE RESULTS ---
    print("\n--- Tuning Complete ---")
    print(f"The lowest perplexity achieved was: {lowest_perplexity:.4f}")
    print(f"The best lambda weights are: {best_lambdas}")

    return best_lambdas, lowest_perplexity


if __name__ == "__main__":
    find_best_lambdas()
