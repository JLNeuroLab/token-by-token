# File: ngrams/gen_text.py

import os
import sys
from collections import defaultdict

# --- This block makes sure Python can find your other project folders ---
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # noqa
sys.path.insert(0, root)  # noqa

# --- Project-specific imports with #noqa to prevent auto-formatting errors ---
from ngrams.ngram_model import build_all_ngram_freqs  # noqa
from BPE.bytepair_encoding import load_and_normalize, BPE_encoder, apply_bpe_merges  # noqa


def predict_next_token_argmax(context: tuple, ngram_freqs: dict, n: int):
    """Predicts the next token by finding the most frequent continuation (argmax)."""
    candidates = defaultdict(int)
    for ngram, freq in ngram_freqs.items():
        if len(ngram) == n and ngram[:-1] == context:
            candidates[ngram[-1]] = freq
    if not candidates:
        return None
    return max(candidates, key=candidates.get)


def generate_text(
    prompt_text: str,
    ngram_freqs: dict,
    n: int,
    merges: list,
    max_length=30,
    stop_token=".",
):
    """Generates a sequence of text starting from a prompt."""
    prompt_tokens = apply_bpe_merges(prompt_text.lower(), merges)
    generated_tokens = list(prompt_tokens)
    print(f"Starting with prompt tokens: {generated_tokens}")

    while len(generated_tokens) < max_length:
        context = tuple(generated_tokens[-(n - 1) :])
        prediction = predict_next_token_argmax(context, ngram_freqs, n)

        # Fallback strategy: if trigram fails, try bigram
        if prediction is None:
            if len(context) > 1:
                context_fallback = context[1:]
                prediction = predict_next_token_argmax(
                    context_fallback, ngram_freqs, n - 1
                )

            # If all fallbacks fail, stop
            if prediction is None:
                print("--- Context not found, stopping generation. ---")
                break

        # Stopping condition
        if prediction == stop_token:
            generated_tokens.append(prediction)
            print("--- Stop token generated. ---")
            break

        generated_tokens.append(prediction)
    return " ".join(generated_tokens)


def train_model(n, max_k=2000):
    """Loads data and trains the BPE and N-gram models."""
    print("--- Training models (this might take a moment) ---")
    datapath_train = os.path.join(root, "data", "Shakespeare_clean_train.txt")
    train_text = load_and_normalize(datapath_train)

    train_tokens, _, _, merges = BPE_encoder(train_text, max_k)
    ngram_freqs = build_all_ngram_freqs(train_tokens, n)
    print("--- Models trained ---")

    return ngram_freqs, merges


def main(n=3, prompt="the king is"):
    """Main function to orchestrate model training and text generation."""
    # 1. Train the model
    ngram_freqs, merges = train_model(n)

    # 2. Generate text using the trained model
    print(f"\n--- Generating text from prompt: '{prompt}' ---")
    generated_sequence = generate_text(
        prompt_text=prompt,
        ngram_freqs=ngram_freqs,
        n=n,
        merges=merges,
        max_length=20,
    )
    print("\nFinal Generated Text: ")
    print(generated_sequence)


if __name__ == "__main__":
    # This block handles command-line arguments and calls the main function
    n_arg = 3
    prompt_arg = "the king is"

    if len(sys.argv) > 1:
        try:
            n_arg = int(sys.argv[1])
        except ValueError:
            print(f"Error: N-gram order must be an integer. Using default n={n_arg}.")

        if len(sys.argv) > 2:
            prompt_arg = " ".join(sys.argv[2:]).lower()

    main(n=n_arg, prompt=prompt_arg)
