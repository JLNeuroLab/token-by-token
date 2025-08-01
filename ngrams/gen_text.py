import os
import sys
import pickle
from collections import defaultdict
import argparse

# --- This block makes sure Python can find your other project folders ---
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # noqa
sys.path.insert(0, root)  # noqa

# --- Project-specific imports with #noqa to prevent auto-formatting errors ---
from ngrams.ngram_model import build_all_ngram_freqs  # noqa
from bpe.bytepair_encoding import load_and_normalize, BPE_encoder, BPE_segmenter  # noqa


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
    prompt_tokens = BPE_segmenter(prompt_text.lower(), merges)
    generated_tokens = list(prompt_tokens)
    print(f"Starting with prompt tokens: {generated_tokens}")

    while len(generated_tokens) < max_length:
        prediction = None

        # This loop automatically tries to predict, backing off from n down to 1 (unigram)
        for current_n in range(n, 0, -1):
            # Get the context needed for the current n-gram size
            context_len = current_n - 1
            if len(generated_tokens) >= context_len:
                context = (
                    tuple(generated_tokens[-context_len:]) if context_len > 0 else ()
                )

                # Try to predict
                prediction = predict_next_token_argmax(context, ngram_freqs, current_n)

                if prediction is not None:
                    # Found a successful prediction, so stop backing off
                    break

        # If no prediction was found even after trying unigrams, stop.
        if prediction is None:
            print("--- Context not found, stopping generation. ---")
            break

        # Stop if the stop token is generated
        if prediction == stop_token:
            generated_tokens.append(prediction)
            print("--- Stop token generated. ---")
            break

        generated_tokens.append(prediction)
    return " ".join(generated_tokens)


def train_model(n, max_k=2000, force_retrain=False):
    """
    Loads data and trains the BPE and N-gram models.
    If -t/--train a retraining is forced.
    """
    model_fname = f"ngram_model_n{n}_k{max_k}.pkl"
    merges_fname = f"BPE_merges_k{max_k}.pkl"

    model_dir = os.path.join(root, "saved_models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, model_fname)
    merges_path = os.path.join(model_dir, merges_fname)

    # basic argument without need of training
    if not force_retrain and os.path.exists(model_path) and os.path.exists(merges_path):
        print(f"--- Loading pre-trained model from {model_path} ---")
        with open(model_path, "rb") as f:
            ngram_freqs = pickle.load(f)
        with open(merges_path, "rb") as f:
            merges = pickle.load(f)

        return ngram_freqs, merges
    
    else:
        # if -t|no file then trains BPE and NG
        if force_retrain:
            print("--- '-t' flag detected. Forcing model retraining... ---")
        else:
            print("--- No pre-trained model found. Training a new one... ---")

        # training block
        print("--- Training models (this might take a moment) ---")
        datapath_train = os.path.join(root, "data", "Shakespeare_clean_train.txt")
        train_text = load_and_normalize(datapath_train)
        train_tokens, _, _, merges = BPE_encoder(train_text, max_k)
        ngram_freqs = build_all_ngram_freqs(train_tokens, n)
        print("--- Models trained ---")

        # Saves model to plk file
        print(f"--- Saving model to {model_path} ---")
        with open(model_path, "wb") as f:
            pickle.dump(ngram_freqs, f)
        with open(merges_path, "wb") as f:
            pickle.dump(merges, f)

        return ngram_freqs, merges


def main(args):
    """Main function to orchestrate model training and text generation."""
    # 1. Train the model or load plk with -t
    ngram_freqs, merges = train_model(n=args.n, force_retrain=args.train)

    # 2. Generate text using the trained model
    print(f"\n--- Generating text from prompt: '{args.prompt}' ---")
    generated_sequence = generate_text(
        prompt_text=args.prompt,
        ngram_freqs=ngram_freqs,
        n=args.n,
        merges=merges,
        max_length=20,
    )
    print("\nFinal Generated Text: ")
    print(generated_sequence)


if __name__ == "__main__":
    # This block handles command-line arguments and calls the main function
    parser = argparse.ArgumentParser(description="Generate text using and N-gram.")

    parser.add_argument(
        "-t", "--train", action="store_true", help="Force model retrain"
    )

    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        nargs="+",  # multi-args after arg
        default=["the", "king", "is"],
        help="the starting prompt for gen",
    )

    parser.add_argument(
        "-n",
        "--ngram",
        type=int,
        nargs="?",  # optional arg
        default=3,
        dest="n",
        help="Order of N-gram (i.e. 2,3,4,5....)",
    )
    args = parser.parse_args()
    # join words in single string
    args.prompt = " ".join(args.prompt).lower()

    """
    OLD IMPLEMENTATION
    if len(sys.argv) > 1:
        try:
            n_arg = int(sys.argv[1])
        except ValueError:
            print(f"Error: N-gram order must be int. Using default n={n_arg}.")

        if len(sys.argv) > 2:
            prompt_arg = " ".join(sys.argv[2:]).lower()
    """

    main(args)
