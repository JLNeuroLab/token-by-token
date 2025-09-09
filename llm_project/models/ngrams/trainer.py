from llm_project.models.configs.configs import NgramConfig
from llm_project.utils.dataloader import load_shakespeare
from llm_project.models.ngrams.model import NGram
from llm_project.utils.file_manager import (
    save_model,
    load_model,
    get_model_path,
    get_project_root,
)

# --- NGramTrainer Setup ---
from llm_project.models.configs.configs import NgramConfig
from llm_project.utils.debugg_utils import Colors
import os

# import time
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm


class NGramTrainer:
    def __init__(self, config, model, tokens, root=None, k=None, final=False):
        self.model = model
        self.tokens = tokens
        self.n = config.n
        self.device = config.device
        if k is None:
            raise ValueError(
                f"{Colors.FAIL}[FAIL]{Colors.ENDC} k must be selected (use max_k from the BPE tokenizer)"
            )
        self.k = k
        self.root = get_project_root()
        self.config = config
        self.train_text = None
        self.valid_text = None
        self.final = final

    def _state_dict(self):
        if self.model is None:
            raise ValueError(f"{Colors.FAIL}[FAIL]{Colors.ENDC} Model not initialized")
        state = {
            "n": self.model.n,
            "tokens": self.model.tokens,
            "ngram_freqs": self.model.ngram_freqs,
            "ngram_dict": dict(self.model.ngram_dict),
            "lambdas": self.model.lambdas,
        }
        return state

    def _save_state(self, subdir="ngram", filename=None, final=None):
        """
        Save model state to disk.

        Args:
            subdir (str): Subfolder inside models
            filename (str): File name for pickle
            final (bool): Save in final folder if True

        Returns:
            str: Full path of saved file
        """
        final_flag = final if final is not None else getattr(self, "final", False)
        state = self._state_dict()

        save_path = save_model(
            state,
            root=self.root,
            subdir=subdir,
            category="models",
            filename=filename,
            final=final_flag,
        )
        # print(f"[DEBUG SAVE] Saving model to: {save_path}")
        # print(f"[DEBUG SAVE] final_flag = {final_flag}, subdir = {subdir}")
        # print(f"[DEBUG SAVE] Model saved successfully at: {save_path}")
        return save_path

    def _load_state(self, filename=None, final=False):
        """
        Load model state from disk and initialize NGram instance.

        Args:
            filename (str): Name of the pickle file
            final (bool): Load from final folder if True

        Returns:
            NGram: Loaded NGram model
        """
        # print(f"[DEBUG LOAD] Trying to load model from: {filename}")
        final_flag = final if final is not None else getattr(self, "final", False)
        # print(f"[DEBUG LOAD] final_flag = {final_flag}, subdir = 'ngram'")

        model_data = load_model(
            root=self.root, filename=filename, final=final_flag, subdir="ngram"
        )

        self.model = NGram(tokens=model_data["tokens"], n=model_data["n"])
        self.model.ngram_freqs = model_data["ngram_freqs"]
        self.model.ngram_dict = model_data["ngram_dict"]
        self.model.lambdas = model_data["lambdas"]

        return self.model

    def train(
        self,
        force_retrain=False,
        tune_lambdas=True,
        train_limit=None,
        valid_limit=None,
        final=None,
    ):
        """Loads data and trains the N-gram model using pre-tokenized input."""
        if self.tokens is None:
            raise ValueError(
                "Tokens must be provided externally; tokenizer is decoupled."
            )

        # Load train/validation text
        full_train_text = "".join(load_shakespeare("train"))
        self.train_text = (
            full_train_text[:train_limit] if train_limit else full_train_text
        )

        full_valid_text = "".join(load_shakespeare("validation"))
        self.valid_text = (
            full_valid_text[:valid_limit] if valid_limit else full_valid_text
        )

        model_fname = f"ngram_model_n{self.n}_k{self.k}.pkl"
        final_flag = final if final is not None else getattr(self, "final", False)
        model_folder = get_model_path(
            self.root, "models", subdir="ngram", final=final_flag
        )
        model_path = os.path.join(model_folder, model_fname)

        if os.path.exists(model_path) and not force_retrain:
            print(f"\n--- Loading pre-trained model from:\n{model_path}")
            return self._load_state(model_fname, final=final_flag)
        else:
            print(
                f"{Colors.WARNING}[WARNING]{Colors.ENDC} No existing model found, training one from scratch"
            )

        print("--- Training N-gram model ---")
        self.model = NGram(self.tokens, self.n)
        self.model.build_all_ngram_freqs()

        if tune_lambdas:
            best_lambdas, _, _ = self.tune_lambdas()
            self.model.lambdas = {"best": best_lambdas}
        else:
            self.model.lambdas = {"default": [1 / self.n] * self.n}

        # Save model and get full path
        saved_path = self._save_state(
            subdir="ngram", filename=model_fname, final=final_flag
        )
        print(f"{Colors.OKGREEN}[OK]{Colors.ENDC} Model saved to: {saved_path}")

        return self.model

    def compute_perplexity(self, test_tokens: list, lambdas: list):
        """Calculates perplexity using on-the-fly probability and vectorized counts."""
        test_ngrams = [
            tuple(test_tokens[i : i + self.n])
            for i in range(len(test_tokens) - self.n + 1)
        ]
        if not test_ngrams:
            return float("inf")
        ngram_counts = Counter(test_ngrams)
        unique_ngrams = list(ngram_counts.keys())
        freqs = np.array([ngram_counts[ng] for ng in unique_ngrams])
        probs = self.model.get_interpolated_prob_batch(unique_ngrams, np.array(lambdas))
        probs[probs == 0] = 1e-10

        log_probs_sum = np.sum(freqs * np.log(probs))
        total_ngrams_in_test = np.sum(freqs)
        avg_log_prob = log_probs_sum / total_ngrams_in_test
        perplexity = np.exp(-avg_log_prob)
        return perplexity

    def plot_lambda_perplexities(self, results, folder="ngram", filename="lambda_perplexity.png", final=False):
        """
        Plots a bar chart of perplexity results for different lambda sets.
        Respects final/pretrained folder logic.
        """
        import matplotlib.pyplot as plt
        from llm_project.utils.file_manager import get_model_path
        import os

        if not results:
            print(f"{Colors.WARNING}[WARN]{Colors.ENDC} No results to plot.")
            return

        labels, perplexities = zip(*results)
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, perplexities, color="skyblue", edgecolor="black")
        ax.set_ylabel("Perplexity (Lower is Better)")
        ax.set_title("Lambda Weighting Schemes vs. Perplexity")
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, yval, f"{yval:.2f}", va="bottom", ha="center")

        # Respect final folder logic
        save_folder = get_model_path(root=self.root, category="models", subdir="ngram", final=final)
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, filename)

        try:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"{Colors.OKGREEN}[OK]{Colors.ENDC} Perplexity comparison plot saved to {save_path}")
        except Exception as e:
            print(f"{Colors.FAIL}[FAIL]{Colors.ENDC} Failed to save plot: {e}")

        plt.close(fig)

    def set_lambdas(self, lambdas, label="custom"):
        """
        Set a custom lambda weighting for interpolated probabilities.

        Args:
            lambdas (list or np.ndarray): Interpolation weights for n-gram orders.
            label (str): Label to identify this set of lambdas.
        """
        if not np.isclose(sum(lambdas), 1.0):
            raise ValueError("Lambdas must sum to 1.")
        self.model.lambdas[label] = lambdas

    def tune_lambdas(self, lambda_candidates=None, valid_limit=None, plot=True):
        """Finds optimal lambda weights by minimizing perplexity."""
        if lambda_candidates is None:
            lambda_candidates = {
                "Trigram Focus": [0.1, 0.3, 0.6],
                "Balanced": [0.33, 0.33, 0.34],
                "Bigram Focus": [0.1, 0.6, 0.3],
                "Unigram Focus": [0.6, 0.3, 0.1],
            }

        validation_tokens = self.tokens  # External tokens expected
        best_lambdas, lowest_perplexity, results = None, float("inf"), []

        for label, current_lambdas in tqdm(
            lambda_candidates.items(), desc="Tuning Lambdas"
        ):
            perplexity = self.compute_perplexity(validation_tokens, current_lambdas)
            results.append((label, perplexity))
            if perplexity < lowest_perplexity:
                lowest_perplexity, best_lambdas = perplexity, current_lambdas

        self.model.lambdas = {"best": best_lambdas}
        # Plot automatically if requested
        if plot and results:
            self.plot_lambda_perplexities(results, folder="ngram", final=getattr(self, "final", False))

        return best_lambdas, lowest_perplexity, results

    def plot_perplexity_comparison(self, results: dict, output_folder: str):
        n_values = sorted(results.keys())
        perplexities = [results[n] for n in n_values]
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(
            [str(n) for n in n_values], perplexities, color="coral", edgecolor="black"
        )
        ax.set_xlabel("N-gram Order (n)")
        ax.set_ylabel("Perplexity (Lower is Better)")
        ax.set_title("N-gram Model Perplexity vs. Order (n)")
        ax.grid(axis="y", linestyle="--")
        for bar in bars:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval,
                f"{yval:.2f}",
                va="bottom",
                ha="center",
            )
        min_perplexity = min(perplexities)
        best_n_index = perplexities.index(min_perplexity)
        bars[best_n_index].set_color("mediumseagreen")
        ax.legend([bars[best_n_index]], ["Best Performance"])
        os.makedirs(output_folder, exist_ok=True)
        save_path = os.path.join(output_folder, "ngram_perplexity_vs_n_comparison.png")
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"{Colors.OKGREEN}[OK]{Colors.ENDC} Plot saved to {save_path}")


if __name__ == "__main__":
    print("--- Running N-gram Performance Test using BPE ---")

    from llm_project.bpe.bytepair_encoding import BPE
    from llm_project.utils.file_manager import save_tokenizer

    # --- Config paths ---
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    datapath = os.path.join(project_root, "data", "raw", "Shakespeare_clean_full.txt")

    # --- BPE Setup ---
    max_k = 2000
    bpe = BPE(max_k=max_k, datapath=datapath)

    # Load and normalize text
    norm_text = bpe.load_and_normalize()

    # Split train/test (10% test)
    test_text, train_text = bpe.split_train_test()
    bpe.train_text = train_text[:10000]
    bpe.test_text = test_text[:1000]

    # Train BPE on train_text
    bpe.text = bpe.train_text
    print("\n--- Training BPE ---\n")
    bpe.BPE_encoder()

    # Tokenize train and test
    train_tokens = bpe.tokens
    test_tokens = bpe.BPE_segmenter(bpe.test_text)
    print(f"Train tokens: {len(train_tokens)}")
    print(f"Test tokens: {len(test_tokens)}\n")

    # --- Save BPE tokenizer (merges) ---
    save_tokenizer(bpe, project_root, f"BPE_merges_k{max_k}.pkl", category="tokenizers")
    print(f"{Colors.OKGREEN}[OK]{Colors.ENDC} BPE tokenizer saved")

    config = NgramConfig(n=3, device="cpu")
    trainer = NGramTrainer(config=config, model=None, tokens=train_tokens, k=max_k)

    # Train N-gram
    ngram_model = trainer.train(
        force_retrain=False, tune_lambdas=True, train_limit=10000, valid_limit=1000
    )
    print(
        f"{Colors.OKGREEN}[OK]{Colors.ENDC} N-gram model saved automatically by trainer."
    )

    # Compute perplexity on test set
    best_lambdas = ngram_model.lambdas["best"]
    perplexity = trainer.compute_perplexity(test_tokens, list(best_lambdas))
    print(f"\nPerplexity on test set: {perplexity:.2f}\n")
