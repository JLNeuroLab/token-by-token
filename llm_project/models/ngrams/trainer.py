import os
import math
import time
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
from llm_project.utils.file_manager import save_item, load_item
from llm_project.models.ngrams.model import NGram
from llm_project.bpe.bytepair_encoding import BPE, normalize_text
from llm_project.utils.dataloader import load_shakespeare


class NGramTrainer:
    def __init__(self, model, tokens, n, max_k, root=None):
        self.model = model
        self.tokens = tokens
        self.n = n
        self.root = (
            root
            if root is not None
            else os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..")
            )
        )
        self.bpe = None
        self.max_k = max_k

    def train(self, force_retrain=False, tune_lambdas=True):
        """Loads data and trains the BPE and N-gram models."""
        model_fname = f"ngram_model_n{self.n}_k{self.max_k}.pkl"
        merges_fname = f"BPE_merges_k{self.max_k}.pkl"
        model_dir = os.path.join(self.root, "experiments", "saved_models", "ngram")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, model_fname)
        merges_path = os.path.join(model_dir, merges_fname)

        if (
            not force_retrain
            and os.path.exists(model_path)
            and os.path.exists(merges_path)
        ):
            print("--- Loading pre-trained model and tokenizer ---")
            model_data = load_item(model_dir, model_fname)
            self.model = NGram(tokens=model_data["tokens"], n=model_data["n"])
            self.model.ngram_freqs = model_data.get("ngram_freqs", {})
            self.model.lambdas = model_data.get("lambdas", {})
            self.bpe = BPE(max_k=self.max_k)
            self.bpe.merges = load_item(model_dir, merges_fname)
            return self.model, self.bpe.merges

        if force_retrain or not os.path.exists(merges_path):
            print("--- Training BPE tokenizer ---")
            train_text = "".join(load_shakespeare("train"))
            self.bpe = BPE(max_k=self.max_k, text=train_text)
            self.bpe.BPE_encoder()
            save_item(self.bpe.merges, model_dir, merges_fname)
            self.tokens = self.bpe.tokens
        else:
            print("--- Loading existing BPE tokenizer ---")
            self.bpe = BPE(max_k=self.max_k)
            self.bpe.merges = load_item(model_dir, merges_fname)
            print("--- Re-tokenizing training text with loaded BPE ---")
            train_text = "".join(load_shakespeare("train"))
            self.tokens = self.bpe.BPE_segmenter(normalize_text(train_text))

        print("--- Training N-gram model ---")
        self.model = NGram(self.tokens, self.n)
        self.model.build_all_ngram_freqs()

        if tune_lambdas:
            best_lambdas, _, _ = self.tune_lambdas()
            self.model.lambdas = {"best": best_lambdas}
        else:
            self.model.lambdas = {"default": [1 / self.n] * self.n}

        model_dict = {
            "n": self.model.n,
            "tokens": self.model.tokens,
            "ngram_freqs": self.model.ngram_freqs,
            "lambdas": self.model.lambdas,
        }
        save_item(model_dict, model_dir, model_fname)
        return self.model, self.bpe.merges

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
        probs = np.array(
            [self.model.get_interpolated_prob(ng, lambdas) for ng in unique_ngrams]
        )
        probs[probs == 0] = 1e-10
        log_probs_sum = np.sum(freqs * np.log(probs))
        total_ngrams_in_test = np.sum(freqs)
        avg_log_prob = log_probs_sum / total_ngrams_in_test
        perplexity = np.exp(-avg_log_prob)
        return perplexity

    def plot_lambda_perplexities(
        self, results, folder, filename="lambda_perplexity.png"
    ):
        """Plots a bar chart of perplexity results for different lambda sets."""
        labels, perplexities = zip(*results)
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, perplexities, color="skyblue", edgecolor="black")
        ax.set_ylabel("Perplexity (Lower is Better)")
        ax.set_title("Lambda Weighting Schemes vs. Perplexity")
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        for bar in bars:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval,
                f"{yval:.2f}",
                va="bottom",
                ha="center",
            )
        save_item(fig, folder, filename, text_version=False)
        print(f"\nPerplexity comparison plot saved to {os.path.join(folder, filename)}")
        plt.close(fig)

    def tune_lambdas(self):
        """Finds the optimal lambda weights by minimizing perplexity."""
        print("\n--- Tuning lambdas ---")
        lambda_candidates = {
            "Trigram Focus": [0.1, 0.3, 0.6],
            "Balanced": [0.33, 0.33, 0.34],
            "Bigram Focus": [0.1, 0.6, 0.3],
            "Unigram Focus": [0.6, 0.3, 0.1],
        }
        valid_text = "".join(load_shakespeare("validation"))
        validation_tokens = self.bpe.BPE_segmenter(normalize_text(valid_text))
        best_lambdas, lowest_perplexity, results = None, float("inf"), []
        for label, current_lambdas in tqdm(
            lambda_candidates.items(), desc="Tuning Lambdas"
        ):
            perplexity = self.compute_perplexity(validation_tokens, current_lambdas)
            results.append((label, perplexity))
            if perplexity < lowest_perplexity:
                lowest_perplexity, best_lambdas = perplexity, current_lambdas
        print(
            f"\n--- Best Lambdas: {best_lambdas} with Perplexity: {lowest_perplexity:.4f} ---"
        )
        plot_folder = os.path.join(
            self.root, "experiments", "plots", "ngram_comparison"
        )
        os.makedirs(plot_folder, exist_ok=True)
        self.plot_lambda_perplexities(results, folder=plot_folder)
        return best_lambdas, lowest_perplexity, None

    # --- NEWLY ADDED EXPERIMENT METHODS ---
    def run_comparison_experiment(self, n_values_to_test=[1, 2, 3, 4], max_k_bpe=2000):
        """Trains and evaluates N-gram models for different n, then plots the results."""
        output_folder = os.path.join(
            self.root, "experiments", "plots", "ngram_comparison"
        )
        os.makedirs(output_folder, exist_ok=True)
        perplexity_results = {}

        print("--- Step 1: Training BPE Tokenizer (once) ---")
        train_text = "".join(load_shakespeare("train"))
        valid_text = "".join(load_shakespeare("validation"))
        bpe = BPE(max_k=max_k_bpe, text=train_text)
        bpe.BPE_encoder()
        train_tokens = bpe.tokens
        validation_tokens = bpe.BPE_segmenter(normalize_text(valid_text))
        print(f"BPE training complete. Vocabulary size: {len(bpe.vocab)}")

        for n in n_values_to_test:
            print(f"\n--- Step 2: Training and Evaluating for n={n} ---")
            temp_trainer = NGramTrainer(
                model=None, tokens=train_tokens, n=n, max_k=max_k_bpe
            )
            temp_trainer.bpe = bpe
            temp_trainer.train(force_retrain=True, tune_lambdas=False)

            # ADJUSTMENT: Pass the default lambdas directly to the new compute_perplexity method
            default_lambdas = [1 / n] * n
            perplexity = temp_trainer.compute_perplexity(
                validation_tokens, lambdas=default_lambdas
            )

            perplexity_results[n] = perplexity
            print(f"✅ n={n}: Perplexity = {perplexity:.4f}")

        print("\n--- Step 3: Generating Perplexity Comparison Plot ---")
        self.plot_perplexity_comparison(perplexity_results, output_folder)
        print("\n--- Comparison Experiment Complete ---")

    def plot_perplexity_comparison(self, results: dict, output_folder: str):
        """Generates a bar chart comparing perplexity for different n."""
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
        plot_filename = "ngram_perplexity_vs_n_comparison.png"
        save_item(fig, output_folder, plot_filename)
        print(f"\nPlot saved to {os.path.join(output_folder, plot_filename)}")
        plt.show()


if __name__ == "__main__":
    # This block now runs your requested n-gram comparison experiment
    print("--- Running N-gram Performance Comparison ---")

    # We can initialize with dummy values as the experiment creates its own trainers
    comparison_runner = NGramTrainer(model=None, tokens=None, n=3, max_k=2000)

    # Call the experiment method
    comparison_runner.run_comparison_experiment(
        n_values_to_test=[2, 3, 4, 5], max_k_bpe=1000
    )
