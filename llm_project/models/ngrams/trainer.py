import os
import numpy as np
import matplotlib.pyplot as plt
from llm_project.utils.file_manager import save_item, load_item
from llm_project.models.ngrams.model import NGram
from llm_project.bpe.bytepair_encoding import BPE, normalize_text
from llm_project.utils.dataloader import load_shakespeare

class NGramTrainer:

    def __init__(self, model, tokens, n, max_k, root = None):
        self.model = model
        self.tokens = tokens
        self.n = n
        self.root = root if root is not None else os.path.abspath(
                                                        os.path.join(
                                                            os.path.dirname(__file__),
                                                            "..",
                                                            "..",
                                                            ".."
                                                        )
                                                    )
        self.bpe = None
        self.merges = None
        self.max_k = max_k

    def train(self, force_retrain=False, tune_lambdas=True):
        """
        Loads data and trains the BPE and N-gram models.
        If -t/--train a retraining is forced.
        """
        model_fname = f"ngram_model_n{self.n}_k{self.max_k}.pkl"
        merges_fname = f"BPE_merges_k{self.max_k}.pkl"

        model_dir = os.path.join(self.root, "experiments", "saved_models", "ngram")
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, model_fname)
        merges_path = os.path.join(model_dir, merges_fname)

        # basic argument without need of training
        if not force_retrain and os.path.exists(model_path) and os.path.exists(merges_path):
            # -----------
            print(f"--- Loading pre-trained model from {model_path} ---")
            model_data = load_item(model_dir, model_fname)
            self.model = NGram(tokens=model_data["tokens"], n=model_data["n"])
            self.model.ngram_freqs = model_data["ngram_freqs"]
            self.model.lambdas = model_data["lambdas"]
            self.model.interpolated_probs = model_data["interpolated_probs"]

            # --- Load BPE merges and recreate BPE instance ---
            self.merges = load_item(model_dir, merges_fname)
            self.bpe = BPE(max_k=self.max_k, text=None)
            self.bpe.merges = self.merges
            self.bpe.tokens = self.model.tokens
            self.tune_lambdas()
            return self.model, self.merges
                
        else:
            # if -t|no file then trains BPE and NG
            if force_retrain:
                print("--- '-t' flag detected. Forcing model retraining... ---")
            else:
                print("--- No pre-trained model found. Training a new one... ---")

            # training bpe 
            print("--- Training BPE (this might take a moment) ---")
            train_text = load_shakespeare("train")
            train_text = train_text[:10000]
            print(f"using only {len(train_text)} of training set")

            self.bpe = BPE(max_k=self.max_k, text=train_text)
            norm_text = self.bpe.load_and_normalize()
            self.bpe.text = norm_text
            self.bpe.BPE_encoder()
            self.merges = self.bpe.merges
            print("--- bpe trained ---")

            # build ngram
            print("--- Training ngram model ---")
            tokens = self.bpe.tokens
            self.model = NGram(tokens, self.n)
            self.model.build_all_ngram_freqs()

            if tune_lambdas:
                print("--- Tuning lambda weights ---")
                best_lambdas, _, _ = self.tune_lambdas()
                self.best_lambdas = best_lambdas
                self.model.lambdas = best_lambdas
                print(f"Best lambdas set in model: {self.model.lambdas}")
            else:
                self.model.interpolated_probs = self.model.interpolate_probs_with_laplace(
                    {"default": [1/self.n]*self.n}
                )

            # Saves model to plk file
            print(f"--- Saving model to {model_path} ---")
            # save model params in a dict to make compatible with save_item
            model_dict = {
                "n": self.model.n,
                "ngram_freqs": self.model.ngram_freqs,
                "tokens": self.model.tokens,
                "lambdas": self.model.lambdas,
                "interpolated_probs": self.model.interpolated_probs
            }
            save_item(model_dict, model_dir, model_fname)
            save_item(self.merges, model_dir, merges_fname)

        return self.model, self.merges
    
    def compute_perplexity(self, test_tokens: list, label="best") :
        """
            Calculates the perplexity score of the ngram model on a text.

            Args:
                test_tokens (list): tokens resulting from word tokenization of the text (BPE)
                ngrams_probs (dict): mapping ngrams to interpolated probabilities
                n (int): ngram order
            Returns:
                Perplexity score
        """
        test_ngrams = []
        for i in range(len(test_tokens) - self.n + 1):
            ngram = tuple(test_tokens[i : i + self.n])
            test_ngrams.append(ngram)

        if not self.model.interpolated_probs:
            raise ValueError("Run interpolate_probs_with_laplace before computing perplexity")
    
        log_probs_sum = 0.0
        counts = 0
        probs = self.model.interpolated_probs[label]
        for ngram in test_ngrams:
            prob = probs.get(ngram, 1e-8)
            log_probs_sum += np.log(prob)
            counts += 1

        avg_log_prob = log_probs_sum / counts if counts > 0 else float('-inf')
        perplexity = np.exp(-avg_log_prob)

        return perplexity
    
    def plot_lambda_perplexities(self, results, folder, filename="lambda_perplexity.png"):
        """
        results: list of (label, perplexity)
        """
        labels, perplexities = zip(*results)
        x = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(x, perplexities, color="skyblue", edgecolor="black")

        # aggiungi valori sopra le barre
        for bar, val in zip(bars, perplexities):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{val:.2f}", ha='center', va='bottom', fontsize=10)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Perplexity")
        ax.set_title("Confronto λ-sets vs Perplexity")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        save_item(fig, folder, filename, text_version=False)
        plt.close(fig)

    
    def tune_lambdas(self):
        """
        Finds the optimal lambda weights for an n-gram model by minimizing
        perplexity on a validation set.
        """
    
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

        valid_text = load_shakespeare("validation")
        valid_text = valid_text[:1000]
        valid_text = normalize_text(valid_text)
        validation_tokens = self.bpe.BPE_segmenter(valid_text)
        
        best_lambdas = None
        best_model_probs = None
        lowest_perplexity = float("inf")

        results = []
        for label, current_lambdas in lambda_candidates.items():
            print(f"\nTesting {label}: {current_lambdas}...")

            # Train the model with the current set of lambdas on the t_toks
            # Wrap the lambdas list in another dictionary
            model_probs = self.model.interpolate_probs_with_laplace({label: current_lambdas})
            # Evaluate the model on the validation tokens
            perplexity = self.compute_perplexity(validation_tokens, label=label)
            print(f"Perplexity on validation set: {perplexity:.4f}")
            results.append((label, perplexity))

            # Check if this is the best result so far
            if perplexity < lowest_perplexity:
                lowest_perplexity = perplexity
                best_lambdas = current_lambdas
                best_model_probs = model_probs[label]
                print(f"!!! New best perplexity found: {lowest_perplexity:.4f} !!!")
                self.model.lambdas = best_lambdas
                self.model.interpolated_probs = { "best": best_model_probs }

        # --- 4. REPORT THE RESULTS ---
        print("\n--- Tuning Complete ---")
        print(f"The lowest perplexity achieved was: {lowest_perplexity:.4f}")
        print(f"The best lambda weights are: {best_lambdas}")
        plot_folder = os.path.join(self.root, "experiments", "saved_models", "ngram")
        self.plot_lambda_perplexities(results, folder=plot_folder)

        return best_lambdas, lowest_perplexity, best_model_probs

if __name__ == "__main__":
    # --- Load datasets ---
    train_text = load_shakespeare("train")[:10000]
    valid_text = load_shakespeare("validation")[:1000]
    print(f"Train size: {len(train_text)}, Validation size: {len(valid_text)}")

    # --- Hyperparameters ---
    max_k = 2000
    n = 3

    # --- Initialize trainer without pre-created BPE ---
    trainer = NGramTrainer(model=None, tokens=None, n=n, max_k=max_k)

    # --- Train BPE and NGram model (train() gestisce tutto) ---
    model, merges = trainer.train(tune_lambdas=True, force_retrain=False)
    print("Training completed.")

    # --- Generate text ---
    prompt_text = "The"
    prompt_text = normalize_text(prompt_text)
    prompt_tokens = trainer.bpe.BPE_segmenter(prompt_text)

    generated_tokens = model.generate_text(prompt_tokens, max_length=20)
    print("\nGenerated text:")
    print(generated_tokens)
