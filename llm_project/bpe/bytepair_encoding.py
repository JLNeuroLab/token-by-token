from tqdm import trange  # at top
import sys
from collections import defaultdict
import re
import os
import numpy as np
import matplotlib.pyplot as plt

# For Debugging steps
import time
import psutil
import platform
from llm_project.utils.debugg_utils import Colors
from llm_project.utils.tracker import track


# For tracking resources during merge steps
def print_resource_usage(self, step: int):
    now = time.time()
    step_duration = now - self._last_step_time
    self._last_step_time = now

    ram_used = psutil.virtual_memory().used / 1024**2
    cpu_p = psutil.cpu_percent()

    print(
        f"""
            [DEBUG]{f" BPE Merge step: {step:>10}m":>25}||{f"{'Step duration:':<8}{step_duration:>8.2f}s":>24}||
            {f"        Ram used: {ram_used:>15.2f}MB":>32}||{f"{'CPU:':>1} {cpu_p:>17.2f}%":>24}||"""
    )
    current_os = platform.system().lower()

    if current_os == "linux":
        if os.path.exists("/kaggle/working"):
            path = "/kaggle/working"
        elif os.path.exists("/content"):
            path = "/content"
        else:
            path = "/"
        print(f"             Disk Usage ({path}): {psutil.disk_usage(path).percent}%")

    elif current_os == "windows":
        root_drive = os.getenv("SystemDrive", "C:") + "\\"
        print(
            f"             Disk Usage ({root_drive}): {psutil.disk_usage(root_drive).percent}%"
        )

    elif current_os == "darwin":
        print(f"             Disk Usage (/): {psutil.disk_usage('/').percent}%")

    else:
        print(f"{Colors.WARNING}[WARN]{Colors.ENDC} Unknown OS, defaulting to '/'")
        print(f"             Disk Usage (/): {psutil.disk_usage('/').percent}%")

    return step_duration


def get_vocab(text):
    vocab = defaultdict(int)
    tokens = text
    for token in tokens:
        vocab[token] += 1
    return dict(vocab)


def normalize_text(text):
    # lower case text
    text = text.lower()
    # compress spaces into single spaces
    text = re.sub(r"\s+", "_", text.strip())
    # remove everything except for characters, spaces and '
    text = re.sub(r"[^\w\s?!.']", "", text)
    # save text
    # save_item(text, "normalized_text", "normalized_shakespeare.txt")
    return text


class BPE:
    def __init__(self, max_k, data_path=None, text=None, track_resource_fn=None):
        # Step timer init for debugg
        self.merge_time_dict = {}  # Tracks times per merge step
        self._last_step_time = time.time()
        self.track_resource_fn = track_resource_fn or print_resource_usage

        # text attributes
        self.datapath = data_path
        self.text = text
        self.train_text = None
        self.test_text = None

        # Mappings
        self.token_to_id = {}
        self.id_to_token = {}

        # bpe attributes
        self.max_k = max_k
        self.norm_text = None
        self.tokens = None
        self.vocab_size_history = []
        self.vocab = {}
        self.merges = []

    def plot_merge_times(self):
        if not self.merge_time_dict:
            print("No merge timings recorded.")
            return

        steps = list(self.merge_time_dict.keys())
        durations = list(self.merge_time_dict.values())

        plt.figure()
        plt.plot(steps, durations, label="Step Duration (s)", color="blue")
        plt.xlabel("Merge Step")
        plt.ylabel("Time (seconds)")
        plt.title("Time per Merge Step")
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.show()

    # Methods enc/dec/map
    def build_token_mappings(self):
        """Creates the token-to-ID and ID-to-token mappings from the vocab."""
        vocab_list = list(self.vocab.keys())
        self.token_to_id = {token: i for i, token in enumerate(vocab_list)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}

    @track(v=True)
    def encode(self, text_to_encode):
        """Encodes a string into a list of token IDs."""
        segmented_tokens = self.BPE_segmenter(text_to_encode)
        return [self.token_to_id.get(t, -1) for t in segmented_tokens]

    @track(v=True)
    def decode(self, ids_to_decode):
        """Decodes a list of token IDs back into a string."""
        text = "".join([self.id_to_token.get(i, "") for i in ids_to_decode])
        return text.replace("_", " ")

    # Split train and test method

    def split_train_test(self, test_ratio=0.1):
        split_point = int(len(self.text) * test_ratio)
        # Returns test, train
        return self.text[:split_point], self.text[split_point:]

    # ----------------Normalize the text: substitute wide spaces with single space, remove every character
    # except for letters, numbers and ', turn the text in lowercase
    def load_and_normalize(self) -> str:
        """
        Loading and normalizing the input text

        """
        if self.text is None and self.datapath is not None:
            with open(self.datapath, "r", encoding="utf-8") as f:
                self.text = f.read()
        elif self.text is None:
            raise ValueError(
                "You must specify a datapath or provide a raw text when importing this module"
            )
        norm_text = normalize_text(self.text)
        return norm_text

    # -------------------Byte-Pair Encoding method-----------------------
    @track(v=True)
    def BPE_encoder(self):
        """
        Perform byte-pair encoding algorithm over the input text

        Args:
            text = string of words separeted by _ spaces
            max_k = number of times the algorithm merges a pair in the text
            pretokens = (add description)

        Returns:
            tokens (list of str): Final list of tokens after BPE merges.
            vocab_size_history (list of int): Vocabulary size at each merge step.
            final_vocab (dict): Final vocabulary as a dictionary {token: frequency}.
            bpe_merges (list of tuples): List of performed merges in the form
                                    [((token1, token2), new_token), ...].
        """
        # Turn the text into a list of characters
        tokens = list(self.text)
        self.tokens = tokens.copy()

        # Start a timer for training time
        start_time = time.time()

        # Training loop over k
        for step in trange(self.max_k, desc="Training BPE", ncols=100):
            # New track for bpe merges
            if (step + 1) % 50 == 0 or step == 0:
                print(f"[BPE] Merge step {step + 1}/{self.max_k}", flush=True)
                if self.track_resource_fn:
                    # duration = print_resource_usage(self, step)
                    duration = self.track_resource_fn(self, step)
                    self.merge_time_dict[step] = duration

                sys.stdout.flush()
                frequencies = defaultdict(int)

            # Iteration over all the tokens in the text, we exclude the last one
            for i in range(len(tokens) - 1):
                # A pair is defined by adjecent tokens in the text
                pair = (tokens[i], tokens[i + 1])
                # Update frequencies dictionary every time is encountered in the text
                frequencies[pair] += 1

            # If the dictionary is empty we do not merge anymore
            if not frequencies:
                print(
                    f"{Colors.UNDERLINE}No more pairs to merge at step {step}{Colors.ENDC}. Stopping early."
                )
                break

            # Select the most frequent pair
            most_freq = max(frequencies, key=frequencies.get)

            # Merge the pair into a new token
            new_token = "".join(most_freq)

            # Append the merge at the merges list
            self.merges.append((most_freq, new_token))

            # -------Segmenter-------
            # Now we rebuild the text using the new token
            m = 0
            new_text = []
            # While loop until we reach the end of the text
            while m < len(tokens):
                # If the current token is not the last one and the pair matches the most frequent pair
                # we substitute the pair with the new token and we add it to the new text
                if m < len(tokens) - 1 and (tokens[m], tokens[m + 1]) == most_freq:
                    new_text.append(new_token)
                    # we skip two positions because now the following token is merged into new token with the previous one
                    m += 2
                # Otherwise just append the token as it is
                else:
                    new_text.append(tokens[m])
                    m += 1
            # Now the new text contains the new token
            # And the loop repeats
            tokens = new_text
            self.tokens = tokens
            # Add the size of the vocabulary to the vocabulary history
            self.vocab_size_history.append(len(get_vocab(tokens)))
            # print(f"step {step}: merged {most_freq} in {new_token}")

        # Retrieves training time
        total_time = time.time() - start_time
        print(f"Total BPE training time: {total_time:.2f} seconds")

        # Obtain the final vocabulary of the tokenized text
        initial_vocab = get_vocab(list(self.text))
        # Create a dictionary of all tokens created during merges
        # Freq isn't critical here
        merged_tokens_vocab = {merge[1]: 0 for merge in self.merges}
        # Combine them to create a complete vocabulary
        full_vocab = {**initial_vocab, **merged_tokens_vocab}
        self.vocab = full_vocab

        print(f"BPE training {Colors.OKGREEN}[DONE]{Colors.ENDC}\n")
        self.build_token_mappings()

    def plot_vocabulary_growth(self):
        """
        Visualize the growth of the vocabulary against the number of merges performed by BPE.

        Args:
            vocab_size: number of types in the vocabulary
            max_k: maximum number of merges

        """
        if not self.vocab_size_history:
            print("No vocabulary size data available to plot.")
            return

        ks_for_plt = np.linspace(10, self.max_k, 100, dtype=int)
        vocab_plot = [
            self.vocab_size_history[k - 1]
            if k - 1 < len(self.vocab_size_history)
            else self.vocab_size_history[-1]
            for k in ks_for_plt
        ]

        fig = plt.figure()
        plt.plot(ks_for_plt, vocab_plot, label="vocabulary size")

        max_index = np.argmax(vocab_plot)
        max_k_value = ks_for_plt[max_index]
        max_vocab_value = vocab_plot[max_index]
        plt.axvline(
            x=max_k_value,
            color="red",
            linestyle="--",
            label=f"Max vocab at k={max_k_value}",
        )
        plt.axhline(y=max_vocab_value, color="gray", linestyle=":", alpha=0.7)

        plt.xlabel("number of merges")
        plt.ylabel("vocabulary size")
        plt.title("vocabulary size over number of merges")
        plt.legend()
        plt.tight_layout()
        return fig
        # save_item(fig, "plots", "vocabulary_growth")

    # ------------------Tokenize test text with the training tokens--------------------------
    def BPE_segmenter(self, text=None):
        """
        Apply a sequence of Byte-Pair Encoding (BPE) merges to the input text.

        This function takes raw text (as a string) and a list of BPE merge operations
        (usually learned from training data), and applies them sequentially to
        reproduce the tokenization on new/test data.

        Args:
            text (str): Input text string to tokenize, typically the test set.
            merges (list of tuples): List of merge operations in the form
                                    [((token1, token2), new_token), ...]
                                    produced during BPE training.

        Returns:
            tokens (list of str): Tokenized version of the input text after applying all merges.

        """
        if text is None:
            if self.test_text is None:
                raise ValueError("No text provided and self.test_text is None.")
            text = self.test_text

        tokens = list(text)
        # We retrieve the merges of the training by retrieving the merges history
        for pair, new_token in self.merges:
            i = 0
            new_tokens = []
            while i < len(tokens):
                # if the pair in the text matches the pair in the merges history
                # we append it to the new test text
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            # Same procedure as the second loop of the BPE method
            tokens = new_tokens
        # test_vocab = get_vocab(tokens)
        # save_item(" ".join(tokens), "test_results", "test_tokenized.txt")
        # save_item(test_vocab, "test_results", "test_final_vocab.pkl")
        return tokens

    # After applying the tokenizer to the test text, evaluate the performance by computing coverage:
    # coverage is defined by the number of tokens in the text, that the learned vocabulary from the training
    # can cover.
    @track(v=True)
    def compute_coverage(self, text=None):
        if text is None:
            if self.test_text is None:
                raise ValueError("No text provided and self.test_text is None.")
            text = self.test_text

        if isinstance(text, list):
            tokenized = text
        else:
            tokenized = self.BPE_segmenter(text)

        covered = sum(token in self.vocab for token in tokenized)
        return covered / len(tokenized)


# -------------------Training----------------------
if __name__ == "__main__":
    # CONFIG
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    experiments_root = os.path.join(project_root, "experiments")
    print("Project root:", project_root)

    from llm_project.utils.file_manager import save_item
    import os

    train_results_path = os.path.join(
        experiments_root, "bpe_results", "train_results"
    )  # o neural_ngram
    plots_path = os.path.join(
        experiments_root,
        "plots",
    )
    test_results_path = os.path.join(experiments_root, "bpe_results", "test_results")

    os.makedirs(train_results_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)
    os.makedirs(test_results_path, exist_ok=True)

    datapath = os.path.join(project_root, "data", "raw", "Shakespeare_clean_full.txt")

    # set k value
    max_k = 2000
    # Initialize tokenizer
    bpe = BPE(max_k=max_k, datapath=datapath)

    # load and normalize text
    norm_text = bpe.load_and_normalize()
    # Split train/test
    test_text, train_text = bpe.split_train_test()
    bpe.train_text = train_text[:10000]
    bpe.test_text = test_text[:1000]
    print(f"Train size: {len(bpe.train_text)} chars")
    print(f"Test size:  {len(bpe.test_text)} chars\n")

    # Train BPE solo sul train_text
    bpe.text = bpe.train_text
    print("\nStarting Byte-Pair Encoding on train text...\n")
    bpe.BPE_encoder()
    print("\nLoop completed, train text tokenized\n")

    # To visualize slow downs
    bpe.plot_merge_times()

    # Plot growth of vocabulary
    bpe.plot_vocabulary_growth()

    # Tokenize test text usando le merges imparate dal train
    print("\nTokenizing test text...\n")
    test_tokens = bpe.BPE_segmenter(bpe.test_text)
    print("\nTokenization of test text completed\n")

    # Compute coverage del test set
    coverage = bpe.compute_coverage(test_tokens)
    print("-" * 80)
    print(f"[RESULTS]:\nCoverage score for k = {max_k}: {coverage:.4f}\n")

    # Save results
    save_item(" ".join(test_tokens), test_results_path, "test_tokenized.txt")
    save_item(test_tokens, test_results_path, "test_tokenized.pkl")

    reconstructed_text = "".join(test_tokens)
    save_item(reconstructed_text, test_results_path, "test_reconstructed.txt")

    save_item(bpe.vocab, train_results_path, "train_final_vocab.pkl")
    save_item(bpe.vocab_size_history, train_results_path, "train_vocab_history.pkl")
    save_item(bpe.merges, train_results_path, "train_bpe_merges.pkl")
    save_item(bpe.merges, train_results_path, "train_bpe_merges.pkl")
