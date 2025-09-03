import time
import psutil
import platform
import os
from collections import defaultdict, Counter
import numpy as np


class NGram:
    def __init__(self, tokens, n, enable_debug=True):
        self.tokens = tokens
        self.n = n
        self.max_n = n
        self.ngram_freqs = {}
        self.ngram_dict = defaultdict(lambda: defaultdict(int))
        self.vocab = set(tokens)
        self.vocab_size = len(self.vocab)
        self.lambdas = {}
        self.debug = enable_debug
        # Improvement: A single cache for context counts, keyed by length
        self.context_counts_cache = defaultdict(lambda: defaultdict(int))
        # Debug tracking
        self._last_step_time = time.time()
        self.step_times = {}

    def _debug_resources(self, step_name):
        if not self.debug:
            return
        now = time.time()
        step_duration = now - self._last_step_time
        self._last_step_time = now

        ram_used = psutil.virtual_memory().used / 1024**2
        cpu_p = psutil.cpu_percent()
        current_os = platform.system().lower()
        disk_usage = 0

        if current_os == "linux":
            path = "/"
            if os.path.exists("/kaggle/working"):
                path = "/kaggle/working"
            elif os.path.exists("/content"):
                path = "/content"
            disk_usage = psutil.disk_usage(path).percent
        elif current_os == "windows":
            root_drive = os.getenv("SystemDrive", "C:") + "\\"
            disk_usage = psutil.disk_usage(root_drive).percent
        elif current_os == "darwin":
            disk_usage = psutil.disk_usage("/").percent

        print(
            f"[DEBUG] Step: {step_name:>25} || Duration: {step_duration:>8.2f}s || "
            f"RAM: {ram_used:>8.2f}MB || CPU: {cpu_p:>6.2f}% || Disk: {disk_usage}%"
        )
        self.step_times[step_name] = step_duration

    def create_ngrams(self, n=None):
        """Generates n-grams from a list of tokens."""
        if n is None:
            n = self.n
        # Use a generator for memory efficiency with large datasets
        ngrams = (tuple(self.tokens[i : i + n]) for i in range(len(self.tokens) - n + 1))
        self._debug_resources("Ngrams generation")
        return ngrams
    
    def build_all_ngram_freqs(self, max_n=None):
        """
        Build n-gram frequency tables up to order `max_n`.

        - Stores all n-gram counts in `self.ngram_freqs`.
        - Populates `self.ngram_dict`, mapping each context to a dict of
        possible next tokens with their frequencies.

        Returns
        -------
        dict
            Mapping n-gram → frequency.
        """
        if max_n is None:
            max_n = self.max_n
        # Initialize a Counter to accumulate frequencies of all n-grams
        self._debug_resources("Start building ngram freqs")
        all_freqs = Counter()

        for n_val in range(1, max_n + 1):
            # For each n (1..max_n), generate n-grams from the token sequence
            # Example: tokens = ['a','b','c'], n_val = 2 → bigrams: ('a','b'), ('b','c')
            # zip(*(self.tokens[i:] for i in range(n_val))) creates a sliding window
            ngrams = zip(*(self.tokens[i:] for i in range(n_val)))

            # Update the Counter with the frequencies of these n-grams
            # Counter handles aggregation automatically
            all_freqs.update(ngrams)

        # Convert Counter to dict and store in self.ngram_freqs
        # This is the master table of all n-grams and their absolute frequencies
        self.ngram_freqs = dict(all_freqs)

        # Populate self.ngram_dict: a nested mapping of context → next token → frequency
        # This is useful for quick lookups when generating text or calculating
        # conditional probabilities like P(token | context)
        for ngram, freq in self.ngram_freqs.items():
            # Only consider n-grams with length ≥ 2, since unigrams have no context
            if len(ngram) > 1:
                context = ngram[:-1]  # all tokens except the last
                token = ngram[-1]     # the last token, which follows the context

                # Increment the count of this token given the context
                # defaultdict(int) ensures that missing keys start at 0
                self.ngram_dict[context][token] += freq
        self._debug_resources("Populated ngram_dict")
        # Return the complete n-gram frequency dictionary
        return self.ngram_freqs


    def build_context_counts_cache(self):
        """
        Input:
            takes ngram_freqs calculated by buld_all_ngram_freqs

        Build a cache of context counts for all n-grams.

        - Stores, for each context length, the total frequency of each context.
        - This cache is useful for computing conditional probabilities
        P(token | context) efficiently.
        """
        # Initialize the context counts cache as an empty dictionary
        self.context_counts_cache = {}
        self.context_denominators = {}
        self._debug_resources("Start building context cache")

        # Iterate over all n-grams and their frequencies
        for ngram, freq in self.ngram_freqs.items():
            # For each n-gram, consider all possible prefix contexts of length 1..n-1
            for context_len in range(1, len(ngram)):
                context = ngram[:context_len]

                # Initialize a defaultdict(int)  for this context length if not present
                if context_len not in self.context_counts_cache:
                    self.context_counts_cache[context_len] = defaultdict(int)

                # Increment the total count for this context by the n-gram frequency
                # This accumulates the number of times the context appears in all n-gr
                self.context_counts_cache[context_len][context] += freq

                # Precompute denominators for Laplace smoothing
        for context_len, mapping in self.context_counts_cache.items():
            for context, count in mapping.items():
                        self.context_denominators[context] = count + self.vocab_size

        self.context_denominators[None] = len(self.tokens) + self.vocab_size
        self._debug_resources("Finished context cache")

    def get_interpolated_prob_single(self, ngram, lambdas: np.ndarray):
        """
        Compute interpolated Laplace-smoothed probability for a single n-gram.

        Uses cached context counts and the nested ngram_dict for efficient probability computation.

        Parameters
        ----------
        ngram : tuple
            The n-gram sequence of tokens.
        lambdas : np.ndarray
            Interpolation weights for each n-gram order (length up to n).

        Returns
        -------
        float
            Interpolated probability of the given n-gram.
        """
        total_prob = 0.0
        context_denominator_cache = {}  # cache denominators to avoid redundant calculations

        # Iterate over n-gram orders (1..n)
        for i in range(1, len(lambdas)+1):
            if len(ngram) >= i:
                sub_ngram = ngram[-i:]      # last i tokens
                context = sub_ngram[:-1]    # all but last token

                # Compute denominator: total count of context + vocab size
                if context:
                    if context not in context_denominator_cache:
                        count_context = self.context_counts_cache[len(context)].get(context, 0)
                        context_denominator_cache[context] = count_context + self.vocab_size
                    denominator = context_denominator_cache[context]
                else:
                    # Unigram case: total number of tokens + vocab size
                    if None not in context_denominator_cache:
                        context_denominator_cache[None] = len(self.tokens) + self.vocab_size
                    denominator = context_denominator_cache[None]

                # Count of token following the context
                count_token = self.ngram_dict[context].get(sub_ngram[-1], 0)

                # Laplace-smoothed probability for this order
                prob = (count_token + 1) / denominator

                # Weighted contribution for interpolation
                total_prob += lambdas[i-1] * prob

        return total_prob


    def get_interpolated_prob_batch(self, ngrams_list, lambdas: np.ndarray):
        """
        Compute interpolated Laplace-smoothed probabilities for a batch of n-grams.

        Parameters
        ----------
        ngrams_list : list of tuples
            Each tuple is an n-gram sequence of tokens.
        lambdas : np.ndarray
            Interpolation weights for each n-gram order.

        Returns
        -------
        np.ndarray
            Array of interpolated probabilities, same order as ngrams_list.
        """
        # Initialize array to store probabilities
        probs = np.zeros(len(ngrams_list), dtype=float)
        self._debug_resources("Start batch interpolated prob")
        # Compute probability for each n-gram individually using single n-gram method
        for idx, ngram in enumerate(ngrams_list):
            probs[idx] = self.get_interpolated_prob_single(ngram, lambdas)
        self._debug_resources("End batch interpolated prob")
        return probs


    def predict_next_token_sampling(self, context, lambdas):
        """Predicts the next token using interpolated probabilities."""
        candidates = {}
        # Find all tokens that can follow the context
        for ngram, freq in self.ngram_freqs.items():
            if len(ngram) == len(context) + 1 and ngram[:-1] == context:
                candidates[ngram] = self.get_interpolated_prob(ngram, lambdas)

        if not candidates:
            return None

        # Normalize probabilities and choose a token
        total_prob = sum(candidates.values())
        norm_probs = [p / total_prob for p in candidates.values()]
        candidate_ngrams = list(candidates.keys())
        chosen_ngram = candidate_ngrams[
            np.random.choice(len(candidate_ngrams), p=norm_probs)
        ]

        return chosen_ngram[-1]

    def generate_text(self, prompt_tokens, max_length=50):
        """Generates text from a prompt, with automatic backoff."""
        if not self.ngram_freqs:
            raise ValueError("Model is not trained. Train the model first.")

        # Use the best lambda weights found during tuning
        best_lambdas = self.lambdas.get("best", [1 / self.n] * self.n)

        generated = list(prompt_tokens)
        for _ in range(max_length):
            next_token = None
            # Backoff loop: try with n-1 context, then n-2, etc.
            for n_context in range(self.n - 1, 0, -1):
                if len(generated) >= n_context:
                    context = tuple(generated[-n_context:])
                    next_token = self.predict_next_token_sampling(context, best_lambdas)
                    if next_token:
                        break

            if next_token:
                generated.append(next_token)
            else:
                # If no context works, stop generation
                break

        return "".join(generated).replace("_", " ")
