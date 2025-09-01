# import os
# import sys
# import numpy as np
# from collections import defaultdict
#
#
# class NGram:
#     def __init__(self, tokens, n):
#         self.tokens = tokens
#         self.n = n
#         self.max_n = n
#         self.ngrams = []
#         self.ngram_freqs = {}
#         self.vocab = set(tokens)
#         self.vocab_size = len(self.vocab)
#         self.total_ngrams = 0
#         self.total_token_count = len(tokens)
#         self.lambdas = {}
#         self.interpolated_probs = {}
#
#     def create_ngrams(self, n=None):
#         """
#         Generate n-grams from a list of tokens.
#
#         Args:
#             tokens (list of str): List of tokens (e.g., output of BPE encoding).
#             n (int): Size of the n-gram (e.g., 1 for unigram, 2 for bigram, etc.).
#
#         Returns:
#             list of tuples: A list containing n-grams represented as tuples of strings.
#         """
#         if n is None:
#             n = self.n
#         ngrams = []
#         # tokens = tokens.split("_")
#         for i in range(len(self.tokens) - n + 1):
#             ngram = tuple(self.tokens[i : i + n])
#             ngrams.append(ngram)
#         self.ngrams = ngrams
#         return ngrams
#
#     def compute_freq(self, ngrams=None):
#         """
#         Count the frequency of each n-gram in the input list.
#
#         Args:
#             ngrams (list of tuples): List of n-grams (tuples of tokens).
#
#         Returns:
#             dict: Dictionary where keys are n-gram tuples and values are their frequency counts.
#         """
#         if ngrams is None:
#             ngrams = self.ngrams
#
#         ngram_freq = defaultdict(int)
#         for ngram in ngrams:
#             ngram_freq[ngram] += 1
#
#         self.ngram_freqs = dict(ngram_freq)
#         self.total_ngrams = len(self.tokens)
#         return dict(ngram_freq)
#
#     def build_all_ngram_freqs(self, max_n=None):
#         """
#         Combine create_ngrams and compute_freq to get n-gram frequencies.
#
#         Args:
#             tokens (list of str): Input token list.
#             max_n (int): Max n-gram size (e.g., 3 for unigrams to trigrams).
#
#         Returns:
#             dict: Maps n-gram tuples to their frequency count.
#         """
#         if max_n is None:
#             max_n = self.max_n
#
#         all_ngrams = []
#         for n in range(1, max_n + 1):
#             all_ngrams.extend(self.create_ngrams(n))
#         return self.compute_freq(all_ngrams)
#
#     def predict_next_token_sampling(self, context):
#         probs = self.interpolated_probs.get("best", {})
#         candidates = {}
#
#         for ngram, p in probs.items():
#             if ngram[:-1] == context:
#                 candidates[ngram[-1]] = candidates.get(ngram[-1], 0) + p
#
#         if not candidates:
#             return None
#
#         tokens, ps = zip(*candidates.items())
#         ps = np.array(ps, dtype=float)
#         ps /= ps.sum()
#         return np.random.choice(tokens, p=ps)
#
#     def laplace_smoothed_probs(self):
#         """
#         Compute conditional probabilities with Laplace (Add-one) smoothing
#
#         Args:
#             ngram_freq = dictionary with ngrams and corresponding counts
#             context_freq = dictionary with (n-1)-grams counts
#             vocab_size = total number of unique tokens
#
#         returns: dictionary with smoothed ngram probabilities
#         """
#         laplace_probs = {}
#         vocab_size = len(self.vocab)
#         for ngram, freq in self.ngram_freqs.items():
#             context = ngram[:-1]
#
#             context_count = self.ngram_freqs.get(context, 0)
#
#             smoothed_prob = (freq + 1) / (context_count + vocab_size)
#
#             laplace_probs[ngram] = smoothed_prob
#
#         return laplace_probs
#
#     def interpolate_probs_with_laplace(self, lambdas):
#         """
#         Perform interpolation of probabilities for n-grams of different lengths.
#
#         Args:
#             lambdas (dict): Dictionary of interpolation weight sets.
#                             Keys are labels (e.g., 'set1'), values are lists/tuples of weights
#                             for each n-gram order.
#                             For example, weights for unigram, bigram, trigram probabilities
#                             respectively.
#
#         Returns:
#             dict: A dictionary where each key is a label from lambdas, and each value is another dictionary.
#                 This inner dictionary maps each n-gram to its interpolated probability.
#         """
#         interpolated_results = {}
#         # Loop over sets of lambads and corresponding weights
#         for label, weights in lambdas.items():
#             probs = {}
#             # Loop over ngrams (keys in dict of ngram frequencies)
#             for ngram in self.ngram_freqs.keys():
#                 # total probability score
#                 total_prob = 0.0
#                 # Loop over each lambda in the set (starting from 1, 0_gram not helpful)
#                 for i in range(1, len(weights) + 1):
#                     # Check if ngram lenghts is greater than idx of weights.
#                     # This avoids considering ngrams of lenght smaller than the number of weights
#                     if len(ngram) >= i:
#                         # Find subgram (n-1_gram) of current ngram
#                         sub_ngram = ngram[-i:]
#                         # Find context of subgram
#                         context = sub_ngram[:-1]
#                         # get the frequencies of both from ngram_freq
#                         count_sub_ngram = self.ngram_freqs.get(sub_ngram, 0)
#                         count_context = sum(
#                             freq
#                             for key, freq in self.ngram_freqs.items()
#                             if key[:-1] == context
#                         )
#
#                         # Applying Laplace Smoothing
#                         prob = (
#                             (count_sub_ngram + 1) / (count_context + self.vocab_size)
#                             if count_context > 0
#                             else 0
#                         )
#                         # Update total probability by adding the weighted sum of lamda times its
#                         # interpolated probabiltiy
#                         # weights is a list of labdas of the correspondig ngram order
#                         total_prob += weights[i - 1] * prob
#                 # Update the dictionary with interpolated probability of the corresponding ngram in the vocabulary
#                 probs[ngram] = total_prob
#             # Add probs to the corresponding set of lambas in the dictionary
#             interpolated_results[label] = probs
#             self.interpolated_probs = interpolated_results
#         return interpolated_results
#
#     def generate_text(
#         self,
#         prompt_tokens: str,
#         max_length=30,
#         stop_token=".",
#     ):
#         """Generates a sequence of text starting from a prompt."""
#         generated_tokens = list(prompt_tokens)
#         print(f"Starting with prompt tokens: {generated_tokens}")
#
#         while len(generated_tokens) < max_length:
#             prediction = None
#
#             # This loop automatically tries to predict, backing off from n down to 1 (unigram)
#             for current_n in range(self.n, 0, -1):
#                 # Get the context needed for the current n-gram size
#                 context_len = current_n - 1
#                 if len(generated_tokens) >= context_len:
#                     context = (
#                         tuple(generated_tokens[-context_len:])
#                         if context_len > 0
#                         else ()
#                     )
#
#                     # Try to predict
#                     prediction = self.predict_next_token_sampling(context)
#
#                     if prediction is not None:
#                         # Found a successful prediction, so stop backing off
#                         break
#
#             # If no prediction was found even after trying unigrams, stop.
#             if prediction is None:
#                 print("--- Context not found, stopping generation. ---")
#                 prediction = np.random.choice(self.tokens)
#
#             # Stop if the stop token is generated
#             if prediction == stop_token:
#                 generated_tokens.append(prediction)
#                 print("--- Stop token generated. ---")
#                 break
#
#             generated_tokens.append(prediction)
#         text = " ".join(generated_tokens)
#         text = text.replace("_", " ")
#         return text


# 33
# 333
####################################################
# 3
# from collections import defaultdict
#
#
# class NGram:
#     def __init__(self, tokens, n):
#         self.tokens = tokens
#         self.n = n
#         self.max_n = n
#         self.ngrams = []
#         self.ngram_freqs = {}
#         self.vocab = set(tokens)
#         self.vocab_size = len(self.vocab)
#         self.context_counts = {}
#         self.total_token_count = len(tokens)
#
#         # --- OPTIMIZATION 1: Add a cache for context counts ---
#         self.context_counts = {}
#         # ---
#
#         self.lambdas = {}
#
#     def create_ngrams(self, n=None):
#         """
#         Generate n-grams from a list of tokens.
#         """
#         if n is None:
#             n = self.n
#         ngrams = []
#         for i in range(len(self.tokens) - n + 1):
#             ngram = tuple(self.tokens[i : i + n])
#             ngrams.append(ngram)
#         self.ngrams = ngrams
#         return ngrams
#
#     def compute_freq(self, ngrams=None):
#         """
#         Count the frequency of each n-gram in the input list.
#         """
#         if ngrams is None:
#             ngrams = self.ngrams
#
#         ngram_freq = defaultdict(int)
#         for ngram in ngrams:
#             ngram_freq[ngram] += 1
#
#         self.ngram_freqs = dict(ngram_freq)
#         return dict(ngram_freq)
#
#     def _precompute_context_counts(self):
#         """
#         --- OPTIMIZATION 1: New helper method ---
#         Calculates the frequency of all possible contexts (n-1 grams) and
#         caches them in a dictionary for fast lookups.
#         """
#         print("Pre-computing context counts for faster lookups...")
#         context_cache = defaultdict(int)
#         # We only need to iterate through n-grams that can BE contexts (size > 1)
#         for ngram, freq in self.ngram_freqs.items():
#             if len(ngram) > 1:
#                 context = ngram[:-1]
#                 context_cache[context] += freq
#         self.context_counts = dict(context_cache)
#         print(f"Cached {len(self.context_counts)} unique contexts.")
#
#     def _precompute_context_counts(self):
#         """
#         Calculates the frequency of all possible contexts (n-1 grams) and
#         caches them in a dictionary for fast lookups. This is a key optimization.
#         """
#         print("Pre-computing context counts for faster lookups...")
#         context_cache = defaultdict(int)
#         # We only need to check n-grams that can BE contexts (i.e., size > 1)
#         for ngram, freq in self.ngram_freqs.items():
#             if len(ngram) > 1:
#                 context = ngram[:-1]
#                 context_cache[context] += freq
#         self.context_counts = dict(context_cache)
#         print(f"Cached {len(self.context_counts)} unique contexts.")
#
#     def build_all_ngram_freqs(self, max_n=None):
#         """
#         Combine create_ngrams and compute_freq to get n-gram frequencies.
#         """
#         if max_n is None:
#             max_n = self.max_n
#
#         all_ngrams = []
#         for n in range(1, max_n + 1):
#             all_ngrams.extend(self.create_ngrams(n))
#
#         self.compute_freq(all_ngrams)
#
#         # --- OPTIMIZATION 1: Call the pre-computation step ---
#         self._precompute_context_counts()
#         # ---
#
#         self.compute_freq(all_ngrams)
#
#         # This is the new line that triggers the caching optimization
#         self._precompute_context_counts()
#
#         return self.ngram_freqs
#
#     # Note: Generation logic is simplified here as it's not the focus of the optimization.
#     # A fully robust generator would also use the on-the-fly probability logic.
#     def generate_text(self, prompt_tokens: list, max_length=50):
#         """Generates text from a prompt."""
#         generated = list(prompt_tokens)
#         for _ in range(max_length):
#             context_len = self.n - 1
#             if len(generated) >= context_len:
#                 context = tuple(generated[-context_len:])
#                 # Simple prediction: find most common follower
#                 candidates = {}
#                 for ngram, freq in self.ngram_freqs.items():
#                     if len(ngram) == self.n and ngram[:-1] == context:
#                         candidates[ngram[-1]] = freq
#
#                 if candidates:
#                     next_token = max(candidates, key=candidates.get)
#                     generated.append(next_token)
#                 else:
#                     break  # Stop if no known continuation
#             else:
#                 break
#
#         text = "".join(generated).replace("_", " ")
#         return text


import numpy as np
from collections import defaultdict


class NGram:
    def __init__(self, tokens, n):
        self.tokens = tokens
        self.n = n
        self.max_n = n
        self.ngram_freqs = {}
        self.vocab = set(tokens)
        self.vocab_size = len(self.vocab)
        self.lambdas = {}
        self.interpolated_probs = {}
        # Improvement: A single cache for context counts, keyed by length
        self.context_counts_cache = {}

    def create_ngrams(self, n=None):
        """Generates n-grams from a list of tokens."""
        if n is None:
            n = self.n
        # Use a generator for memory efficiency with large datasets
        return (tuple(self.tokens[i : i + n]) for i in range(len(self.tokens) - n + 1))

    def compute_freq(self, ngrams):
        """Counts the frequency of each n-gram."""
        ngram_freq = defaultdict(int)
        for ngram in ngrams:
            ngram_freq[ngram] += 1
        return ngram_freq

    def build_all_ngram_freqs(self, max_n=None):
        """Builds frequencies for all n-grams up to max_n."""
        if max_n is None:
            max_n = self.max_n

        all_freqs = defaultdict(int)
        for n_val in range(1, max_n + 1):
            for ngram in self.create_ngrams(n_val):
                all_freqs[ngram] += 1
        self.ngram_freqs = dict(all_freqs)
        return self.ngram_freqs

    def get_context_count(self, context):
        """Calculates and caches the frequency of a given context."""
        context_len = len(context)
        if context_len not in self.context_counts_cache:
            # If not cached, calculate counts for all contexts of this length
            counts = defaultdict(int)
            for ngram, freq in self.ngram_freqs.items():
                if len(ngram) == context_len + 1:
                    counts[ngram[:-1]] += freq
            self.context_counts_cache[context_len] = counts

        return self.context_counts_cache[context_len].get(context, 0)

    def get_interpolated_prob(self, ngram, weights):
        """Computes interpolated Laplace probability for a single n-gram."""
        total_prob = 0.0
        for i in range(1, len(weights) + 1):
            if len(ngram) >= i:
                sub_ngram = ngram[-i:]
                context = sub_ngram[:-1]

                count_sub_ngram = self.ngram_freqs.get(sub_ngram, 0)
                count_context = (
                    self.get_context_count(context) if context else len(self.tokens)
                )

                prob = (count_sub_ngram + 1) / (count_context + self.vocab_size)
                total_prob += weights[i - 1] * prob
        return total_prob

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
