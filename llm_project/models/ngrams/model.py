import os
import sys
import numpy as np
from collections import defaultdict

class NGram:

    def __init__(self, tokens, n):
        self.tokens = tokens
        self.n = n
        self.max_n = n
        self.ngrams = []
        self.ngram_freqs = {}
        self.vocab = set(tokens)
        self.vocab_size = len(self.vocab)
        self.total_ngrams = 0
        self.lambdas = {}
        self.interpolated_probs = {}

    def create_ngrams(self, n=None):
        """
        Generate n-grams from a list of tokens.

        Args:
            tokens (list of str): List of tokens (e.g., output of BPE encoding).
            n (int): Size of the n-gram (e.g., 1 for unigram, 2 for bigram, etc.).

        Returns:
            list of tuples: A list containing n-grams represented as tuples of strings.
        """
        if n is None:
            n = self.n
        ngrams = []
        # tokens = tokens.split("_")
        for i in range(len(self.tokens) - n + 1):
            ngram = tuple(self.tokens[i : i + n])
            ngrams.append(ngram)
        self.ngrams = ngrams
        return ngrams

    def compute_freq(self, ngrams=None):
        """
        Count the frequency of each n-gram in the input list.

        Args:
            ngrams (list of tuples): List of n-grams (tuples of tokens).

        Returns:
            dict: Dictionary where keys are n-gram tuples and values are their frequency counts.
        """
        if ngrams is None:
            ngrams = self.ngrams

        ngram_freq = defaultdict(int)
        for ngram in ngrams:
            ngram_freq[ngram] += 1
        
        self.ngram_freqs = dict(ngram_freq)
        self.total_ngrams = sum(ngram_freq.values())
        return dict(ngram_freq)


    def build_all_ngram_freqs(self, max_n=None):
        """
        Combine create_ngrams and compute_freq to get n-gram frequencies.

        Args:
            tokens (list of str): Input token list.
            max_n (int): Max n-gram size (e.g., 3 for unigrams to trigrams).

        Returns:
            dict: Maps n-gram tuples to their frequency count.
        """
        if max_n is None:
            max_n = self.max_n

        all_ngrams = []
        for n in range(1, max_n + 1):
            all_ngrams.extend(self.create_ngrams(n))
        return self.compute_freq(all_ngrams)

    def predict_next_token_sampling(self, context):
        probs = self.interpolated_probs.get("best", {})
        candidates = {}

        for ngram, p in probs.items():
            if ngram[:-1] == context:
                candidates[ngram[-1]] = candidates.get(ngram[-1], 0) + p

        if not candidates:
            return None

        tokens, ps = zip(*candidates.items())
        ps = np.array(ps, dtype=float)
        ps /= ps.sum() 
        return np.random.choice(tokens, p=ps)


    def laplace_smoothed_probs(self):
        """
        Compute conditional probabilities with Laplace (Add-one) smoothing

        Args:
            ngram_freq = dictionary with ngrams and corresponding counts
            context_freq = dictionary with (n-1)-grams counts
            vocab_size = total number of unique tokens

        returns: dictionary with smoothed ngram probabilities
        """
        laplace_probs = {}
        vocab_size = len(self.vocab)
        for ngram, freq in self.ngram_freqs.items():
            context = ngram[:-1]

            context_count = sum(freq for key, freq in self.ngram_freqs.items()
                                if key[:-1] == context)

            smoothed_prob = (freq + 1) / (context_count + vocab_size)

            laplace_probs[ngram] = smoothed_prob

        return laplace_probs

    def interpolate_probs_with_laplace(self, lambdas):
        """
        Perform interpolation of probabilities for n-grams of different lengths.

        Args:
            lambdas (dict): Dictionary of interpolation weight sets.
                            Keys are labels (e.g., 'set1'), values are lists/tuples of weights
                            for each n-gram order.
                            For example, weights for unigram, bigram, trigram probabilities
                            respectively.

        Returns:
            dict: A dictionary where each key is a label from lambdas, and each value is another dictionary.
                This inner dictionary maps each n-gram to its interpolated probability.
        """
        interpolated_results = {}
        # Loop over sets of lambads and corresponding weights
        for label, weights in lambdas.items():
            probs = {}
            # Loop over ngrams (keys in dict of ngram frequencies)
            for ngram in self.ngram_freqs.keys():
                # total probability score
                total_prob = 0.0
                # Loop over each lambda in the set (starting from 1, 0_gram not helpful)
                for i in range(1, len(weights) + 1):
                    # Check if ngram lenghts is greater than idx of weights.
                    # This avoids considering ngrams of lenght smaller than the number of weights
                    if len(ngram) >= i:
                        # Find subgram (n-1_gram) of current ngram
                        sub_ngram = ngram[-i:]
                        # Find context of subgram
                        context = sub_ngram[:-1]
                        # get the frequencies of both from ngram_freq
                        count_sub_ngram = self.ngram_freqs.get(sub_ngram, 0)
                        count_context = sum(freq for key, freq in self.ngram_freqs.items()
                                if key[:-1] == context)

                        # Applying Laplace Smoothing
                        prob = (
                            (count_sub_ngram + 1) / (count_context + self.vocab_size)
                            if count_context > 0
                            else 0
                        )
                        # Update total probability by adding the weighted sum of lamda times its
                        # interpolated probabiltiy
                        # weights is a list of labdas of the correspondig ngram order
                        total_prob += weights[i - 1] * prob
                # Update the dictionary with interpolated probability of the corresponding ngram in the vocabulary
                probs[ngram] = total_prob
            # Add probs to the corresponding set of lambas in the dictionary
            interpolated_results[label] = probs
            self.interpolated_probs = interpolated_results
        return interpolated_results

    def generate_text(self,
        prompt_tokens: str,
        max_length=30,
        stop_token=".",
        ):
        """Generates a sequence of text starting from a prompt."""
        generated_tokens = list(prompt_tokens)
        print(f"Starting with prompt tokens: {generated_tokens}")

        while len(generated_tokens) < max_length:
            prediction = None

            # This loop automatically tries to predict, backing off from n down to 1 (unigram)
            for current_n in range(self.n, 0, -1):
                # Get the context needed for the current n-gram size
                context_len = current_n - 1
                if len(generated_tokens) >= context_len:
                    context = (
                        tuple(generated_tokens[-context_len:]) if context_len > 0 else ()
                    )

                    # Try to predict
                    prediction = self.predict_next_token_sampling(context)

                    if prediction is not None:
                        # Found a successful prediction, so stop backing off
                        break

            # If no prediction was found even after trying unigrams, stop.
            if prediction is None:
                print("--- Context not found, stopping generation. ---")
                prediction = np.random.choice(self.tokens)


            # Stop if the stop token is generated
            if prediction == stop_token:
                generated_tokens.append(prediction)
                print("--- Stop token generated. ---")
                break

            generated_tokens.append(prediction)
        text = " ".join(generated_tokens)
        text = text.replace("_", " ") 
        return text