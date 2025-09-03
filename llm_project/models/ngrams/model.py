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
