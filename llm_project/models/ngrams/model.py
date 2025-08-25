import os
import sys
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
        for i in range(len(self.tokens) - self.n + 1):
            ngram = tuple(self.tokens[i : i + self.n])
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


    def build_all_ngram_freqs(self, max_n):
        """
        Combine create_ngrams and compute_freq to get n-gram frequencies.

        Args:
            tokens (list of str): Input token list.
            max_n (int): Max n-gram size (e.g., 3 for unigrams to trigrams).

        Returns:
            dict: Maps n-gram tuples to their frequency count.
        """
        all_ngrams = []
        for n in range(1, max_n + 1):
            all_ngrams.extend(self.create_ngrams(n))
        self.max_n = max_n
        return self.compute_freq(all_ngrams)


    def predict(self, context):
        """
        Predict next word from an input list of words

        Args:
            words = list of words or subwords
            freq_dict = dictionary of counts returned from compute_freq
            n = length of ngrams

            returns most probable word
        """
        ngram_pred = {}
        if len(context) == self.n - 1:
            for ngram, frequency in self.ngram_freqs.items():
                if list(ngram[:-1]) == context:
                    ngram_pred[ngram] = frequency
            if ngram_pred:
                best_ngram = max(ngram_pred, key=ngram_pred.get)
                prediction = best_ngram[-1]
            else:
                best_ngram = max(self.ngram_freqs, key=self.ngram_freqs.get)
                prediction = best_ngram[-1]
                print(
                    "No ngram found to match the word, fallback to most frequent ngram's final word"
                )
        else:
            raise ValueError("The word provided does not match the ngram length")
        return prediction


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
                            (count_sub_ngram + 1) / (count_context + vocab_size)
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

        return interpolated_results


    def fit(self, tokens, max_n, lambdas):
        """
        Train an n-gram language model with Laplace interpolation.

        Combines `build_all_ngram_freqs` and `interpolate_probs_with_laplace`
        to compute smoothed n-gram probabilities.

        Args:
            tokens (list of str): Input token list.
            max_n (int): Maximum n-gram size.
            lambdas (list of float): Interpolation weights for each n-gram order.

        Returns:
            dict: Mapping of n-grams to interpolated probabilities.
        """
        ngram_freqs = self.build_all_ngram_freqs(tokens, self.max_n)
        vocab_size = len(set(tokens))
        interpolated_probs = self.interpolate_probs_with_laplace(
            ngram_freqs, lambdas, vocab_size
        )
        return interpolated_probs


if __name__ == "__main__":
    # text_path = os.path.join(module_path, "data", "Shakespeare_clean_full.txt")
    # norm_text = load_and_normalize(text_path)
    # tokens, vocab_history, vocab, bpe_merges = BPE_encoder(norm_text, 10)
    tokens = ["the", "cat", "sat", "on", "the", "mat"]
    max_n = 3
    n_grams = create_ngrams(tokens, max_n)

    ngram_freq = build_all_ngram_freqs(tokens, max_n)

    # 3. Lambda per interpolazione (somma 1)
    lambdas = {"example": [0.1, 0.3, 0.6]}

    # 4. Calcola vocab_size
    vocab = set(tokens)
    vocab_size = len(vocab)

    # 5. Usa funzione con Laplace smoothing integrato
    interpolated_probs = interpolate_probs_with_laplace(ngram_freq, lambdas, vocab_size)

    # 6. Stampa alcune probabilità di prova
    print("Interpolated probabilities for 'the cat sat':")
    test_ngram = ("the", "cat", "sat")
    print(interpolated_probs["example"].get(test_ngram, "N-gram not found"))

    # 7. Controllo rapido (esempio)
    for ngram, prob in interpolated_probs["example"].items():
        assert 0 <= prob <= 1, f"Probability out of range for {ngram}: {prob}"

    print("Test passed: all probabilities between 0 and 1")
