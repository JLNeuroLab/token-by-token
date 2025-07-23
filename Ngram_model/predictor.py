
from Ngram_model.ngram_model import *

def predict_next_word(words: list, freq_dict, n):
    ngram_pred = defaultdict(int)
    if len(words) == n - 1:
        for ngram, frequency in freq_dict.items():
            if list(ngram[:-1]) == words:
                ngram_pred[ngram] = frequency
        if ngram_pred:
            best_ngram = max(ngram_pred, key=ngram_pred.get)
            prediction = best_ngram[-1]
        else:
            best_ngram = max(freq_dict, key=freq_dict.get)
            prediction = best_ngram[-1]
            print("No ngram found to match the word, fallback to most frequent ngram's final word")
    else:
        raise ValueError("The word provided does not match the ngram length")
    return prediction

