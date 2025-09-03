import numpy as np
from llm_project.models.ngrams.model import NGram
from llm_project.bpe.bytepair_encoding import BPE  

def main():
    print("===== Mini NGram Trainer with BPE-Tokenized Shakespeare =====")

    # -------------------------------
    # 1. Load raw dataset
    # -------------------------------
    with open("data/processed/Shakespeare_clean_train.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"Loaded raw Shakespeare text, {len(raw_text)} chars")

    # -------------------------------
    # 2. Train BPE tokenizer
    # -------------------------------
    max_k = 2000
    bpe = BPE(max_k=max_k, text=raw_text)
    norm_text = bpe.load_and_normalize()
    test_text, train_text = bpe.split_train_test(test_ratio=0.1)
    bpe.text = train_text[:10000]  # limit for speed
    bpe.BPE_encoder()

    # -------------------------------
    # 3. Tokenize train and test text
    # -------------------------------
    train_tokens = bpe.tokens  # already BPE-tokenized
    test_tokens = bpe.BPE_segmenter(test_text[:1000])

    print(f"Train tokens: {len(train_tokens)}, Test tokens: {len(test_tokens)}")

    # -------------------------------
    # 4. Initialize NGram model
    # -------------------------------
    n = 3
    model = NGram(train_tokens, n, enable_debug=False)
    lambdas = np.array([0.2, 0.3, 0.5])

    # -------------------------------
    # 5. Build n-gram frequencies
    # -------------------------------
    model.build_all_ngram_freqs(max_n=n)
    print(f"Sample n-gram frequencies:")
    for i, (ng, freq) in enumerate(list(model.ngram_freqs.items())[:10]):
        print(f"{ng}: {freq}")

    # -------------------------------
    # 6. Build context counts cache
    # -------------------------------
    model.build_context_counts_cache()
    print("Context counts cache (sample):")
    for length, ctx_map in model.context_counts_cache.items():
        print(f"Length {length}, sample 5 contexts: {dict(list(ctx_map.items())[:5])}")

    # -------------------------------
    # 7. Compute interpolated probabilities
    # -------------------------------
    ngrams_list = list(model.create_ngrams(n))
    probs = model.get_interpolated_prob_batch(ngrams_list, lambdas)
    print("Interpolated probabilities (first 10 n-grams):")
    for ng, p in zip(ngrams_list[:10], probs[:10]):
        print(f"{ng}: {p:.6f}")

    print("===== Mini Training Complete =====")

if __name__ == "__main__":
    main()
