# Let's code a simple Byte Pair Encoding (BPE) implementation.
from collections import defaultdict 
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

# ------------------------Method to automatically save results------------------------------
def save_item(item, folder = str , name = str, text_version=True, base_dir = None):
    """
    Save different type of files and creates an according folder if it does not exist
    
        Args:
            item: type of file (e.g., string, list, dictionary, matplotlib figure)
            folder: name of the folder in which the file is saved in the form of string
            name: name that we assign to the file
            text_version: parameter that saves some files in a reading format if set to true
            base_dir: takes the base directory of the current file
    
    """
    if base_dir is None:
        base_dir = os.getcwd()
    folder = os.path.join(base_dir, folder)
    os.makedirs(folder, exist_ok=True)
    output_file = os.path.join(folder, name)
    if isinstance(item, str):
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(item)
        print(f"text normalized and saved: {os.path.basename(folder)}/{name}")

    elif isinstance(item, plt.Figure):
        item.savefig(output_file)
        print(f"plot saved in: {os.path.basename(folder)}/{name}")

    elif isinstance(item, (list, tuple, set, dict)):
        with open(output_file, "wb") as f:
            pickle.dump(item, f)
        print(f"data saved in: {os.path.basename(folder)}/{name}")

        if text_version is True:
            text_file = os.path.splitext(output_file)[0] + ".txt"
            with open(text_file, "w", encoding="utf-8") as f:

                if isinstance(item, dict):
                 
                    for k, v in item.items():
                        f.write(f"{k}\t{v}\n")
                else:
                    for elem in item:
                        if isinstance(elem, tuple):
                            
                            f.write("\t".join(map(str, elem)) + "\n")
                        else:
                            f.write(str(elem) + "\n")
                            
            print(f"data saved in readable text: {os.path.basename(folder)}/{os.path.basename(text_file)}")
    else:
        raise TypeError(f"Unsupported type: {type(item)}")
        

def get_vocab(text):
    vocab = defaultdict(int)
    tokens = text
    for token in tokens:
        vocab[token] += 1
    return dict(vocab)  

# Split train and test method
def split_train_test(text, test_ratio=0.1):
    split_point = int(len(text) * test_ratio)
    # Returns test, train
    return text[:split_point], text[split_point:]

# ----------------Normalize the text: substitute wide spaces with single space, remove every character
# except for letters, numbers and ', turn the text in lowercase
def load_and_normalize(datapath=None) -> str:
    """
        Loading and normalizing the input text

    """
    if datapath is None:
        raise ValueError("You must specify a datapath when importing this module")
    with open(datapath, "r", encoding="utf-8") as f:
        text = f.read()
    # lower case text
    text = text.lower()
    # compress spaces into single spaces
    text = re.sub(r"\s+", "_", text.strip())
    # remove everything except for characters, spaces and '
    text = re.sub(r"[^\w\s']", "", text)
    # save text
    # save_item(text, "normalized_text", "normalized_shakespeare.txt")
    return text

# -------------------Byte-Pair Encoding method-----------------------
def BPE_encoder(text, max_k):
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
    tokens = list(text)
    # Initialize list for storing the vocabulary size at each step of the training
    vocab_size_history = []
    # Initialize list for storing the merges done at each step (pair, new token)
    bpe_merges = []
    # Training loop over k
    for step in range(max_k):
        # Initialize default dictionary
        frequencies = defaultdict(int)
        # Iteration over all the tokens in the text, we exclude the last one 
        for i in range(len(tokens) - 1):
            # A pair is defined by adjecent tokens in the text
            pair = (tokens[i], tokens[i + 1])
            # Update frequencies dictionary every time is encountered in the text
            frequencies[pair] += 1
        # If the dictionary is empty we do not merge anymore
        if not frequencies:
            print(f"No more pairs to merge at step {step}. Stopping early.")
            break
        # Select the most frequent pair 
        most_freq = max(frequencies, key=frequencies.get)
        # Merge the pair into a new token
        new_token = "".join(most_freq)
        # Append the merge at the merges list
        bpe_merges.append((most_freq, new_token))
        
        # -------Segmenter-------
        # Now we rebuild the text using the new token
        m = 0
        new_text = []
        # While loop until we reach the end of the text
        while m < len(tokens):
            # If the current token is not the last one and the pair matches the most frequent pair
            # we substitute the pair with the new token and we add it to the new text
            if m < len(tokens) - 1 and (tokens[m], tokens[m+1]) == most_freq:
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
        vocab = get_vocab(tokens)
        # Add the size of the vocabulary to the vocabulary history
        vocab_size_history.append(len(vocab))
        #print(f"step {step}: merged {most_freq} in {new_token}")
    # Obtain the final vocabulary of the tokenized text
    final_vocab = get_vocab(tokens)
    # Transform the list of tokens into a string of tokens separeted by spaces
    joined_tokens = " ".join(tokens)
    # Saving all the results
    save_item(joined_tokens, "train_results", f"train_tokenized_k{max_k}.txt")
    save_item(vocab_size_history, "train_results", "train_vocab_history.pkl", text_version=False)
    save_item(final_vocab, "train_results", "train_final_vocab.pkl")
    save_item(bpe_merges, "train_results", "train_bpe_merges.pkl")

    return tokens, vocab_size_history, final_vocab, bpe_merges

def plot_vocabulary_growth(vocab_size: int, max_k: int):
    """
        Visualize the growth of the vocabulary against the number of merges performed by BPE.

        Args:
            vocab_size: number of types in the vocabulary
            max_k: maximum number of merges
    
    """
    if vocab_size is not None:
        ks_for_plt = np.linspace(10, max_k, 100, dtype=int)
        vocab_plot = [vocab_size[k - 1] if k - 1 < len(vocab_size) else vocab_size[-1] for k in ks_for_plt]
        
        fig = plt.figure()
        plt.plot(ks_for_plt, vocab_plot, label="vocabulary size")

        max_index = np.argmax(vocab_plot)
        max_k_value = ks_for_plt[max_index]
        max_vocab_value = vocab_plot[max_index]
        plt.axvline(x=max_k_value, color='red', linestyle='--', label=f'Max vocab at k={max_k_value}')
        plt.axhline(y=max_vocab_value, color='gray', linestyle=':', alpha=0.7)

        plt.xlabel("number of merges")
        plt.ylabel("vocabulary size")
        plt.title("vocabulary size over number of merges")
        plt.legend()
        plt.tight_layout()
        save_item(fig, "plots", "vocabulary_growth")
    else:
        print("No vocabulary size data available to plot.")

# ------------------Tokenize test text with the training tokens--------------------------
def BPE_segmenter(text, merges):
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
    tokens = list(text)
    # We retrieve the merges of the training by retrieving the merges history
    for (pair, new_token) in merges:
        i = 0
        new_tokens = []
        while i < len(tokens):
            # if the pair in the text matches the pair in the merges history
            # we append it to the new test text
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        # Same procedure as the second loop of the BPE method
        tokens = new_tokens
    test_vocab = get_vocab(tokens)
    save_item(" ".join(tokens), "test_results", "test_tokenized.txt")
    save_item(test_vocab, "test_results", "test_final_vocab.pkl")
    return tokens

# After applying the tokenizer to the test text, evaluate the performance by computing coverage:
# coverage is defined by the number of tokens in the text, that the learned vocabulary from the training 
# can cover.
def compute_coverage(test_text, vocab, merges):
    test_tokens = BPE_segmenter(test_text, merges)
    # We sum the number of tokens that are matched both in the vocabulary and the test text
    covered = sum(token in vocab for token in test_tokens)
    # Total number of tokens in the test text
    total = len(test_tokens)
    # fraction of covered tokens over total number of tokens
    coverage = covered / total
    return coverage
    
# -------------------Training----------------------
if __name__ == "__main__":
    # Obtain normalized text
    # base_dir = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))                     # -> /project_root
    datapath = os.path.join(project_root, "data", "Shakespeare_clean_full.txt")
    text_norm = load_and_normalize(datapath)
    # Split between train and test sets
    t_test, t_train = split_train_test(text_norm)
    t_train = t_train[:10000]
    t_test = t_test[:1000]
    print(f"test size (char):{len(t_test)}\ntrain size (char):{len(t_train)}")
    # set k value
    max_k = 2000
    # Train BPE on training set
    print("\nstarting Byte-Pair Encoding on train text...\n")
    tokens, vocab_size, final_vocab, merges = BPE_encoder(t_train, max_k=max_k)
    print("\nloop completed, train text tokenized\n")

    plot_vocabulary_growth(vocab_size, max_k)

    print("\nTokenizing test text...\n")
    coverage = compute_coverage(t_test, final_vocab, merges)
    print("\nTokenization of test text completed\n")
    print("-" * 80)
    print(f"[RESULTS]:\nCoverage score for k = {max_k}: {coverage:.4f}\n")

#sorted_vocab = dict(sorted(vocab_size.items(), key= lambda item: item[1], reverse=True))