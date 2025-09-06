import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from llm_project.utils.dataloader import load_shakespeare

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


def get_batch(data_ids, block_size, batch_size):
    """
    Creates a batch of input sequences and corresponding target sequences
    from a list of token IDs.

    Args:
        data_ids (list of int): sequence of integer token IDs.
        block_size (int): length of each input sequence (context window).
        batch_size (int): number of sequences in the batch.

    Returns:
        X (np.ndarray): array of shape (batch_size, block_size) containing input sequences.
        y (np.ndarray): array of shape (batch_size, block_size) containing target sequences
                        (inputs shifted by one token).
    """
    X, y = [], []
    for _ in range(batch_size):
        start_idx = np.random.randint(0, len(data_ids) - block_size - 1)
        x_block = data_ids[start_idx : start_idx + block_size]
        y_block = data_ids[start_idx + 1 : start_idx + block_size + 1]

        X.append(x_block)
        y.append(y_block)

    return np.array(X), np.array(y)


class NeuralEmbed:
    def __init__(self, n, vocab_size, embd_dim=32, seed=35):
        np.random.seed(seed)
        self.n = n
        self.vocab_size = vocab_size
        self.embd_dim = embd_dim
        # Initialize embedding matrix
        self.embeddings = (
            np.random.randn(vocab_size, embd_dim) * 0.01
        )  # (vocab_size, embedding_dim)
        # Initialize weights and biases
        self.W = np.random.randn(embd_dim, vocab_size)
        self.b = np.zeros((1, 1, vocab_size))

    # -------------- FORWARD / LOSS -----------------

    def forward(self, X):
        """
        Forward pass

        Args:
            X (np.ndarray): batches of id sequences, shape (batch_size, block_size).

        Returns:
        logits (np.ndarray): output logits per token, shape (batch_size, block_size, vocab_size)
        """
        # X: (batch_size, block_size)
        # B = batch_size
        # L = block_size
        # E = embedding_dim
        # C = vocab_size
        batch_embeddings = self.embeddings[X]  # (B, L, E)
        logits = batch_embeddings @ self.W + self.b  # (B, L, C)
        return logits

    def cross_entropy_loss(self, logits, y):
        """
        Computes the cross-entropy loss and softmax probabilities given logits and target indices.

        Args:
            logits (np.ndarray): array of shape (batch_size, block_size, vocab_size), raw output scores from the model.
            y (np.ndarray): array of shape (batch_size, block_size), target token indices for each sequence.

        Returns:
            loss (float): mean cross-entropy loss over the batch.
            probs (np.ndarray): array of shape (batch_size, block_size, vocab_size), softmax probabilities
            corresponding to logits.
        """
        B, L, _ = logits.shape
        # Substract the highest value from each logit of each sequence
        logits = logits.astype(np.float32)
        exps = np.exp(logits - np.max(logits, axis=2, keepdims=True))
        exps_sum = np.sum(exps, axis=2, keepdims=True)
        # applying softmax, (batch_size, block_size, vocab_size)
        probs = exps / exps_sum
        # extracting indexes of correct tokens for each position in each sequence
        # (batch_size, block_size)
        true_prob = probs[np.arange(B)[:, None], np.arange(L), y]
        # cross-entropy loss: L = -1/N * Σ log(p(y_i))
        loss = -np.mean(np.log(true_prob + 1e-9))
        return loss, probs

    # ---------------- BACKWARD ------------------
    def backward(self, X_batch, y_batch, probs, lr=0.01):
        """
        Performs a backward pass for a batch: computes gradients of the cross-entropy loss
        w.r.t. the model's weights and updates them using SGD.

        Args:
            X (np.ndarray): input batch of shape (batch_size, block_size), containing token ids.
            y (np.ndarray): target batch of shape (batch_size, block_size), containing token ids.
            probs (np.ndarray): softmax probabilities from forward pass, shape (batch_size, block_size, vocab_size).
            learning_rate (float): step size for gradient descent.

        Returns:
            None. Updates model weights in place.
        """
        # B = batch_size
        # L = block_size
        # E = embedding_dim
        # C = vocab_size
        B, L, C = probs.shape
        E = self.embd_dim

        # gradient of cross entropy and softmax
        dZ = probs.copy()  # (B, L, C)
        dZ[np.arange(B)[:, None], np.arange(L), y_batch] -= 1
        dZ /= B * L

        # gradient of weights and biases
        batch_embeddings = self.embeddings[X_batch]  # (B, L, E)
        batch_flat = batch_embeddings.reshape(B * L, E)  # (B*L, E)
        dZ_flat = dZ.reshape(B * L, C)  # (B*L, C)

        dW = batch_flat.T @ dZ_flat  # (E, C)
        db = np.sum(dZ_flat, axis=0, keepdims=True).reshape(1, 1, C)  # (1,1,C)

        # gradient of embeddings
        dE_flat = dZ_flat @ self.W.T  # (B*L, E)
        dE = dE_flat.reshape(B, L, E)  # (B, L, E)

        # update weights
        self.W -= lr * dW
        self.b -= lr * db

        # update embeddings (vectorized, safer)
        np.add.at(self.embeddings, X_batch, -lr * dE)

    #  ---------------- PLOTTING LOSS CURVE -------------------

    def plot_loss(self, train_losses, val_losses=None):
        plt.figure(figsize=(8, 6))
        plt.plot(train_losses, label="Train loss", color="blue")

        if val_losses:
            steps_per_epoch = len(train_losses) // len(val_losses)
            x_vals = [i * steps_per_epoch for i in range(len(val_losses))]

            plt.plot(x_vals, val_losses, label="Validation loss", color="red")

        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

    # ---------------- GENERATION -------------------
    def predict_next_token_sampling(self, logits, top_k=None, top_p=None, temperature=1.0, unk_id=None):
        """
        Sample the next token from logits using temperature scaling, top-k, and top-p (nucleus) sampling.

        Args:
            logits (np.ndarray): 1D array of raw model logits for all tokens.
            top_k (int, optional): Keep only top_k most probable tokens for sampling.
            top_p (float, optional): Keep the minimal set of tokens with cumulative probability >= top_p.
            temperature (float): Temperature for scaling logits. Higher values increase randomness.

        Returns:
            int: Index of the sampled token.
        """
        # --- Apply temperature scaling ---
        logits = logits / temperature

        # --- Convert logits to probabilities using softmax ---
        exps = np.exp(logits - np.max(logits))  # subtract max for numerical stability
        probs = exps / np.sum(exps)             # softmax probabilities

        # --- Top-K sampling ---
        if top_k is not None:
            # Find indices of the top_k highest probability tokens
            top_k_ids = np.argsort(probs)[-top_k:]
            top_k_probs = probs[top_k_ids]
            # Normalize the probabilities of top_k tokens
            top_k_probs /= np.sum(top_k_probs)
            # Zero out all other token probabilities
            probs_filtered = np.zeros(probs.shape)
            probs_filtered[top_k_ids] = top_k_probs
            probs = probs_filtered  # update probs to filtered top_k

        # --- Top-P (nucleus) sampling ---
        if top_p is not None:
            # Sort token indices in descending order of probability
            sorted_ids = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_ids]
            # Compute cumulative probabilities
            cumulative_probs = np.cumsum(sorted_probs)
            # Find the cutoff index where cumulative probability exceeds top_p
            cutoff = np.searchsorted(cumulative_probs, top_p, side="right") + 1
            # Keep only the nucleus tokens
            nucleus_ids = sorted_ids[:cutoff]
            nucleus_probs = sorted_probs[:cutoff]
            # Normalize probabilities of nucleus tokens
            nucleus_probs /= np.sum(nucleus_probs)
            # Zero out all other probabilities
            probs_filtered = np.zeros(probs.shape)
            probs_filtered[nucleus_ids] = nucleus_probs
            probs = probs_filtered

        # --- Filter UNK ---
        if unk_id is not None:
            probs[unk_id] = 0.0
            probs /= np.sum(probs)  # renormalize

        # --- Sample the next token from the filtered probabilities ---
        next_id = int(np.random.choice(len(probs), p=probs))

        return next_id

    def generate(
        self,
        start_ids,
        id2token=None,
        max_new_tokens=20,
        top_k=None,
        top_p=None,
        temperature=1.0,
        stop_ids=None,
        stop_words=None,
        block_size=None,
        unk_id = None
    ):
        """
        Generate a sequence of tokens starting from start_ids using top-k / top-p sampling.

        Args:
            start_ids (list of int): Initial token IDs to start generation.
            id2token (dict, optional): Mapping from IDs to tokens for decoding.
            max_new_tokens (int): Maximum number of tokens to generate.
            top_k (int, optional): Top-k sampling parameter.
            top_p (float, optional): Top-p (nucleus) sampling parameter.
            temperature (float): Temperature for sampling.
            stop_ids (set, optional): Token IDs which stop generation if produced.
            stop_words (set, optional): Token strings which stop generation if produced.
            block_size (int, optional): Context window size.

        Returns:
            tuple: (generated_ids, generated_tokens, generated_text)
        """
        if start_ids is None or len(start_ids) == 0:
            raise ValueError("start_ids must be provided and non-empty.")

        generated_ids = list(start_ids.copy())
        stop_ids = stop_ids or set()
        stop_words = stop_words or set()

        for _ in range(max_new_tokens):
            # Use last 'block_size' tokens as context
            context = generated_ids[-block_size:] if block_size is not None else generated_ids
            X = np.array(context, dtype=np.int64)[None, :]  # shape: (1, block_size)
            logits = self.forward(X)[0, -1]  # take logits of last token 

            # Sample next token using top-k / top-p / temperature
            next_id = self.predict_next_token_sampling(
                logits,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                unk_id=unk_id
            )

            generated_ids.append(next_id)

            # Stop if next token is in stop_ids or stop_words
            if next_id in stop_ids:
                break
            if id2token is not None and id2token.get(next_id) in stop_words:
                break

        # Convert IDs to tokens and string if id2token is provided
        if id2token is not None:
            generated_tokens = [id2token.get(i, "UNK") for i in generated_ids]
            generated_text = " ".join(generated_tokens)
            return generated_ids, generated_tokens, generated_text

        return generated_ids

if __name__ == "__main__":
    from llm_project.utils.token_mapping import token_id_mapping
    from llm_project.bpe.bytepair_encoding import BPE

    # ---------------- CONFIG -----------------
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    )
    saved_models_dir = os.path.join(
        project_root, "experiments", "saved_models", "neural_ngram"
    )
    os.makedirs(saved_models_dir, exist_ok=True)
    train_path = os.path.join(
        project_root, "data", "processed", "Shakespeare_clean_train.txt"
    )
    val_path = os.path.join(
        project_root, "data", "processed", "Shakespeare_clean_valid.txt"
    )

    # N-gram model params
    n = 3
    block_size = 4
    batch_size = 32
    epochs = 5
    lr = 0.01
    max_k = 2000
    checkpoint_dir = os.path.join(saved_models_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ---------------- LOAD AND NORMALIZE TEXT -----------------
    bpe = BPE(datapath=train_path, max_k=max_k)
    t_train = bpe.load_and_normalize()
    bpe.text = t_train[:10000]

    t_valid = load_shakespeare("validation")
    bpe.test_text = t_valid[:1000]
    print(f"Train size (chars): {len(t_train)}, Test size (chars): {len(t_valid)}")

    # ---------------- TRAIN BPE -----------------
    bpe.BPE_encoder()
    print(f"DEBUG: Number of tokens after BPE: {len(bpe.tokens)}")
    # bpe.plot_vocabulary_growth()

    # ---------------- CONVERT TEXT TO IDS -----------------
    # Build simple mapping token <-> id
    token2id, id2token, _ = token_id_mapping(bpe.tokens)

    # Encode training and validation text
    train_ids = [token2id[tok] for tok in bpe.tokens if tok in token2id]
    test_tokens = bpe.BPE_segmenter(bpe.test_text)
    val_ids = [token2id[tok] for tok in test_tokens if tok in token2id]
    print(
        f"DEBUG: Length of train_ids: {len(train_ids)}, Length of val_ids: {len(val_ids)}"
    )
    print(f"DEBUG: Sample of train_ids: {train_ids[:20]}")

    # ---------------- INITIALIZE MODEL -----------------
    model = NeuralEmbed(
        n=n, vocab_size=len(token2id), block_size=block_size, batch_size=batch_size
    )
    print(f"DEBUG: Model initialized with vocab size {len(token2id)}")
    # ---------------- TRAIN MODEL -----------------
    print("DEBUG: Starting training...")
    losses, val_losses = model.fit(
        data_ids=train_ids,
        val_ids=val_ids,
        epochs=epochs,
        lr=lr,
        checkpoint_dir=checkpoint_dir,
        print_every=50,
        max_checkpoints=3,
        patience=3,
        load_ckpt_name=None,
        force_train=True,
    )
    print("DEBUG: Training finished")
    import glob

    ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pkl")))
    if ckpt_files:
        last_ckpt = os.path.basename(ckpt_files[-1])
        losses, val_losses = model.load_checkpoint(checkpoint_dir, last_ckpt)
    model.plot_loss(losses, val_losses)
    print("Training completed!")

    # ---------------- GENERATE SAMPLE -----------------
    start_ids = train_ids[:block_size]
    stop_ids_set = {token2id[tok] for tok in ".?!" if tok in token2id}
    generated_ids, generated_tokens = model.generate(
        start_ids=start_ids, max_new_tokens=20, id2token=id2token, stop_ids=stop_ids_set
    )
    print("Generated token IDs:", generated_ids)
    print("Generated tokens:", generated_tokens)
