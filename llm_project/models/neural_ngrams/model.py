from llm_project.utils.file_manager import save_item, load_item
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


class NeuralNgram:
    def __init__(self, n, vocab_size, embedding_dim=32, seed=35):
        np.random.seed(seed)
        self.n = n
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # Initialize embedding matrix
        self.embeddings = (
            np.random.randn(vocab_size, embedding_dim) * 0.01
        )  # (vocab_size, embedding_dim)
        # Initialize weights and biases
        self.W = np.random.randn(embedding_dim, vocab_size)
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
        E = self.embedding_dim

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

    # ------------- MODEL STATES ------------------
    def state_dict(self):
        """
        Returns model parameters as a dictionary
        """
        return {"embeddings": self.embeddings, "W": self.W, "b": self.b}

    def load_state_dict(self, state):
        """
        Loads parameters from dictionary
        """
        self.embeddings = state["embeddings"]
        self.W = state["W"]
        self.b = state["b"]

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

    def generate(
        self,
        start_ids,
        id2token=None,
        max_new_tokens=20,
        stochastic=True,
        stop_ids=None,
        stop_words=None,
    ):
        generated_ids = list(start_ids.copy())

        stop_ids = stop_ids or set()
        stop_words = stop_words or set()

        for _ in range(max_new_tokens):
            context = generated_ids[-self.block_size :]
            X = np.array(context, dtype=np.int64)[None, :]
            # generate logits
            # Pick the last token of the first and only example
            logits = self.forward(X)[0, -1]

            # Numerical stability
            logits = logits - np.max(logits)

            exps = np.exp(logits)
            probs = exps / exps.sum()
            print("DEBUG generate:", len(probs), self.vocab_size)

            assert len(probs) == self.vocab_size, f"{len(probs)} vs {self.vocab_size}"
            if stochastic:
                next_id = int(np.random.choice(self.vocab_size, p=probs))
            else:
                next_id = int(np.argmax(probs))

            generated_ids.append(next_id)

            if next_id in stop_ids:
                break
            if id2token is not None and id2token[next_id] in stop_words:
                break

        if id2token is not None:
            generated_tokens = [id2token[i] for i in generated_ids]
            return generated_ids, generated_tokens

        return generated_ids

    # -------------- PERPLEXITY METRIC ------------------
    def compute_perplexity(self, data_ids):
        """
        Computes the perplexity of the model on a given dataset.

        Args:
            data_ids (list[int]): sequence of token ids to evaluate.

        Returns:
            float: perplexity score (lower is better).
        """
        X_batch, y_batch = get_batch(
            data_ids, block_size=self.block_size, batch_size=self.batch_size
        )
        logits = self.forward(X_batch)
        loss, _ = self.cross_entropy_loss(logits, y_batch)
        return float(np.exp(loss))


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
    model = NeuralNgram(
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
