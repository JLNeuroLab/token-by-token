import numpy as np
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utils.file_manager import save_item, load_item

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
        x_block = data_ids[start_idx: start_idx + block_size]
        y_block = data_ids[start_idx + 1: start_idx + block_size + 1]

        X.append(x_block)
        y.append(y_block)

    return np.array(X), np.array(y)

class NeuralNgram:

    def __init__(self, n, vocab_size, embedding_dim=32, batch_size=32, block_size=4, seed=35):
        np.random.seed(seed)
        self.n = n
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.block_size = block_size
        # Initialize embedding matrix
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01 # (vocab_size, embedding_dim)
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
        probs = exps / exps_sum # applying softmax, (batch_size, block_size, vocab_size)
        # extracting indexes of correct tokens for each position in each sequence
        true_prob = probs[np.arange(B)[:, None], np.arange(L), y] # (batch_size, block_size)
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
        dZ /= (B * L)

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

    # ------------- CHECKPOINTS ------------------
    def save_checkpoint(self, folder, name):
        """
        Usa save_item per salvare i parametri del modello
        """
        print("saving checkpoint")
        state = {
            "embeddings": self.embeddings,
            "W": self.W,
            "b": self.b
        }
        save_item(state, folder=folder, name=name, text_version=False)

    def load_checkpoint(self, folder, name, base_dir=None):
        """
        Load a model checkpoint previously saved with save_checkpoint.

        Args:
            folder (str): folder where checkpoint is saved
            name (str): checkpoint filename
            base_dir (str|None): optional base directory

        Returns:
            None. Updates model weights in place.
        """
        checkpoint = load_item(folder, name, base_dir=base_dir)
        
        if not all(k in checkpoint for k in ("embeddings", "W", "b")):
            raise ValueError(f"Checkpoint {name} does not contain all required keys.")
        
        self.embeddings = checkpoint["embeddings"]
        self.W = checkpoint["W"]
        self.b = checkpoint["b"]
        print(f"Checkpoint loaded: {os.path.join(folder, name)}")

    # ------------- TRAINING --------------
    def fit(self, data_ids,
            epochs=3,
            lr=0.01,
            print_every=100,
            val_ids=None,
            checkpoint_dir = "checkpoints",
            max_checkpoints = 3,
            patience = 3,
            load_ckpt_name = None,
            force_train = False
            ):
        """
        Trains the model on the given data using mini-batch SGD.

        Args:
            data_ids (list[int]): list of token ids for training.
            epochs (int): number of passes over the training dataset.
            batch_size (int): number of sequences per batch.
            block_size (int): length of each input sequence.
            learning_rate (float): learning rate for SGD updates.
            val_data_ids (list[int]|None): optional validation data for early stopping.
            patience (int): number of epochs to wait before stopping if validation loss does not improve.
            max_checkpoints (int): maximum number of model checkpoints to save.

        Returns:
            None. Model weights are updated in place. Saves checkpoints to disk if enabled.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        if load_ckpt_name is not None and not force_train:
            checkpoint_path = os.path.join(checkpoint_dir, load_ckpt_name)
            if os.path.exists(checkpoint_path):
                self.load_checkpoint(checkpoint_dir, load_ckpt_name)
                print("Checkpoint loaded, skipping training.")
                return [], []

        losses, val_losses = [], []
        best_val_loss = float("inf")
        step = 0
        patience_counter = 0
        checkpoints = []
        #------ TRAINING-----------
        for epoch in range(epochs):

            n_batches = len(data_ids) // self.batch_size

            for _ in range(n_batches):
            
                X_batch, y_batch = get_batch(data_ids, block_size=self.block_size, batch_size=self.batch_size)
                logits = self.forward(X_batch)
                loss, probs = self.cross_entropy_loss(logits, y_batch)
                self.backward(X_batch, y_batch, probs, lr=lr)
                losses.append(loss)

                if step % print_every == 0:
                    print(f"epoch {epoch + 1}/{epochs}, step {step}, loss: {loss:.4f}")
                step += 1

            # -------- VALIDATION --------  
            if val_ids is not None:
                X_val, y_val = get_batch(val_ids, block_size=self.block_size, batch_size=self.batch_size)
                val_logits = self.forward(X_val)
                val_loss, _ = self.cross_entropy_loss(val_logits, y_val)
                val_losses.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    ckpt_path = f"val={val_loss:.4f}_epoch={epoch+1}.pkl"
                    self.save_checkpoint(checkpoint_dir, ckpt_path)
                    checkpoints.append((val_loss, ckpt_path))
                    checkpoints.sort(key=lambda x: x[0])

                    if len(checkpoints) > max_checkpoints:
                        _, worst_ckpt = checkpoints.pop(-1)
                        os.remove(os.path.join(checkpoint_dir, worst_ckpt))
                    print(f"saved checkpoints in {ckpt_path}")
                
                else:
                    patience_counter += 1
                    print(f"no improvement (patience {patience_counter}/{patience})")

                if patience_counter >= patience:
                    print(f"early stopping triggered")
                    return losses, val_losses

        return losses, val_losses
    
    # ---------------- GENERATION -------------------
    def generate(self,
                start_ids, 
                id2token=None, 
                max_new_tokens=20, 
                stochastic=True,
                stop_ids=None,
                stop_words=None):

        generated_ids = list(start_ids.copy())
        
        stop_ids = stop_ids or set()
        stop_words = stop_words or set()

        for _ in range(max_new_tokens):
            context = generated_ids[-self.block_size:]
            X = np.array(context, dtype=np.int64)[None, :]
            # generate logits
            logits = self.forward(X)[0, -1] # Pick the last token of the first and only example

            # Numerical stability
            logits = (logits - np.max(logits))

            exps = np.exp(logits)
            probs = exps / exps.sum()

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
        X_batch, y_batch = get_batch(data_ids, block_size=self.block_size, batch_size=self.batch_size)
        logits = self.forward(X_batch)
        loss, _ = self.cross_entropy_loss(logits, y_batch)
        return float(np.exp(loss))

if __name__ == "__main__":
    import os
    from utils.file_manager import save_item
    from utils.token_mapping import token_id_mapping
    from bpe.bytepair_encoding import load_and_normalize, BPE_encoder, BPE_segmenter, plot_vocabulary_growth

    # ---------------- CONFIG -----------------
    project_root = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    train_path = os.path.join(project_root, "data", "Shakespeare_clean_train.txt")
    val_path = os.path.join(project_root, "data", "Shakespeare_clean_valid.txt")

    # N-gram model params
    n = 3
    block_size = 4
    batch_size = 32
    epochs = 5
    lr = 0.01
    max_k = 2000
    checkpoint_dir = os.path.join(os.getcwd(), "saved_models", "neural_ngram", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    # ---------------- LOAD AND NORMALIZE TEXT -----------------
    t_train = load_and_normalize(train_path)
    t_valid = load_and_normalize(val_path)

    # (Optional) truncate for faster testing
    t_train = t_train[:10000]
    t_valid = t_valid[:1000]
    print(f"Train size (chars): {len(t_train)}, Test size (chars): {len(t_valid)}")

    # ---------------- TRAIN BPE -----------------
    tokens, vocab_size_history, final_vocab, merges = BPE_encoder(t_train, max_k=max_k)
    plot_vocabulary_growth(vocab_size_history, max_k)

    # ---------------- CONVERT TEXT TO IDS -----------------
    # Build simple mapping token <-> id
    token2id = token_id_mapping(tokens)
    id2token = token_id_mapping(token2id, decode=True)

    # Encode training and validation text
    train_ids = [token2id[tok] for tok in tokens if tok in token2id]
    test_tokens = BPE_segmenter(t_valid, merges)
    val_ids = [token2id[tok] for tok in test_tokens if tok in token2id]

    # ---------------- INITIALIZE MODEL -----------------
    model = NeuralNgram(n=n, vocab_size=len(token2id), block_size=block_size, batch_size=batch_size)

    # ---------------- TRAIN MODEL -----------------
    losses, val_losses = model.fit(
        data_ids=train_ids,
        val_ids=val_ids,
        epochs=epochs,
        lr=lr,
        checkpoint_dir=checkpoint_dir,
        print_every=50,
        max_checkpoints=3,
        patience=3,
        load_ckpt_name="val=6.3290_epoch=5.pkl",
        force_train=False
    )
    print("Training completed!")

    # ---------------- GENERATE SAMPLE -----------------
    start_ids = train_ids[:block_size]
    stop_ids_set = {token2id[tok] for tok in ".?!" if tok in token2id}
    generated_ids, generated_tokens = model.generate(start_ids=start_ids, max_new_tokens=20, id2token=id2token, stop_ids=stop_ids_set)
    print("Generated token IDs:", generated_ids)
    print("Generated tokens:", generated_tokens)
