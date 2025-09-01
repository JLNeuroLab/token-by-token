import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from llm_project.utils.dataloader import load_shakespeare
from llm_project.utils.file_manager import save_item, load_item
from llm_project.models.neural_ngrams.model import NeuralNgram
from llm_project.bpe.bytepair_encoding import normalize_text, BPE
from llm_project.utils.debugg_utils import Colors


class NeuralNgramTrainer:
    def __init__(
        self,
        model,
        n,
        train_text,
        valid_text,
        max_k,
        root=None,
        checkpoint_dir="checkpoints",
        print_every=100,
        batch_size=32,
        block_size=4,
        embedding_dim=16,
        autoload=True,
    ):
        self.model = model
        self.n = n
        self.train_text = train_text
        self.valid_text = valid_text
        self.print_every = print_every
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../..")
        )
        self.root = root or project_root

        self.checkpoint_dir = (
            checkpoint_dir
            if checkpoint_dir != "checkpoints"
            else os.path.join(self.root, "experiments", "saved_models", "neural_ngram")
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.batch_size = batch_size
        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self.max_k = max_k

        self.bpe = None
        self.merges = None
        self.train_ids = None
        self.val_ids = None
        self.autoload = autoload

    def get_batch(self, data_ids):
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
        for _ in range(self.batch_size):
            start_idx = np.random.randint(0, len(data_ids) - self.block_size - 1)
            x_block = data_ids[start_idx : start_idx + self.block_size]
            y_block = data_ids[start_idx + 1 : start_idx + self.block_size + 1]

            X.append(x_block)
            y.append(y_block)

        return np.array(X), np.array(y)

    def save_checkpoint(self, folder, name, losses=None, val_losses=None):
        """
        Uses save_item for saving the  parameters of the model

        Args:
            folder (str): folder where checkpoint is saved
            name (str): checkpoint filename
        """
        print("saving checkpoint")
        state = {
            "model_state": self.model.state_dict(),
            "losses": losses,
            "val_losses": val_losses,
        }
        save_item(state, folder_path=folder, name=name, text_version=False)

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
        if "model_state" not in checkpoint:
            raise ValueError(f"Checkpoint {name} does not contain 'model_state'.")
        self.model.load_state_dict(checkpoint["model_state"])
        losses = checkpoint.get("losses", [])
        val_losses = checkpoint.get("val_losses", [])
        print(f"Checkpoint loaded: {os.path.join(folder, name)}")
        return losses, val_losses

    def prepare_bpe(self, force_retrain=False, load_bpe_name=None):
        bpe_dir = os.path.join(self.root, "experiments", "saved_models", "neural_ngram")
        os.makedirs(bpe_dir, exist_ok=True)
        bpe_file = load_bpe_name or os.path.join(
            bpe_dir, f"BPE_merges_k{self.max_k}.pkl"
        )

        if os.path.exists(bpe_file) and not force_retrain:
            print(f"Loading existing BPE from {bpe_file}...")
            bpe_data = load_item(bpe_dir, os.path.basename(bpe_file))
            self.bpe = BPE(max_k=bpe_data["max_k"], text=" ".join(self.train_text))
            self.bpe.merges = bpe_data["merges"]
            self.bpe.tokens = bpe_data["tokens"]
            self.bpe.text = bpe_data["text"]
        else:
            print("Training new BPE...")
            self.bpe = BPE(max_k=self.max_k, text=" ".join(self.train_text))
            self.bpe.text = self.bpe.load_and_normalize()
            self.bpe.BPE_encoder()
            bpe_dict = {
                "max_k": self.bpe.max_k,
                "merges": self.bpe.merges,
                "tokens": self.bpe.tokens,
                "text": self.bpe.text,
            }
            save_item(bpe_dict, bpe_dir, os.path.basename(bpe_file))
            save_item(self.bpe.tokens, bpe_dir, "bpe_tokens")
            fig = self.bpe.plot_vocabulary_growth()
            if fig is not None:
                save_item(fig, bpe_dir, "vocabulary_growth.png")
                plt.close(fig)
            print(f"BPE saved to {bpe_file}.")

        # mapping token <-> id
        self.bpe.token_to_id = {tok: i for i, tok in enumerate(self.bpe.tokens)}
        self.bpe.id_to_token = {i: tok for i, tok in enumerate(self.bpe.tokens)}
        self.merges = self.bpe.merges
        print(f"BPE ready with {len(self.bpe.tokens)} tokens.")

    def plot_perplexity(
        self, train_ids=None, val_ids=None, folder=None, filename="perplexity_curve.png"
    ):
        """
        Plots the perplexity on training and validation sets.

        Args:
            train_ids (list[int]): token IDs for training data
            val_ids (list[int]): token IDs for validation data
        """
        train_ids = train_ids or self.train_ids
        val_ids = val_ids or self.val_ids

        # Compute perplexity in chunks to reduce computation time
        train_ppl = [
            self.compute_perplexity(train_ids[i : i + self.block_size * 10])
            for i in range(0, len(train_ids), self.block_size * 10)
        ]

        val_ppl = (
            [
                self.compute_perplexity(val_ids[i : i + self.block_size * 10])
                for i in range(0, len(val_ids), self.block_size * 10)
            ]
            if val_ids
            else []
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(train_ppl, label="Train Perplexity", color="blue")
        if val_ppl:
            ax.plot(val_ppl, label="Validation Perplexity", color="red")
        ax.set_xlabel("Step")
        ax.set_ylabel("Perplexity")
        ax.set_title("Perplexity over Training")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

        folder = folder or self.checkpoint_dir
        save_item(fig, folder, filename, text_version=False)
        plt.close(fig)

    def compute_perplexity(self, data_ids):
        """
        Computes the perplexity of the model on a given dataset.
        """
        X_batch, y_batch = self.get_batch(data_ids)
        logits = self.model.forward(X_batch)
        loss, _ = self.model.cross_entropy_loss(logits, y_batch)
        return float(np.exp(loss))

    def fit(
        self,
        batch_size=None,
        block_size=None,
        epochs=3,
        lr=0.01,
        force_retrain=False,
        load_ckpt_name=None,
        load_bpe_name=None,
        patience=3,
        max_checkpoints=3,
    ):
        """
        Trains the neural ngram model using mini-batch SGD.
        """

        if batch_size is not None:
            self.batch_size = batch_size
        if block_size is not None:
            self.block_size = block_size

        # ---------------- PATHS ----------------------
        ckpt_dir = self.checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        # ---------------- BPE -----------------------
        self.prepare_bpe(load_bpe_name=load_bpe_name, force_retrain=force_retrain)

        if self.model is None or self.model.vocab_size != len(self.bpe.tokens):
            print("Initializing model with vocab size:", len(self.bpe.tokens))
            self.model = NeuralNgram(
                n=self.n,
                vocab_size=len(self.bpe.tokens),
                embedding_dim=self.embedding_dim,
            )

        # ---------------- Convert texts to IDs ----------------

        def text_to_ids(text_list):
            ids = []
            for t in text_list:
                tokens = self.bpe.BPE_segmenter(normalize_text(t))
                # Solo token già presenti nel vocabolario
                ids.extend(
                    [
                        self.bpe.token_to_id[tok]
                        for tok in tokens
                        if tok in self.bpe.token_to_id
                    ]
                )
            return ids

        self.train_ids = text_to_ids(self.train_text)
        self.val_ids = text_to_ids(self.valid_text) if self.valid_text else None

        # ------------------- Automatic checkpoint loading -------------------
        if self.autoload and not force_retrain:
            ckpts = glob.glob(os.path.join(self.checkpoint_dir, "*.pkl"))
            ckpts = [ckpt for ckpt in ckpts if "BPE" not in os.path.basename(ckpt)]

            if ckpts:
                ckpts = sorted(ckpts, key=lambda x: os.path.getmtime(x), reverse=True)
                load_ckpt_name = os.path.basename(ckpts[0])
                print(
                    f"Latest checkpoint found: {load_ckpt_name}, loading automatically."
                )
                try:
                    losses, val_losses = self.load_checkpoint(
                        self.checkpoint_dir, load_ckpt_name
                    )
                    # Stop immediately if no training data (generation-only mode)
                    return losses, val_losses

                except Exception as e:
                    print(f"Error loading checkpoint: {e}. Initializing a new model.")
            else:
                print(
                    f"No checkpoints found in {self.checkpoint_dir}. Initializing a new model."
                )

        # ---------------- TRAINING LOOP -------------------
        losses, val_losses = [], []
        best_val_loss = float("inf")
        step = 0
        patience_counter = 0
        checkpoints = []
        val_perplexities = []

        for epoch in range(epochs):
            n_batches = len(self.train_ids) // self.batch_size
            for _ in range(n_batches):
                X_batch, y_batch = self.get_batch(self.train_ids)
                logits = self.model.forward(X_batch)
                loss, probs = self.model.cross_entropy_loss(logits, y_batch)
                self.model.backward(X_batch, y_batch, probs, lr=lr)
                losses.append(loss)

                if step % self.print_every == 0:
                    print(f"epoch {epoch + 1}/{epochs}, step {step}, loss: {loss:.4f}")
                step += 1

            # ---------------- VALIDATION -------------------
            if self.val_ids:
                X_val, y_val = self.get_batch(self.val_ids)
                val_logits = self.model.forward(X_val)
                val_loss, _ = self.model.cross_entropy_loss(val_logits, y_val)
                val_losses.append(val_loss)
                val_perplexities.append(np.exp(val_loss))

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    ckpt_path = f"val={val_loss:.4f}_epoch={epoch + 1}.pkl"
                    self.save_checkpoint(
                        self.checkpoint_dir,
                        ckpt_path,
                        losses=losses,
                        val_losses=val_losses,
                    )
                    checkpoints.append((val_loss, ckpt_path))
                    checkpoints.sort(key=lambda x: x[0])

                    if len(checkpoints) > max_checkpoints:
                        _, worst_ckpt = checkpoints.pop(-1)
                        os.remove(os.path.join(self.checkpoint_dir, worst_ckpt))
                    print(f"Saved checkpoint: {ckpt_path}")
                else:
                    patience_counter += 1
                    print(f"No improvement (patience {patience_counter}/{patience})")

                if patience_counter >= patience:
                    print("Early stopping triggered")
                    return losses, val_losses
        # plotting and saving loss function plot
        # self.plot_loss(train_losses=losses, val_losses=val_losses)
        self.plot_perplexity()
        # --- Plot val perplexity per epoch ---
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(val_perplexities, marker="o", label="Validation Perplexity")
        ax.set_title("Validation Perplexity per Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Perplexity")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

        folder = self.checkpoint_dir
        save_item(fig, folder, "val_perplexity_by_epoch.png", text_version=False)
        plt.close(fig)

        return losses, val_losses

    def ensure_model_initialized(self):
        if self.model is None:
            vocab_size = len(self.bpe.tokens)
            self.model = NeuralNgram(
                n=self.n, vocab_size=vocab_size, embedding_dim=self.embedding_dim
            )

    def generate(
        self,
        prompt,
        max_new_tokens=100,
        stochastic=True,
        stop_words=None,
        batch_size=16,
        block_size=8,
    ):
        self.ensure_model_initialized()

        if prompt is None:
            raise ValueError("You should at least provide 'prompt' ")
        if self.bpe is None:
            raise ValueError(
                f"{Colors.FAIL}[ERROR]{Colors.FAIL}BPE not initialized. Execute prepare_bpe() first."
            )
            # Convert prompt in start_ids
            start_tokens = self.bpe.BPE_segmenter(normalize_text(prompt))
            start_ids = [
                self.bpe.token_to_id[tok]
                for tok in start_tokens
                if tok in self.bpe.token_to_id
            ]

        generated_ids = list(start_ids.copy())

        stop_words = stop_words or set()

        stop_words = stop_words or set()
        vocab_size = self.model.embeddings.shape[0]

        for _ in range(max_new_tokens):
            context = generated_ids[-self.block_size :]
            X = np.array(context, dtype=np.int64)[None, :]
            # generate logits
            # Pick the last token of the first and only example
            logits = self.model.forward(X)[0, -1]

            # Numerical stability
            logits = logits - np.max(logits)

            exps = np.exp(logits)
            probs = exps / exps.sum()
            assert len(probs) == vocab_size, f"{len(probs)} vs {vocab_size}"
            if stochastic:
                next_id = int(np.random.choice(vocab_size, p=probs))
            else:
                next_id = int(np.argmax(probs))

            generated_ids.append(next_id)

            if next_id in stop_ids:
                break
            if id2token is not None and id2token[next_id] in stop_words:
                break

            print(
                f"[DEBUG] Generation with batch_size={batch_size}, block_size={block_size}"
            )

        if self.bpe:
            generated_tokens = [self.bpe.id_to_token[i] for i in generated_ids]
            generated_text = " ".join(generated_tokens).replace("_", "")
            return generated_ids, generated_text

        return generated_ids


if __name__ == "__main__":
    # --- Load dataset ---
    train_text = load_shakespeare("train")[:10000]
    valid_text = load_shakespeare("validation")[:1000]

    # --- Hyperparameters ---
    embedding_dim = 16
    max_k = 2000
    batch_size = 32
    block_size = 8
    print_every = 50
    n = 4

    # --- Initialize trainer without model ---
    trainer = NeuralNgramTrainer(
        model=None,
        n=n,  # model will be created after BPE
        train_text=train_text,
        valid_text=valid_text,
        max_k=max_k,
        root=None,
        batch_size=batch_size,
        block_size=block_size,
        print_every=print_every,
        embedding_dim=embedding_dim,
    )

    # --- Start training ---
    losses, val_losses = trainer.fit(epochs=5, lr=0.01, patience=2)

    print("Training completed.")

    # --- Prepare id2token mapping for generation ---
    id2token = {i: tok for i, tok in enumerate(trainer.bpe.tokens)}

    # --- Select some starting tokens ---
    start_text = "Once upon a time"
    start_tokens = trainer.bpe.BPE_segmenter(normalize_text(start_text))
    start_ids = [
        trainer.bpe.token_to_id[tok]
        for tok in start_tokens
        if tok in trainer.bpe.token_to_id
    ]

    # --- Generate text ---
    generated_ids, generated_tokens = trainer.generate(
        start_ids=start_ids,
        id2token=id2token,
        max_new_tokens=50,
        stochastic=True,
        stop_words={".", "\n"},
    )

    # --- Show generated text ---
    generated_text = "".join(generated_tokens)
    print("\nGenerated text:")
    print(generated_text)
