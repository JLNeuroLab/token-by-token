import os
import numpy as np
import matplotlib.pyplot as plt

from llm_project.utils.file_manager import (
    save_model,
    load_model,
    get_model_path,
    get_project_root,
)
from llm_project.models.neural_embeddings.model import NeuralEmbed
from llm_project.utils.debugg_utils import Colors, get_proc_mem_mb, get_proc_cpu_percent


class NeuralEmbedTrainer:
    def __init__(
        self,
        model,
        epochs,
        lr,
        tokens,
        batch_size,
        train_text,
        valid_text,
        config,
        max_k,
        root=None,
        print_every=100,
    ):
        # trainer parameters
        self.model = model
        self.n = config.n
        self.train_text = train_text
        self.valid_text = valid_text
        self.print_every = print_every

        self.root = root or get_project_root()
        self.model_dir = os.path.join(
            self.root, "experiments", "models", "neuralslow"
        )
        os.makedirs(self.model_dir, exist_ok=True)

        self.batch_size = batch_size
        self.block_size = config.block_size
        self.embedding_dim = config.embd_dim
        self.max_k = max_k

        # Provided but unused for training (we train on ids). Kept for compatibility.
        self.tokens = tokens

        # Pipeline must assign these before calling train()
        self.train_ids = None
        self.val_ids = None

        self.epochs = epochs
        self.lr = lr

        # mapping provided by pipeline
        self.token2id = getattr(self, "token2id", None)
        self.id2token = getattr(self, "id2token", None)
        self.unk_id = getattr(self, "unk_id", None)

        # whether to load/save from final folder (pipeline may set this attribute too)
        self.final = getattr(self, "final", False)

    # ---- vocab wiring from pipeline ----
    def set_vocab(self, token2id, id2token, unk_id):
        """Called from pipeline before training/generation."""
        self.token2id = token2id
        self.id2token = id2token
        self.unk_id = 0 if unk_id is None else int(unk_id)

    # ---- batching over id sequences ----
    def get_batch(self, data_ids):
        """
        Create a batch of (X, y) from a list of token IDs.
        X: (batch_size, block_size)
        y: (batch_size, block_size) = X shifted by one
        """
        data_ids = np.asarray(data_ids, dtype=np.int64)

        if data_ids is None or len(data_ids) <= self.block_size + 1:
            raise ValueError(
                f"Not enough data for block_size={self.block_size} (len={0 if data_ids is None else len(data_ids)})"
            )

        # Preallocate as int64
        X = np.empty((self.batch_size, self.block_size), dtype=np.int64)
        y = np.empty((self.batch_size, self.block_size), dtype=np.int64)

        max_start = data_ids.size - self.block_size - 1
        for b in range(self.batch_size):
            start_idx = np.random.randint(0, max_start + 1)
            X[b, :] = data_ids[start_idx : start_idx + self.block_size]
            y[b, :] = data_ids[start_idx + 1 : start_idx + 1 + self.block_size]

        return np.array(X), np.array(y)

    # ---- serialization helpers ----
    def _state_dict(self):
        if self.model is None:
            raise ValueError(f"{Colors.FAIL}[FAIL]{Colors.ENDC} Model not initialized")
        return {
            "n": self.model.n,
            "vocab_size": self.model.vocab_size,
            "embd_dim": self.model.embd_dim,
            "params": {
                "embeddings": self.model.embeddings,
                "W": self.model.W,
                "b": self.model.b,
            },
            # persist vocab so generation works out of process
            "id2token": getattr(self, "id2token", None),
            "token2id": getattr(self, "token2id", None),
            "unk_id": getattr(self, "unk_id", 0),
        }

    def _save_state(self, subdir=None, filename=None, final=None):
        final_flag = final if final is not None else getattr(self, "final", False)
        model_subdir = "neuralslow"
        target_subdir = os.path.join(
            model_subdir, "final" if final_flag else "checkpoints"
        )
        state = self._state_dict()
        full_path = save_model(
            state,
            root=self.root,
            subdir=target_subdir,
            filename=filename,
            category="models",
            final=final_flag,
        )
        return full_path

    def _load_state(self, filename=None, final=False, subdir=None):
        final_flag = final if final is not None else getattr(self, "final", False)
        filename = filename or "best_model.pkl"
        subdir = subdir or os.path.join(
            "neuralslow", "final" if final_flag else "checkpoints"
        )

        model_data = load_model(
            root=self.root, filename=filename, final=final_flag, subdir=subdir
        )

        # Rebuild the model
        self.model = NeuralEmbed(
            n=model_data["n"],
            vocab_size=model_data["vocab_size"],
            embd_dim=model_data["embd_dim"],
        )
        self.model.embeddings = model_data["params"]["embeddings"]
        self.model.W = model_data["params"]["W"]
        self.model.b = model_data["params"]["b"]

        # vocab + unk
        self.id2token = model_data.get("id2token", None)
        self.token2id = model_data.get("token2id", None)
        self.unk_id = model_data.get("unk_id", 0)

        if self.id2token is None or self.token2id is None:
            raise ValueError(
                f"{Colors.FAIL}[FAIL]{Colors.ENDC} Loaded model does not contain vocab!"
            )

        print(f"{Colors.OKGREEN}[OK]{Colors.ENDC} Model loaded from {filename}")
        return self.model

    # ---- evaluation ----

    def compute_perplexity(self, data_ids):
        """
        Computes the perplexity of the model on the entire dataset in chunks.

        Args:
            data_ids (list[int]): sequence of token ids to evaluate.

        Returns:
            float: perplexity score (lower is better)
        """
        total_loss = 0.0
        total_tokens = 0
        batch_size = self.batch_size
        block_size = self.block_size

        # Walk the dataset in steps of block_size
        for start_idx in range(0, len(data_ids) - block_size, block_size):
            X_block, Y_block = [], []
            for i in range(
                start_idx,
                min(start_idx + batch_size * block_size, len(data_ids) - block_size),
                block_size,
            ):
                X_block.append(data_ids[i : i + block_size])
                Y_block.append(data_ids[i + 1 : i + 1 + block_size])

            if not X_block:
                continue

            X_block = np.array(X_block, dtype=np.int64)
            Y_block = np.array(Y_block, dtype=np.int64)

            logits = self.model.forward(X_block)
            loss, _ = self.model.cross_entropy_loss(logits, Y_block)

            total_loss += loss * X_block.shape[0]
            total_tokens += X_block.shape[0] * block_size

        avg_loss = total_loss / max(1, total_tokens)
        return float(np.exp(avg_loss))

    def plot_perplexity(
        self, train_ids=None, val_ids=None, folder=None, filename="perplexity_curve.png"
    ):
        """
        Plot perplexity over the dataset in chunks (train and validation).
        """
        train_ids = train_ids or self.train_ids
        val_ids = val_ids or self.val_ids

        train_ppl, val_ppl = [], []
        chunk_size = self.block_size * self.batch_size * 10  # tweak as you like

        for i in range(0, len(train_ids), chunk_size):
            train_ppl.append(self.compute_perplexity(train_ids[i : i + chunk_size]))

        if val_ids:
            for i in range(0, len(val_ids), chunk_size):
                val_ppl.append(self.compute_perplexity(val_ids[i : i + chunk_size]))

        fig, ax = plt.subplots(figsize=(8, 6))
        if train_ppl:
            ax.plot(train_ppl, label="Train Perplexity")
        if val_ppl:
            ax.plot(val_ppl, label="Validation Perplexity")
        ax.set_xlabel("Step")
        ax.set_ylabel("Perplexity")
        ax.set_title("Perplexity over dataset")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

        folder = folder or self.model_dir
        os.makedirs(folder, exist_ok=True)
        save_path = os.path.join(folder, filename)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"{Colors.OKGREEN}[OK]{Colors.ENDC} Perplexity plot saved to {save_path}")

    def plot_loss_curves(
        self, train_losses, val_losses, root, filename="loss_curves.png", final=False
    ):
        """
        Plot and save training and validation loss curves per epoch.

        Args:
            train_losses (list or array): List of training losses per epoch.
            val_losses (list or array): List of validation losses per epoch.
            root (str): Project root path.
            filename (str): Name of the saved figure.
            final (bool): Whether to save in final folder (saved_models) or experiments.
        """
        import matplotlib.pyplot as plt
        from llm_project.utils.file_manager import get_model_path
        from llm_project.utils.debugg_utils import Colors
        import os

        # Make sure losses are lists of epoch averages
        train_losses = list(train_losses)
        val_losses = list(val_losses)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(train_losses, marker="o", color="blue", label="Train Loss")
        ax.plot(val_losses, marker="o", color="red", label="Validation Loss")
        ax.set_title("Train and Validation Loss per Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

        folder = get_model_path(root, "models", subdir="neuralslow", final=final)
        os.makedirs(folder, exist_ok=True)
        save_path = os.path.join(folder, filename)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(
            f"{Colors.OKGREEN}[OK]{Colors.ENDC} Train/Validation loss plot saved to {save_path}"
        )

    def plot_val_perplexity_per_epoch(
        self, val_perplexities, filename="val_perplexity_by_epoch.png", final=False
    ):
        """
        Plot and save validation perplexity per epoch.

        Args:
            val_perplexities (list): List of validation perplexities per epoch.
            filename (str): Name of the saved figure.
            final (bool): Whether to save in final folder (saved_models) or experiments.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(
            val_perplexities, marker="o", color="blue", label="Validation Perplexity"
        )
        ax.set_title("Validation Perplexity per Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Perplexity")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

        # Determine folder based on final flag
        folder = get_model_path(self.root, "models", subdir="neuralslow", final=final)
        os.makedirs(folder, exist_ok=True)
        save_path = os.path.join(folder, filename)

        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(
            f"{Colors.OKGREEN}[OK]{Colors.ENDC} Validation perplexity plot saved to {save_path}"
        )

    # ---- training ----
    def train(
        self,
        epochs=None,
        lr=None,
        force_retrain=False,
        patience=3,
        max_checkpoints=5,
        final=False,
    ):
        """
        Train the neural n-gram model using mini-batch SGD with checkpointing
        and save the best model.
        """
        # Must have vocab + ids from pipeline
        if not (self.token2id and self.id2token and self.unk_id is not None):
            raise RuntimeError(
                f"{Colors.WARNING}[WARNING]{Colors.ENDC}Vocabulary mapping not set. Call set_vocab(...) from the pipeline before training."
            )
        if not self.train_ids:
            raise RuntimeError(
                f"{Colors.WARNING}[WARNING]{Colors.ENDC} train_ids not set. Pipeline must provide integer id sequences."
            )

        epochs = epochs or self.epochs
        lr = lr or self.lr
        final_flag = self.final if final is None else final
        print(
            f"{Colors.OKCYAN}[INFO]{Colors.ENDC} Training for {epochs} epochs, lr={lr}"
        )
        # Guard: ensure integers before casting
        if isinstance(self.train_ids[0], str):
            raise TypeError(
                f"{Colors.FAIL}[FAIL]{Colors.ENDC} Trainer received string tokens. Fix mapping in pipeline.prepare_tokens() so neural models get int IDs."
            )
        # Start process cpu measurement window
        get_proc_cpu_percent(prime=True)
        self.train_ids = np.asarray(self.train_ids, dtype=np.int64).ravel().tolist()
        if self.val_ids is not None:
            self.val_ids = np.asarray(self.val_ids, dtype=np.int64).ravel().tolist()

        # Try to load a final/best model unless forced to retrain
        if not force_retrain:
            try:
                model = self._load_state(filename="best_model.pkl", final=final)
                print(
                    f"{Colors.OKGREEN}[OK]{Colors.ENDC} Loading .pkl from final saved model."
                )
                return model  # Not return model so it keeps going with resume training
            except FileNotFoundError:
                print(
                    f"{Colors.WARNING}[WARN]{Colors.ENDC} No final model found, training from scratch."
                )

        # Model init (use vocab from mapping)
        vocab_size = len(self.token2id)
        if self.model is None or getattr(self.model, "vocab_size", -1) != vocab_size:
            print(
                f"{Colors.OKCYAN}[INFO]{Colors.ENDC} Initializing model with vocab size:",
                vocab_size,
            )
            self.model = NeuralEmbed(
                n=self.n, vocab_size=vocab_size, embd_dim=self.embedding_dim
            )

        # Keep model aware of sizes used elsewhere
        self.model.block_size = self.block_size
        self.model.batch_size = self.batch_size

        train_losses_per_epoch, val_losses_per_epoch = [], []
        val_perplexities = []
        best_val_loss = float("inf")
        patience_counter = 0
        step = 0
        checkpoint_list = []  # list[(val_loss, filename)]

        for epoch in range(epochs):
            n_batches = max(1, len(self.train_ids) // self.batch_size)
            epoch_loss = 0.0

            for _ in range(n_batches):
                X_batch, y_batch = self.get_batch(self.train_ids)
                logits = self.model.forward(X_batch)
                loss, probs = self.model.cross_entropy_loss(logits, y_batch)
                epoch_loss += loss

                self.model.backward(X_batch, y_batch, probs, lr=lr)

                if step % self.print_every == 0:
                    ram_mb, _ = get_proc_mem_mb()
                    cpu_p = get_proc_cpu_percent()
                    print(
                        f"[train] Epoch {epoch + 1}/{epochs} | Step {step} | Loss: {loss:.4f} | RAM: {ram_mb:.2f}mb ({ram_mb / 1024:.2f}Gb) | CPU: {cpu_p:.1f}%"
                    )
                    get_proc_cpu_percent(prime=True)
                step += 1

            avg_epoch_loss = epoch_loss / n_batches
            train_losses_per_epoch.append(avg_epoch_loss)
            print(f"Epoch {epoch + 1}/{epochs} | Avg Train Loss: {avg_epoch_loss:.4f}")

            # Validation + checkpoint
            if self.val_ids:
                X_val, y_val = self.get_batch(self.val_ids)
                val_logits = self.model.forward(X_val)
                val_loss, _ = self.model.cross_entropy_loss(val_logits, y_val)
                val_losses_per_epoch.append(val_loss)
                val_perplexities.append(float(np.exp(val_loss)))
                print(
                    f"Validation Loss: {val_loss:.4f}, Perplexity: {val_perplexities[-1]:.4f}"
                )

                cpu_p = get_proc_cpu_percent()
                ram_mb, _ = get_proc_mem_mb()
                print(
                    f"[valid] Epoch {epoch + 1}/{epochs} | Step {step} | Validation loss {val_loss} | RAM: {ram_mb:.2f}mb ({ram_mb / 1024:.2f}Gb) | CPU: {cpu_p:.1f}%"
                )
                get_proc_cpu_percent(prime=True)

                ckpt_name = f"epoch{epoch + 1}_val{val_loss:.4f}.pkl"
                self._save_state(subdir="checkpoints", filename=ckpt_name, final=False)
                checkpoint_list.append((val_loss, ckpt_name))
                checkpoint_list.sort(key=lambda x: x[0])

                # Keep only the best K checkpoints on disk
                if len(checkpoint_list) > max_checkpoints:
                    _, remove_ckpt = checkpoint_list.pop(-1)
                    remove_path = os.path.join(
                        self.root,
                        "experiments",
                        "models",
                        "neuralslow",
                        "checkpoints",
                        remove_ckpt,
                    )
                    try:
                        os.remove(remove_path)
                    except FileNotFoundError:
                        pass

                # Save best model into final/
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    full_path = self._save_state(
                        subdir="final", filename="best_model.pkl", final=final
                    )
                    self.model_path = full_path

                    print(
                        f"{Colors.OKGREEN}[OK]{Colors.ENDC} Best model updated at epoch {epoch + 1}"
                    )
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f"No improvement (patience {patience_counter}/{patience})")

                if patience_counter >= patience:
                    print(
                        f"{Colors.WARNING}[WARN]{Colors.ENDC} Early stopping at epoch {epoch + 1}"
                    )
                    break

            if patience_counter >= patience:
                break
        
        self.plot_loss_curves(
            train_losses=train_losses_per_epoch,
            val_losses=val_losses_per_epoch,
            root=self.root,
            final=final,
        )

        self.plot_val_perplexity_per_epoch(
            val_perplexities=val_perplexities, final=final
        )

        return train_losses_per_epoch, val_losses_per_epoch
