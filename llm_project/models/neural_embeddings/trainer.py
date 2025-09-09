import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from llm_project.utils.dataloader import load_shakespeare
from llm_project.utils.file_manager import (
    save_model,
    load_model,
    get_model_path,
    get_project_root,
)
from llm_project.models.neural_embeddings.model import NeuralEmbed
from llm_project.utils.debugg_utils import Colors


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
            self.root, "experiments", "models", "neural_ngrams"
        )
        os.makedirs(self.model_dir, exist_ok=True)

        self.batch_size = batch_size
        self.block_size = config.block_size
        self.embedding_dim = config.embd_dim
        self.max_k = max_k

        self.tokens = tokens
        self.train_ids = None
        self.val_ids = None

        self.epochs = epochs
        self.lr = lr

        # mapping provided by pipeline
        self.token2id = getattr(self, "token2id", None)
        self.id2token = getattr(self, "id2token", None)
        self.unk_id = getattr(self, "unk_id", None)

    def set_vocab(self, token2id, id2token, unk_id):
        """Called from pipeline before training/generation."""
        self.token2id = token2id
        self.id2token = id2token
        self.unk_id = unk_id

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
        if len(data_ids) <= self.block_size + 1:
            raise ValueError(
                f"Not enough data for block_size={self.block_size} (len={len(data_ids)})"
            )
        X, y = [], []
        for _ in range(self.batch_size):
            start_idx = np.random.randint(0, len(data_ids) - self.block_size - 1)
            x_block = data_ids[start_idx : start_idx + self.block_size]
            y_block = data_ids[start_idx + 1 : start_idx + self.block_size + 1]

            X.append(x_block)
            y.append(y_block)

        return np.array(X), np.array(y)

    def _state_dict(self):
        if self.model is None:
            raise ValueError(f"{Colors.FAIL}[FAIL]{Colors.ENDC} Model not initialized")
        state = {
            "n": self.model.n,
            "vocab_size": self.model.vocab_size,
            "embd_dim": self.model.embd_dim,
            "params": {
                "embeddings": self.model.embeddings,
                "W": self.model.W,
                "b": self.model.b,
            },
            "id2token": getattr(self, "id2token", None),
            "token2id": getattr(self, "token2id", None),
        }
        return state

    def _save_state(self, subdir=None, filename=None, final=None):
        final_flag = final if final is not None else getattr(self, "final", False)

        # scegli la sottocartella in base a final
        model_subdir = "neural_ngrams"
        target_subdir = (
            os.path.join(model_subdir, "checkpoints")
            if not final_flag
            else os.path.join(model_subdir, "final")
        )

        state = self._state_dict()

        # salva il file
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
            "neural_ngrams", "final" if final else "checkpoints"
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

        self.id2token = model_data["id2token"]
        self.token2id = model_data["token2id"]

        if self.id2token is None or self.token2id is None:
            raise ValueError("Loaded model does not contain vocab!")

        return self.model

    def compute_perplexity(self, data_ids):
        """
        Computes the perplexity of the model on the entire dataset in chunks.

        Args:
            data_ids (list[int]): sequence of token ids to evaluate.

        Returns:
            float: perplexity score (lower is better)
        """
        total_loss = 0
        total_tokens = 0
        batch_size = self.batch_size
        block_size = self.block_size

        # Loop over the dataset in non-overlapping chunks
        for start_idx in range(0, len(data_ids) - block_size, block_size):
            end_idx = start_idx + block_size
            X_block = []
            Y_block = []
            # Prepare batch
            for i in range(
                start_idx,
                min(start_idx + batch_size * block_size, len(data_ids) - block_size),
                block_size,
            ):
                X_block.append(data_ids[i : i + block_size])
                Y_block.append(data_ids[i + 1 : i + 1 + block_size])

            X_block = np.array(X_block)
            Y_block = np.array(Y_block)
            if len(X_block) == 0:
                continue

            logits = self.model.forward(X_block)
            loss, _ = self.model.cross_entropy_loss(logits, Y_block)

            # peso per numero di sequenze
            total_loss += loss * X_block.shape[0]
            total_tokens += X_block.shape[0] * block_size

        avg_loss = total_loss / total_tokens
        return float(np.exp(avg_loss))

    def plot_perplexity(
        self, train_ids=None, val_ids=None, folder=None, filename="perplexity_curve.png"
    ):
        """
        Plots the perplexity over the dataset in chunks (train and validation).
        """
        train_ids = train_ids or self.train_ids
        val_ids = val_ids or self.val_ids

        # Compute perplexity in chunks of block_size*batch_size
        train_ppl = []
        val_ppl = []

        chunk_size = self.block_size * self.batch_size * 10  # configurable

        for i in range(0, len(train_ids), chunk_size):
            train_ppl.append(self.compute_perplexity(train_ids[i : i + chunk_size]))

        if val_ids:
            for i in range(0, len(val_ids), chunk_size):
                val_ppl.append(self.compute_perplexity(val_ids[i : i + chunk_size]))

        print("DEBUG: train_ppl =", train_ppl)
        print("DEBUG: val_ppl =", val_ppl)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(train_ppl, label="Train Perplexity", color="blue")
        if val_ppl:
            ax.plot(val_ppl, label="Validation Perplexity", color="red")
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
        Trains the neural n-gram model using mini-batch SGD with automatic checkpointing
        and saving of the final/best model.
        """

        epochs = epochs or self.epochs
        lr = lr or self.lr
        print(f"[INFO] Training for {epochs} epochs, lr={lr}")

        # ---------------- MODEL INIT ----------------------
        if not (self.token2id and self.id2token and self.unk_id is not None):
            raise RuntimeError(
                "Vocabulary mapping not set. Call set_vocab(...) from the pipeline before training."
            )

        vocab_size = len(self.token2id)
        if self.model is None or getattr(self.model, "vocab_size", -1) != vocab_size:
            print("Initializing model with vocab size:", vocab_size)
            self.model = NeuralEmbed(
                n=self.n,
                vocab_size=vocab_size,
                embd_dim=self.embedding_dim,
            )
        # keep model aware of sizes used elsewhere
        self.model.block_size = self.block_size
        self.model.batch_size = self.batch_size
        # ---------------- LOAD FINAL MODEL ----------------
        if not force_retrain:
            try:
                model = self._load_state(filename="best_model.pkl", final=final)
                print(
                    f"{Colors.OKGREEN}[OK]{Colors.ENDC}\n--- Loaded final saved model ---"
                )
                return model
            except FileNotFoundError:
                print(
                    f"{Colors.WARNING}[WARN]{Colors.ENDC} \n--- No final model found, training from scratch ---"
                )

        # ---------------- TRAINING LOOP -------------------
        losses, val_losses = [], []
        best_val_loss = float("inf")
        patience_counter = 0
        step = 0
        checkpoint_list = []
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
                    print(f"Epoch {epoch + 1}/{epochs}, Step {step}, Loss: {loss:.4f}")
                step += 1

            # ---------------- VALIDATION -------------------
            if self.val_ids:
                X_val, y_val = self.get_batch(self.val_ids)
                val_logits = self.model.forward(X_val)
                val_loss, _ = self.model.cross_entropy_loss(val_logits, y_val)
                val_losses.append(val_loss)
                val_perplexities.append(np.exp(val_loss))

                # ---------------- SAVE CHECKPOINT ----------------
                ckpt_name = f"epoch{epoch + 1}_val{val_loss:.4f}.pkl"
                self._save_state(subdir="checkpoints", filename=ckpt_name, final=final)
                checkpoint_list.append((val_loss, ckpt_name))
                # sort by validation loss
                checkpoint_list.sort(key=lambda x: x[0])

                # Keep only the best `max_checkpoints`
                if len(checkpoint_list) > max_checkpoints:
                    _, remove_ckpt = checkpoint_list.pop(-1)
                    remove_path = os.path.join(
                        self.root,
                        "experiments",
                        "models",
                        "neural_ngrams",
                        "checkpoints",
                        remove_ckpt,
                    )
                    try:
                        os.remove(remove_path)
                    except FileNotFoundError:
                        pass

                # ---------------- SAVE BEST MODEL ----------------
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_state(
                        subdir="final", filename="best_model.pkl", final=final
                    )
                    print(
                        f"{Colors.OKGREEN}[OK]{Colors.ENDC} Best model updated at epoch {epoch + 1}"
                    )

                else:
                    patience_counter += 1
                    print(f"No improvement (patience {patience_counter}/{patience})")

                # ---------------- EARLY STOPPING ----------------
                if patience_counter >= patience:
                    print(
                        f"{Colors.WARNING}[WARN]{Colors.ENDC} Early stopping triggered at epoch {epoch + 1}"
                    )
                    break

            if patience_counter >= patience:
                break

        self.plot_perplexity()

        # Plot val perplexity per epoch
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(val_perplexities, marker="o", label="Validation Perplexity")
        ax.set_title("Validation Perplexity per Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Perplexity")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

        folder = self.model_dir
        os.makedirs(folder, exist_ok=True)
        save_path = os.path.join(folder, "val_perplexity_by_epoch.png")
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(
            f"{Colors.OKGREEN}[OK]{Colors.ENDC} Validation perplexity plot saved to {save_path}"
        )

        return losses, val_losses


#
# if __name__ == "__main__":
#     from llm_project.models.configs.configs import NeuralConfig
#     from llm_project.models.neural_embeddings.trainer import NeuralNgramTrainer
#
#     # --- Hyperparameters ---
#     embedding_dim = 8
#     block_size = 3
#     batch_size = 2
#     n = 2
#     max_k = 10
#     device = "cpu"
#     print_every = 1
#
#     # --- Minimal test dataset ---
#     train_seq = ["A", "B", "C", "D", "E", "F", "G"]
#     val_seq = ["A", "B", "C", "D", "E"]
#
#     # Create simple token mapping
#     tokens = sorted(list(set(train_seq + val_seq)))
#     token_to_id = {tok: i for i, tok in enumerate(tokens)}
#     id_to_token = {i: tok for tok, i in token_to_id.items()}
#
#     train_ids = [token_to_id[tok] for tok in train_seq]
#     val_ids = [token_to_id[tok] for tok in val_seq]
#
#     # --- Config ---
#     config = NeuralConfig(
#         n=n,
#         vocab_size=len(tokens),
#         block_size=block_size,
#         embd_dim=embedding_dim,
#         device=device,
#     )
#
#     # --- Initialize trainer ---
#     trainer = NeuralEmbedTrainer(
#         model=None,
#         tokens=tokens,
#         batch_size=batch_size,
#         train_text=None,
#         valid_text=None,
#         config=config,
#         max_k=max_k,
#         root=None,
#         print_every=print_every,
#         autoload=False,
#     )
#
#     # --- Assign ids ---
#     trainer.train_ids = train_ids
#     trainer.val_ids = val_ids
#
#     # --- Train ---
#     losses, val_losses = trainer.train(epochs=5, lr=0.01, patience=2)
#     print("Training completed.")
#
#     # --- Generate text from a start token ---
#     start_tokens = ["A", "B"]
#     start_ids = [token_to_id[tok] for tok in start_tokens]
#
#     generated_ids, generated_text = trainer.model.generate(
#         start_ids=start_ids,
#         id2token=id_to_token,
#         max_new_tokens=5,
#         stochastic=True,
#         stop_words=set(),
#         block_size=config.block_size,
#     )
#
#     print("Generated text:", generated_text)
#
#
# def _batch_loss(self, X_batch, Y_batch):
#     """
#     Computes cross-entropy loss for a single batch.
#
#     Args:
#         X_batch (np.ndarray): shape (batch_size, block_size)
#         Y_batch (np.ndarray): shape (batch_size, block_size)
#
#     Returns:
#         float: average loss for this batch
#     """
#     logits = self.model.forward(X_batch)
#     loss, _ = self.model.cross_entropy_loss(logits, Y_batch)
#     return loss
#
#
# def compute_perplexity(self, data_ids):
#     """
#     Computes the perplexity over the entire dataset.
#
#     Args:
#         data_ids (list[int]): sequence of token ids to evaluate.
#
#     Returns:
#         float: perplexity score (lower is better)
#     """
#     total_loss = 0.0
#     total_tokens = 0
#
#     # iterate dataset in non-overlapping batches
#     for start_idx in range(0, len(data_ids) - self.block_size, self.block_size):
#         X_batch = []
#         Y_batch = []
#         for i in range(
#             start_idx,
#             min(
#                 start_idx + self.batch_size * self.block_size,
#                 len(data_ids) - self.block_size,
#             ),
#             self.block_size,
#         ):
#             X_batch.append(data_ids[i : i + self.block_size])
#             Y_batch.append(data_ids[i + 1 : i + 1 + self.block_size])
#
#         if not X_batch:
#             continue
#
#         X_batch = np.array(X_batch)
#         Y_batch = np.array(Y_batch)
#         batch_loss = self._batch_loss(X_batch, Y_batch)
#
#         total_loss += batch_loss * X_batch.shape[0]
#         total_tokens += X_batch.shape[0] * self.block_size
#
#     avg_loss = total_loss / total_tokens
#     perplexity = float(np.exp(avg_loss))
#     return perplexity
#
#
# def plot_perplexity(
#     self, train_ids=None, val_ids=None, folder=None, filename="perplexity_curve.png"
# ):
#     """
#     Plots perplexity per epoch using compute_perplexity for global dataset evaluation.
#     """
#     train_ids = train_ids or self.train_ids
#     val_ids = val_ids or self.val_ids
#
#     train_ppl, val_ppl = [], []
#
#     # Compute perplexity globally per epoch
#     train_ppl.append(self.compute_perplexity(train_ids))
#     if val_ids:
#         val_ppl.append(self.compute_perplexity(val_ids))
#
#     fig, ax = plt.subplots(figsize=(8, 6))
#     ax.plot(train_ppl, label="Train Perplexity", color="blue")
#     if val_ppl:
#         ax.plot(val_ppl, label="Validation Perplexity", color="red")
#     ax.set_xlabel("Epoch")
#     ax.set_ylabel("Perplexity")
#     ax.set_title("Perplexity over dataset")
#     ax.grid(True, linestyle="--", alpha=0.5)
#     ax.legend()
#
#     folder = folder or self.model_dir
#     os.makedirs(folder, exist_ok=True)
#     save_path = os.path.join(folder, filename)
#     fig.savefig(save_path, bbox_inches="tight", dpi=150)
#     plt.close(fig)
#     print(f"{Colors.OKGREEN}[OK]{Colors.ENDC} Perplexity plot saved to {save_path}")
