import torch

# import torch.nn as nn
# import torch.nn.functional as F

import os
import matplotlib.pyplot as plt
from llm_project.utils.debugg_utils import Colors, get_proc_mem_mb
from llm_project.models.neural_fast.model import NeuralLanguageModel
from llm_project.utils.file_manager import (
    get_project_root,
    load_model,
    save_model,
    get_model_path,
)


class NeuralTrainer:
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
        max_checkpoints=5,
        patience=3,
    ):
        self.config = config
        self.device = self.config.device
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.max_checkpoints = max_checkpoints
        self.print_every = print_every
        self.patience = patience
        self.tokens = tokens

        self.batch_size = batch_size
        self.block_size = config.block_size
        self.embd_dim = config.embd_dim
        self.max_k = max_k
        self.train_text = train_text
        self.valid_text = valid_text

        self.train_ids = None
        self.valid_ids = None

        self.root = root or get_project_root()
        self.model_dir = os.path.join(self.root, "experiments", "models", "neuralfast")
        os.makedirs(self.model_dir, exist_ok=True)

        self.checkpoints = []

        self.id2token = None
        self.token2id = None

    # ------------------- BATCHES -------------------
    def get_batch(self, data_ids):
        X, y = [], []
        for _ in range(self.batch_size):
            start_idx = torch.randint(
                0, len(data_ids) - self.block_size - 1, (1,)
            ).item()
            x_block = data_ids[start_idx : start_idx + self.block_size]
            y_block = data_ids[start_idx + 1 : start_idx + self.block_size + 1]
            X.append(x_block)
            y.append(y_block)
        X = torch.tensor(X, dtype=torch.long).to(self.device)
        y = torch.tensor(y, dtype=torch.long).to(self.device)
        return X, y

    # ------------------- SAVE / LOAD -------------------
    def _save_state(self, subdir=None, filename=None, final=None):
        """
        Salva lo stato completo del modello, optimizer, vocab e config.
        """
        final_flag = final if final is not None else getattr(self, "final", False)
        # Se non viene passato subdir, scegli il percorso corretto
        model_subdir = "neuralfast"
        target_subdir = (
            os.path.join(model_subdir, "checkpoints")
            if not final_flag
            else os.path.join(model_subdir, "final")
        )
        # Crea lo stato da salvare
        state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "id2token": getattr(self, "id2token", None),
            "token2id": getattr(self, "token2id", None),
            "config": self.config,
        }

        full_path = save_model(
            state,
            root=self.root,
            category="models",
            subdir=target_subdir,
            filename=filename,
            final=final_flag,
        )
        print(f"{Colors.OKGREEN}[OK]{Colors.ENDC} Model saved to {full_path}")
        return full_path

    def _load_state(self, filename=None, subdir=None, final=None):
        """
        Carica lo stato completo del modello, optimizer, vocab e config.
        """
        final_flag = final if final is not None else getattr(self, "final", False)
        filename = filename or "best_model.pkl"
        # Usa il subdir passato oppure default coerente con final_flag
        target_subdir = subdir or os.path.join(
            "neuralfast", "final" if final else "checkpoints"
        )

        state = load_model(
            root=self.root,
            category="models",
            subdir=target_subdir,
            filename=filename,
            final=final_flag,
        )

        self.model = state["model"].to(self.device)
        self.optimizer = state["optimizer"]
        self.id2token = state.get("id2token", None)
        self.token2id = state.get("token2id", None)
        self.config = state.get("config", self.config)

        if self.id2token is None or self.token2id is None:
            raise ValueError(
                f"{Colors.FAIL}[FAIL]{Colors.ENDC} Loaded model does not contain vocab!"
            )

        print(f"{Colors.OKGREEN}[OK]{Colors.ENDC} Model loaded from {filename}")
        return self.model

    # ------------------- TRAINING STEP -------------------

    def train_step(self, X_batch, y_batch):
        self.model.train()
        logits, loss = self.model.forward(X_batch, targets=y_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    # ------------------- PERPLEXITY -------------------
    @torch.no_grad()
    def batch_perplexity(self, data_ids):
        self.model.eval()
        total_loss, total_tokens = 0.0, 0

        for start_id in range(0, len(data_ids) - self.block_size, self.block_size):
            X_batch, y_batch = [], []
            for i in range(
                start_id,
                min(
                    start_id + self.batch_size * self.block_size,
                    len(data_ids) - self.block_size,
                ),
                self.block_size,
            ):
                X_batch.append(data_ids[i : i + self.block_size])
                y_batch.append(data_ids[i + 1 : i + 1 + self.block_size])

            if not X_batch:
                continue

            X_batch = torch.tensor(X_batch, dtype=torch.long).to(self.device)
            y_batch = torch.tensor(y_batch, dtype=torch.long).to(self.device)
            _, loss = self.model.forward(X_batch, targets=y_batch)

            total_loss += loss.item() * X_batch.shape[0]
            total_tokens += X_batch.shape[0] * self.block_size

        avg_loss = total_loss / total_tokens
        return torch.exp(torch.tensor(avg_loss))

    # ------------------- PLOTTING -------------------
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

        folder = get_model_path(root, "models", subdir="neuralfast", final=final)
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
        folder = get_model_path(self.root, "models", subdir="neuralfast", final=final)
        os.makedirs(folder, exist_ok=True)
        save_path = os.path.join(folder, filename)

        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(
            f"{Colors.OKGREEN}[OK]{Colors.ENDC} Validation perplexity plot saved to {save_path}"
        )

    # ------------------- TRAIN LOOP -------------------

    def train(
        self,
        epochs=None,
        lr=None,
        force_retrain=False,
        patience=3,
        max_checkpoints=5,
        final=False,
    ):
        import numpy as np

        if self.model is None:
            if getattr(self, "token2id", None):
                V = len(self.token2id)
            elif self.tokens:
                V = len(self.tokens)
            elif self.train_ids and len(self.train_ids) > 0:
                V = int(max(self.train_ids)) + 1
            else:
                raise ValueError(
                    f"{Colors.FAIL}[ERROR]{Colors.ENDC} Cannot infer vocab size: provide token2id, tokens (vocab cover), or train_ids."
                )
            self.config.vocab_size = V
            print(
                f"{Colors.OKCYAN}[MODEL]{Colors.ENDC} Initializing NeuralFast model with vocab size {V}"
            )
            self.model = NeuralLanguageModel(config=self.config).to(self.device)

        epochs = self.epochs if None else epochs
        lr = self.lr if None else lr
        print(
            f"{Colors.OKBLUE}[INFO]{Colors.ENDC} Training for {epochs} epochs, lr={lr}"
        )

        # Try loading final model first
        if self.model is None:
            if not self.tokens:
                raise ValueError(
                    "Tokens must be set to initialize the model automatically."
                )
            print(
                f"{Colors.OKCYAN}[MODEL]{Colors.ENDC} Initializing NeuralFast model with vocab size {len(self.tokens)}"
            )
            self.config.vocab_size = len(self.tokens)
            self.model = NeuralLanguageModel(config=self.config).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

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
        train_losses_per_epoch, val_losses_per_epoch = [], []
        val_perplexities = []
        best_val_loss = float("inf")
        patience_counter = 0
        step = 0
        checkpoint_list = []

        for epoch in range(epochs):
            n_batches = len(self.train_ids) // self.batch_size
            epoch_loss = 0.0

            for _ in range(n_batches):
                X_batch, y_batch = self.get_batch(self.train_ids)
                loss = self.train_step(X_batch, y_batch)
                epoch_loss += loss

                if step % self.print_every == 0:
                    ram_mb, _ = get_proc_mem_mb()
                    print(
                        f"Epoch {epoch + 1}/{epochs} | Step {step} | Loss: {loss:.4f} | RAM: {ram_mb:.2f}mb ({ram_mb / 1024:.2f}Gb)"
                    )
                step += 1

            avg_epoch_loss = epoch_loss / n_batches
            train_losses_per_epoch.append(avg_epoch_loss)
            print(f"Epoch {epoch + 1}/{epochs} - Avg Train Loss: {avg_epoch_loss:.4f}")

            # ---------------- VALIDATION -------------------
            if self.valid_ids:
                X_val, y_val = self.get_batch(self.valid_ids)
                X_val = X_val.to(self.device)
                y_val = y_val.to(self.device)

                self.model.eval()
                with torch.no_grad():
                    _, val_loss_tensor = self.model.forward(X_val, targets=y_val)
                val_loss = val_loss_tensor.item()
                val_losses_per_epoch.append(val_loss)
                val_perplexities.append(np.exp(val_loss))

                print(
                    f"Validation Loss: {val_loss:.4f}, Perplexity: {val_perplexities[-1]:.4f}"
                )

                # ---------------- SAVE CHECKPOINT ----------------
                ckpt_name = f"epoch{epoch + 1}_val{val_loss:.4f}.pkl"
                ckpt_path = self._save_state(
                    subdir="checkpoints", filename=ckpt_name, final=final
                )
                checkpoint_list.append((val_loss, ckpt_path))
                # sort by validation loss
                checkpoint_list.sort(key=lambda x: x[0])

                # Keep only the best `max_checkpoints`
                if len(checkpoint_list) > max_checkpoints:
                    _, remove_ckpt = checkpoint_list.pop(-1)
                    if os.path.exists(remove_ckpt):
                        os.remove(remove_ckpt)

                # ---------------- SAVE BEST MODEL ----------------
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_state(
                        subdir="final", filename="best_model.pkl", final=final
                    )
                    print(
                        f"{Colors.OKGREEN}[OK]{Colors.ENDC} Best model updated at epoch {epoch + 1}"
                    )
                    patience_counter = 0
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

        # ---------------- PLOTTING -------------------
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
