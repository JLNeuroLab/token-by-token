import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import matplotlib.pyplot as plt
from llm_project.utils.debugg_utils import Colors
from llm_project.utils.file_manager import get_project_root, load_model, save_model


class NeuralTrainer:

    def __init__(self,
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
                 patience=3):

        self.config = config
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
        self.model_dir = os.path.join(self.root, "experiments", "models", "neural_ngrams")
        os.makedirs(self.model_dir, exist_ok=True)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.checkpoints = []

        self.id2token = None
        self.token2id = None

    # ------------------- BATCHES -------------------
    def get_batch(self, data_ids):
        X, y = [], []
        for _ in range(self.batch_size):
            start_idx = torch.randint(0, len(data_ids) - self.block_size - 1, (1,)).item()
            x_block = data_ids[start_idx : start_idx + self.block_size]
            y_block = data_ids[start_idx + 1 : start_idx + self.block_size + 1]
            X.append(x_block)
            y.append(y_block)
        return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    # ------------------- SAVE / LOAD -------------------
    def _save_state(self, subdir=None, filename="best_model.pkl", final=False):
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'id2token': self.id2token,
            'token2id': self.token2id
        }
        saved_path = save_model(
            state,
            root=self.root,
            category="models",
            subdir=subdir,
            filename=filename,
            final=final
        )
        print(f"[OK] Model saved to {saved_path}")
        return saved_path

    def _load_state(self, filename="best_model.pkl", subdir=None, final=False):
        state = load_model(
            root=self.root,
            filename=filename,
            category="models",
            subdir=subdir,
            final=final
        )
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.id2token = state.get('id2token', None)
        self.token2id = state.get('token2id', None)
        print(f"[OK] Model loaded from {filename}")
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
            for i in range(start_id,
                           min(start_id + self.batch_size * self.block_size, len(data_ids) - self.block_size),
                           self.block_size):
                X_batch.append(data_ids[i: i + self.block_size])
                y_batch.append(data_ids[i+1: i+1 + self.block_size])

            if not X_batch:
                continue

            X_batch = torch.tensor(X_batch, dtype=torch.long)
            y_batch = torch.tensor(y_batch, dtype=torch.long)
            _, loss = self.model.forward(X_batch, targets=y_batch)

            total_loss += loss.item() * X_batch.shape[0]
            total_tokens += X_batch.shape[0] * self.block_size

        avg_loss = total_loss / total_tokens
        return torch.exp(torch.tensor(avg_loss))

    # ------------------- PLOTTING -------------------
    def plot_perplexity(self, train_ids=None, val_ids=None, filename="perplexity_curve.png"):
        train_ids = train_ids or self.train_ids
        val_ids = val_ids or self.valid_ids

        train_ppl = self.batch_perplexity(train_ids) if train_ids else None
        val_ppl = self.batch_perplexity(val_ids) if val_ids else None

        fig, ax = plt.subplots(figsize=(8, 6))
        if train_ppl: ax.plot([train_ppl], label="Train Perplexity", color="blue", marker="o")
        if val_ppl: ax.plot([val_ppl], label="Validation Perplexity", color="red", marker="o")

        ax.set_xlabel("Step")
        ax.set_ylabel("Perplexity")
        ax.set_title("Perplexity over dataset")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        save_path = os.path.join(self.root, filename)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"{Colors.OKGREEN}[OK]{Colors.ENDC} Perplexity plot saved to {save_path}")

    # ------------------- TRAIN LOOP -------------------
    def train(self, force_retrain=False):
        # Try loading final model first
        if not force_retrain:
            try:
                self._load_state(filename="best_model.pkl", subdir="final", final=True)
                print("[INFO] Loaded existing final model.")
                return self.batch_perplexity(self.valid_ids) if self.valid_ids else None
            except FileNotFoundError:
                print("[INFO] No final model found, training from scratch.")

        best_val_loss = float("inf")
        patience_counter = 0
        val_perplexities = []

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            n_batches = len(self.train_ids) // self.batch_size
            for _ in range(n_batches):
                X_batch, y_batch = self.get_batch(self.train_ids)
                loss = self.train_step(X_batch, y_batch)
                epoch_loss += loss
            avg_epoch_loss = epoch_loss / n_batches
            print(f"Epoch {epoch}/{self.epochs} - Avg Loss: {avg_epoch_loss:.4f}")

            # Validation
            if self.valid_ids:
                val_ppl = self.batch_perplexity(self.valid_ids)
                val_perplexities.append(val_ppl.item())
                print(f"Validation Perplexity: {val_ppl:.4f}")

                # Save checkpoint
                self._save_state(subdir="checkpoints", filename=f"epoch{epoch}_val{val_ppl:.4f}.pkl")
                if len(self.checkpoints) > self.max_checkpoints:
                    self.checkpoints.pop(0)

                # Early stopping
                if val_ppl < best_val_loss:
                    best_val_loss = val_ppl
                    patience_counter = 0
                    self._save_state(subdir="final", filename="best_model.pkl", final=True)
                else:
                    patience_counter += 1
                    print(f"[INFO] Patience {patience_counter}/{self.patience}")
                    if patience_counter >= self.patience:
                        print(f"[INFO] Early stopping at epoch {epoch}")
                        break

        # Final plot
        self.plot_perplexity(train_ids=self.train_ids, val_ids=self.valid_ids)
        return val_perplexities
