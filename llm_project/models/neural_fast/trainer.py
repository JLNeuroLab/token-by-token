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
                    max_checkpoints=5
    ):

        self.config = config
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.max_checkpoints = max_checkpoints
        self.print_every = print_every
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

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.checkpoints = []

        self.model_dir = os.path.join(self.root, "experiments", "models", "neural_ngrams")
        os.makedirs(self.model_dir, exist_ok=True)


    def get_batch(self, data_ids):
        """
        Creates a batch of input sequences and corresponding target sequences
        from a list of token IDs.

        Args:
            data_ids (list[int] or torch.Tensor): sequence of integer token IDs.

        Returns:
            X (torch.LongTensor): shape (batch_size, block_size)
            y (torch.LongTensor): shape (batch_size, block_size)
        """
        X, y = [], []
        for _ in range(self.batch_size):
            start_idx = torch.randint(0, len(data_ids) - self.block_size - 1, (1,)).item()
            x_block = data_ids[start_idx : start_idx + self.block_size]
            y_block = data_ids[start_idx + 1 : start_idx + self.block_size + 1]

            X.append(x_block)
            y.append(y_block)

        X = torch.tensor(X, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)

        return X, y
    
    # ------------------- STATE -------------------
    def _state_dict(self):
        return {
            "vocab_size": len(self.tokens),
            "embd_dim": self.embd_dim,
            "params": {
                "embeddings": self.model.embeddings.detach().cpu().numpy(),
                "W": self.model.W.detach().cpu().numpy(),
                "b": self.model.b.detach().cpu().numpy()
            },
            "id2token": self.id2token,
            "token2id": self.token2id
        }
    
    def _save_state(self, subdir=None, filename="best_model.pkl", final=False):
        """
        Saves the full model state (parameters, optimizer, vocab) using pickle
        in a way compatible with the provided save_model function.
        """
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'id2token': getattr(self, "id2token", None),
            'token2id': getattr(self, "token2id", None)
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
        """
        Loads a saved model (parameters, optimizer, vocab) using pickle
        in a way compatible with load_model.
        """
        state = load_model(
            root=self.root,
            filename=filename,
            category="models",
            subdir=subdir,
            final=final
        )

        # Load parameters into the model
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.id2token = state.get('id2token', None)
        self.token2id = state.get('token2id', None)

        print(f"[OK] Model loaded from {filename}")
        return self.model

    def train_step(self, X_batch, y_batch):
        
        self.model.train()
        logits, loss = self.model.forward(X_batch, targets=y_batch)
        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    @torch.no_grad
    def _batch_perplexity(self, data_ids):
        self.model.evel()
        total_loss = 0.0
        total_tokens = 0

        for start_id in range(0, len(data_ids) - self.block_size, self.block_size):
            X_batch = []
            y_batch = []
            for i in range(start_id,
                           min(start_id + self.batch_size * self.block_size, len(data_ids) - self.block_size),
                           self.block_size
                           ):
                X_batch.append(data_ids[i: i + self.block_size])
                y_batch.append(data_ids[i+1: i+1 + self.block_size])

            if len(X_batch) == 0:
                continue

            X_batch = torch.tensor(X_batch, dtype=torch.long)
            y_batch = torch.tensor(y_batch, dtype=torch.long)

            logits, loss = self.model.forward(X_batch, y_batch)
            total_loss += loss.item() * X_batch.shape[0]
            total_tokens += X_batch.shape[0] * self.block_size
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(avg_loss)
        
        return perplexity
    
    def plot_perplexity(self, train_ids=None, val_ids=None, filename="perplexity_curve.png"):
        train_ids = train_ids or self.train_ids
        val_ids = val_ids or self.val_ids

        train_ppl = self._batch_perplexity(train_ids) if train_ids else None
        val_ppl = self._batch_perplexity(val_ids) if val_ids else None

        fig, ax = plt.subplots(figsize=(8, 6))
        if train_ppl:
            ax.plot([train_ppl], label="Train Perplexity", color="blue", marker="o")
        if val_ppl:
            ax.plot([val_ppl], label="Validation Perplexity", color="red", marker="o")

        ax.set_xlabel("Step")
        ax.set_ylabel("Perplexity")
        ax.set_title("Perplexity over dataset")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

        save_path = os.path.join(self.root, filename)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"{Colors.OKGREEN}[OK]{Colors.ENDC} Perplexity plot saved to {save_path}")

    def train(self):

        perplexity = []

        # --- TRAINING LOOP ---
        for epoch in range(1, self.epochs + 1): 
            epoch_loss = 0.0
            n_batches = len(self.train_ids) // self.batch_size

            for _ in range(n_batches):
                X_batch, y_batch = self.get_batch(self.train_ids)
                loss = self.train_step(X_batch=X_batch, y_batch=y_batch)
                epoch_loss += loss

            avg_epoch_loss = epoch_loss / n_batches
            print(f"Epoch {epoch}/{self.epochs} - Avg Loss: {avg_epoch_loss:.4f}")

            # --- VALIDATON ---
            if self.valid_ids:
                val_ppl = self._batch_perplexity(self.valid_ids)
                perplexity.append(val_ppl)
                print(f"Validation Perplexity: {val_ppl:.4f}")

        # Plot final perplexity
        self.plot_perplexity(train_ids=self.train_ids, val_ids=self.val_ids)

        return perplexity
        