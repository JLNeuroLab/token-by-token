# -------------- Imports for GPT ----------------
from llm_project.models.configs.configs import GPTConfig
from llm_project.models.gpt.model import GPT
from llm_project.models.gpt.generator import Generator
import torch
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch import nn
import json
from llm_project.utils.debugg_utils import print_resource_usage, Colors

# -------------- Imports for file handling ----------------
from llm_project.utils.dataloader import load_shakespeare
from llm_project.utils.file_manager import (
    save_model,
    load_model,
    get_model_path,
    get_project_root,
)

# -------------- Imports Misc ----------------
import os
import psutil

# Reproducibility block


def set_seed(seed: int = 1717):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# def main(max_iters, embd_dim, n_layer, dropout, max_k, device, force_retrain=False):


class GPTTrainConfig:
    # --- model ---
    embd_dim: int = 384
    n_layer: int = 4
    n_head: int = 6
    block_size: int = 64
    dropout: float = 0.2

    # --- training knobs (constants) ---
    batch_size: int = 32  # BATCH_SIZE
    max_iters: int = 5000  # MAX_ITERS
    eval_interval: int = 500  # EVAL_INTERVAL
    eval_iters: int = 200  # EVAL_ITERS
    learning_rate: float = 3e-4  # LEARNING_RATE
    weight_decay: float = 0.0
    grad_clip: [float] = None  # e.g. 1.0 or None
    log_interval: int = 500  # log_interval

    # --- provenance / IO ---
    device: [str] = None  # "cuda"|"mps"|"cpu"|None(auto)
    pipeline_id: [str] = None  # e.g. "shakespeare-k2000"
    max_k: [int] = None  # BPE merges (for metadata only)
    vocab_size_hint: [int] = None

    model_name: str = "gpt_shakespeare"
    save_dir: str = "experiments/saved_models/gpt"

    # --- preview generation ---
    preview_prompt: str = "ROMEO:"
    preview_tokens: int = 100
    preview_top_k: int = 40
    preview_temperature: float = 0.9
    save_best_only: bool = False


class GptTrainer:
    """
    NGram-style GPT trainer:
      - __init__(config, model, tokens, root=None, k=None)
      - train(force_retrain=False, train_limit=None, valid_limit=None) -> returns model
      - _state_dict/_save_state/_load_state
      - compute_perplexity on a token list
      - plot_loss_curve
    Expects `tokens` to be a dict with keys {"train","validation"} of token-id lists.
    """

    def __init__(self, config, tokens, model, root=None, k=None):
        self.model = model
        self.tokens = tokens
        self.config = config
        if k is None:
            raise ValueError("k must be selected (use max_k from the BPE tokenizer)")
        self.k = int(k)
        self.root = root or get_project_root()
        # device + AMP
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # set False to disable AMP entirely
        self.use_amp = self.device.type == "cuda"
        self.device_str = self.device.type  # "cuda" | "cpu" | "mps"

        self.batch_size = int(getattr(config, "batch_size", 32))
        self.block_size = int(getattr(config, "block_size", 64))
        self.max_iters = int(getattr(config, "max_iters", 5000))
        self.eval_interval = int(getattr(config, "eval_interval", 500))
        self.eval_iters = int(getattr(config, "eval_iters", 200))
        self.learning_rate = float(getattr(config, "learning_rate", 3e-4))
        self.log_interval = int(getattr(config, "log_interval", 500))
        self.weight_decay = float(getattr(config, "weight_decay", 0.0))
        self.grad_clip = getattr(config, "grad_clip", None)

        # convenience tensors kept on CPU; batches will be moved to self.device
        self.train_ids = torch.tensor(self.tokens["train"], dtype=torch.long)
        self.val_ids = torch.tensor(self.tokens["validation"], dtype=torch.long)

        # track which training positions were sampled (optional)
        self.train_range = max(0, len(self.train_ids) - self.block_size - 1)
        self.coverage = torch.zeros(train_range, dtype=torch.bool)

        # build/save paths like N-gram does (under experiments/saved_models/gpt/)
        self.model_folder = get_model_path(self.root, "saved_models", subdir="gpt")
        os.makedirs(self.model_folder, exist_ok=True)
        # filename embeds k (like ngram) + key arch bits + n_layer
        ed = int(getattr(config, "embd_dim", 384))
        nl = int(getattr(config, "layer_dim", getattr(config, "n_layer", 4)))
        self.model_fname = f"gpt_model_k{self.k}_ed{ed}_nl{nl}.pth"
        self.model_path = os.path.join(self.model_folder, self.model_fname)

        # plots folder
        self.plots_folder = os.path.join(self.root, "experiments", "plots", "gpt")
        os.makedirs(self.plots_folder, exist_ok=True)

        # ensure we have/construct a model object
        if self.model is None:
            self._build_model()

    # ---------- state/IO ----------

    def _state_dict(self):
        if self.model is None:
            raise ValueError("Model not initialized")
        # for GPT we store the torch state_dict plus a small meta
        return {
            "state_dict": self.model.state_dict(),
            "meta": {
                "k": self.k,
                "embd_dim": getattr(self.config, "embd_dim", None),
                "layer_dim": getattr(self.config, "layer_dim", None),
                "n_heads": getattr(self.config, "n_heads", None),
                "block_size": getattr(self.config, "block_size", None),
                "dropout": getattr(self.config, "dropout", None),
                "device": self.device_str,
                "vocab_size": int(self._vocab_size()),
            },
        }

    def _save_state(self, subdir="gpt", filename=None):
        # keep the same interface: save under experiments/saved_models/gpt/<filename>
        filename = filename or self.model_fname
        # save weights as .pth and the meta as .json next to it
        torch.save(self.model.state_dict(), os.path.join(self.model_folder, filename))
        meta = self._state_dict()["meta"]
        with open(
            os.path.join(self.model_folder, filename.replace(".pth", ".json")),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(meta, f, indent=2)

    def _load_state(self, file_path):
        # file_path should be a .pth (relative or absolute)
        full_path = (
            file_path
            if os.path.isabs(file_path)
            else os.path.join(self.model_folder, file_path)
        )
        self._build_model()  # ensure model is constructed
        self.model.load_state_dict(torch.load(full_path, map_location=self.device))
        return self.model

    # ---------- model stuff ----------
    def _vocab_size(self):
        tok2id = getattr(self, "tokenizer_token_to_id", None)
        # we don't store a tokenizer here; derive vocab_size from tokens set if needed
        # BUT: you should pass vocab via config or attach tokenizer externally before calling build if you prefer.
        # safest: infer from max id + 1 in training tokens
        vs = int(torch.max(self.train_ids).item()) + 1
        return vs

    def _build_model(self):
        # Map your config (configs/configs.py) → model config
        vocab_size = self._vocab_size()
        embd_dim = int(getattr(self.config, "embd_dim", 384))
        layer_dim = int(
            getattr(self.config, "layer_dim", 4)
        )  # number of transformer blocks
        n_heads = int(getattr(self.config, "n_heads", 6))
        dropout = float(getattr(self.config, "dropout", 0.2))
        block_size = int(self.block_size)

        # guard: multi-head attn requires embd_dim % n_heads == 0
        if embd_dim % n_heads != 0:
            raise ValueError("embd_dim must be divisible by n_heads")

        # Build project’s GPTConfig (the param names must match class)
        gcfg = GptConfig(
            vocab_size=vocab_size,
            n_heads=n_heads,
            layer_dim=layer_dim,
            embd_dim=embd_dim,
            block_size=block_size,
            dropout=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            resid_pdrop=dropout,
            device=self.device_str,
        )
        # compat aliases in case model.py reads n_head/n_layer
        if not hasattr(gcfg, "n_head"):  # avoid overwriting if already present
            setattr(gcfg, "n_head", n_heads)
        if not hasattr(gcfg, "n_layer"):
            setattr(gcfg, "n_layer", layer_dim)

        self.model = GPT(gcfg).to(self.device)

    def _get_batch(self, split: str):
        data = self.train_ids if split == "train" else self.val_ids
        max_start = len(data) - (self.block_size + 1)
        if max_start <= 0:
            raise RuntimeError(
                f"Not enough tokens for block_size={self.block_size} in split={split} (len={len(data)})"
            )
        ix = torch.randint(0, max_start, (self.batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])
        if split == "train" and self.train_range > 0:
            # mark sampled starts as covered (clamp to valid range)
            start_ix = ix.clamp_min(0).clamp_max(train_range - 1)
            self.coverage[start_ix] = True
        return x.to(self.device), y.to(self.device)

    # ---------- GPT API part ----------

    def train(
        self,
        force_retrain=False,
        train_limit=None,
        valid_limit=None,
    ):
        """
        Trains GPT on the provided token splits.
        - If a checkpoint under this k/arch combo exists and force_retrain=False, it loads and returns it.
        - Otherwise trains and saves.
        """
        ram_log = []
        tokens_per_iter = self.batch_size * self.block_size  # rough estimate
        tokens_seen = 0

        if self.tokens is None:
            raise ValueError(
                "Tokens must be provided externally; tokenizer is decoupled."
            )

        # trim tokens for quick runs
        if train_limit:
            self.train_ids = self.train_ids[:train_limit]
        if valid_limit:
            self.val_ids = self.val_ids[:valid_limit]

        if (not force_retrain) and os.path.exists(self.model_path):
            print("--- Loading pre-trained GPT ---")
            self._load_state(self.model_path)
            return self.model

        print("--- Training GPT model ---")
        model = self.model
        opt = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        steps, train_curve, val_curve = [], [], []
        best_val = float("inf")

        for it in range(self.max_iters):
            # periodic eval
            if it % self.eval_interval == 0:
                losses = self._estimate_loss(self.eval_iters)
                tr, vl = losses["train"], losses["val"]
                ppl = math.exp(vl)
                print(
                    f"[step {it:>5}] train {tr:.4f} | val {vl:.4f} | val ppl {ppl:.2f}"
                )
                steps.append(it)
                train_curve.append(tr)
                val_curve.append(vl)
                if vl < best_val:
                    best_val = vl
                    self._save_state(subdir="gpt", filename=self.model_fname)
                    # also write a small meta JSON with latest val
                    meta = self._state_dict()["meta"]
                    meta.update({"val_loss": float(vl), "step": int(it)})
                    with open(
                        self.model_path.replace(".pth", ".json"), "w", encoding="utf-8"
                    ) as f:
                        json.dump(meta, f, indent=2)

            # one step
            X, Y = self._get_batch("train")
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = model(X)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if self.grad_clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            scaler.step(opt)
            scaler.update()
            tokens_seen += tokens_per_iter

            if self.log_interval and (it % self.log_interval == 0):
                print(f"[train] step {it:>5} | loss {loss.item():.4f}")
                print_resource_usage(it)
                # Append ram log
                ram_mb = psutil.virtual_memory().used / 1024**2
                ram_log.append(ram_mb)

        # final save
        self._save_state(subdir="gpt", filename=self.model_fname)
        print(f"{Colors.OKGREEN}[OK]{Colors.ENDC} Model saved to: {self.model_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"{Colors.FAIL}[FAIL]{Colors.ENDC}Model not found at {self.model_path}"
            )

        # plot and return model
        self.plot_loss_curve(steps, train_curve, val_curve)
        self.plot_val_perplexity(steps, val_curve)

        # optional RAM plot
        if ram_log:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(ram_log, marker=".")
            ax.set_title("RAM Usage (MB)")
            ax.set_xlabel("Logged Steps")
            ax.set_ylabel("MB")
            ax.grid(True, alpha=0.4)
            ram_path = os.path.join(self.plots_folder, f"ram_usage_k{self.k}.png")
            fig.tight_layout()
            fig.savefig(ram_path)
            plt.close(fig)
            print(f"RAM plot saved to {ram_path}")

        return self.model

    def plot_val_perplexity(self, steps, val_losses):
        if not steps or not val_losses:
            print("No eval points; skipping perplexity plot")
            return
        val_ppl = [math.exp(float(v)) for v in val_losses]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(steps, val_ppl, marker="^", label="Validation Perplexity")
        ax.set_title("Validation Perplexity per Interval")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Perplexity")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        out = os.path.join(self.plots_folder, f"val_perplexity_k{self.k}.png")
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        print(f"Perplexity plot saved to {out}")

    def _estimate_loss(self, iters: int):
        out = {}
        self.model.eval()
        with torch.no_grad():
            for sp in ("train", "val"):
                losses = []
                for _ in range(iters):
                    X, Y = self._get_batch(sp)
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        logits = self.model(X)
                        loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)), Y.view(-1)
                        )
                    losses.append(float(loss.item()))
                out[sp] = sum(losses) / len(losses)
        self.model.train()
        return out

        def compute_perplexity(self, test_tokens, batch_size=None):
            """Compute perplexity on a custom token list (e.g., held-out test)."""
            if not test_tokens:
                return float("inf")
            bs = int(batch_size or self.batch_size)
            data = torch.tensor(test_tokens, dtype=torch.long)
            max_start = len(data) - (self.block_size + 1)
            if max_start <= 0:
                return float("inf")
            losses = []
            self.model.eval()
            with torch.no_grad():
                # iterate in chunks to avoid OOM on very long sequences
                i = 0
                while i < max_start:
                    take = min(bs, max_start - i)
                    ix = torch.arange(i, i + take)
                    X = torch.stack([data[j : j + self.block_size] for j in ix])
                    Y = torch.stack([data[j + 1 : j + self.block_size + 1] for j in ix])
                    X, Y = X.to(self.device), Y.to(self.device)
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        logits = self.model(X)
                        loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)), Y.view(-1)
                        )
                    losses.append(float(loss.item()))
                    i += take

        self.model.train()
        mean_loss = sum(losses) / max(1, len(losses))
        return math.exp(mean_loss)

    def plot_loss_curve(self, steps, train, val):
        if not steps:
            print("No eval points; skipping plot")
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps, train, marker="o", label="Train loss")
        ax.plot(steps, val, marker="s", label="Val loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("GPT Loss Curve")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        save_path = os.path.join(self.plots_folder, f"gpt_loss_k{self.k}.png")
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Plot saved to {save_path}")

    # Optional: quick sample stage
    def generate_preview(
        self,
        prompt: str = "ROMEO:",
        max_new_tokens: int = 100,
        top_k: int = 40,
        temperature: float = 0.9,
    ):
        gen = Generator(
            model=self.model,
            tokenizer=None,  # generator can accept raw IDs; we pass IDs directly
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            temperature=temperature,
            mode="top_k",
            eos_token_id=None,
        )
        ctx_ids = self._encode_prompt(prompt)
        out_ids = gen.generate(ctx_ids)
        # to decode, need the tokenizer the tokens came from (pipeline should handle decode/print)
        return out_ids

    def _encode_prompt(self, prompt: str):
        # quick & dirty: use training vocabulary to map unknown chars to 0
        # (better: pass the tokenizer in from pipeline if you want decoding too)
        # Here we just start from an empty context for consistency
        return []


if __name__ == "__main__":
    """
    Standalone GPT smoke test using its own BPE (mirrors the N-gram demo).
    For the real experiments, prefer running via the pipeline so all models
    share the same tokenizer. This is just a self-contained check.
    """
    import os
    from llm_project.bpe.bytepair_encoding import BPE
    from llm_project.utils.file_manager import save_tokenizer

    # Repro & colors already in this file
    set_seed(3108)

    # --- Paths ---
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    datapath = os.path.join(project_root, "data", "raw", "Shakespeare_clean_full.txt")

    # --- BPE setup ---
    max_k = 1000
    bpe = BPE(max_k=max_k, datapath=datapath)

    # Load + normalize, split (10% test)
    _ = bpe.load_and_normalize()
    test_text, train_text = bpe.split_train_test()
    bpe.train_text = train_text[:10_000]
    bpe.test_text = test_text[:2_000]

    # Train BPE on the training slice
    bpe.text = bpe.train_text
    print("\n--- Training BPE ---\n")
    bpe.BPE_encoder()

    # Save tokenizer (merges)
    save_tokenizer(bpe, project_root, f"BPE_merges_k{max_k}.pkl", category="tokenizers")
    print(f"{Colors.OKGREEN}[OK]{Colors.ENDC} BPE tokenizer saved")

    # Tokenize splits
    train_tokens = bpe.tokens
    valid_text = train_text[10_000:12_000]  # small holdout slice
    valid_tokens = bpe.BPE_segmenter(valid_text)
    test_tokens = bpe.BPE_segmenter(bpe.test_text)

    print(f"Train tokens: {len(train_tokens)}")
    print(f"Valid tokens: {len(valid_tokens)}")
    print(f"Test  tokens: {len(test_tokens)}\n")

    # --- Trainer config (small/fast) ---
    cfg = GPTTrainConfig()
    cfg.embd_dim = 128
    cfg.n_layer = 2
    cfg.n_head = 4
    cfg.block_size = 64
    cfg.dropout = 0.2

    cfg.batch_size = 32
    cfg.max_iters = 200
    cfg.eval_interval = 50
    cfg.eval_iters = 50
    cfg.learning_rate = 3e-4
    cfg.log_interval = 50
    cfg.device = None  # auto: cuda->mps->cpu

    # --- Train GPT ---
    tokens_dict = {"train": train_tokens, "validation": valid_tokens}
    trainer = GptTrainer(config=cfg, tokens=tokens_dict, model=None, k=max_k)
    _ = trainer.train(force_retrain=False)

    # --- Perplexity on test set ---
    if len(test_tokens) >= (cfg.block_size + 1):
        ppl = trainer.compute_perplexity(test_tokens)
        print(f"\nPerplexity on test set: {ppl:.2f}\n")
    else:
        print(
            f"{Colors.WARNING}[!!!]{Colors.ENDC} Skipping test perplexity (test set too short for block_size={cfg.block_size})."
        )


#     config = GPTConfig(vocab_size, BLOCK_SIZE)
#     model = GPT(config).to(device)
#
#     optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
#     scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
#
#     train_losses, val_losses = [], []
#     eval_steps = []
#
#     best_val = float("inf")
#     best_path = os.path.join("saved_models", "gpt_shakespeare-best.pth")
#     os.makedirs("saved_models", exist_ok=True)
#
#     @torch.no_grad()
#     def _estimate_loss(self, split="val", iters=100):
#         out = {}
#         model.eval()
#         for split in ["train", "val"]:
#             losses = torch.zeros(EVAL_ITERS)
#             for k in range(EVAL_ITERS):
#                 X, Y = get_batch(split)
#                 with torch.cuda.amp.autocast(enabled=use_amp):
#                     logits = model(X)
#                     loss = F.cross_entropy(
#                         logits.view(-1, logits.size(-1)), Y.view(-1))
#                 losses[k] = loss.item()
#             out[split] = losses.mean()
#         model.train()
#         return out
#
#     print("Starting training...")
#     tokens_seen = 0
#     ram_log = []
#     for iter_num in range(MAX_ITERS):
#         if iter_num % EVAL_INTERVAL == 0 or iter_num == MAX_ITERS - 1:
#             losses = estimate_loss()
#             print(
#                 f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}",
#                 flush=True,
#             )
#             train_losses.append(losses["train"])
#             val_losses.append(losses["val"])
#             eval_steps.append(iter_num)
#
#             # Save best Val Loss
#             cur_val = float(losses["val"])
#             if cur_val < best_val:
#                 best_val = cur_val
#                 torch.save(model.state_dict(), best_path)
#                 print(
#                     f"[checkpoint] new best val loss {best_val:.4f} saved -> {best_path}"
#                 )
#
#             # Progress/coverage
#             epoch_equiv = tokens_seen / max(1, DATASET_TOKENS)
#             cov = coverage.float().mean().item() * 100.0
#             print(
#                 f"[progress] ~epochs={epoch_equiv:.2f}  coverage≈{cov:.1f}%  "
#                 f"steps/epoch≈{EST_STEPS_PER_EPOCH}"
#             )
#
#         xb, yb = get_batch("train")
#         with torch.cuda.amp.autocast(enabled=use_amp):
#             logits = model(xb)
#             loss = F.cross_entropy(
#                 logits.view(-1, logits.size(-1)), yb.view(-1))
#
#         optimizer.zero_grad(set_to_none=True)
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         tokens_seen += TOKENS_PER_ITER
#
#         if iter_num > 0 and iter_num % log_interval == 0:
#             ram = psutil.virtual_memory().used / 1024**2
#             ram_log.append(ram)
#             print_resource_usage_compat(step=iter_num)
#
#     print("Training finished.")
#
#     # RAM plot
#     plt.plot(ram_log, label="RAM Usage (MB)")
#     plt.title("RAM Usage over Training")
#     plt.xlabel("Logged Steps")
#     plt.ylabel("RAM (MB)")
#     plt.grid(True)
#     plt.legend()
#     os.makedirs(os.path.join("experiments", "saved_models"), exist_ok=True)
#     plt.savefig(os.path.join("experiments", "saved_models", "ram_usage.png"))
#     plt.close()
#
#     # Save the Model
#     save_path = os.path.join("saved_models")
#     os.makedirs(save_path, exist_ok=True)
#
#     debug_suffix = "-debug" if os.getenv("TTB_DEBUG") == "1" else ""
#     model_save_path = os.path.join(
#         save_path, f"gpt_shakespeare{debug_suffix}.pth")
#     print(f"Saving model to {model_save_path}")
#     torch.save(model.state_dict(), model_save_path)
#
#     # Generation Preview
#     start_text = "ROMEO:"
#     context_ids = tokenizer.encode(start_text)
#     generator = Generator(
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens=100,
#         temperature=0.9,
#         top_k=40,
#         eos_token_id=None,
#         mode="top_k",
#     )
#     output_ids = generator.generate(context_ids)
#     generated_text = tokenizer.decode(output_ids)
#
#     print("\n=== Generation Preview ===")
#     print(generated_text)
#     self.plot_val_perplexity(steps, val_curve)

#
#     preview_path = os.path.join("experiments", "plots", "gpt")
#     os.makedirs(preview_path, exist_ok=True)
#     with open(
#         os.path.join(preview_path, "generation_preview.txt"), "w", encoding="utf-8"
#     ) as f:
#         f.write(generated_text)
#
#     # Plots
#
#     # Create folder for plots
#     plot_folder = os.path.join("experiments", "plots", "gpt")
#     os.makedirs(plot_folder, exist_ok=True)
#
#     # Safety: lengths must match
#     assert len(eval_steps) == len(train_losses) == len(val_losses), (
#         f"Lengths mismatch: steps={len(eval_steps)} "
#         f"train={len(train_losses)} val={len(val_losses)}"
#     )
#
#     # 1) Loss curves
#     plt.figure(figsize=(8, 6))
#     plt.plot(
#         eval_steps, [float(t) for t in train_losses], label="Train Loss", marker="o"
#     )
#     plt.plot(eval_steps, [float(v)
#              for v in val_losses], label="Val Loss", marker="s")
#     plt.title("Train and Validation Loss per Interval")
#     plt.xlabel("Iterations")
#     plt.ylabel("Loss")
#     plt.grid(True, linestyle="--", alpha=0.6)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(plot_folder, "loss_curve.png"))
#     plt.close()
#
#     # 2) Perplexity
#     # CHANGE: avoid wrapping tensors inside torch.tensor(...) (causes warning)
#
#     def _to_float(x):  # ADD: works for both torch.Tensor and float
#         return x.item() if hasattr(x, "item") else float(x)
#
#     val_perplexities = [math.exp(_to_float(vl)) for vl in val_losses]  # CHANGE
#     plt.figure(figsize=(8, 6))
#     plt.plot(
#         eval_steps, val_perplexities, label="Validation Perplexity", marker="^"
#     )  # keep eval_steps on x
#     plt.title("Validation Perplexity per Interval")
#     plt.xlabel("Iterations")
#     plt.ylabel("Perplexity")
#     plt.grid(True, linestyle="--", alpha=0.6)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(plot_folder, "val_perplexity.png"))
#     plt.close()
#
#
# if __name__ == "__main__":
#     os.environ.setdefault("TTB_DEBUG", "1")
#     set_seed(3108)
#
#     main(
#         max_iters=200,
#         embd_dim=128,
#         n_layer=2,
#         dropout=0.2,
#         max_k=1000,
#         device="cuda",
#         force_retrain=False,  # keep False so you reuse cached BPE
#     )
