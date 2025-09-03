import os
import torch
import math
from torch.nn import functional as F
import psutil
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from llm_project.bpe.bytepair_encoding import BPE
from llm_project.models.gpt.model import GPT
from llm_project.utils.file_manager import save_item, load_item
from llm_project.utils.debugg_utils import print_resource_usage


def set_seed(seed: int = 1717):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(
    max_iters,
    embd_dim,
    n_layer,
    dropout,
    max_k,
    device,
    force_retrain=False,
    learning_rate=3e-4,
    block_size=64,
    batch_size=32,
):
    # --- Configuration ---
    BATCH_SIZE = batch_size
    BLOCK_SIZE = block_size
    MAX_ITERS = max_iters
    EVAL_INTERVAL = 100
    LEARNING_RATE = learning_rate
    EVAL_ITERS = 200
    log_interval = 50

    _pru = print_resource_usage

    def print_resource_usage_compat(step: int) -> None:
        try:
            return _pru(step=step)
        except TypeError:
            try:
                return _pru(None, step)
            except Exception:
                ram_mb = psutil.virtual_memory().used / 1024**2
                cpu = psutil.cpu_percent(interval=0.0)
                disk = psutil.disk_usage(os.getcwd()).percent
                tqdm.write(
                    f"[DEBUG] step {step:>7} || RAM {ram_mb:>8.2f} MB || CPU {cpu:>5.1f}% || Disk {disk:>5.1f}%"
                )

    if isinstance(device, str) and device in {"cuda", "mps", "cpu"}:
        dev_str = device
    else:
        dev_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {dev_str}", flush=True)
    device = torch.device(dev_str)
    use_amp = device.type == "cuda"

    datapath = os.path.join("data", "raw", "Shakespeare_clean_full.txt")
    with open(datapath, "r", encoding="utf-8") as f:
        text = f.read()

    print("Training tokenizer...", flush=True)
    tokenizer = BPE(data_path=None, text=text, max_k=max_k)

    project_root = os.getcwd()
    bpe_results_folder = os.path.join("experiments", "bpe_results", "train_results")
    os.makedirs(os.path.join(project_root, bpe_results_folder), exist_ok=True)
    merges_file = "train_bpe_merges.pkl"
    vocab_file = "train_final_vocab.pkl"
    merges_path = os.path.join(project_root, bpe_results_folder, merges_file)
    vocab_path = os.path.join(project_root, bpe_results_folder, vocab_file)
    merges_exist = os.path.exists(merges_path)
    vocab_exist = os.path.exists(vocab_path)

    tokenizer_full_file = "tokenizer_full.pkl"
    tokenizer_full_path = os.path.join(
        project_root, bpe_results_folder, tokenizer_full_file
    )

    tokenizer_full_file = "tokenizer_full.pkl"
    tokenizer_full_path = os.path.join(
        project_root, bpe_results_folder, tokenizer_full_file
    )

    if force_retrain or not os.path.exists(tokenizer_full_path):
        print(">>Training tokenizer from scratch...", flush=True)
        tokenizer.BPE_encoder()
        tokenizer.build_token_mappings()  # ← ensure mappings are built before save
    else:
        print(f">>Loaded FULL tokenizer from {tokenizer_full_path}", flush=True)
        tokenizer = load_item(bpe_results_folder, tokenizer_full_file)
        tokenizer.build_token_mappings()  # ← re-build after load just to be safe

    vocab_size = len(tokenizer.token_to_id)
    print(f"Vocabulary size: {vocab_size}", flush=True)

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    DATASET_TOKENS = len(train_data) - 1
    TOKENS_PER_ITER = BATCH_SIZE * BLOCK_SIZE
    EST_STEPS_PER_EPOCH = (
        math.ceil(DATASET_TOKENS / TOKENS_PER_ITER) if TOKENS_PER_ITER > 0 else 0
    )
    print(
        f"[data] train tokens={DATASET_TOKENS:,} | tokens/iter={TOKENS_PER_ITER} | ~steps/epoch: {EST_STEPS_PER_EPOCH}",
        flush=True,
    )

    coverage = torch.zeros(len(train_data) - BLOCK_SIZE, dtype=torch.bool)

    def get_batch(split):
        data_split = train_data if split == "train" else val_data
        ix = torch.randint(len(data_split) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([data_split[i : i + BLOCK_SIZE] for i in ix])
        y = torch.stack([data_split[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
        x, y = x.to(device), y.to(device)
        if split == "train":
            coverage[ix] = True
        return x, y

    class GPTConfig:
        def __init__(self, vocab_size, block_size):
            self.vocab_size = vocab_size
            self.block_size = block_size
            self.embd_dim = embd_dim
            self.n_layer = n_layer
            self.n_head = 4
            self.dropout = dropout
            self.embd_pdrop = dropout
            self.attn_pdrop = dropout
            self.resid_pdrop = dropout

    config = GPTConfig(vocab_size, BLOCK_SIZE)
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    train_losses, val_losses, eval_steps = [], [], []
    best_val = float("inf")
    best_path = os.path.join("experiments", "saved_models", f"gpt_k{max_k}.pth")
    os.makedirs(os.path.dirname(best_path), exist_ok=True)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(EVAL_ITERS)
            for k in range(EVAL_ITERS):
                X, Y = get_batch(split)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(X)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    print("Starting training...", flush=True)
    tokens_seen = 0
    # =================== START OF THE FIX ===================
    # Added the missing range() function here
    for iter_num in tqdm(range(MAX_ITERS), desc="Training GPT"):
        # ==================== END OF THE FIX ====================
        if iter_num % EVAL_INTERVAL == 0 or iter_num == MAX_ITERS - 1:
            losses = estimate_loss()
            tqdm.write(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            train_losses.append(losses["train"])
            val_losses.append(losses["val"])
            eval_steps.append(iter_num)

            epoch_equiv = tokens_seen / max(1, DATASET_TOKENS)
            cov = coverage.float().mean().item() * 100.0
            # Replaced special character '≈' with '~' for Windows compatibility
            tqdm.write(
                f"[progress] ~epochs={epoch_equiv:.2f}  coverage~{cov:.1f}%  steps/epoch~{EST_STEPS_PER_EPOCH}"
            )

            if float(losses["val"]) < best_val:
                best_val = float(losses["val"])
                torch.save(model.state_dict(), best_path)
                tqdm.write(
                    f"[checkpoint] new best val loss {best_val:.4f} saved -> {best_path}"
                )

        xb, yb = get_batch("train")
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        tokens_seen += TOKENS_PER_ITER
        if iter_num > 0 and iter_num % log_interval == 0:
            print_resource_usage_compat(step=iter_num)

            # 2125
    print("Training finished.", flush=True)
    save_path = os.path.join("experiments", "saved_models")
    os.makedirs(save_path, exist_ok=True)
    model_save_path = os.path.join(save_path, f"gpt_final_k{max_k}.pt")
    print(f"Saving model to {model_save_path}", flush=True)
    torch.save(model.state_dict(), model_save_path)
    tokenizer_vocab_path = os.path.join(save_path, f"gpt_vocab_k{max_k}.pkl")
    tokenizer_merges_path = os.path.join(save_path, f"gpt_merges_k{max_k}.pkl")
    save_item(tokenizer.vocab, save_path, os.path.basename(tokenizer_vocab_path))
    save_item(tokenizer.merges, save_path, os.path.basename(tokenizer_merges_path))
    print(f"Saved frozen tokenizer vocab and merges for k={max_k}", flush=True)

    print(f"Saving model to {model_save_path}", flush=True)
    torch.save(model.state_dict(), model_save_path)

    # --- SAVING BPE STATE SOLUTION
    # Save the tokenizer state alongside the model for perfect consistency
    tokenizer_vocab_path = os.path.join(save_path, f"gpt_vocab_k{max_k}.pkl")
    tokenizer_merges_path = os.path.join(save_path, f"gpt_merges_k{max_k}.pkl")
    save_item(
        tokenizer.vocab,
        os.path.dirname(tokenizer_vocab_path),
        os.path.basename(tokenizer_vocab_path),
    )
    save_item(
        tokenizer.merges,
        os.path.dirname(tokenizer_merges_path),
        os.path.basename(tokenizer_merges_path),
    )
    print(f"Saved tokenizer vocab and merges for k={max_k}", flush=True)

    # Plotting and saving results
    plot_folder = os.path.join("experiments", "plots", "gpt")
    os.makedirs(plot_folder, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(
        eval_steps, [float(t) for t in train_losses], label="Train Loss", marker="o"
    )
    plt.plot(eval_steps, [float(v) for v in val_losses], label="Val Loss", marker="s")
    plt.title("Train and Validation Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, "loss_curve.png"))
    plt.close()

    val_perplexities = [math.exp(v) for v in val_losses if v is not None]
    plt.figure(figsize=(8, 6))
    plt.plot(eval_steps, val_perplexities, label="Validation Perplexity", marker="^")
    plt.title("Validation Perplexity")
    plt.xlabel("Iterations")
    plt.ylabel("Perplexity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, "val_perplexity.png"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT Model Trainer")
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--embd_dim", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_k", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force_retrain", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    set_seed(1717)
    main(
        max_iters=args.max_iters,
        embd_dim=args.embd_dim,
        n_layer=args.n_layer,
        dropout=args.dropout,
        max_k=args.max_k,
        device=args.device,
        force_retrain=args.force_retrain,
        learning_rate=args.learning_rate,
        block_size=args.block_size,
        batch_size=args.batch_size,
    )
    best_path = os.path.join("experiments", "saved_models", f"gpt_k{args.max_k}.pth")
