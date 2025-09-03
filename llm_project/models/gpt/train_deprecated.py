# from llm_project.bpe.bytepair_encoding import BPE
# from llm_project.models.gpt.model import GPT
# from llm_project.utils.file_manager import save_item, load_item
# from llm_project.utils.debugg_utils import print_resource_usage
# from llm_project.models.gpt.generator import Generator
# import torch
# from torch.nn import functional as F
# import os
# import psutil
# import matplotlib.pyplot as plt
#
#
# def main(max_iters, embd_dim, n_layer, dropout, max_k, device, force_retrain=False):
#     # --- Configuration ---
#     BATCH_SIZE = 32
#     BLOCK_SIZE = 64
#     MAX_ITERS = max_iters
#     EVAL_INTERVAL = 500
#     LEARNING_RATE = 3e-4
#     EVAL_ITERS = 200
#     # --- End of Config ---
#     log_interval = 500
#
#     # Device Setup
#     if torch.cuda.is_available():
#         device = "cuda"
#     elif torch.backends.mps.is_available():
#         device = "mps"
#     else:
#         device = "cpu"
#     print(f"Using device: {device}")
#     device = torch.device(device)
#
#     # Data Loading and Preparation
#     datapath = os.path.join("data", "raw", "Shakespeare_clean_full.txt")
#     with open(datapath, "r", encoding="utf-8") as f:
#         text = f.read()
#
#     print("Training tokenizer...")
#     tokenizer = BPE(data_path=None, text=text, max_k=2000)  # Pass text directly
#
#     project_root = os.getcwd()
#     bpe_results_folder = os.path.join("experiments", "bpe_results", "train_results")
#     merges_file = "train_bpe_merges.pkl"
#     vocab_file = "train_final_vocab.pkl"
#
#     # Paths
#     merges_path = os.path.join(bpe_results_folder, merges_file)
#     vocab_path = os.path.join(bpe_results_folder, vocab_file)
#
#     merges_exist = os.path.exists(merges_path)
#     vocab_exist = os.path.exists(vocab_path)
#
#     # Retrains if force retrain or files are missing
#     if force_retrain or not (merges_exist and vocab_exist):
#         print(">>Training tokenizer from scratch...")
#         tokenizer.BPE_encoder()
#
#         # Save results
#         print(f">>Saving tokenizer to {bpe_results_folder}...")
#         save_item(
#             tokenizer.merges,
#             folder=bpe_results_folder,
#             name=merges_file,
#             base_dir=project_root,
#         )
#         save_item(
#             tokenizer.vocab,
#             folder=bpe_results_folder,
#             name=vocab_file,
#             base_dir=project_root,
#         )
#     else:
#         print(f">>Loaded pre-trained tokenizer from {bpe_results_folder}")
#         tokenizer.merges = load_item(
#             folder_path=bpe_results_folder, name=merges_file, base_dir=project_root
#         )
#         tokenizer.vocab = load_item(
#             folder_path=bpe_results_folder, name=vocab_file, base_dir=project_root
#         )
#         tokenizer.build_token_mappings()
#
#     vocab_size = len(tokenizer.token_to_id)
#     print(f"Vocabulary size: {vocab_size}")
#
#     data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
#
#     n = int(0.9 * len(data))
#     train_data = data[:n]
#     val_data = data[n:]
#
#     # Data Batching
#     def get_batch(split):
#         data_split = train_data if split == "train" else val_data
#         ix = torch.randint(len(data_split) - BLOCK_SIZE, (BATCH_SIZE,))
#         x = torch.stack([data_split[i : i + BLOCK_SIZE] for i in ix])
#         y = torch.stack([data_split[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
#         x, y = x.to(device), y.to(device)
#         return x, y
#
#     # Model and Optimizer
#
#     class GPTConfig:
#         def __init__(self, vocab_size, block_size):
#             self.vocab_size = vocab_size
#             self.block_size = block_size
#             self.embd_dim = embd_dim
#             self.n_layer = n_layer
#             self.n_head = 4
#             self.dropout = dropout
#             self.embd_pdrop = dropout
#             self.attn_pdrop = dropout
#             self.resid_pdrop = dropout
#
#     config = GPTConfig(vocab_size, BLOCK_SIZE)
#     model = GPT(config)
#     model.to(device)
#
#     optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
#
#     use_amp = device == "cuda"
#     scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
#
#     # Tracking loss
#     train_losses = []
#     val_losses = []
#
#     @torch.no_grad()
#     def estimate_loss():
#         out = {}
#         model.eval()
#         for split in ["train", "val"]:
#             losses = torch.zeros(EVAL_ITERS)
#             for k in range(EVAL_ITERS):
#                 X, Y = get_batch(split)
#                 with torch.cuda.amp.autocast(enabled=use_amp):
#                     logits = model(X)
#                     loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
#                 losses[k] = loss.item()
#             out[split] = losses.mean()
#         model.train()
#         return out
#
#     # Training Loop
#     print("Starting training...")
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
#
#         xb, yb = get_batch("train")
#
#         with torch.cuda.amp.autocast(enabled=use_amp):
#             logits = model(xb)
#             loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
#
#         optimizer.zero_grad(set_to_none=True)
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         if iter_num % log_interval == 0:
#             ram = psutil.virtual_memory().used / 1024**2
#             ram_log.append(ram)
#             print_resource_usage(step=iter)
#
#     print("Training finished.")
#
#     plt.plot(ram_log, label="RAM Usage (MB)")
#     plt.title("RAM Usage over Training")
#     plt.xlabel("Logged Steps")
#     plt.ylabel("RAM (MB)")
#     plt.grid(True)
#     plt.legend()
#     plt.savefig("experiments/saved_models/ram_usage.png")
#     plt.close()
#
#     # Save the Model
#     save_path = os.path.join("saved_models")
#     os.makedirs(save_path, exist_ok=True)
#     model_save_path = os.path.join(save_path, "gpt_shakespeare.pth")
#     print(f"Saving model to {model_save_path}")
#     torch.save(model.state_dict(), model_save_path)
#
#     # Generation Preview
#     # Prepare context (e.g., starting phrase)
#     start_text = "ROMEO:"
#     context_ids = tokenizer.encode(start_text)
#
#     # Setup generator
#     generator = Generator(
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens=100,
#         temperature=0.9,
#         top_k=40,
#         eos_token_id=None,
#         mode="top_k",
#     )
#
#     # Generate
#     output_ids = generator.generate(context_ids)
#     generated_text = tokenizer.decode(output_ids)
#
#     print("\n=== Generation Preview ===")
#     print(generated_text)
#
#     # Save it
#     preview_path = os.path.join("experiments", "plots", "gpt")
#     os.makedirs(preview_path, exist_ok=True)
#     with open(
#         os.path.join(preview_path, "generation_preview.txt"), "w", encoding="utf-8"
#     ) as f:
#         f.write(generated_text)
#
#     # Plotting
#     # Create folder for plots
#     plot_folder = os.path.join("experiments", "plots", "gpt")
#     os.makedirs(plot_folder, exist_ok=True)
#
#     # 1. Loss Plot
#     fig, ax = plt.subplots(figsize=(8, 6))
#     ax.plot(
#         range(0, MAX_ITERS, EVAL_INTERVAL), train_losses, label="Train Loss", marker="o"
#     )
#     ax.plot(
#         range(0, MAX_ITERS, EVAL_INTERVAL), val_losses, label="Val Loss", marker="s"
#     )
#     ax.set_title("Train and Validation Loss per Interval")
#     ax.set_xlabel("Iterations")
#     ax.set_ylabel("Loss")
#     ax.grid(True, linestyle="--", alpha=0.6)
#     ax.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(plot_folder, "loss_curve.png"))
#     plt.close()
#
#     # 2. PP Plot
#     val_perplexities = [torch.exp(torch.tensor(vl)).item() for vl in val_losses]
#     fig, ax = plt.subplots(figsize=(8, 6))
#     ax.plot(
#         range(0, MAX_ITERS, EVAL_INTERVAL),
#         val_perplexities,
#         label="Validation Perplexity",
#         marker="^",
#     )
#     ax.set_title("Validation Perplexity per Interval")
#     ax.set_xlabel("Iterations")
#     ax.set_ylabel("Perplexity")
#     ax.grid(True, linestyle="--", alpha=0.6)
#     ax.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(plot_folder, "val_perplexity.png"))
#     plt.close()


# --- imports unchanged ---
from llm_project.bpe.bytepair_encoding import BPE
from llm_project.models.gpt.model import GPT
from llm_project.utils.file_manager import save_item, load_item
from llm_project.utils.debugg_utils import print_resource_usage
from llm_project.utils.debugg_utils import Colors
from llm_project.models.gpt.generator import Generator
import torch
import math
from torch.nn import functional as F
import os
import psutil
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse

# Reproducibility block


def set_seed(seed: int = 1717):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(max_iters, embd_dim, n_layer, dropout, max_k, device, force_retrain=False):
    # --- Configuration ---
    BATCH_SIZE = 32
    BLOCK_SIZE = 64
    MAX_ITERS = max_iters
    EVAL_INTERVAL = 100
    LEARNING_RATE = 3e-4
    EVAL_ITERS = 200
    # --- End of Config ---
    log_interval = 100

    _pru = print_resource_usage

    def print_resource_usage_compat(step: int) -> None:
        try:
            # case 1: module-level function: def print_resource_usage(step)
            return _pru(step=step)
        except TypeError:
            # case 2: class method signature: def print_resource_usage(self, step)
            try:
                return _pru(None, step)  # pass dummy self
            except Exception:
                # fallback: minimal inline resource print
                ram_mb = psutil.virtual_memory().used / 1024**2
                cpu = psutil.cpu_percent(interval=0.0)
                disk = psutil.disk_usage(os.getcwd()).percent
                print(
                    f"[DEBUG] step {step:>7} || RAM {ram_mb:>8.2f} MB || CPU {cpu:>5.1f}% || Disk {disk:>5.1f}%"
                )

    # Device Setup: honor CLI device if valid; otherwise auto-detect
    if isinstance(device, str) and device in {"cuda", "mps", "cpu"}:
        dev_str = device
    else:
        if torch.cuda.is_available():
            dev_str = "cuda"
        elif torch.backends.mps.is_available():
            dev_str = "mps"
        else:
            dev_str = "cpu"
    print(f"Using device: {dev_str}")
    device = torch.device(dev_str)
    use_amp = device.type == "cuda"

    # Data Loading and Preparation
    datapath = os.path.join("data", "raw", "Shakespeare_clean_full.txt")
    with open(datapath, "r", encoding="utf-8") as f:
        text = f.read()

    print("Training tokenizer...")
    # use CLI max_k instead of hard-coded 2000
    tokenizer = BPE(data_path=None, text=text, max_k=max_k)

    project_root = os.getcwd()
    bpe_results_folder = os.path.join("experiments", "bpe_results", "train_results")
    os.makedirs(os.path.join(project_root, bpe_results_folder), exist_ok=True)

    merges_file = "train_bpe_merges.pkl"
    vocab_file = "train_final_vocab.pkl"

    # Full paths for save/load
    merges_path = os.path.join(project_root, bpe_results_folder, merges_file)
    vocab_path = os.path.join(project_root, bpe_results_folder, vocab_file)

    merges_exist = os.path.exists(merges_path)
    vocab_exist = os.path.exists(vocab_path)

    # Retrain if forced or files missing
    if force_retrain or not (merges_exist and vocab_exist):
        print(">>Training tokenizer from scratch...")
        tokenizer.BPE_encoder()

        # Save results (path-only API)
        print(f">>Saving tokenizer to {bpe_results_folder}...")
        save_item(tokenizer.merges, bpe_results_folder, merges_file)
        save_item(tokenizer.vocab, bpe_results_folder, vocab_file)  # <- was vocab_path
    else:
        print(f">>Loaded pre-trained tokenizer from {bpe_results_folder}")
        tokenizer.merges = load_item(
            bpe_results_folder,
            merges_file,
        )
        tokenizer.vocab = load_item(
            bpe_results_folder,
            vocab_file,
        )
        tokenizer.build_token_mappings()

    vocab_size = len(tokenizer.token_to_id)
    print(f"Vocabulary size: {vocab_size}")

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # Sanity & dataset stats ######## !!!!!!!!!
    print(f"{Colors.OKCYAN}[sanity]{Colors.ENDC} raw chars in file: {len(text):,}")
    # encode→decode roundtrip check (proves we can reconstruct the whole file)
    try:
        recon_ok = tokenizer.decode(tokenizer.encode(text)) == text
    except Exception:
        recon_ok = False
    print(
        f"{Colors.OKCYAN}[sanity]{Colors.ENDC} decode(encode(text)) == original? {recon_ok}"
    )

    TOTAL_TOKENS = len(data)
    print(
        f"{Colors.OKCYAN}[sanity]{Colors.ENDC} total tokens (after BPE): {TOTAL_TOKENS:,}  | vocab_size: {vocab_size}"
    )

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # === Epoch-equivalent stats ===
    DATASET_TOKENS = len(train_data) - 1  # usable next-token contexts
    TOKENS_PER_ITER = BATCH_SIZE * BLOCK_SIZE
    EST_STEPS_PER_EPOCH = math.ceil(DATASET_TOKENS / TOKENS_PER_ITER)
    print(
        f"[data] train tokens={DATASET_TOKENS:,} | tokens/iter={TOKENS_PER_ITER} "
        f"| ~steps/epoch: {EST_STEPS_PER_EPOCH}"
    )

    # Coverage of starting positions
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

    train_losses, val_losses = [], []
    eval_steps = []

    best_val = float("inf")
    best_path = os.path.join("experiments", "saved_models", "gpt_shakespeare-best.pth")
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

    print("Starting training...")
    tokens_seen = 0
    ram_log = []
    for iter_num in MAX_ITERS:
        if iter_num % EVAL_INTERVAL == 0 or iter_num == MAX_ITERS - 1:
            losses = estimate_loss()
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            train_losses.append(losses["train"])
            val_losses.append(losses["val"])
            eval_steps.append(iter_num)

            # Save best Val Loss
            cur_val = float(losses["val"])
            if cur_val < best_val:
                best_val = cur_val
                torch.save(model.state_dict(), best_path)
                print(
                    f"[checkpoint] new best val loss {best_val:.4f} saved -> {best_path}"
                )

            # Progress/coverage
            epoch_equiv = tokens_seen / max(1, DATASET_TOKENS)
            cov = coverage.float().mean().item() * 100.0
            print(
                f"[progress] ~epochs={epoch_equiv:.2f}  coverage≈{cov:.1f}%  "
                f"steps/epoch≈{EST_STEPS_PER_EPOCH}"
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
            ram = psutil.virtual_memory().used / 1024**2
            ram_log.append(ram)
            print_resource_usage_compat(step=iter_num)

    print("Training finished.")

    # RAM plot
    plt.plot(ram_log, label="RAM Usage with GPT trainer (MB)")
    plt.title("RAM Usage over Training")
    plt.xlabel("Logged Steps")
    plt.ylabel("RAM (MB)")
    plt.grid(True)
    plt.legend()
    os.makedirs(os.path.join("experiments", "saved_models"), exist_ok=True)
    plt.savefig(os.path.join("experiments", "saved_models", "ram_usage_gpt.png"))
    plt.close()

    # Save the Model
    save_path = os.path.join("saved_models")
    os.makedirs(save_path, exist_ok=True)

    model_save_path = os.path.join(save_path, "gpt_final.pt")
    print(f"Saving model to {model_save_path}")
    torch.save(model.state_dict(), model_save_path)

    # Generation Preview
    start_text = "ROMEO:"
    context_ids = tokenizer.encode(start_text)
    generator = Generator(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        temperature=0.9,
        top_k=40,
        eos_token_id=None,
        mode="top_k",
    )
    output_ids = generator.generate(context_ids)
    generated_text = tokenizer.decode(output_ids)

    print("\n=== Generation Preview ===")
    print(generated_text)

    preview_path = os.path.join("experiments", "plots", "gpt")
    os.makedirs(preview_path, exist_ok=True)
    with open(
        os.path.join(preview_path, "generation_preview.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(generated_text)

    # Plots
    # Create folder for plots
    plot_folder = os.path.join("experiments", "plots", "gpt")
    os.makedirs(plot_folder, exist_ok=True)

    # Safety: lengths must match
    assert len(eval_steps) == len(train_losses) == len(val_losses), (
        f"Lengths mismatch: steps={len(eval_steps)} "
        f"train={len(train_losses)} val={len(val_losses)}"
    )

    # Loss curves
    plt.figure(figsize=(8, 6))
    plt.plot(
        eval_steps, [float(t) for t in train_losses], label="Train Loss", marker="o"
    )
    plt.plot(eval_steps, [float(v) for v in val_losses], label="Val Loss", marker="s")
    plt.title("Train and Validation Loss per Interval")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, "loss_curve.png"))
    plt.close()

    # Perplexity smalll thing
    def _to_float(x):  # ADD: works for both torch.Tensor and float
        return x.item() if hasattr(x, "item") else float(x)

    val_perplexities = [math.exp(_to_float(vl)) for vl in val_losses]  # CHANGE
    plt.figure(figsize=(8, 6))
    plt.plot(eval_steps, val_perplexities, label="Validation Perplexity", marker="^")

    @torch.no_grad()
    def compute_perplexity():
        model.eval()
        total_loss = 0.0
        total_tokens = 0

        for i in range(0, len(val_data) - BLOCK_SIZE, BLOCK_SIZE):
            x = val_data[i : i + BLOCK_SIZE].unsqueeze(0).to(device)
            y = val_data[i + 1 : i + BLOCK_SIZE + 1].unsqueeze(0).to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    reduction="sum",
                )
            total_loss += loss.item()
            total_tokens += y.numel()

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        return perplexity

    final_ppl = compute_perplexity()
    print(f"[final] Full validation perplexity: {final_ppl:.4f}")
    # save to file
    with open("experiments/saved_models/gpt_val_ppl.txt", "w") as f:
        f.write(f"{final_ppl:.4f}\n")

    plt.axhline(final_ppl, color="red", linestyle="--", label="Full Val Perplexity")
    plt.title("Validation Perplexity per Interval")
    plt.xlabel("Iterations")
    plt.ylabel("Perplexity")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, "val_perplexity.png"))
    plt.close()


if __name__ == "__main__":
    set_seed(1717)

    main(
        max_iters=200,
        embd_dim=128,
        n_layer=2,
        dropout=0.2,
        max_k=1000,
        device="cuda",
        force_retrain=False,  # keep False so you reuse cached BPE
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT Model Trainer")
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--embd_dim", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_k", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force_retrain", action="store_true", default=False)
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
    )
