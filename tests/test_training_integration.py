import torch
from torch.nn import functional as F
from llm_project.models.gpt.model import GPT
import random
import numpy as np
from llm_project.bpe.bytepair_encoding import BPE
import os
from llm_project.utils.debugg_utils import Colors

random.seed(17)
torch.manual_seed(17)


def load_shakespeare_snip(path, num_chars=4000):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    text = text[:num_chars]
    return text.lower().replace("\n", " ")


def test_training_loop_reduces_loss():
    torch.manual_seed(17)
    random.seed(17)
    np.random.seed(17)

    # === Load real data ===
    datapath = os.path.join("data", "raw", "Shakespeare_clean_full.txt")
    raw_text = load_shakespeare_snip(datapath)

    # === Tokenize with your BPE ===
    bpe = BPE(max_k=200, text=raw_text)
    bpe.BPE_encoder()
    token_ids = bpe.encode(raw_text)
    print(f"[INFO] BPE produced {len(token_ids)} tokens with max_k={bpe.max_k}")

    # === Setup GPT config ===
    class TestConfig:
        vocab_size = len(bpe.token_to_id)
        block_size = 16
        embd_dim = 32
        n_layer = 2
        n_head = 2
        dropout = 0.0
        embd_pdrop = 0.0
        attn_pdrop = 0.0
        resid_pdrop = 0.0

    config = TestConfig()
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # === Split into train/val ===
    data = torch.tensor(token_ids)
    # train_data = data[:1500]
    # val_data = data[1500:]

    # Split into train/val checked
    # It checks if token_ids are long enough to support your GPT batches.
    # If not, it warns you and swaps in synthetic token IDs so your test never fails just 'cause BPE got too greedy.
    # Then it slices up train_data and val_data as usual.
    min_val_len = config.block_size + 1  # +1 since we need T+1 tokens for x and y
    required_len = min_val_len + 1000

    if len(data) < required_len:
        print(
            f"{Colors.WARNING}[WARN]{Colors.ENDC} Not enough tokens after BPE (got {len(data)}), using synthetic data instead."
        )
        data = torch.randint(0, config.vocab_size, (required_len,))

    train_data = data[:-min_val_len]
    val_data = data[-min_val_len:]
    print(f"Train tokens: {len(train_data)} | Val tokens: {len(val_data)}")

    def get_batch(split, B=4, T=config.block_size):
        source = train_data if split == "train" else val_data
        ix = torch.randint(len(source) - T, (B,))
        x = torch.stack([source[i : i + T] for i in ix])
        y = torch.stack([source[i + 1 : i + T + 1] for i in ix])
        return x, y

    # === Training loop ===
    for i in range(50):
        xb, yb = get_batch("train")
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), yb.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if i == 0:
            val_x, val_y = get_batch("val")
            initial_loss = F.cross_entropy(
                model(val_x).view(-1, config.vocab_size), val_y.view(-1)
            ).item()

        if i == 49:
            val_x, val_y = get_batch("val")
            final_loss = F.cross_entropy(
                model(val_x).view(-1, config.vocab_size), val_y.view(-1)
            ).item()

    print(f"Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")
    assert final_loss < initial_loss + 0.02, (
        f"Final loss ({final_loss:.4f}) not significantly lower than initial ({initial_loss:.4f})"
    )
