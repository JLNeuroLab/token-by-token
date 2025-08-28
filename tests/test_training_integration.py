import torch
from torch.nn import functional as F
from llm_project.models.gpt.model import GPT


def test_training_loop_reduces_loss():
    """
    An integration test to confirm that a short training run reduces the
    validation loss, proving that the end-to-end process is working.
    """

    # ARRANGE: Set up a miniature training environment
    # Use a small config to make the test run faster
    class TestConfig:
        vocab_size = 500  # A reasonable guess for a small vocab
        block_size = 16
        embd_dim = 16
        n_layer = 2
        n_head = 2
        dropout = 0.0
        embd_pdrop = 0.0
        attn_pdrop = 0.0
        resid_pdrop = 0.0

    config = TestConfig()
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Create a tiny, random dataset for the test
    B, T = 4, config.block_size  # Batch, Time
    train_data = torch.randint(0, config.vocab_size, (100,))
    val_data = torch.randint(0, config.vocab_size, (100,))

    def get_test_batch(split):
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - T, (B,))
        x = torch.stack([data[i : i + T] for i in ix])
        y = torch.stack([data[i + 1 : i + T + 1] for i in ix])
        return x, y

    # ACT: Run a very short training loop
    initial_loss = -1.0
    final_loss = -1.0

    for i in range(20):  # Only 20 iterations for a quick test
        # Get data and run a training step
        xb, yb = get_test_batch("train")
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), yb.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Record initial and final validation loss
        if i == 0:
            val_xb, val_yb = get_test_batch("val")
            logits = model(val_xb)
            initial_loss = F.cross_entropy(
                logits.view(-1, config.vocab_size), val_yb.view(-1)
            ).item()

        if i == 19:
            val_xb, val_yb = get_test_batch("val")
            logits = model(val_xb)
            final_loss = F.cross_entropy(
                logits.view(-1, config.vocab_size), val_yb.view(-1)
            ).item()

    # ASSERT
    print(f"Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")
    assert final_loss < initial_loss
