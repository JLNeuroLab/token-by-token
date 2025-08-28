import torch
from torch.nn import functional as F
from llm_project.models.gpt.model import GPT

# (You can keep the MockGPTConfig class from the previous test)


class MockGPTConfig:
    def __init__(
        self,
        block_size,
        vocab_size,
        embd_dim,
        n_layer,
        n_head,
        dropout,
        embd_pdrop,
        attn_pdrop,
        resid_pdrop,
    ):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.embd_dim = embd_dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop


def test_gpt_model_forward_pass():
    """
    Smoke test to ensure the full GPT model can perform a forward pass
    and that the output shape is correct.
    """
    # ARRANGE
    config = MockGPTConfig(
        block_size=8,
        vocab_size=10,
        embd_dim=16,
        n_layer=2,
        n_head=4,
        dropout=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
    )
    model = GPT(config)

    B, T = 2, 5
    idx = torch.randint(0, config.vocab_size, (B, T))

    # ACT
    logits = model(idx)

    # ASSERT
    assert logits.shape == (B, T, config.vocab_size)
    assert logits.dtype == torch.float32


def test_gpt_model_can_overfit_on_a_small_batch():
    """
    A 'really good' test: confirms the model can learn by forcing it to
    overfit on a single batch. If the loss doesn't decrease significantly,
    there's likely an issue with the model's ability to train.
    """
    # ARRANGE
    B, T, C = 2, 5, 10  # Batch, Time, Vocab Size
    config = MockGPTConfig(
        block_size=8,
        vocab_size=C,
        embd_dim=16,
        n_layer=2,
        n_head=4,
        dropout=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
    )
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Create a single, consistent batch of data
    x = torch.randint(0, C, (B, T))
    y = torch.randint(0, C, (B, T))

    # ACT
    final_loss = None
    for i in range(500):  # Train for 100 steps
        logits = model(x)
        B_logits, T_logits, C_logits = logits.shape
        loss = F.cross_entropy(
            logits.view(B_logits * T_logits, C_logits), y.view(B * T)
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if i == 499:  # Store the loss from the final step
            final_loss = loss.item()

    # ASSERT
    # The initial loss will be around -ln(1/C) = -ln(0.1) = 2.3
    # After 100 steps, it should be significantly lower.
    print(f"Final loss after overfitting: {final_loss}")
    assert final_loss is not None and final_loss < 0.2


def test_gpt_model_is_sensitive_to_position():
    """
    Tests that the model's output changes when the order of input tokens
    is changed, confirming that positional embeddings are effective.
    """
    # ARRANGE
    B, T, C = 2, 5, 10
    config = MockGPTConfig(
        block_size=8,
        vocab_size=C,
        embd_dim=16,
        n_layer=2,
        n_head=4,
        dropout=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
    )
    model = GPT(config)

    # Create two input tensors with the same tokens but in a different order
    idx1 = torch.randint(0, C, (B, T))
    # Reverse the sequence along the time dimension
    idx2 = torch.flip(idx1, dims=[1])

    # Ensure the tensors are not identical (i.e., not a palindrome)
    if torch.equal(idx1, idx2):
        idx1[0, 0] = (idx1[0, 0] + 1) % C  # Tweak to ensure difference

    # ACT
    output1 = model(idx1)
    output2 = model(idx2)

    # ASSERT
    # The outputs should NOT be close to each other if positional embeddings work
    assert not torch.allclose(output1, output2, atol=1e-6)
