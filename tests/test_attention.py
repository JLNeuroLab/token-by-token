import torch
from llm_project.models.gpt.attention import CausalSelfAttention

# A mock config class to initialize the attention module for testing


class MockGPTConfig:
    def __init__(self, block_size, embd_dim, n_head, attn_pdrop, resid_pdrop, dropout):
        self.block_size = block_size
        self.embd_dim = embd_dim
        self.n_head = n_head
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.dropout = dropout


def test_attention_output_shape():
    """
    Tests that the output of the attention module has the same shape as the input.
    """
    # ARRANGE
    config = MockGPTConfig(
        block_size=8,
        embd_dim=16,
        n_head=4,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        dropout=0.0,
    )
    attention = CausalSelfAttention(config)
    # Create a random input tensor: (batch_size=1, sequence_length=5, embedding_dim=16)
    # Sequence length is less than block_size to test padding.
    x = torch.randn(1, 5, config.embd_dim)

    # ACT
    output = attention(x)

    # ASSERT
    assert output.shape == x.shape


def test_attention_is_causal():
    """
    Tests that the causal mask is working correctly. A token at position i
    should not depend on any token at position j > i.
    """
    # ARRANGE
    config = MockGPTConfig(
        block_size=8,
        embd_dim=16,
        n_head=4,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        dropout=0.0,
    )
    attention = CausalSelfAttention(config)
    # Create a random input tensor of size (1, 3, 16)
    x1 = torch.randn(1, 3, config.embd_dim)

    # ACT (First Pass)
    # Get the output for the first token
    output1 = attention(x1)
    output1_first_token = output1[:, 0, :]  # Shape (1, 16)

    # ACT (Second Pass)
    # Create a second input that is identical to the first, but with the
    # LAST token changed.
    x2 = x1.clone()
    x2[:, -1, :] = torch.randn(config.embd_dim)  # Change the last token

    # Get the new output
    output2 = attention(x2)
    output2_first_token = output2[:, 0, :]  # Shape (1, 16)

    # ASSERT
    # The output for the first token should be IDENTICAL in both passes,
    # because it should not have been influenced by the change in a future token.
    assert torch.allclose(output1_first_token, output2_first_token)

