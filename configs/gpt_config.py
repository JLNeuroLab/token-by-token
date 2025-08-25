from base_config import BaseConfig

class GPTConfig(BaseConfig):
    """
    Base configuration class for GPT models.
    
    Args:
        vocab_size (int): Size of the vocabulary (number of unique tokens).
        **kwargs: Additional keyword arguments passed to BaseConfig.
    """
    def __init__(self, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size  # Needed for embedding layers


class GPTCustomConfig(GPTConfig):
    """
    Custom GPT configuration with default hyperparameters.
    
    These defaults are suitable for small-scale experiments and can be tuned.
    """

    # Transformer architecture
    n_layer = 8           # Number of decoder blocks in the GPT model
    n_head = 8            # Number of attention heads per block
    n_embd = 256          # Dimension of token embeddings and hidden states

    # Training parameters
    batch_size = 32       # Number of sequences per batch
    block_size = 64       # Maximum sequence length (context window)
    max_iters = 2e4       # Maximum number of training iterations
    learning_rate = 6e-4  # AdamW learning rate
    betas = (0.9, 0.95)   # AdamW betas for momentum and RMS
    grad_norm_clip = 1.0  # Gradient clipping to prevent exploding gradients

    # Dropout / Regularization
    embd_pdrop = 0.1      # Dropout on token embeddings
    resid_pdrop = 0.1     # Dropout on residual connections
    attn_pdrop = 0.1      # Dropout on attention weights
    dropout = 0.1         # Global dropout

    # Device configuration
    device = "cuda"        # Training device: "cpu", "cuda", or "mps"

    # Notes / optional parameters for generation
    # temperature = 1.0
    # top_k = None
    # n_positions = block_size  # Useful for positional embeddings
