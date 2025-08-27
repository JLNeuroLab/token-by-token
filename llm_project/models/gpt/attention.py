import torch
import torch.nn as nn
import torch.nn.functional as F 
import math


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        """
        Initializes the CausalSelfAttention module.

        Args:
            config: A configuration object (e.g., GPTCustomConfig) containing
                hyperparameters and settings for the attention layer.

        Attributes:
            n_heads (int): Number of attention heads.
            embd_dim (int): Dimensionality of input embeddings.
            head_dim (int): Dimensionality per attention head (embd_dim / n_heads).
            c_attn (nn.Linear): Linear layer projecting input embeddings into concatenated queries, keys, and values.
            c_proj (nn.Linear): Output linear projection after attention.
            attn_dropout (nn.Dropout): Dropout applied to attention weights.
            resid_dropout (nn.Dropout): Dropout applied to the residual connection.
            dropout (float): General dropout rate from config.
            mask (torch.Tensor, registered buffer): Lower-triangular causal mask 
                to prevent attention to future tokens.

        Raises:
            AssertionError: If embd_dim is not divisible by n_heads.

        Explanation:
            - head_dim ensures that each attention head has an equal share of embedding dimensions.
            - c_attn computes queries, keys, and values in a single projection for efficiency.
            - c_proj maps the multi-head output back to the original embedding dimension.
            - attn_dropout and resid_dropout implement regularization to reduce overfitting.
            - mask ensures causal attention by blocking access to future tokens in the sequence.
        """
        super().__init__()  # Call parent constructor for nn.Module initialization

        # Store number of attention heads and embedding dimension
        self.n_heads = config.n_head
        self.embd_dim = config.embd_dim

        # Compute dimension per head and ensure divisibility
        self.head_dim = self.embd_dim // self.n_heads
        assert self.head_dim * self.n_heads == self.embd_dim, "embed_dim must be divisible by n_heads"

        # Linear projection for queries, keys, and values in a single matrix multiplication
        self.c_attn = nn.Linear(self.embd_dim, 3 * self.embd_dim)

        # Linear projection to map concatenated multi-head outputs back to embedding dimension
        self.c_proj = nn.Linear(self.embd_dim, self.embd_dim)

        # Dropout applied to attention weights for regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)

        # Dropout applied after residual connection for regularization
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # Global dropout value (could be used in Flash Attention or elsewhere)
        self.dropout = config.dropout

        # Register a lower-triangular mask buffer to enforce causal attention
        # Shape: (1, 1, block_size, block_size) to broadcast across batch and heads
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        """
        Forward pass of the Causal Self-Attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, E), where
                B = batch size
                T = sequence length (block size, number of tokens)
                E = embedding dimension (embd_dim)

        Returns:
            torch.Tensor: Output tensor of shape (B, T, E) after applying
            causal self-attention and linear projection.
        """

        B, T, E = x.shape # (batch_size, block_size, embd_dim)

        # Project input embeddings into queries (q), keys (k), and values (v)
        # Each has shape (B, T, E)
        q, k, v = self.c_attn(x).split(self.embd_dim, dim=2) # (batch_size, block_size, embd_dim)

        # Reshape k, q, v for multi-head attention
        # (B, T, E) -> (B, T, n_heads, head_dim) -> transpose -> (B, n_heads, T, head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute raw attention scores
        # Matrix multiplication between q and k^T, scaled by sqrt(head_dim)
        # Resulting shape: (B, n_heads, T, T) = how much each query token attends to all key tokens

        # batch matrix multiplication in PyTorch:
        # The last two tensor dimensions are interpreted as matrices, 
        # while all the previous dimensions are considered batches.
        # The matmul happens only between the last two dimensions of the tensor.
        attn_score = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B, n_heads, T, T) =  q: (B, n_heads, T, head_dim) @  k: (B, n_heads, head_dim, T)

        # Apply causal mask to prevent attending to future tokens
        # self.mask has shape (block_size, block_size)
        # [:, :, :T, :T] selects the relevant portion of the mask for current sequence length
        # masked_fill sets positions with 0 in the mask to -inf so that softmax ignores them
        attn_score = attn_score.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        # Normalize scores into probabilities
        # Softmax along the last dimension (keys dimension T)
        # Each query token now has a probability distribution over all key tokens  
        attn_probs = F.softmax(attn_score, dim=-1) # (B, n_heads, T, T)
        attn_probs = self.attn_dropout(attn_probs)

        # Weighted sum of values v, according to attention probabilities
        # (B, n_heads, T, T) @ (B, n_heads, T, head_dim) -> (B, n_heads, T, head_dim)
        y = attn_probs @ v #(B, n_heads, T, T) @ (B, n_heads, T, head_dim) -> (B, n_heads, T, head_dim)
        # Recombine all heads back into a single embedding
        # (B, n_heads, T, head_dim) -> (B, T, n_heads, head_dim) -> (B, T, E)
        y = y.transpose(1, 2) # (B, T, n_heads, head_dim)
        y = y.contiguous().view(B, T, E)
        # Final linear projection and dropout
        y = self.c_proj(y)
        y = self.resid_dropout(y)

        return y
    
if __name__ == "__main__":
    from configs.gpt_config import GPTCustomConfig
    # create config instance
    config = GPTCustomConfig(vocab_size=5000)

    # create attention layer
    attn = CausalSelfAttention(config)
    
    # create a fake input: batch_size=2, seq_len=block_size, embedding_dim=n_embd
    x = torch.randn(2, config.block_size, config.embd_dim)
    
    # forward pass
    y = attn.forward(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    
    # check for NaNs
    if torch.isnan(y).any():
        print("Warning: output contains NaNs!")
    else:
        print("Output seems fine.")
