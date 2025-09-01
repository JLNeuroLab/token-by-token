import torch
import torch.nn as nn
from llm_project.models.gpt.attention import CausalSelfAttention
import math
import matplotlib.pyplot as plt


class Gelu(nn.Module):
    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))
            )
        )


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.embd_dim, 4 * config.embd_dim),
            Gelu(),
            nn.Linear(4 * config.embd_dim, config.embd_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embd_dim)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.embd_dim)
        self.ffwd = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.embd_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.embd_dim))
        self.drop = nn.Dropout(config.embd_pdrop)

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.embd_dim)
        self.head = nn.Linear(config.embd_dim, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.block_size, (
            f"Cannot forward sequence of length {T}, block size is {self.block_size}"
        )

        tok_emb = self.tok_emb(idx)  # (B, T, C)
        pos_emb = self.pos_emb[:, :T, :]  # (1, T, C)
        x = self.drop(tok_emb + pos_emb)  # (B, T, C)

        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.head(x)  # (B, T, vocab_size)

        return logits

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def load_weights(self, path):
        # placeholder: adjust as needed
        self.load_state_dict(torch.load(path))

    def eval(self):
        return super().eval()


if __name__ == "__main__":
    gelu = Gelu()
    x = torch.linspace(-3, 3, steps=100)
    y = gelu(x)

    plt.plot(x.numpy(), y.detach().numpy())
    plt.title("Custom GELU Activation")
    plt.grid(True)
    plt.show()
