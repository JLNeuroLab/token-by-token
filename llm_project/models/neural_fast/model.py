import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralLanguageModel(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        self.n = config.n
        self.embd_dim = config.embd_dim
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(self.vocab_size, self.embd_dim)

        self.fc = nn.Linear(self.embd_dim, self.vocab_size)

    def forward(self, x, targets=None):
        embeds = self.embeddings(x)
        logits = self.fc(embeds)

        loss = None
        if targets is not None:
            B, L, C = logits.shape
            loss = F.cross_entropy(logits.view(B*L, C), targets.view(B*L))
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, start_ids, max_new_tokens=20, stochastic=True):
        generated = list(start_ids)
        for _ in range(max_new_tokens):
            context = torch.tensor([generated[-self.n:]], dtype=torch.long)  # (1, L)
            logits, _ = self.forward(context)                                  # (1, L, C)
            last_logits = logits[0, -1]                                        # (C,)
            probs = F.softmax(last_logits, dim=-1)

            if stochastic:
                next_id = torch.multinomial(probs, num_samples=1).item()
            else:
                next_id = torch.argmax(probs).item()

            generated.append(next_id)
        return generated
