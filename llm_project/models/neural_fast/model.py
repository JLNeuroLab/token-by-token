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
    def predict_next_token_sampling(self, logits, top_k=None, top_p=None, temperature=1.0, unk_id=None):
        """
        Sample the next token from logits using temperature, top-k, top-p, and optional UNK filtering.

        Args:
            logits (torch.Tensor): 1D tensor of logits (C,)
            top_k (int, optional): Keep only top_k most probable tokens.
            top_p (float, optional): Keep minimal set of tokens with cumulative probability >= top_p.
            temperature (float): Temperature for scaling logits.
            unk_id (int, optional): Token id to forbid (set probability 0).

        Returns:
            int: Index of the sampled token.
        """
        # --- Temperature scaling ---
        logits = logits / temperature

        # --- Top-K filtering ---
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            topk_values, topk_indices = torch.topk(logits, top_k)
            mask = torch.full_like(logits, float('-inf'))
            mask[topk_indices] = logits[topk_indices]
            logits = mask

        # --- Convert to probabilities ---
        probs = F.softmax(logits, dim=-1)

        # --- Top-P (nucleus) filtering ---
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least one token
            sorted_indices_to_remove[..., 0] = 0
            probs[sorted_indices[sorted_indices_to_remove]] = 0
            probs = probs / probs.sum()  # renormalize

        # --- Filter UNK token ---
        if unk_id is not None:
            probs[unk_id] = 0
            probs = probs / probs.sum()

        # --- Sample next token ---
        next_id = torch.multinomial(probs, num_samples=1).item()
        return next_id

    @torch.no_grad()
    def generate(self, 
                 start_ids, 
                 max_new_tokens=20, 
                 stochastic=True, 
                 top_k=None, 
                 top_p=None, 
                 temperature=1.0,
                 block_size = None,
                 id2token = None,
                 unk_id=None):
        """
        Generate tokens starting from start_ids, using stochastic sampling or greedy.

        Args:
            start_ids (list): List of starting token IDs
            max_new_tokens (int): Number of tokens to generate
            stochastic (bool): Use sampling if True, else greedy
            top_k (int): Top-K sampling
            top_p (float): Top-P (nucleus) sampling
            temperature (float): Temperature for scaling logits
            unk_id (int): Token ID to forbid

        Returns:
            list: Generated token IDs
        """
        generated_ids = list(start_ids)
        context_size = block_size if block_size is not None else self.n

        for _ in range(max_new_tokens):
            device = next(self.parameters()).device 

            flat_ids = []
            for gid in generated_ids[-context_size:]:
                if isinstance(gid, torch.Tensor):
                    flat_ids.extend(gid.view(-1).tolist())  
                else:
                    flat_ids.append(int(gid))

            context_ids = flat_ids[-context_size:]  
            context = torch.tensor([context_ids], dtype=torch.long, device=device)

            logits, _ = self.forward(context)
            last_logits = logits[0, -1]

            if stochastic:
                next_id = self.predict_next_token_sampling(
                    last_logits, top_k=top_k, top_p=top_p, temperature=temperature, unk_id=unk_id
                )
            else:
                next_id = torch.argmax(last_logits).item()

            generated_ids.append(next_id)

        # Convert IDs to tokens se id2token fornito
        if id2token is not None:
            generated_tokens = [id2token.get(i, "UNK") for i in generated_ids]
            generated_text = " ".join(generated_tokens)
            return generated_ids, generated_tokens, generated_text

        return generated_ids