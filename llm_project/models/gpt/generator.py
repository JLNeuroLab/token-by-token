import numpy as np
import torch


class Generator:
    """
    Class for autorregressive text generation.
    Provides configuration for temperature scaling, top-k filtering and early stopping.

    """

    def __init__(
        self,
        model,
        tokenizer,
        max_new_tokens=50,
        temperature=1,
        top_k=0,
        top_p=1,
        eos_token_id=None,
        mode="auto",
    ):
        """
        model: Your scratch GPT model with a forward() method that input logits
        tokenizer: Tokenizer object with `.encode()` and `.decode()` methods.
        max_new_tokens(int): Max tokens to generate.
        temperature(float): Softmax temperature. Lower is more deterministic
        top_k(int): If > 0, restrict to top-k logits.
        top_p (float): If < 1, use top-p nucleus sampling.
        eos_token_id(int or None): Optional ID for end of sequence tk - stop condition.
        mode (str): One of "auto", "top_k", "top_p", or "full". Determines sampling method.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.eos_token_id = eos_token_id
        self.mode = self._resolve_mode(mode)

    def _resolve_mode(self, mode):
        if mode != "auto":
            return mode
        if self.top_p < 1.0:
            return "top_p"
        elif self.top_k > 0:
            return "top_k"
        else:
            return "full"

    ####################
    # Sampling Methods #
    ####################

    def softmax(self, logits):
        # Apply softmax to convert logits into probs.
        # Substracts np.max(logits) and prevents overflow for big logit values.
        # This refers to: softmax(x_i) = exp(x_i) / sum_j(exp(x_j)), applied element wise.
        return np.exp(logits - np.max(logits)) / np.sum(np.exp(logits - np.max(logits)))

    def top_p_sample(self, logits):
        # Softmax for Top-p
        prob = self.softmax(logits)

        # Top-p filtering
        sorted_indices = np.argsort(prob)[::-1]
        sorted_probs = prob[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumulative_probs, self.top_p)
        top_indices = sorted_indices[: cutoff + 1]
        top_probs = prob[top_indices]
        top_probs /= np.sum(top_probs)
        return np.random.choice(top_indices, p=top_probs)

    def top_k_sample(self, logits):
        # get top_k logits directly (avoids full softmax over vocab)
        top_k_indices = np.argpartition(logits, -self.top_k)[-self.top_k :]
        top_k_logits = logits[top_k_indices]
        prob = self.softmax(top_k_logits)
        return np.random.choice(top_k_indices, p=prob)

    def full_sample(self, logits):
        prob = self.softmax(logits)
        return np.random.choice(len(prob), p=prob)

    def generate(self, context_ids, seed=None):
        """
        Generate a sequence of token IDs based on the input context.

        The model predicts one token at a time and adds it to the sequence.
        Depending on the chosen mode, the next token is selected using:
        - "top_k": sample from the k most likely tokens
        - "top_p": sample from the smallest group of tokens whose total probability is >= p
        - "greedy": pick the most likely token
        - "full": sample from the entire vocabulary

        Generation stops after a fixed number of tokens or when the end-of-sequence token is reached.

        Args:
            context_ids (List[int]): The starting sequence of token IDs.
            seed (int, optional): Optional seed for random sampling.

        Returns:
            List[int]: The complete sequence including the original context and generated tokens.
        """

        if seed is not None:
            np.random.seed(seed)

        # reuse array shape preallocate speed on long text
        generated = context_ids[:]
        input_array = np.zeros((1, self.model.block_size), dtype=np.int32)

        # Decoding strategy dispatch table
        strategy_map = {
            "top_k": self.top_k_sample,
            "top_p": self.top_p_sample,
            "full": self.full_sample,
        }
        if self.mode not in strategy_map:
            raise ValueError(f"Unknown dec mode: '{self.mode}'")
        sample_fn = strategy_map[self.mode]

        for _ in range(self.max_new_tokens):
            # Slice context to block size
            block = generated[-self.model.block_size :]
            input_array[0, -len(block) :] = block

            # Tensor fix
            # Build the "bridge" to PyTorch: Convert the NumPy array to a PyTorch tensor.
            # Get the model's device (cpu or gpu)
            device = next(self.model.parameters()).device
            input_tensor = torch.tensor(input_array, dtype=torch.long).to(device)

            # Call the PyTorch model. We add torch.no_grad() for efficiency.
            with torch.no_grad():
                logits_tensor = self.model.forward(input_tensor)

            # Convert the output tensor back to a NumPy array.
            logits = logits_tensor.cpu().numpy()

            # Get the logits for the very last token
            next_tk_logits = logits[0, -1, :] / self.temperature

            # Use method for picking next tk
            next_tk = sample_fn(next_tk_logits)

            # # Apply softmax and Top-k/p sampling
            # if self.top_p < 1.0:
            #     next_tk = self.top_p_sample(next_tk_logits)
            # elif self.top_k > 0:
            #     next_tk = self.top_k_sample(next_tk_logits)
            # else:
            #     # Append and check for stop condition
            #     next_tk = self.full_sample(next_tk_logits)

            generated.append(next_tk.item())
            if self.eos_token_id is not None and next_tk == self.eos_token_id:
                break

        return generated
