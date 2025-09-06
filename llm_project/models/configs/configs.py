class BaseConfig:
    def __init__(self, device, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.device = device


class GPTConfig(BaseConfig):
    """
    Base configuration class for GPT models.

    Args:
        vocab_size (int): Size of the vocabulary (number of unique tokens).
        **kwargs: Additional keyword arguments passed to BaseConfig.
    """

    def __init__(
        self,
        vocab_size,
        n_heads,
        layer_dim,
        embd_dim,
        block_size,
        dropout,
        embd_pdrop,
        attn_pdrop,
        resid_pdrop,
        device,
    ):
        super().__init__(device=device)
        self.vocab_size = vocab_size
        self.embd_dim = embd_dim
        self.layer_dim = layer_dim
        self.n_heads = n_heads
        self.block_size = block_size
        self.dropout = dropout
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop

    def with_defaults(
        cls,
        *,
        vocab_size: int,
        n_heads: int = 6,
        layer_dim: int = 4,  # number of transformer blocks
        embd_dim: int = 384,
        block_size: int = 64,
        dropout: float = 0.2,
        embd_pdrop: float | None = None,
        attn_pdrop: float | None = None,
        resid_pdrop: float | None = None,
        device: str | None = None,
        n_head: int | None = None,
        n_layer: int | None = None,
    ) -> "GPTConfig":
        """
        Supplies defaults. Use this in trainers instead of calling __init__ directly.
        """
        if n_head is not None:
            n_heads = n_head
        if n_layer is not None:
            layer_dim = n_layer

        # If per-submodule dropouts are not provided, fall back to global dropout
        if embd_pdrop is None:
            embd_pdrop = dropout
        if attn_pdrop is None:
            attn_pdrop = dropout
        if resid_pdrop is None:
            resid_pdrop = dropout

        return cls(
            vocab_size=vocab_size,
            n_heads=n_heads,
            layer_dim=layer_dim,
            embd_dim=embd_dim,
            block_size=block_size,
            dropout=dropout,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            device=device,
        )

    def display(self):
        print("GPT Custom Configuration:")
        for k, v in self.__dict__.items():
            print(f"{k}: {v}")


class NgramConfig(BaseConfig):
    def __init__(
        self,
        n,
        device,
        max_n=None,
        lambdas=None,
    ):
        super().__init__(device=device)

        self.n = n
        self.max_n = max_n if max_n is not None else n
        self.lambdas = lambdas if lambdas is not None else [1 / self.n] * self.n

    def display(self):
        print("N-gram configuration")
        for k, v in self.__dict__.items():
            print(f"{k}, {v}")


class NeuralConfig(BaseConfig):
    def __init__(
        self,
        n,
        vocab_size,
        embd_dim,
        block_size,
        device,
        max_n=None,
    ):
        super().__init__(device=device)

        self.n = n
        self.max_n = max_n if max_n is not None else n
        self.vocab_size = vocab_size
        self.embd_dim = embd_dim
        self.block_size = block_size

    def display(self):
        print("Neural n-gram configuration")
        for k, v in self.__dict__.items():
            print(f"{k}, {v}")


class NeuralFastConfig(BaseConfig):
    def __init__(
        self,
        n,
        vocab_size,
        embd_dim,
        block_size,
        device,
        max_n=None,
    ):
        super().__init__(device=device)
        self.n = n
        self.max_n = max_n if max_n is not None else n
        self.vocab_size = vocab_size
        self.embd_dim = embd_dim
        self.block_size = block_size

    def display(self):
        print("Neural n-gram configuration")
        for k, v in self.__dict__.items():
            print(f"{k}, {v}")
