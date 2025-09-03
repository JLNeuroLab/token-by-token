import os
import matplotlib.pyplot as plt
from llm_project.models.ngrams.trainer import NGramTrainer
from llm_project.bpe.bytepair_encoding import BPE
from llm_project.utils.file_manager import load_tokenizer, save_tokenizer, get_project_root
from llm_project.utils.dataloader import load_shakespeare
from llm_project.models.configs.configs import NgramConfig

class LM_Pipeline:

    def __init__(self, model_type, config, tokenizer=None, project_root=None):
        self.model_type = model_type
        self.config = config
        self.tokenizer = tokenizer
        self.device = self.config.device
        self.model = None
        self.trainer = None
        self.project_root = project_root or get_project_root()

    def prepare_tokens(self, train_text=None, max_k=2000, force_retrain=False, train_limit=None):
        """
        Prepare or train a BPE tokenizer and return limited tokens.

        Args:
            train_text (str): Text to train the tokenizer if needed.
            max_k (int): Maximum number of merges for BPE.
            force_retrain (bool): If True, retrain the tokenizer even if saved.
            train_limit (int): Maximum number of tokens to use from the training text.
        """
        if self.tokenizer is not None and not force_retrain:
            print("Using provided tokenizer.")
            tokens = self.tokenizer.tokens[:train_limit] if train_limit else self.tokenizer.tokens
            print(f"DEBUG: tokenizer.tokens length = {len(tokens)}")
            return tokens

        tokenizer_filename = f"BPE_merges_k{max_k}.pkl"
        category = "tokenizers"
        save_path = os.path.join(self.project_root, "experiments", category, tokenizer_filename)

        if not force_retrain and os.path.exists(save_path):
            print(f"--- Loading existing BPE tokenizer from:\n{save_path}")
            self.tokenizer, tokens = load_tokenizer(save_path)
            if train_limit:
                tokens = tokens[:train_limit]
                self.tokenizer.tokens = tokens
            print(f"DEBUG: loaded tokenizer.tokens length = {len(tokens)}")
            return tokens

        if train_text is None:
            raise ValueError("train_text must be provided to train a new BPE tokenizer.")

        # Limit the training text before BPE to save time
        if train_limit:
            train_text = train_text[:train_limit]

        print("Training new BPE tokenizer...")
        bpe = BPE(max_k=max_k, text=train_text)
        bpe.BPE_encoder()
        tokens = bpe.tokens
        self.tokenizer = bpe

        # Ensure plot directory exists
        plots_dir = os.path.join(self.project_root, "experiments", "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Plot and save
        fig = bpe.plot_vocabulary_growth()
        if fig is not None:
            plot_path = os.path.join(plots_dir, f"vocab_growth_k{max_k}.png")
            fig.savefig(plot_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            print(f"Vocabulary growth plot saved to: {plot_path}") 
            save_path = save_tokenizer(bpe, root=self.project_root, filename=tokenizer_filename, category=category)
        print(f"BPE tokenizer saved to: {save_path}")
        print(f"DEBUG: newly trained tokenizer.tokens length = {len(tokens)}")
        return tokens

    def setup_trainer(self, train_tokens, force_retrain=False, max_k=None):
        """Setup and train the N-gram model."""
        model_type = self.model_type.lower()
        if model_type == "ngram":
            ngram_trainer = NGramTrainer(config=self.config, model=None, tokens=train_tokens, k=max_k)
            self.model = ngram_trainer.train(force_retrain=force_retrain, tune_lambdas=True,
                                             train_limit=None, valid_limit=None)
        else:
            raise NotImplementedError(f"Model type '{self.model_type}' not implemented")

    def run(self, train_text=None, valid_text=None, max_k=2000,
            force_retrain_tokenizer=False, force_retrain_model=False,
            train_limit=10000, valid_limit=1000):
        """Full automatic pipeline: tokenizer → tokens → trainer → model"""
        if train_text is None:
            raise ValueError("train_text must be provided")

        # Step 1: tokenizer + train tokens
        train_tokens = self.prepare_tokens(train_text=train_text,
                                           max_k=max_k,
                                           force_retrain=force_retrain_tokenizer,
                                           train_limit=train_limit)
        print(f"DEBUG: train_tokens length = {len(train_tokens)}")

        # Step 2: valid tokens
        if valid_text is not None and valid_limit:
            # Approximate: use first N characters to produce roughly `valid_limit` tokens
            valid_text = valid_text[:valid_limit]  
        valid_tokens = self.tokenizer.BPE_segmenter(valid_text) if valid_text else None
        if valid_tokens is not None and valid_limit:
            valid_tokens = valid_tokens[:valid_limit]
        if valid_tokens is not None:
            print(f"DEBUG: valid_tokens length = {len(valid_tokens)}")

        # Step 3: train model
        self.setup_trainer(train_tokens=train_tokens, force_retrain=force_retrain_model, max_k=max_k)

        return self.model, train_tokens, valid_tokens


if __name__ == "__main__":
    config = NgramConfig(n=3, device="cpu")
    train_text = load_shakespeare(version="train")
    valid_text = load_shakespeare(version="validation")

    pipeline = LM_Pipeline(model_type="ngram", config=config)
    model, train_tokens, valid_tokens = pipeline.run(
        train_text=train_text,
        valid_text=valid_text,
        max_k=800,
        force_retrain_tokenizer=False,
        force_retrain_model=False,
        train_limit=100000,
        valid_limit=10000
    )

    print(f"Train tokens (sample): {train_tokens[:20]}")
    print(f"Validation tokens (sample): {valid_tokens[:10]}")
