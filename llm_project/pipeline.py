from llm_project.models.configs.configs import NgramConfig
from llm_project.utils.dataloader import load_shakespeare
from llm_project.utils.file_manager import load_tokenizer, save_tokenizer, get_project_root, get_model_path
from llm_project.utils.file_manager import (
    load_tokenizer,
    save_tokenizer,
    get_project_root,
)
import os
import matplotlib.pyplot as plt
from llm_project.models.ngrams.trainer import NGramTrainer
from llm_project.bpe.bytepair_encoding import BPE


class LM_Pipeline:
    def __init__(self, model_type, config, tokenizer=None, project_root=None):
        self.model_type = model_type
        self.config = config
        self.tokenizer = tokenizer
        self.device = self.config.device
        self.model = None
        self.trainer = None
        self.project_root = project_root or get_project_root()
        self.subdir = None
        self.final = False

    def prepare_tokens(
        self, train_text=None, max_k=2000, force_retrain=False, train_limit=None
    ):
        """
        Prepare or train a BPE tokenizer and return limited tokens.

        Args:
            train_text (str): Text to train the tokenizer if needed.
            max_k (int): Maximum number of merges for BPE.
            force_retrain (bool): If True, retrain the tokenizer even if saved.
            train_limit (int): Maximum number of tokens to use from the training text.
        """
        # --- CASE 1: tokenizer available (in the eventuality you want to add one) and train not forced
        if self.tokenizer is not None and not force_retrain:
            # Load the existing tokenizer
            print("Using provided tokenizer.")
            tokens = (
                self.tokenizer.tokens[:train_limit]
                if train_limit
                else self.tokenizer.tokens
            )
            print(f"DEBUG: tokenizer.tokens length = {len(tokens)}")
            # Returns tokens, will be useful for setting up the trainer
            return tokens

        # Preparing parameters for saving and loading tokenizers
        tokenizer_filename = f"BPE_merges_k{max_k}.pkl"
        category = "tokenizers"
        save_dir = get_model_path(
            root=self.project_root, category=category, subdir=self.subdir, final=self.final)
        save_path = save_dir / tokenizer_filename

        # CASE 2: training is not forced and saved tokenizer available
        if not force_retrain and os.path.exists(save_path):
            print(f"--- Loading existing BPE tokenizer from:\n{save_path}")
            # Load the tokenizer
            self.tokenizer, tokens = load_tokenizer(root=self.project_root,
                                                    filename=save_path,
                                                    )
            if train_limit:
                tokens = tokens[:train_limit]
                self.tokenizer.tokens = tokens
            print(f"DEBUG: loaded tokenizer.tokens length = {len(tokens)}")
            return tokens

        if train_text is None:
            raise ValueError(
                "train_text must be provided to train a new BPE tokenizer."
            )

        # Limit the training text before BPE to save time
        if train_limit:
            train_text = train_text[:train_limit]

        # CASE 3: Train a new tokenizer
        print("Training new BPE tokenizer...")
        bpe = BPE(max_k=max_k, text=train_text)
        bpe.BPE_encoder()
        tokens = bpe.tokens
        self.tokenizer = bpe

        # Ensure plot directory exists
        plots_dir = os.path.join(self.project_root, "experiments", "plots")
        os.makedirs(plots_dir, exist_ok=True)

        plot_path = os.path.join(plots_dir, f"vocabulary_growth_k{max_k}.png")
        bpe.plot_vocabulary_growth(save_path=plot_path)

        # Finally, save the tokenizer using save_tokenizer
        save_path = save_tokenizer(bpe,
                                   root=self.project_root,
                                   filename=tokenizer_filename,
                                   category=category,
                                   )
        print(f"BPE tokenizer saved to: {save_path}")
        print(f"DEBUG: newly trained tokenizer.tokens length = {len(tokens)}")
        return tokens

    def setup_trainer(self, train_tokens, force_retrain=False, max_k=None):
        """Setup and train the model."""
        model_type = self.model_type.lower()

        # CLASSICAL N-GRAM MODEL
        if model_type == "ngram":
            ngram_trainer = NGramTrainer(
                config=self.config, model=None, tokens=train_tokens, k=max_k
            )
            self.model = ngram_trainer.train(
                force_retrain=force_retrain,
                tune_lambdas=True,
                train_limit=None,
                valid_limit=None,
            )
        else:
            raise NotImplementedError(
                f"Model type '{self.model_type}' not implemented")

        if model_type == "neural":
            neural_trainer = NeuralTrainer(
                config=self.config, model=None, tokens=train_tokens, k=max_k
            )
            self.model = ngram_trainer.train(
                force_retrain=force_retrain,
                tune_lambdas=True,
                train_limit=None,
                valid_limit=None,
            )
        else:
            raise NotImplementedError(
                f"Model type '{self.model_type}' not implemented")

        if model_type == "gpt":
            ngram_trainer = GPTTrainer(
                config=self.config, model=None, tokens=train_tokens, k=max_k
            )
            self.model = ngram_trainer.train(
                force_retrain=force_retrain,
                tune_lambdas=True,
                train_limit=None,
                valid_limit=None,
            )
        else:
            raise NotImplementedError(
                f"Model type '{self.model_type}' not implemented")

    def run(
        self,
        train_text=None,
        valid_text=None,
        max_k=2000,                             # -> force_retrain of bpe
        force_retrain_tokenizer=False,          # -> force_retrain of the model
        force_retrain_model=False,
        train_limit=10000,
        valid_limit=1000):
        """Full automatic pipeline: tokenizer → tokens → trainer → model"""

        if train_text is None:
            raise ValueError("train_text must be provided")

        # STEP 1: tokenizer + train tokens
        train_tokens = self.prepare_tokens(train_text=train_text,
                                            max_k=max_k,
                                            force_retrain=force_retrain_tokenizer,
                                            train_limit=train_limit)
        print(f"DEBUG: train_tokens length = {len(train_tokens)}")

        # STEP 2: valid tokens
        if valid_text is not None and valid_limit:
            valid_text = valid_text[:valid_limit]
        # Applying BPE merges to validation text

        valid_tokens = self.tokenizer.BPE_segmenter(
            valid_text) if valid_text else None
        if valid_tokens is not None and valid_limit:
                valid_tokens = valid_tokens[:valid_limit]
            if valid_tokens is not None:
                print(f"DEBUG: valid_tokens length = {len(valid_tokens)}")

        # STEP 3: train model
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
        train_limit=10000,
        valid_limit=1000
    )
    print(f"Train tokens (sample): {train_tokens[:20]}")
    print(f"Validation tokens (sample): {valid_tokens[:10]}")
