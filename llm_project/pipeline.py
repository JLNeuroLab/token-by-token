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
from llm_project.utils.file_manager import load_tokenizer, save_tokenizer, get_project_root, get_model_path, save_model
from llm_project.utils.dataloader import load_shakespeare
from llm_project.models.configs.configs import NgramConfig
from llm_project.utils.debugg_utils import Colors


class LM_Pipeline:
    def __init__(self, model_type, config, tokenizer=None, project_root=None, final=False):
        self.model_type = model_type
        self.config = config
        self.tokenizer = tokenizer
        self.device = self.config.device
        self.model = None
        self.trainer = None
        self.project_root = project_root or get_project_root()
        self.subdir = None

    def prepare_tokens(self, train_text=None, max_k=2000, force_retrain=False, train_limit=None, final=None):
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
        final_flag = final if final is not None else self.final
        save_dir = get_model_path(
            root=self.project_root, category=category, subdir=self.subdir, final=final_flag)
        save_path = save_dir / tokenizer_filename

        # CASE 2: training is not forced and saved tokenizer available
        if not force_retrain and os.path.exists(save_path):
            print(f"\n--- Loading existing BPE tokenizer from:\n{save_path}")
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

        plot_path = save_dir / f"vocabulary_growth_k{max_k}.png"
        bpe.plot_vocabulary_growth(save_path=plot_path)

        # Finally, save the tokenizer using save_tokenizer
        tokenizer_file_path = save_tokenizer(bpe,
                                   root=self.project_root,
                                   filename=tokenizer_filename,
                                   category=category,
                                   final=final_flag,
                                   )
        print(f"BPE tokenizer saved to: {tokenizer_file_path}")
        print(f"DEBUG: newly trained tokenizer.tokens length = {len(tokens)}")
        return tokens

    def setup_trainer(self, train_tokens, force_retrain=False, max_k=None):
        """Setup and train the model."""
        model_type = self.model_type.lower()

        # CLASSICAL N-GRAM MODEL
        if model_type.lower() == "ngram":
            self.trainer = NGramTrainer(config=self.config,
                                         model=None,
                                         tokens=train_tokens,
                                         k=max_k)
            self.trainer.final = self.final
            self.model = self.trainer.train(force_retrain=force_retrain,
                                             tune_lambdas=True,
                                             train_limit=None,
                                             valid_limit=None,
                                             final=self.final)
        else:
            raise NotImplementedError(
                f"Model type '{self.model_type}' not implemented")

        if model_type == "neural":
            neural_trainer = Neur(
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

    def train(self,
            train_text=None,
            valid_text=None,
            max_k=2000,
            force_retrain_tokenizer=False,  # -> Handle force_retrain of bpe
            force_retrain_model=False,  # -> Handle force_retrain of the model
            train_limit=10000,
            valid_limit=1000
            ):
        
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
                print(f"DEBUG: valid_tokens length = {len(valid_tokens)}")

        # STEP 3: train model
        self.setup_trainer(train_tokens=train_tokens, force_retrain=force_retrain_model, max_k=max_k)
        return self.model, train_tokens, valid_tokens

    def generate(self, prompt, max_length=50, from_pretrained=False):
        """
        Generate a sequence of tokens starting from a prompt using the selected LM.
        
        Args:
            prompt (str or list): Prompt text or token list.
            max_length (int): Maximum number of tokens to generate.
            from_pretrained (bool): If True, attempt to load a pre-trained model from final folder.
            
        Returns:
            list: Generated token sequence.
        """
        # --- Ensure prompt is tokenized ---
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("Tokenizer is not initialized. Cannot encode string prompt.")
            prompt_tokens = self.tokenizer.BPE_segmenter(prompt)
        else:
            prompt_tokens = prompt

        # --- Load pre-trained model if requested ---
        if from_pretrained:
            if self.model_type.lower() == "ngram":
                model_fname = f"ngram_model_n{self.config.n}_k{self.tokenizer.max_k}.pkl"
                model_folder = get_model_path(self.project_root, 
                                              category="models", 
                                              subdir="ngram", 
                                              final=True
                                )
                model_path = os.path.join(model_folder, model_fname)
                if os.path.exists(model_path):
                    print(f"Loading pre-trained N-gram model from: {model_path}")
                    trainer = NGramTrainer(config=self.config, model=None, tokens=prompt_tokens, k=self.config.n)
                    self.model = trainer._load_state(model_path, final=True)
                else:
                    print("No pre-trained N-gram model found in final folder. Using current model.")

        # --- Check that model exists ---
        if self.model is None:
            raise ValueError("No model available. Train a model first or set from_pretrained=True with an existing model.")

        # --- Generation for N-gram ---
        if self.model_type.lower() == "ngram":
            generated_text = self.model.generate_text(prompt_tokens, max_length=max_length)
            return generated_text

        raise NotImplementedError(f"Generation for model type '{self.model_type}' is not implemented.")


if __name__ == "__main__":
    # --- Config ---
    config = NgramConfig(n=3, device="cpu")

    # --- Load data ---
    train_text = load_shakespeare(version="train")
    valid_text = load_shakespeare(version="validation")

    # --- Initialize pipeline ---
    pipeline = LM_Pipeline(model_type="ngram", config=config, final=False)

    # --- Select mode ---
    mode = "generate"  # Options: "train", "generate", "generate_pretrained"

    if mode == "train":
        # --- Train model and prepare tokenizer ---
        model, train_tokens, valid_tokens = pipeline.train(
            train_text=train_text,
            valid_text=valid_text,
            max_k=800,
            force_retrain_tokenizer=True,
            force_retrain_model=True,
            train_limit=100000,
            valid_limit=10000
        )
        print(f"Train tokens (sample): {train_tokens[:20]}")
        print(f"Validation tokens (sample): {valid_tokens[:10]}")

    elif mode == "generate":
        # --- Ensure pipeline has tokenizer and model trained in this session ---
        pipeline.train(
            train_text=train_text,
            valid_text=valid_text,
            max_k=800,
            force_retrain_tokenizer=False,
            force_retrain_model=False,
            train_limit=100000,
            valid_limit=10000
        )
        prompt = "to be or not to"
        generated_text = pipeline.generate(prompt, max_length=1)
        print(f"\nGenerated text:\n{generated_text}")

    elif mode == "generate_pretrained":
        # --- Load tokenizer from disk ---
        project_root = pipeline.project_root
        tokenizer_path = project_root / "experiments" / "tokenizers" / "BPE_merges_k800.pkl"
        pipeline.tokenizer, _ = load_tokenizer(root=project_root, filename=tokenizer_path)

        # --- Generate from pre-trained model ---
        prompt = "To be or not to"
        generated_text = pipeline.generate(prompt, max_length=100, from_pretrained=True)
        print(f"\n{Colors.O}Generated text (pre-trained model):\n{generated_text}")
