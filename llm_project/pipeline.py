# -------------- TRAINERS IMPORTS ----------------
from llm_project.models.ngrams.trainer import NGramTrainer
from llm_project.models.neural_ngrams.trainer import NeuralNgramTrainer
from llm_project.models.neural_embeddings.trainer import NeuralEmbedTrainer
from llm_project.models.gpt.trainer import GptTrainer

# -------------- MODEL IMPORTS ----------------
from llm_project.models.neural_ngrams.model import NeuralNgram

# -------------- UTILS IMPORTS ----------------
from llm_project.utils.debugg_utils import Colors
from llm_project.utils.dataloader import load_shakespeare
from llm_project.utils.file_manager import (
    load_tokenizer,
    save_tokenizer,
    get_project_root,
    get_model_path,
)

# -------------- CONFIG IMPORTS ----------------
from llm_project.models.configs.configs import NgramConfig, NeuralConfig, GptConfig

# --------------GENERATOR IMPORTS ----------------
from llm_project.models.gpt.generator import Generator

# -------------- BPE IMPORTS ----------------
from llm_project.bpe.bytepair_encoding import BPE

# -------------- OTHER IMPORTS ----------------
import os
import matplotlib.pyplot as plt


class LM_Pipeline:
    def __init__(
        self, model_type, config, tokenizer=None, project_root=None, final=False
    ):
        self.model_type = model_type
        self.config = config
        self.tokenizer = tokenizer
        self.device = self.config.device
        self.model = None
        self.trainer = None
        self.project_root = project_root or get_project_root()
        self.subdir = None
        self.final = final

    def prepare_tokens(
        self,
        train_text=None,
        max_k=2000,
        force_retrain=False,
        train_limit=None,
        final=None,
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
        final_flag = final if final is not None else self.final
        save_dir = get_model_path(
            root=self.project_root,
            category=category,
            subdir=self.subdir,
            final=final_flag,
        )
        save_path = save_dir / tokenizer_filename

        # CASE 2: training is not forced and saved tokenizer available
        if not force_retrain and os.path.exists(save_path):
            print(f"\n--- Loading existing BPE tokenizer from:\n{save_path}")
            # Load the tokenizer
            self.tokenizer, tokens = load_tokenizer(
                root=self.project_root,
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
        tokenizer_file_path = save_tokenizer(
            bpe,
            root=self.project_root,
            filename=tokenizer_filename,
            category=category,
            final=final_flag,
        )
        print(f"BPE tokenizer saved to: {tokenizer_file_path}")
        print(f"DEBUG: newly trained tokenizer.tokens length = {len(tokens)}")
        return tokens

    def setup_trainer(
        self, train_tokens, batch_size, val_tokens=None, force_retrain=False, max_k=None
    ):
        """Setup and train the model."""
        model_type = self.model_type.lower()

        unique_tokens = sorted(set(train_tokens))
        self.token_to_id = {tok: i for i, tok in enumerate(unique_tokens)}
        self.id2token = {i: tok for tok, i in self.token_to_id.items()}

        self.config.vocab_size = len(self.token_to_id)

        ##########################
        # CLASSICAL N-GRAM MODEL #
        ##########################
        if model_type == "ngram":
            self.trainer = NGramTrainer(
                config=self.config, model=None, tokens=train_tokens, k=max_k
            )
            self.trainer.train()
            self.trainer.final = self.final
            self.model = self.trainer.model

        #######################
        # NEURAL N-GRAM MODEL #
        #######################
        elif model_type == "neural":
            self.trainer = NeuralEmbedTrainer(
                model=None,
                batch_size=batch_size,
                epochs=3,
                lr=0.01,
                tokens=train_tokens,
                train_text=None,
                valid_text=None,
                config=self.config,
                max_k=max_k,
                root=self.project_root,
                print_every=50,
            )
            # Converte tokens → ids
            self.trainer.train_ids = [
                self.token_to_id[tok] for tok in train_tokens if tok in self.token_to_id
            ]
            if val_tokens:
                self.trainer.val_ids = [
                    self.token_to_id[tok]
                    for tok in val_tokens
                    if tok in self.token_to_id
                ]

            # For some reason j rm this if
            if self.trainer.model is None:
                self.trainer.model = NeuralNgram(
                    n=self.config.n,
                    vocab_size=len(self.token_to_id),
                    embd_dim=self.config.embd_dim,
                )

            self.trainer.id2token = self.id2token
            self.trainer.token2id = self.token_to_id

            self.trainer.train(
                epochs=self.trainer.epochs,
                lr=self.trainer.lr,
                patience=getattr(self.config, "patience", 3),
                force_retrain=force_retrain,
            )
            self.model = self.trainer.model

        #############
        # GPT MODEL #
        #############
        elif model_type == "gpt":
            self.trainer = GptTrainer(
                config=self.config, model=None, tokens=train_tokens, k=max_k
            )
            self.model =

            self.train(
                force_retrain=force_retrain,
                tune_lambdas=True,
                train_limit=None,
                valid_limit=None,
            )
        else:
            raise NotImplementedError(
                f"Model type '{self.model_type}' not implemented")

    def train(
        self,
        train_text=None,
        valid_text=None,
        max_k=2000,
        force_retrain_tokenizer=False,  # -> Handle force_retrain of bpe
        force_retrain_model=False,  # -> Handle force_retrain of the model
        train_limit=10000,
        valid_limit=1000,
    ):
        """Full automatic pipeline: tokenizer → tokens → trainer → model"""

        if train_text is None:
            raise ValueError("train_text must be provided")

        # STEP 1: tokenizer + train tokens
        train_tokens = self.prepare_tokens(
            train_text=train_text,
            max_k=max_k,
            force_retrain=force_retrain_tokenizer,
            train_limit=train_limit,
        )
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
        self.setup_trainer(
            train_tokens=train_tokens,
            val_tokens=valid_tokens,
            batch_size=getattr(self.config, "batch_size", 64),
            force_retrain=force_retrain_model,
            max_k=max_k,
        )
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
                raise ValueError(
                    "Tokenizer is not initialized. Cannot encode string prompt."
                )
            prompt_tokens = self.tokenizer.BPE_segmenter(prompt)
        else:
            prompt_tokens = prompt

        # --- Load pre-trained model if requested ---
        if from_pretrained:
            if self.model_type.lower() == "ngram":
                model_fname = (
                    f"ngram_model_n{self.config.n}_k{self.tokenizer.max_k}.pkl"
                )
                model_folder = get_model_path(
                    self.project_root, category="models", subdir="ngram", final=True
                )
                model_path = os.path.join(model_folder, model_fname)
                if os.path.exists(model_path):
                    print(
                        f"Loading pre-trained N-gram model from: {model_path}")
                    trainer = NGramTrainer(
                        config=self.config,
                        model=None,
                        tokens=prompt_tokens,
                        k=self.config.n,
                    )
                    self.model = trainer._load_state(model_path, final=True)
                else:
                    print(
                        "No pre-trained N-gram model found in final folder. Using current model."
                    )

            else:
                raise NotImplementedError(
                    "No other model ready from retrieving pretrained models")

    # --- Check that model exists ---
    if self.model is None:
        if self.trainer is None:
            raise ValueError(
                "No trainer available. Train a model first or set from_pretrained=True."
            )

        model_folder = get_model_path(
            root=self.trainer.root,
            category="models",
            subdir=self.model_type.lower(),  # ngram / neural / gpt
            final=False,
        )
        print(f"Trying to load existing model from: {model_folder}")

        model_path = os.path.join(model_folder, "best_model.pkl")
        print(f"model path is: {model_path}")

        if os.path.exists(model_path):
            self.model = self.trainer._load_state(
                filename=model_path, final=True)
            print(
                f"[INFO] Loaded saved model automatically from saved_models for generation: {model_path}"
            )
        else:
            raise FileNotFoundError(
                f"No model found in saved_models: {model_path}")

    # --- Generation for N-gram ---
    if self.model_type.lower() == "ngram":
        generated_text = self.model.generate_text(
            prompt_tokens, max_length=max_length
        )
        return generated_text
    elif self.model_type.lower() == "neural":
        # Convert prompt tokens -> ids
        prompt_ids = [
            self.token_to_id[tok]
            for tok in prompt_tokens
            if tok in self.token_to_id
        ]

        unk_id = self.token_to_id.get("UNK", None)
        generated_ids, generated_tokens, generated_text = self.model.generate(
            prompt_ids,
            max_new_tokens=max_length,
            block_size=self.config.block_size,
            id2token=self.id2token,
            top_k=50,
            top_p=0.9,
            unk_id=unk_id
        )
        return generated_text

    raise NotImplementedError(
        f"Generation for model type '{self.model_type}' is not implemented."
    )


# ------------------ TEST -------------------


if __name__ == "__main__":

    from llm_project.models.configs.configs import NgramConfig, NeuralConfig

    train_text = load_shakespeare(version="train")
    valid_text = load_shakespeare(version="validation")

    model = "neural"
    if model == "ngram":
        # --- Ngram model ---
        ngram_config = NgramConfig(n=3,
                                   device="cpu",
                                   lambdas=None)

        ngram_pipeline = LM_Pipeline(model_type=model,
                                     config=ngram_config,
                                     )
        ngram_pipeline.train(train_text=train_text,
                             valid_text=valid_text,
                             max_k=2000,
                             force_retrain_model=False,
                             force_retrain_tokenizer=False,
                             train_limit=10000,
                             valid_limit=1000,
                             )

        prompt = "To be, or not to be"
        generated_ngram = ngram_pipeline.generate(
            prompt, max_length=50, from_pretrained=False)
        print("\n N-gram generated text:")
        print(generated_ngram)

    elif model == "neural":
        # --- Neural N-gram ---
        neural_config = NeuralConfig(n=3,
                                     device="cpu",
                                     vocab_size=None,
                                     embd_dim=256,
                                     block_size=8,
                                     )
        pipeline_neural = LM_Pipeline("neural",
                                      neural_config,
                                      final=False)

        model_neural, train_tokens_neural, valid_tokens_neural = pipeline_neural.train(
            train_text,
            valid_text,
            max_k=1000,
            force_retrain_tokenizer=False,
            force_retrain_model=False,
            train_limit=10000,
            valid_limit=1000
        )

    prompt = "To be, or not to be"
    generated_neural = pipeline_neural.generate(prompt,
                                                max_length=50,
                                                from_pretrained=False)
    print("\nNeural N-gram generated text:")
    print(generated_neural)
