from llm_project.utils.dataloader import load_shakespeare
from llm_project.utils.file_manager import (
    load_tokenizer,
    save_tokenizer,
    get_project_root,
    get_model_path
)
import os
import torch
import matplotlib.pyplot as plt
from llm_project.models.ngrams.trainer import NGramTrainer
from llm_project.models.neural_fast.trainer import NeuralTrainer
from llm_project.models.neural_embeddings.trainer import NeuralEmbedTrainer
from llm_project.models.gpt.train import GptTrainer
from llm_project.bpe.bytepair_encoding import BPE
from llm_project.utils.debugg_utils import Colors


class LM_Pipeline:
    def __init__(self, model_type, config, tokenizer=None, project_root=None, final=False):
        self.model_type = model_type
        self.config = config
        self.tokenizer = tokenizer
        self.device = self.config.device
        print(f"[INFO] Using device: {self.device}")
        self.model = None
        self.trainer = None
        self.project_root = project_root or get_project_root()
        self.subdir = None
        self.final = final

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
        tokenizer_dir = get_model_path(
            root=self.project_root, category=category, subdir=self.subdir, final=final_flag)
        save_path = tokenizer_dir / tokenizer_filename

        # CASE 2: training is not forced and saved tokenizer available
        if not force_retrain and os.path.exists(save_path):
            print(f"\n--- Loading existing BPE tokenizer from:\n{save_path}")
            # Load the tokenizer
            self.tokenizer, tokens = load_tokenizer(root=self.project_root,
                                                    filename=save_path,
                                                    final=final_flag)
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

        plot_path = tokenizer_dir / f"vocabulary_growth_k{max_k}.png"
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

    def setup_trainer(self, train_tokens, batch_size, val_tokens=None, force_retrain=False, max_k=None):
        """Setup and train the model with consistent token <-> id mapping."""

        model_type = self.model_type.lower()

        # --- USE BPE TOKENS AS DEFINITIVE VOCAB ---
        self.token_to_id = {tok: i for i, tok in enumerate(self.tokenizer.tokens)}
        self.id2token = {i: tok for tok, i in self.token_to_id.items()}
        self.config.vocab_size = len(self.token_to_id)

        # Convert train/val tokens to IDs
        self.train_ids = [self.token_to_id[tok] for tok in train_tokens if tok in self.token_to_id]
        self.valid_ids = [self.token_to_id[tok] for tok in val_tokens if val_tokens and tok in self.token_to_id] if val_tokens else None

        # --- MODEL TRAINING / LOADING ---
        if model_type == "ngram":
            self.trainer = NGramTrainer(config=self.config,
                                        model=None,
                                        tokens=train_tokens,
                                        k=max_k,
                                        final=self.final)
            self.trainer.train(final=self.final)
            self.model = self.trainer.model

        elif model_type == "neural":
            self.trainer = NeuralEmbedTrainer(model=None,
                                            batch_size=batch_size,
                                            epochs=10,
                                            lr=0.01,
                                            tokens=train_tokens,
                                            train_text=None,
                                            valid_text=None,
                                            config=self.config,
                                            max_k=max_k,
                                            root=self.project_root,
                                            print_every=50)
            self.trainer.train_ids = self.train_ids
            self.trainer.valid_ids = self.valid_ids
            self.trainer.id2token = self.id2token
            self.trainer.token2id = self.token_to_id

            self.trainer.train(epochs=self.trainer.epochs,
                            lr=self.trainer.lr,
                            patience=getattr(self.config, "patience", 3),
                            force_retrain=force_retrain)
            self.model = self.trainer.model

        elif model_type == "neuralfast":
            self.trainer = NeuralTrainer(model=None,
                                        batch_size=batch_size,
                                        epochs=10,
                                        lr=3e-4,
                                        tokens=train_tokens,
                                        train_text=None,
                                        valid_text=None,
                                        config=self.config,
                                        max_k=max_k,
                                        root=self.project_root,
                                        print_every=50,
                                        patience=3)
            self.trainer.train_ids = self.train_ids
            self.trainer.valid_ids = self.valid_ids
            self.trainer.id2token = self.id2token
            self.trainer.token2id = self.token_to_id

            self.trainer.train(epochs=self.trainer.epochs,
                            force_retrain=force_retrain,
                            final=self.final)
            self.model = self.trainer.model

        elif model_type == "gpt":
            gpt_trainer = GptTrainer(config=self.config, model=None, tokens=train_tokens, k=max_k)
            self.model = gpt_trainer.train(force_retrain=force_retrain,
                                        tune_lambdas=True,
                                        train_limit=None,
                                        valid_limit=None)
        else:
            raise NotImplementedError(f"Model type '{self.model_type}' not implemented.")


    def train(self,
            train_text=None,
            valid_text=None,
            max_k=2000,
            force_retrain_tokenizer=False,  # -> Handle force_retrain of bpe
            force_retrain_model=False,  # -> Handle force_retrain of the model
            train_limit=None,
            valid_limit=None
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
        self.setup_trainer(train_tokens=train_tokens, 
                           val_tokens=valid_tokens, 
                           batch_size=getattr(self.config, "batch_size", 64),
                           force_retrain=force_retrain_model, 
                           max_k=max_k)
        return self.model, train_tokens, valid_tokens
    
    def load_pretrained(self):
        """Tentativo di caricamento modello pretrained dal final folder."""
        model_type = self.model_type.lower()
        
        # Load tokenizer
        """Load pretrained tokenizer and model from final folder."""
        model_type = self.model_type.lower()

        # --- Load tokenizer ---
        tokenizer_folder = get_model_path(self.project_root, "tokenizers", final=True)
        tokenizer_file = os.path.join(tokenizer_folder, "best_tokenizer.pkl")
        if os.path.exists(tokenizer_file):
            print(f"[INFO] Loading pretrained tokenizer from {tokenizer_file}")
            self.tokenizer, tokens = load_tokenizer(root=self.project_root,
                                                    filename=tokenizer_file,
                                                    final=True)
            self.token_to_id = {tok: i for i, tok in enumerate(self.tokenizer.tokens)}
            self.id2token = {i: tok for tok, i in self.token_to_id.items()}
            self.config.vocab_size = len(self.token_to_id)
        else:
            raise FileNotFoundError(f"No pretrained tokenizer found at {tokenizer_file}")
        
        # Loading model 
        if model_type == "ngram":
            fname = f"ngram_model_n{self.config.n}_k{self.tokenizer.max_k}.pkl"
            folder = get_model_path(self.project_root, "models", "ngram", final=True)
            path = os.path.join(folder, fname)
            if os.path.exists(path):
                print(f"[INFO] Loading pretrained N-gram from {path}")
                trainer = NGramTrainer(config=self.config, model=None, tokens=[], k=self.config.n)
                self.model = trainer._load_state(path, final=True)  # <-- assign to self.model
                return self.model
        
        elif model_type == "neuralfast":
            fname = "best_model.pkl"
            folder = get_model_path(self.project_root, "models", "neuralfast", final=True)
            path = os.path.join(folder, "final", fname)
            if os.path.exists(path):
                print(f"[INFO] Loading pretrained NeuralFast from {path}")
                trainer = NeuralTrainer(
                    model=None,
                    batch_size=getattr(self.config, "batch_size", 64),
                    epochs=getattr(self.config, "epochs", 10),
                    lr=getattr(self.config, "lr", 3e-4),
                    tokens=[], train_text=None, valid_text=None,
                    config=self.config, max_k=getattr(self.tokenizer, "max_k", 1000),
                    root=self.project_root
                )
                self.model = trainer._load_state(filename=fname, final=True)
                
                return self.model
        else:
            raise NotImplementedError(f"No pretrained loader for {model_type}")
        
        return None


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
       # --- Load pre-trained model if requested ---
        if from_pretrained and self.model is None:
            self.load_pretrained()
            if self.model is None:
                raise ValueError("No pretrained model found.")

        # --- Ensure model is present ---
        if self.model is None:
            raise ValueError("No model available. Train a model first or set from_pretrained=True.")

        # --- Tokenize prompt ---
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("Tokenizer is not initialized. Cannot encode string prompt.")
            prompt_tokens = self.tokenizer.BPE_segmenter(prompt)
        else:
            prompt_tokens = prompt

        # --- Convert prompt tokens to IDs, map unknowns to UNK ---
        unk_id = self.token_to_id.get("UNK", 0)
        prompt_ids = [self.token_to_id.get(tok, unk_id) for tok in prompt_tokens]
        print(f"[DEBUG] vocab size of model: {self.model.embeddings.num_embeddings}")
        print(f"[DEBUG] len(tokenizer.tokens): {len(self.tokenizer.tokens)}")
        print(f"[DEBUG] max token ID in prompt_ids: {max(prompt_ids)}")
        print(f"[DEBUG] sample prompt_ids: {prompt_ids[:20]}")

        # --- Generation for different model types ---
        if self.model_type.lower() == "ngram":
            return self.model.generate_text(prompt_tokens, max_length=max_length)

        elif self.model_type.lower() in ["neural", "neuralfast"]:
            # Decide which generate method to use
            if self.model_type.lower() == "neural":
                generated_ids, generated_tokens, generated_text = self.model.generate(
                    start_ids=prompt_ids,
                    max_new_tokens=max_length,
                    stochastic=True,
                    top_k=30,
                    top_p=0.9,
                    temperature=1.0,
                    block_size=getattr(self.config, "block_size", self.model.n),
                    id2token=self.id2token,
                    unk_id=unk_id
                )
            else:  # neuralfast
                device = next(self.model.parameters()).device
                prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
                generated_ids, generated_tokens, generated_text = self.model.generate(
                    start_ids=prompt_tensor[0].tolist(),
                    max_new_tokens=max_length,
                    stochastic=True,
                    top_k=30,
                    top_p=0.9,
                    temperature=1.0,
                    block_size=getattr(self.config, "block_size", self.model.n),
                    id2token=self.id2token,
                    unk_id=unk_id
                )

            return generated_text

        else:
            raise NotImplementedError(f"Generation for model type '{self.model_type}' is not implemented.")
    
# ------------------ TEST -------------------

if __name__ == "__main__":

    from llm_project.models.configs.configs import NgramConfig, NeuralConfig, NeuralFastConfig

    train_text = load_shakespeare(version="train")
    valid_text = load_shakespeare(version="validation")

    model = "neuralfast"

    if model == "ngram":
    # --- Ngram model ---
        ngram_config = NgramConfig(n=3,
                               device="cpu",
                               lambdas=None)
    
        ngram_pipeline = LM_Pipeline(model_type=model,
                                    config=ngram_config,
                                    final=True
        )
        ngram_pipeline.train(train_text=train_text,
                             valid_text=valid_text,
                             max_k=2000,
                             force_retrain_model=False,
                             force_retrain_tokenizer=False,
                             train_limit=10000,
                             valid_limit=1000,
                        )
        
        """prompt = "To be, or not to be"
        generated_ngram = ngram_pipeline.generate(prompt, max_length=50, from_pretrained=True)
        print("\n N-gram generated text:")
        print(generated_ngram)"""

    elif model == "neuralfast":
    # --- Neural N-gram ---
        neural_config = NeuralFastConfig(n=3, 
                                    device=None,
                                    vocab_size=None,
                                    embd_dim=256, 
                                    block_size=32,
        )
        pipeline_neural = LM_Pipeline("neuralfast",
                                    neural_config, 
                                    final=True)
        
        model_neural, train_tokens_neural, valid_tokens_neural = pipeline_neural.train(
                                                                                train_text, 
                                                                                valid_text, 
                                                                                max_k=2000, 
                                                                                force_retrain_tokenizer=False, 
                                                                                force_retrain_model=False, 
                                                                                train_limit=None, 
                                                                                valid_limit= None
        )

        prompt = "To be, or not to be"
        generated_neural = pipeline_neural.generate(prompt, 
                                                max_length=100, 
                                                from_pretrained=True)
        print("\nNeural N-gram generated text:")
        print(generated_neural)

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
                                                                                force_retrain_tokenizer=True, 
                                                                                force_retrain_model=True, 
                                                                                train_limit=10000, 
                                                                                valid_limit=1000
        )

        prompt = "To be, or not to be"
        generated_neural = pipeline_neural.generate(prompt, 
                                                max_length=100, 
                                                from_pretrained=False)
        print("\nNeural N-gram generated text:")
        print(generated_neural)


    