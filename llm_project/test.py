import os
import torch

from llm_project.utils.debugg_utils import Colors
from llm_project.utils.dataloader import load_shakespeare
from llm_project.utils.file_manager import load_tokenizer, get_model_path, get_project_root
from llm_project.bpe.bytepair_encoding import normalize_text
from llm_project.models.ngrams.trainer import NGramTrainer
from llm_project.models.neural_fast.trainer import NeuralTrainer
from llm_project.models.configs.configs import (NgramConfig,
                                                NeuralConfig,
                                                GptConfig,
                                                NeuralFastConfig)

class NgramTester:
    def __init__(self, config):
        self.test_text = load_shakespeare("test")
        self.test_tokens = None
        self.config = config
        self.root = get_project_root()

        self.tokenizer = None
        self.tokenizer_tokens = None
        self.trainer = None
        self.model = None
        self.max_k = None

        self.token_to_id = None
        self.id2token = None
        self.unk_id = None

    def prepare_tokens(self):
        """
        Load an existing BPE tokenizer.
        """
        category = "tokenizers"
        tokenizer_dir = get_model_path(root=self.root, category=category)

        # List all files in tokenizer_dir
        tokenizer_files = sorted([f for f in os.listdir(tokenizer_dir) if f.endswith(".pkl")], reverse=True)

        if len(tokenizer_files) == 0:
            raise FileNotFoundError(f"{Colors.FAIL}[ERROR]{Colors.ENDC} No tokenizer file found in {tokenizer_dir}")
        if len(tokenizer_files) > 1:
            print(f"{Colors.WARNING}[WARN]{Colors.ENDC} Multiple tokenizer files found: {tokenizer_files}. Using the first one.")
        tokenizer_path = tokenizer_dir / tokenizer_files[0]
        print(f"\n{Colors.OKCYAN}[TOKENIZER]{Colors.ENDC} Loading BPE tokenizer from {tokenizer_path}")
        self.tokenizer, tokens = load_tokenizer(root=self.root, 
                                                filename=tokenizer_path
                                                )
        self.tokenizer_tokens = tokens
        # Enforce max_k in tokenizer
    
        self.max_k = self.tokenizer.max_k
        
        # ----------------- tokenize test_text -------------------
        print(f"{Colors.OKCYAN}[TOKENIZER]{Colors.ENDC} Applying {self.max_k} BPE merges on test text:")
        self.test_tokens = self.tokenizer.BPE_segmenter(self.test_text)
        return self.tokenizer, self.test_tokens

    def load_model(self):
        """
        Load a pre-trained N-gram model using its trainer.
        """
        if self.tokenizer_tokens is None:
            raise RuntimeError(f"{Colors.FAIL}[ERROR]{Colors.ENDC} Tokenizer not ready. Call prepare_tokens() before load_model().")

        self.trainer = NGramTrainer(
            config=self.config,
            model=None,
            tokens=self.tokenizer_tokens,
            k=self.max_k,
        )

        model_filename = f"ngram_model_n{self.config.n}_k{self.max_k}.pkl"
        self.model = self.trainer._load_state(filename=model_filename)
        print(f"{Colors.OKGREEN}[MODEL]{Colors.ENDC} N-gram model loaded successfully from {model_filename}")

    def sync(self):
        """
        Synchronize tokenizer, trainer, and model.
        Must be called before evaluate().
        """
        self.prepare_tokens()
        self.load_model()
        print(f"{Colors.OKGREEN}[TESTER]{Colors.ENDC} Tester synchronized: tokenizer and model ready")

    def evaluate(self):
        """
        Compute perplexity on the test set.
        """
        if self.model_type != "ngram":
            raise NotImplementedError(f"Evaluate not implemented for {self.model_type}")

        lambdas = self.model.lambdas.get("best")
        if lambdas is None:
            raise ValueError("No 'best' lambdas found in model")

        perplexity = self.trainer.compute_perplexity(self.test_tokens, list(lambdas))
        print(f"{Colors.OKCYAN}[RESULT]{Colors.ENDC} Perplexity on test set: {perplexity:.2f}")
        return perplexity

class NeuralTester:

    def __init__(self,
                config,
                ):
        self.test_text = load_shakespeare("test")
        self.test_tokens = None
        self.config = config
        self.root = get_project_root()

        self.tokenizer = None
        self.tokenizer_tokens = None
        self.trainer = None
        self.model = None
        self.max_k = None

        self.token_to_id = None
        self.id2token = None
        self.unk_id = None

    def prepare_tokens(self):
        """
        Load an existing BPE tokenizer.
        """
        category = "tokenizers"
        tokenizer_dir = get_model_path(root=self.root, category=category)

        # List all files in tokenizer_dir
        tokenizer_files = sorted([f for f in os.listdir(tokenizer_dir) if f.endswith(".pkl")], reverse=True)

        if len(tokenizer_files) == 0:
            raise FileNotFoundError(f"{Colors.FAIL}[ERROR]{Colors.ENDC} No tokenizer file found in {tokenizer_dir}")
        if len(tokenizer_files) > 1:
            print(f"{Colors.WARNING}[WARN]{Colors.ENDC} Multiple tokenizer files found: {tokenizer_files}. Using the first one.")
        tokenizer_path = tokenizer_dir / tokenizer_files[0]
        print(f"\n{Colors.OKCYAN}[TOKENIZER]{Colors.ENDC} Loading BPE tokenizer from {tokenizer_path}")
        self.tokenizer, tokens = load_tokenizer(root=self.root, 
                                                filename=tokenizer_path
                                                )
        self.tokenizer_tokens = tokens
        # Enforce max_k in tokenizer
    
        self.max_k = self.tokenizer.max_k
        
        # ----------------- tokenize test_text -------------------
        print(f"{Colors.OKCYAN}[TOKENIZER]{Colors.ENDC} Applying {self.max_k} BPE merges on test text:")
        self.test_tokens = self.tokenizer.BPE_segmenter(self.test_text)
        return self.tokenizer, self.test_tokens
    
    def load_model(self):
        """
        Load a pre-trained N-gram model using its trainer.
        """
        if self.tokenizer_tokens is None:
            raise RuntimeError(f"{Colors.FAIL}[ERROR]{Colors.ENDC} Tokenizer not ready. Call prepare_tokens() first.")

        self.trainer = NeuralTrainer(
            model=None,
            epochs=1,
            lr=None,
            tokens=self.tokenizer_tokens,
            batch_size=8,
            train_text=None,
            valid_text=None,
            config=self.config,
            max_k=self.max_k,
        )

        try:
            self.model = self.trainer._load_state()
        except FileNotFoundError:
            raise FileNotFoundError(f"{Colors.FAIL}[ERROR]{Colors.ENDC} No final model checkpoint found in. Train first.")

        # Sync vocabolario
        self.id2token = self.trainer.id2token
        self.token_to_id = self.trainer.token2id
        self.unk_id = self.token_to_id.get("<UNK>", None)

        print(f"{Colors.OKGREEN}[MODEL]{Colors.ENDC} Neural model loaded and synchronized")
        return self.model

    def sync(self):
        """
        Synchronize tokenizer and model before evaluation.
        """
        self.prepare_tokens()
        self.load_model()
        print(f"{Colors.OKGREEN}[TESTER]{Colors.ENDC} Tester synchronized: tokenizer and model ready")

    def evaluate(self):
        """
        Compute perplexity on the test set.
        """
        if self.model is None or self.trainer is None:
            raise RuntimeError(f"{Colors.FAIL}[ERROR]{Colors.ENDC} Model not loaded. Call sync() first.")

        # mappo token -> ids usando vocabolario del modello
        test_ids = []
        for tok in self.test_tokens:
            if tok in self.token_to_id:
                test_ids.append(self.token_to_id[tok])
            elif self.unk_id is not None:
                test_ids.append(self.unk_id)
            # se non c'è UNK, skippa
        test_ids = torch.tensor(test_ids, dtype=torch.long)

        perplexity = self.trainer.batch_perplexity(test_ids.tolist())
        print(f"{Colors.OKCYAN}[RESULT]{Colors.ENDC} Perplexity on test set: {perplexity:.2f}")
        return perplexity


if __name__ == "__main__":
    # ---------------- Neural Configuration ----------------
    neural_config = NeuralFastConfig(
        n=3,
        vocab_size=0,
        block_size=64,
        embd_dim=384,
        device="cpu",   # usa "cuda" se disponibile
    )

    # ---------------- Create Tester instance ----------------
    tester = NeuralTester(config=neural_config)

    # ---------------- Synchronize tokenizer + model ----------------
    print(f"{Colors.OKBLUE}[INFO]{Colors.ENDC} Preparing tokenizer and loading model...")
    tester.sync()

    # ---------------- Evaluate on test set ----------------
    print(f"{Colors.OKBLUE}[INFO]{Colors.ENDC} Evaluating Neural model on test set...")
    perplexity = tester.evaluate()
    print(f"\n{Colors.OKGREEN}[Test Complete]{Colors.ENDC} Neural model perplexity on test set: {perplexity:.2f}")

