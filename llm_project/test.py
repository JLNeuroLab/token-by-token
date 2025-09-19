import os
import torch

from llm_project.utils.debugg_utils import Colors
from llm_project.utils.dataloader import load_shakespeare
from llm_project.utils.file_manager import load_tokenizer, get_model_path, get_project_root
from llm_project.bpe.bytepair_encoding import normalize_text
from llm_project.models.ngrams.trainer import NGramTrainer
from llm_project.models.neural_fast.trainer import NeuralTrainer
from llm_project.models.gpt.trainer import GptTrainer
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
        self.test_text = load_shakespeare("test")[:10000]
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
        unique_tokens = set(self.tokenizer_tokens)
        print(f"[DEBUG] True tokenizer vocab size: {len(unique_tokens)}")
    
        self.max_k = self.tokenizer.max_k
        
        # ----------------- tokenize test_text -------------------
        print(f"{Colors.OKCYAN}[TOKENIZER]{Colors.ENDC} Applying {self.max_k} BPE merges on test text:")
        self.test_tokens = self.tokenizer.BPE_segmenter(self.test_text)
        print(f"[TOKENIZER] Tokenized test set length: {len(self.test_tokens)}")
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
        """ Compute perplexity on the test text and debug unknown tokens"""
        if self.model is None or self.trainer is None:
            raise RuntimeError(f"{Colors.FAIL}[ERROR]{Colors.ENDC} Model not loaded. Call sync() first.")

        test_ids = []
        unk_count = 0

        checkpoint_vocab = set(self.trainer.id2token.values())
        tokenizer_vocab = set(self.tokenizer_tokens)
        print(f"[DEBUG] Checkpoint vocab size: {len(checkpoint_vocab)}")
        print(f"[DEBUG] Tokenizer vocab size: {len(tokenizer_vocab)}")
        print(f"[DEBUG] Tokens in checkpoint not in tokenizer: {len(checkpoint_vocab - tokenizer_vocab)}")
        print(f"[DEBUG] Tokens in tokenizer not in checkpoint: {len(tokenizer_vocab - checkpoint_vocab)}")


        for tok in self.test_tokens:
            if tok in self.token_to_id:
                test_ids.append(self.token_to_id[tok])
            elif self.unk_id is not None:
                test_ids.append(self.unk_id)
                unk_count += 1

        total_tokens = len(test_ids)
        unk_ratio = unk_count / total_tokens if total_tokens > 0 else 0

        print(f"{Colors.OKBLUE}[INFO]{Colors.ENDC} Test tokens: {total_tokens}, Unknown tokens mapped to <UNK>: {unk_count}")
        print(f"{Colors.WARNING}[DEBUG]{Colors.ENDC} Ratio of <UNK> tokens: {unk_ratio:.4f}")

        if total_tokens == 0:
            raise RuntimeError(f"{Colors.FAIL}[ERROR]{Colors.ENDC} No valid tokens to evaluate.")

        test_ids_tensor = torch.tensor(test_ids, dtype=torch.long)
        perplexity = self.trainer.batch_perplexity(test_ids_tensor.tolist())
        print(f"{Colors.OKCYAN}[RESULT]{Colors.ENDC} Perplexity on test set: {perplexity:.2f}")
        return perplexity

class GptTester:
    def __init__(self, config):
        self.test_text = load_shakespeare("test")[:1000]
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
        tokenizer_files = sorted([f for f in os.listdir(tokenizer_dir) if f.endswith(".pkl")], reverse=True)

        if not tokenizer_files:
            raise FileNotFoundError(f"{Colors.FAIL}[ERROR]{Colors.ENDC} No tokenizer file found in {tokenizer_dir}")
        if len(tokenizer_files) > 1:
            print(f"{Colors.WARNING}[WARN]{Colors.ENDC} Multiple tokenizer files found: {tokenizer_files}. Using the first one.")

        tokenizer_path = tokenizer_dir / tokenizer_files[0]
        print(f"\n{Colors.OKCYAN}[TOKENIZER]{Colors.ENDC} Loading BPE tokenizer from {tokenizer_path}")
        self.tokenizer, tokens = load_tokenizer(root=self.root, filename=tokenizer_path)
        self.tokenizer_tokens = tokens

        self.max_k = getattr(self.tokenizer, "max_k", None)
        print(f"[DEBUG] True tokenizer vocab size: {len(self.tokenizer_tokens)}")

        # Tokenize test text
        print(f"{Colors.OKCYAN}[TOKENIZER]{Colors.ENDC} Applying {self.max_k} BPE merges on test text:")
        self.test_tokens = self.tokenizer.BPE_segmenter(self.test_text)
        print(f"[TOKENIZER] Tokenized test set length: {len(self.test_tokens)}")
        print(f"[DEBUG] Unique tokens in test set: {len(set(self.test_tokens))}")
        return self.tokenizer, self.test_tokens

    def load_model(self):
        """
        Load pre-trained GPT model using its trainer.
        """
        if self.tokenizer_tokens is None:
            raise RuntimeError(f"{Colors.FAIL}[ERROR]{Colors.ENDC} Tokenizer not ready. Call prepare_tokens() first.")

        self.token_to_id = {tok: i for i, tok in enumerate(self.tokenizer_tokens)}
        self.id2token = {i: tok for tok, i in self.token_to_id.items()}
        self.unk_id = self.token_to_id.get("<UNK>", None)

        # converti i token in ID numerici
        train_ids = [self.token_to_id[tok] for tok in self.tokenizer_tokens]
        valid_ids = [self.token_to_id[tok] for tok in self.tokenizer_tokens]

        tokens_dict = {"train": train_ids, "validation": valid_ids}

        self.trainer = GptTrainer(model=None, tokens=tokens_dict, config=self.config, k=self.max_k)

        try:
            self.model = self.trainer._load_state(self.trainer.model_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"{Colors.FAIL}[ERROR]{Colors.ENDC} No GPT checkpoint found. Train first.")

        print(f"{Colors.OKGREEN}[MODEL]{Colors.ENDC} GPT model loaded and synchronized")
        return self.model

    def sync(self):
        """
        Synchronize tokenizer and model.
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

        test_ids = []
        unk_count = 0
        vocab_size = self.trainer.config.vocab_size  # vocab size del modello
        unk_id = self.unk_id or 0

        for tok in self.test_tokens:
            if tok in self.token_to_id:
                tok_id = self.token_to_id[tok]
                # clamp ID all’interno del range del modello
                if tok_id >= vocab_size:
                    test_ids.append(unk_id)
                    unk_count += 1
                else:
                    test_ids.append(tok_id)
            else:
                test_ids.append(unk_id)
                unk_count += 1

        total_tokens = len(test_ids)
        ratio_unk = unk_count / max(1, total_tokens)
        print(f"{Colors.OKBLUE}[INFO]{Colors.ENDC} Test tokens: {total_tokens}, Unknown tokens mapped to <UNK>: {unk_count}")
        print(f"[DEBUG] Ratio of <UNK> tokens: {ratio_unk:.4f}")

        if total_tokens == 0:
            raise RuntimeError(f"{Colors.FAIL}[ERROR]{Colors.ENDC} No valid tokens to evaluate.")

        test_ids_tensor = torch.tensor(test_ids, dtype=torch.long)
        perplexity = self.trainer.compute_perplexity(test_ids_tensor.tolist())
        print(f"{Colors.OKCYAN}[RESULT]{Colors.ENDC} Perplexity on test set: {perplexity:.2f}")
        return perplexity



if __name__ == "__main__":
    gpt_config = GptConfig(
    vocab_size=None,   
    n_heads=6,   
    layer_dim=4,      
    embd_dim=384,      
    block_size=64,     
    embd_pdrop=0.2,
    attn_pdrop=0.2,
    resid_pdrop=0.2,
    dropout=0.2,      
    device="cuda" if torch.cuda.is_available() else "cpu",
)

    tester = GptTester(config=gpt_config)
    print(f"{Colors.OKBLUE}[INFO]{Colors.ENDC} Preparing tokenizer and loading model...")
    tester.sync()

    print(f"{Colors.OKBLUE}[INFO]{Colors.ENDC} Evaluating GPT model on test set...")
    ppl = tester.evaluate()
    print(f"\n{Colors.OKGREEN}[Test Complete]{Colors.ENDC} GPT model perplexity on test set: {ppl:.2f}")
