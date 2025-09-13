# -------------- UTILS IMPORTS ----------------
from llm_project.utils.debugg_utils import Colors
from llm_project.utils.dataloader import load_shakespeare
from llm_project.utils.file_manager import (
    load_tokenizer,
    save_tokenizer,
    get_project_root,
    get_model_path,
    save_model,
)

# -------------- TRAINERS IMPORTS ----------------
from llm_project.models.ngrams.trainer import NGramTrainer
from llm_project.models.neural_fast.trainer import NeuralTrainer
from llm_project.models.neural_embeddings.trainer import NeuralEmbedTrainer
from llm_project.models.gpt.trainer import GptTrainer

# -------------- MODEL IMPORTS ----------------
# from llm_project.models.neural_ngrams.model import NeuralNgram

# -------------- CONFIG IMPORTS ----------------
from llm_project.models.configs.configs import (
    NgramConfig,
    NeuralConfig,
    GptConfig,
    NeuralFastConfig,
)

# --------------GENERATOR IMPORT ----------------
from llm_project.models.gpt.generator import Generator

# -------------- BPE IMPORTS ----------------
from llm_project.bpe.bytepair_encoding import BPE

# -------------- OTHER IMPORTS ----------------
import os
import torch
import matplotlib.pyplot as plt
import inspect
import re

# Model name aliases
NGRAM_NAMES = {
    "ngram",
    "ngram_classic",
}
NEURAL_SLOW_NAMES = {
    "neural_numpy",
    "neural_slow",
    "neural_cpu",
}
NEURAL_FAST_NAMES = {
    "neural_torch",
    "neural_fast",
    "neural_gpu",
}
GPT_NAMES = {
    "gpt",
}
NEURAL_ALL = NEURAL_SLOW_NAMES | NEURAL_FAST_NAMES
ALL_MODEL_CHOICES = sorted(list(NGRAM_NAMES | NEURAL_ALL | GPT_NAMES))


class LM_Pipeline:
    def __init__(
        self, model_type, config, tokenizer=None, project_root=None, final=False
    ):
        self.model_type = model_type
        self.config = config
        self.tokenizer = tokenizer
        self.device = self.config.device
        print(
            f"{Colors.OKBLUE}[INFO]{Colors.ENDC}{Colors.HEADER} Using device: {str(self.device).upper()}{Colors.ENDC}"
        )
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
        mt = (self.model_type or "").lower()  # Making sure it exists

        # --- CASE 1: tokenizer available (in the eventuality you want to add one) and train not forced
        if self.tokenizer is not None and not force_retrain:
            # Load the existing tokenizer
            print(f"{Colors.OKCYAN}[TOKENIZER]{Colors.ENDC} Using provided tokenizer.")
            tokens = (
                self.tokenizer.tokens[:train_limit]
                if train_limit
                else self.tokenizer.tokens
            )

            print(
                f"{Colors.OKCYAN}[DEBUG]{Colors.ENDC} tokenizer.tokens length = {len(tokens)}"
            )
            # Returns tokens, will be useful for setting up the trainer

            # Neural slow recives string instead of int ids somehow
            # now not asume ints
            mt = (self.model_type or "").lower()
            if mt in NEURAL_ALL:
                sample = tokens[0] if tokens else None
                if isinstance(sample, int):
                    # BPE emits ints → identity mapping
                    unique_ids = list(dict.fromkeys(tokens))
                    self.token_to_id = {tid: tid for tid in unique_ids}
                    self.id2token = {tid: tid for tid in unique_ids}
                    self.unk_id = (
                        0 if getattr(self, "unk_id", None) is None else int(self.unk_id)
                    )
                else:
                    # BPE emits strings → build stoi/itos
                    unique_toks = list(dict.fromkeys(tokens))
                    self.token_to_id = {t: i for i, t in enumerate(unique_toks)}
                    self.id2token = {i: t for i, t in enumerate(unique_toks)}
                    self.unk_id = 0

            # ensure k is available to the rest of the pipeline (checkpoint naming, etc.)
            self.max_k = int(getattr(self.tokenizer, "k", max_k))
            setattr(self.tokenizer, "k", self.max_k)

            return tokens

        # Preparing parameters for saving and loading tokenizers
        tokenizer_filename = f"BPE_merges_k{max_k}.pkl"
        category = "tokenizers"
        final_flag = final if final is not None else self.final

        tokenizer_dir = get_model_path(
            root=self.project_root,
            category=category,
            subdir=self.subdir,
            final=final_flag,
        )
        save_path = tokenizer_dir / tokenizer_filename
        # -----------------------------------------------------------------------

        # CASE 2: training is not forced and saved tokenizer available
        if not force_retrain and os.path.exists(save_path):
            print(
                f"\n{Colors.OKCYAN}[TOKENIZER]{Colors.ENDC} Loading existing BPE tokenizer from:\n{save_path}"
            )
            # Load the tokenizer
            self.tokenizer, tokens = load_tokenizer(
                root=self.project_root,
                filename=save_path,
                final=final_flag,
            )
            if train_limit:
                tokens = tokens[:train_limit]
                self.tokenizer.tokens = tokens
            print(
                f"{Colors.OKCYAN}[DEBUG]{Colors.ENDC}: loaded tokenizer.tokens length = {len(tokens)}"
            )
            try:
                setattr(self.tokenizer, "k", int(max_k))
            except Exception:
                pass

            # --- keep k consistent everywhere ---
            self.max_k = int(getattr(self.tokenizer, "k", max_k))
            setattr(self.tokenizer, "k", self.max_k)

            # --- build identity vocab for neural variants when loading a tokenizer ---
            if mt in NEURAL_ALL:
                sample = tokens[0] if tokens else None
                if isinstance(sample, int):
                    # BPE emits ints → identity mapping
                    unique_ids = list(dict.fromkeys(tokens))
                    self.token_to_id = {tid: tid for tid in unique_ids}
                    self.id2token = {tid: tid for tid in unique_ids}
                    self.unk_id = (
                        0 if getattr(self, "unk_id", None) is None else int(self.unk_id)
                    )
                else:
                    # BPE emits strings → build stoi/itos
                    unique_toks = list(dict.fromkeys(tokens))
                    self.token_to_id = {t: i for i, t in enumerate(unique_toks)}
                    self.id2token = {i: t for i, t in enumerate(unique_toks)}
                    self.unk_id = 0

            return tokens

        if train_text is None:
            raise ValueError(
                f"{Colors.FAIL}[ERROR]{Colors.ENDC} train_text must be provided to train a new BPE tokenizer."
            )
        # If dataloader returned list/tuple, flatten to a single string IDK LETS SEE
        # if isinstance(train_text, (list, tuple)):
        #    train_text = "".join(train_text)

        # Limit the training text before BPE to save time
        if train_limit:
            train_text = train_text[:train_limit]
        # -----------------------------------------------------------------------

        # CASE 3: Train a new tokenizer
        print(f"{Colors.OKCYAN}[TOKENIZER]{Colors.ENDC} Training new BPE tokenizer...")
        bpe = BPE(max_k=max_k, text=train_text)
        bpe.BPE_encoder()
        tokens = bpe.tokens
        self.tokenizer = bpe

        # ---- make k available everywhere (trainer filename glue, lazy load) ----
        self.max_k = int(max_k)
        # also attach to tokenizer so other codepaths can read it
        try:
            setattr(self.tokenizer, "k", int(max_k))
        except Exception:
            pass

        # Fix UNK part
        # Build vocab mapping ONLY for neural trainers (identity over BPE ids)
        if mt in NEURAL_ALL:
            sample = tokens[0] if tokens else None
            if isinstance(sample, int):
                # BPE emits ints → identity mapping
                unique_ids = list(dict.fromkeys(tokens))
                self.token_to_id = {tid: tid for tid in unique_ids}
                self.id2token = {tid: tid for tid in unique_ids}
                self.unk_id = (
                    0 if getattr(self, "unk_id", None) is None else int(self.unk_id)
                )
            else:
                # BPE emits strings → build stoi/itos
                unique_toks = list(dict.fromkeys(tokens))
                self.token_to_id = {t: i for i, t in enumerate(unique_toks)}
                self.id2token = {i: t for i, t in enumerate(unique_toks)}
                self.unk_id = 0
            final_flag = getattr(self, "final", False)

            # optional: persist mapping next to tokenizer
            try:
                # vocab_dir = get_model_path(
                #     self.project_root,
                #     "tokenizers",
                #     subdir=self.subdir,
                #     final=final_flag,
                # ) not used for now anymore
                save_model(
                    {"vocab_tokens": list(self.token_to_id.keys())},
                    root=self.project_root,
                    category="tokenizers",
                    subdir=self.subdir,
                    filename=f"BPE_vocab_k{max_k}.pkl",
                )
            except Exception as e:
                print(f"{Colors.WARNING}[WARN]{Colors.ENDC} Skipped saving vocab: {e}")
        # Fix UNK part, end of block.

        plot_path = tokenizer_dir / f"vocabulary_growth_k{max_k}.png"
        bpe.plot_vocabulary_growth(save_path=plot_path)

        # Finally, save the tokenizer using save_tokenizer
        tokenizer_file_path = save_tokenizer(
            bpe,
            root=self.project_root,
            filename=tokenizer_filename,
            category=category,
            final=final_flag,
        )
        print(
            f"{Colors.OKGREEN}[OK]{Colors.ENDC} BPE tokenizer saved to: {tokenizer_file_path}"
        )
        print(
            f"{Colors.OKCYAN}[DEBUG]{Colors.ENDC} Newly trained tokenizer.tokens length = {len(tokens)}"
        )
        return tokens

    def setup_trainer(
        self,
        train_tokens,
        batch_size,
        val_tokens=None,
        force_retrain=False,
        max_k=None,
        train_ids=None,
        valid_ids=None,
    ):
        """Setup and train the model with consistent token <-> id mapping."""
        model_type = self.model_type.lower()

        # Ensure vocab_size for trainers that need it (i.e. neurals)
        if hasattr(self, "token_to_id"):
            self.config.vocab_size = len(self.token_to_id)

        ##########################
        # CLASSICAL N-GRAM MODEL #
        ##########################
        if model_type in NGRAM_NAMES:
            self.trainer = NGramTrainer(
                config=self.config, model=None, tokens=train_tokens, k=max_k
            )
            self.trainer.train()
            self.trainer.final = self.final
            # final=self.final why inside paramms and not mine NEW FROM MERGE
            self.model = self.trainer.model
            # keep k/max_k coherent for filename glue and later saves
            if getattr(self, "max_k", None) is None and hasattr(self.trainer, "k"):
                self.max_k = self.trainer.k
            return

        #######################
        # NEURAL SLOW (NUMPY) #
        #######################
        elif model_type in NEURAL_SLOW_NAMES:
            self.trainer = NeuralEmbedTrainer(
                model=None,
                epochs=getattr(self.config, "epochs", 3),
                lr=getattr(self.config, "lr", 0.01),
                tokens=None,  # we train on ids now
                batch_size=batch_size,
                train_text=None,
                valid_text=None,
                config=self.config,
                max_k=max_k,
                root=self.project_root,
                print_every=50,
            )
            # Now inject mapping + ids
            self.trainer.set_vocab(self.token_to_id, self.id2token, self.unk_id)
            self.trainer.train_ids = train_ids
            self.trainer.val_ids = valid_ids

            # Lil Sanity check (UNK rate). Should be 0 on train and very little on val
            unk = self.unk_id
            ti, vi = self.trainer.train_ids or [], self.trainer.val_ids or []
            unk_train = sum(i == unk for i in ti) / max(1, len(ti))
            unk_val = sum(i == unk for i in vi) / max(1, len(vi))
            print(
                f"{Colors.OKCYAN}[CHECK OF SANIDAD]{Colors.ENDC} Good, you're not crazy. UNK train={unk_train:.2%}  UNK val={unk_val:.2%}"
            )

            self.trainer.train(
                epochs=getattr(self.config, "epochs", 3),
                lr=getattr(self.config, "lr", 0.01),
                patience=getattr(self.config, "patience", 3),
                force_retrain=force_retrain,
            )
            self.model = self.trainer.model
            return

        #######################
        # NEURAL FAST (TORCH) #
        #######################
        elif model_type in NEURAL_FAST_NAMES:
            # vocab size with prefer mapping, else ids, else distinct tokens
            if hasattr(self, "token_to_id") and self.token_to_id:
                V = len(self.token_to_id)
            elif train_ids is not None and len(train_ids) > 0:
                V = int(max(train_ids)) + 1
            else:
                V = len(set(train_tokens))
            self.config.vocab_size = V

            # Pass a vocab-cover list so trainer uses correct V
            tokens_vocab = list(range(V))

            if NeuralTrainer is None:
                raise RuntimeError(
                    f"{Colors.FAIL}[ERROR]{Colors.ENDC} NeuralTrainer (fast) not found. Check llm_project/models/neural_fast/trainer.py"
                )
            self.trainer = NeuralTrainer(
                model=None,
                batch_size=batch_size,
                epochs=getattr(self.config, "epochs", 10),
                lr=getattr(self.config, "lr", 3e-4),
                tokens=train_tokens,  # Remember it's Not used if train_ids are present now
                train_text=None,
                valid_text=None,
                config=self.config,
                max_k=max_k,
                root=self.project_root,
                print_every=50,
                patience=getattr(self.config, "patience", 3),
            )

            if hasattr(self.trainer, "set_vocab"):
                self.trainer.set_vocab(self.token_to_id, self.id2token, self.unk_id)
            else:
                # fallback if trainer has no helper
                self.trainer.token2id = self.token_to_id
                self.trainer.id2token = self.id2token
                self.trainer.unk_id = self.unk_id

            ids_train = train_ids if train_ids is not None else train_tokens
            ids_val = valid_ids if valid_ids is not None else val_tokens
            self.trainer.train_ids = ids_train
            self.trainer.val_ids = ids_val  # common name in slow trainer
            self.trainer.valid_ids = ids_val  # some fast trainers use this name

            self.trainer.train(
                epochs=getattr(self.config, "epochs", 10),
                lr=getattr(self.config, "lr", 3e-4),
                force_retrain=force_retrain,
                final=self.final,
            )
            self.model = self.trainer.model
            return

        #############
        # GPT MODEL #
        #############
        elif model_type in GPT_NAMES:
            bs = int(getattr(self.config, "block_size", 64))

            # Choose source sequences
            tr_seq = train_ids if train_ids is not None else train_tokens
            va_seq = valid_ids if valid_ids is not None else valid_tokens

            # --- ensure integer ids, prefer tokenizer's own mapping if present ---
            def _ensure_int_ids(seq, build=False):
                if seq is None:
                    return None
                if len(seq) > 0 and isinstance(seq[0], int):
                    # already ids
                    return seq

                # Try to use tokenizer's mapping (so we can decode with tokenizer.decode later)
                if (
                    hasattr(self, "tokenizer")
                    and hasattr(self.tokenizer, "token_to_id")
                    and hasattr(self.tokenizer, "id_to_token")
                ):
                    if not hasattr(self, "gpt_stoi"):
                        self.gpt_stoi = dict(self.tokenizer.token_to_id)
                        self.gpt_itos = dict(self.tokenizer.id_to_token)
                        self.decode_mode = "bpe"
                else:
                    # Build a local, stable mapping from the training+validation streams (identity over order)
                    if build or not hasattr(self, "gpt_stoi"):
                        uniq = []
                        seen = set()
                        for t in (train_tokens or []) + (valid_tokens or []):
                            if t not in seen:
                                seen.add(t)
                                uniq.append(t)
                        self.gpt_stoi = {t: i for i, t in enumerate(uniq)}
                        self.gpt_itos = {i: t for i, t in enumerate(uniq)}
                        self.decode_mode = "local"
                return [self.gpt_stoi[t] for t in seq if t in self.gpt_stoi]

            # Build ids (and mapping) once using training first
            tr_ids = _ensure_int_ids(tr_seq, build=True)
            va_ids = _ensure_int_ids(va_seq, build=False)

            # Make sure validation is long enough for the block size
            if va_ids is None or len(va_ids) < (bs + 1):
                split_idx = max(int(0.9 * len(tr_ids)), bs + 1)
                tr_ids, va_ids = tr_ids[:split_idx], tr_ids[split_idx:]
                if len(va_ids) < (bs + 1):
                    tail = bs + 2
                    tr_ids = tr_ids[:-tail]
                    va_ids = tr_ids[-tail:]
                    print(
                        f"{Colors.OKCYAN}[OK]{Colors.ENDC} Validation (val_ids) > Block size"
                    )

            # Vocab size for GPT = size of stoi we just made
            self.config.vocab_size = len(self.gpt_stoi)
            tokens_dict = {"train": tr_ids, "validation": va_ids}

            self.trainer = GptTrainer(
                config=self.config, model=None, tokens=tokens_dict, k=max_k
            )
            self.model = self.trainer.train(
                force_retrain=force_retrain,
                train_limit=None,
                valid_limit=None,
            )
            print(
                f"{Colors.OKCYAN}[MODEL]{Colors.ENDC} Will load GPT from: {self.trainer.model_path}, vocab_size={self.config.vocab_size}"
            )
            return

        else:
            raise NotImplementedError(
                f"{Colors.FAIL}[ERROR]{Colors.ENDC} Model type '{self.model_type}' could not implemented."
            )

    # =============================================================================================
    #                                      TRAINING BLOCK                                        #
    # =============================================================================================

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
            raise ValueError(
                f"{Colors.FAIL}[ERROR]{Colors.ENDC} train_text must be provided"
            )

        # STEP 1: tokenizer + train tokens
        train_tokens = self.prepare_tokens(
            train_text=train_text,
            max_k=max_k,
            force_retrain=force_retrain_tokenizer,
            train_limit=train_limit,
        )
        print(
            f"{Colors.OKCYAN}[DEBUG]{Colors.ENDC}: train_tokens length = {len(train_tokens)}"
        )

        # STEP 2: valid tokens
        if valid_text is not None and valid_limit:
            valid_text = valid_text[:valid_limit]
        # Applying BPE merges to validation text

        valid_tokens = self.tokenizer.BPE_segmenter(valid_text) if valid_text else None
        if valid_tokens is not None and valid_limit:
            valid_tokens = valid_tokens[:valid_limit]
            print(
                f"{Colors.OKCYAN}[DEBUG]{Colors.ENDC}: valid_tokens length = {len(valid_tokens)}"
            )

        # Fix UNK part 2
        def to_ids(toks):
            if toks is None:
                return None
            if hasattr(self, "token_to_id"):  # only set for neural_slow
                unk = getattr(self, "unk_id", 0)
                return [self.token_to_id.get(t, unk) for t in toks]
            return toks  # n-gram & GPT already operate on integer BPE ids

        train_ids = to_ids(train_tokens)
        valid_ids = to_ids(valid_tokens)
        # End of fix

        # STEP 3: train model
        self.setup_trainer(
            train_tokens=train_tokens,
            val_tokens=valid_tokens,
            batch_size=getattr(self.config, "batch_size", 64),
            force_retrain=force_retrain_model,
            max_k=max_k,
            train_ids=train_ids,  # New
            valid_ids=valid_ids,  # New
        )
        return self.model, train_tokens, valid_tokens

    def load_pretrained(self):
        """Load pretrained tokenizer + model using each trainer's own model_path."""
        # --- tokenizer (final) ---
        tok_dir = get_model_path(self.project_root, "tokenizers", final=True)
        tok_file = os.path.join(tok_dir, "best_tokenizer.pkl")
        if not os.path.exists(tok_file):
            raise FileNotFoundError(
                f"{Colors.FAIL}[ERROR]{Colors.ENDC} No pretrained tokenizer found at {tok_file}"
            )
        self.tokenizer, tokens = load_tokenizer(
            root=self.project_root, filename=tok_file, final=True
        )
        # identity vocab for neural
        if (self.model_type or "").lower() in NEURAL_ALL:
            uniq = list(dict.fromkeys(tokens))
            self.token_to_id = {tid: tid for tid in uniq}
            self.id2token = {tid: tid for tid in uniq}
            self.unk_id = 0

        # --- trainer stub to get model_path ---
        mt = (self.model_type or "").lower()
        k = getattr(self.tokenizer, "k", getattr(self, "max_k", None))
        if mt in NGRAM_NAMES:
            tr = NGramTrainer(
                config=self.config,
                model=None,
                tokens=[],
                k=getattr(self.config, "n", 3),
            )
        elif mt in NEURAL_SLOW_NAMES:
            tr = NeuralEmbedTrainer(
                model=None,
                epochs=1,
                lr=0.01,
                tokens=None,
                batch_size=1,
                train_text=None,
                valid_text=None,
                config=self.config,
                max_k=k,
                root=self.project_root,
                print_every=50,
            )
            tr.set_vocab(
                getattr(self, "token_to_id", {}),
                getattr(self, "id2token", {}),
                getattr(self, "unk_id", 0),
            )
        elif mt in NEURAL_FAST_NAMES:
            V = len(getattr(self, "token_to_id", {})) or 1
            tr = NeuralTrainer(
                model=None,
                epochs=1,
                lr=3e-4,
                tokens=list(range(V)),
                batch_size=1,
                train_text=None,
                valid_text=None,
                config=self.config,
                max_k=k,
                root=self.project_root,
                print_every=50,
                patience=1,
            )
        elif mt in GPT_NAMES:
            # minimal cover so GptTrainer computes consistent model_path
            self.config.vocab_size = getattr(self.config, "vocab_size", 0) or 2
            tokens_dict = {
                "train": [0, 1, 0, 1, 0, 1, 0],
                "validation": [1, 0, 1, 0, 1, 0, 1],
            }
            tr = GptTrainer(config=self.config, model=None, tokens=tokens_dict, k=k)
        else:
            raise NotImplementedError(
                f"{Colors.FAIL}[ERROR]{Colors.ENDC} Unknown model_type: {mt}"
            )

        model_path = getattr(tr, "model_path", None)
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"{Colors.FAIL}[ERROR]{Colors.ENDC} No pretrained checkpoint at: {model_path}"
            )

        if hasattr(tr, "_load_state"):
            tr._load_state(model_path)
            self.model = tr.model
        elif hasattr(tr, "load"):
            self.model = tr.load(model_path)
        else:
            raise RuntimeError(
                f"{Colors.FAIL}[ERROR]{Colors.ENDC} Trainer has no loader method (_load_state/load)."
            )
        return self.model

    def _postprocess_text(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"([,.;:!?])([A-Za-z])", r"\1 \2", text)
        return text.strip()

    def legacy_generate(self, prompt, max_length=50, from_pretrained=False):
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
            raise ValueError(
                "No model available. Train a model first or set from_pretrained=True."
            )

        # --- Tokenize prompt ---
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError(
                    f"{Colors.FAIL}[ERROR]{Colors.ENDC} Tokenizer is not initialized. Cannot encode string prompt."
                )
            prompt_tokens = self.tokenizer.BPE_segmenter(prompt)
        else:
            prompt_tokens = prompt

        # << << << < HEAD
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
                    print(f"Loading pre-trained N-gram model from: {model_path}")
                    trainer = NGramTrainer(
                        config=self.config,
                        model=None,
                        tokens=prompt_tokens,
                        k=self.config.n,
                    )
                    self.model = trainer._load_state(model_path, final=True)
                else:
                    print(
                        f"{Colors.WARNING}[!!!]{Colors.ENDC} No pre-trained N-gram model found in final folder. Using current model."
                    )

            else:
                raise NotImplementedError(
                    f"{Colors.WARNING}[!!!]{Colors.ENDC} No other model ready from retrieving pretrained models"
                )

    # =============================================================================================
    #                                       GENERATE BLOCK                                       #
    # =============================================================================================
    def generate(self, prompt, max_length=50, from_pretrained=False):
        """
        Generate a sequence using the selected LM.
        """
        mt = (self.model_type or "").lower()  # makes sure mt exists

        # --- Ensure prompt is tokenized ---
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError(
                    f"{Colors.FAIL}[ERROR]{Colors.ENDC} Tokenizer is not initialized. Cannot encode string prompt."
                )
            prompt_tokens = self.tokenizer.BPE_segmenter(prompt)
        else:
            prompt_tokens = prompt

        ##########################
        # CLASSICAL N-GRAM MODEL #
        ##########################

        if mt in NGRAM_NAMES:
            # 1) Try to load a pre-trained n-gram if requested
            if from_pretrained and self.model is None:
                k_eff = getattr(
                    getattr(self, "tokenizer", None), "k", getattr(self, "max_k", None)
                )
                model_fname = f"ngram_model_n{self.config.n}_k{k_eff}.pkl"
                model_folder = get_model_path(
                    self.project_root, category="models", subdir="ngram", final=True
                )
                model_path = os.path.join(model_folder, model_fname)
                if os.path.exists(model_path):
                    print(f"[OK] Loading pre-trained N-gram model from: {model_path}")
                    # When loading, trainer can be created without prompt tokens;
                    # it needs config and (optionally) corpus tokens if its loader expects them.
                    loader = NGramTrainer(
                        config=self.config, model=None, tokens=None, k=self.config.n
                    )
                    self.model = loader._load_state(model_path, final=True)

            # 2) If still no model, lazy-build it from the corpus tokens prepared earlier
            if self.model is None:
                corpus_tokens = getattr(
                    getattr(self, "tokenizer", None), "tokens", None
                )
                if not corpus_tokens:
                    raise RuntimeError(
                        f"{Colors.FAIL}[ERROR]{Colors.ENDC} No corpus tokens for n-gram. "
                        "Ensure prepare_tokens() ran before generate()."
                    )
                self.trainer = NGramTrainer(
                    config=self.config,
                    model=None,
                    tokens=corpus_tokens,
                    k=self.config.n,
                )
                self.trainer.train()
                self.model = self.trainer.model

            # 3) Generate with the prompt tokens
            txt = self.model.generate_text(prompt_tokens, max_length=max_length)
            return self._postprocess_text(txt)

        #######################
        # NEURAL N-GRAM MODEL #
        #######################
        elif mt in NEURAL_ALL:
            # ---- lazy-load neural checkpoints if needed ----
            if self.model is None:
                k = getattr(self, "max_k", getattr(self.tokenizer, "k", None))
                if mt in NEURAL_FAST_NAMES:
                    # fast trainer: pass vocab-cover so vocab_size is right
                    V = len(getattr(self, "token_to_id", {}))
                    tokens_vocab = list(range(V)) if V else None
                    self.trainer = NeuralTrainer(
                        model=None,
                        epochs=getattr(self.config, "epochs", 10),
                        lr=getattr(self.config, "lr", 3e-4),
                        tokens=tokens_vocab,
                        batch_size=getattr(self.config, "batch_size", 32),
                        train_text=None,
                        valid_text=None,
                        config=self.config,
                        max_k=k,
                        root=self.project_root,
                        print_every=50,
                        patience=getattr(self.config, "patience", 3),
                    )
                    if hasattr(self.trainer, "set_vocab"):
                        self.trainer.set_vocab(
                            self.token_to_id, self.id2token, getattr(self, "unk_id", 0)
                        )
                    else:
                        self.trainer.token2id = getattr(self, "token_to_id", None)
                        self.trainer.id2token = getattr(self, "id2token", None)
                else:
                    # neural slow
                    self.trainer = NeuralEmbedTrainer(
                        model=None,
                        epochs=getattr(self.config, "epochs", 3),
                        lr=getattr(self.config, "lr", 0.01),
                        tokens=None,
                        batch_size=getattr(self.config, "batch_size", 32),
                        train_text=None,
                        valid_text=None,
                        config=self.config,
                        max_k=k,
                        root=self.project_root,
                        print_every=50,
                    )
                    self.trainer.set_vocab(
                        self.token_to_id, self.id2token, getattr(self, "unk_id", 0)
                    )

                # Try to load the final best model (no need for a path) via API
                # so we don't need to rely completly on model_path
                # ----- load checkpoint depending on neural family -----
                loaded = False
                if mt in NEURAL_FAST_NAMES:
                    # FAST variant: keep the original model_path-based loader.
                    model_path = getattr(self.trainer, "model_path", None)
                    if not model_path or not os.path.exists(model_path):
                        cpkt_dir = os.path.join(
                            self.project_root,
                            "experiments",
                            "models",
                            "neuralfast",
                            "checkpoints",
                        )
                        best = os.path.join(cpkt_dir, "best_model.pkl")
                        if os.path.exists(best):
                            model_path = best
                        else:
                            import glob

                            candidates = sorted(
                                glob.glob(os.path.join(cpkt_dir, "*.pkl"))
                            )
                            if candidates:
                                model_path = candidates[-1]  # the last/newest
                            else:
                                model_path = None

                    if not model_path or not os.path.exists(model_path):
                        raise FileNotFoundError(
                            f"{Colors.FAIL}[ERROR]{Colors.ENDC} No {mt} checkpoint found at: {model_path}. Train {mt} first."
                        )

                    if hasattr(self.trainer, "_load_state"):
                        # fast trainer expects full path
                        self.trainer._load_state(model_path)
                        self.model = self.trainer.model
                        loaded = True
                    elif hasattr(self.trainer, "load"):
                        self.model = self.trainer.load(model_path)
                        loaded = True
                    else:
                        raise RuntimeError(
                            "Trainer has no loader method (_load_state/load)."
                        )
                else:
                    # SLOW variant: standard location final/best_model.pkl via the trainer API.
                    if hasattr(self.trainer, "_load_state"):
                        try:
                            # experiments/models/neural_ngrams/final/best_model.pkl
                            self.model = self.trainer._load_state(
                                filename="best_model.pkl", final=True
                            )
                            loaded = True
                        except FileNotFoundError:
                            # Fallback: newest checkpoint under experiments/models/neural_ngrams/checkpoints/
                            try:
                                import glob

                                ckpt_dir = os.path.join(
                                    self.project_root,
                                    "experiments",
                                    "models",
                                    "neural_ngrams",
                                    "checkpoints",
                                )
                                candidates = sorted(
                                    glob.glob(os.path.join(ckpt_dir, "*.pkl"))
                                )
                                if candidates:
                                    latest = os.path.basename(candidates[-1])
                                    self.model = self.trainer._load_state(
                                        filename=latest, final=False
                                    )
                                    loaded = True
                            except FileNotFoundError:
                                pass

                if not loaded:
                    raise FileNotFoundError(
                        f"{Colors.FAIL}[ERROR]{Colors.ENDC} No {mt} checkpoint found. Train {mt} first."
                    )

            # ---- encode prompt and call correct signature ----
            prompt_ids = [
                self.token_to_id[t] for t in prompt_tokens if t in self.token_to_id
            ]
            unk_id = getattr(self, "unk_id", 0)

            # sampling knobs
            s = getattr(self, "sampling", {}) or {}
            top_k = int(s.get("top_k", 50))
            top_p = float(s.get("top_p", 0.9))
            temperature = float(s.get("temperature", 0.9))
            block_size = getattr(self.config, "block_size", getattr(self.model, "n", 8))

            # --- Robust, signature-aware call for neural variants ---
            # One neural model accepts "stochastic", other accepts "stop_ids",
            # "start_ids"/"max_new_tokens", or "start"/"max_length". Now inspects the
            # callable and pass only what it supports to avoid TypeError mismatches.
            params = set(inspect.signature(self.model.generate).parameters.keys())

            kwargs = {}

            # 1) Start tokens (exact name varies across implementations)
            if "start_ids" in params:
                kwargs["start_ids"] = prompt_ids
            elif "start" in params:
                kwargs["start"] = prompt_ids
            elif "input_ids" in params:
                kwargs["input_ids"] = prompt_ids

            # 2) Length / sampling knobs (only set those the function declares)
            if "max_new_tokens" in params:
                kwargs["max_new_tokens"] = max_length
            elif "max_length" in params:
                kwargs["max_length"] = max_length

            if "top_k" in params:
                kwargs["top_k"] = top_k
            if "top_p" in params:
                kwargs["top_p"] = top_p
            if "temperature" in params:
                kwargs["temperature"] = temperature
            if "block_size" in params:
                kwargs["block_size"] = block_size
            if "id2token" in params:
                kwargs["id2token"] = self.id2token

            # 3) Special controls: stochastic vs explicit stop_ids vs unk handling
            if "stochastic" in params:
                kwargs["stochastic"] = True
            if "unk_id" in params:
                kwargs["unk_id"] = unk_id
            if "stop_ids" in params:
                # IMPORTANT: stop_ids must be an iterable; use a small set containing unk_id
                kwargs["stop_ids"] = {unk_id}

            # 4) Call with filtered kwargs. If a downstream model still has a quirky positional
            #    signature, you can add a positional fallback—but in practice this kwargs call
            #    is enough for both neural_slow and neuralfast you're running.
            # Primary attempt with filtered kwargs
            try:
                generated_ids, generated_tokens, generated_text = self.model.generate(
                    **kwargs
                )
            except TypeError:
                print(
                    f"{Colors.WARNING}[WARNING]{Colors.ENDC} Generating from the except block"
                )
                # Last-resort positional fallback (ensure stop_ids is a list at the tail)
                generated_ids, generated_tokens, generated_text = self.model.generate(
                    prompt_ids,
                    max_length,
                    top_k
                    if "stochastic" not in params
                    else True,  # tolerate odd 3rd arg slots
                    top_p,
                    temperature,
                    block_size,
                    self.id2token,
                    [unk_id],  # iterable, not int
                )
            generated_text = self._postprocess_text(generated_text)
            return generated_text

        #######################
        #      GPT  MODEL     #
        #######################
        elif mt in GPT_NAMES:
            # Infer k (BPE merges) for checkpoint naming
            k = getattr(self, "max_k", None)
            if k is None and hasattr(self, "tokenizer"):
                for attr in ("k", "max_k", "K"):
                    v = getattr(self.tokenizer, attr, None)
                    if v is not None:
                        k = int(v)
                        break
                if k is None:
                    merges = getattr(self.tokenizer, "merges", None)
                    if merges is not None:
                        try:
                            k = int(len(merges))
                        except Exception:
                            pass
            if k is None:
                raise ValueError(
                    f"{Colors.FAIL}[ERROR]{Colors.ENDC} Cannot infer BPE k. Make sure prepare_tokens() ran and set self.max_k."
                )

            # Ensure vocab_size matches training (from tokenizer mapping)
            if (
                hasattr(self, "tokenizer")
                and hasattr(self.tokenizer, "token_to_id")
                and hasattr(self.tokenizer, "id_to_token")
            ):
                self.gpt_stoi = dict(self.tokenizer.token_to_id)
                self.gpt_itos = dict(self.tokenizer.id_to_token)
                self.decode_mode = "bpe"
                self.config.vocab_size = len(self.gpt_stoi)
            else:
                raise RuntimeError(
                    f"{Colors.FAIL}[ERROR]{Colors.ENDC} Tokenizer has no token_to_id/id_to_token. Run a short GPT train once so the vocab can be reconstructed."
                )

            # lazy-load GPT checkpoint (trainer *requires* a tokens dict)
            if self.model is None:
                bs = int(getattr(self.config, "block_size", 64))
                V = int(self.config.vocab_size)
                # << << << < HEAD
                # COVER the vocab: ensure max id is V-1 so trainer infers the right vocab size.
                cover = list(range(V))  # [0, 1, ..., V-1]
                # ensure length >= bs+2
                pad = max(0, (bs + 2) - len(cover))
                train_seq = cover + ([0] * pad)
                # validation: at least bs+1 long, and also contains the max id
                val_base = cover[: max(0, bs + 1)]
                if (V - 1) not in val_base:
                    val_base.append(V - 1)
                if len(val_base) < (bs + 1):
                    val_base += [0] * ((bs + 1) - len(val_base))
                tokens_dict = {"train": train_seq, "validation": val_base}

                try:
                    self.trainer = GptTrainer(
                        config=self.config, tokens=tokens_dict, model=None, k=k
                    )
                    print(
                        f"{Colors.OKCYAN}[MODEL]{Colors.ENDC} Program will load GPT from: {self.trainer.model_path}, vocab_size={self.config.vocab_size}"
                    )

                except TypeError:
                    # older signature (config, tokens, model, root=None, k=None)
                    self.trainer = GptTrainer(self.config, tokens_dict, None, k=k)
                    print(
                        f"{Colors.OKWARN}[ERROR]{Colors.ENDC} While trying to load GPT from: {self.trainer.model_path}, vocab_size={self.config.vocab_size}"
                    )

                # Optional: sanity print to confirm the expected checkpoint path
                model_path = getattr(self.trainer, "model_path", None)
                if not model_path or not os.path.exists(model_path):
                    raise FileNotFoundError(
                        f"{Colors.FAIL}[ERROR]{Colors.ENDC} GPT checkpoint not found. Expected at: {model_path}. "
                        "Train first (or pass --force_model), then run --mode generate."
                    )

                # Align vocab_size to checkpoint (peek rows in tok_emb) fix the emb mismatch stuff
                try:
                    sd = torch.load(model_path, map_location="cpu")
                    # unwrap common containers
                    if isinstance(sd, dict) and "tok_emb.weight" not in sd:
                        for k in ("model_state_dict", "state_dict", "model"):
                            if k in sd and isinstance(sd[k], dict):
                                sd = sd[k]
                                break
                    ckpt_vocab = sd["tok_emb.weight"].shape[0]

                    # force conf to ckp
                    if ckpt_vocab != int(self.config.vocab_size):
                        print(
                            f"{Colors.WARNING}[WARNING]{Colors.ENDC} GPT vocab mismatch "
                            f"(ckpt {ckpt_vocab} vs tokenizer {self.config.vocab_size}). "
                            f"Rebuilding model with checkpoint vocab."
                        )
                        # rebuild trainer/model with the checkpoint vocab size
                        self.config.vocab_size = int(ckpt_vocab)

                        # reset V to cpk size
                        V = self.config.vocab_size

                        # ensures trainer sees V tokens
                        train_seq = list(range(V))
                        bs = int(getattr(self.config, "block_size", 64))
                        val_base = list(range(min(V, bs + 1)))
                        if (V - 1) not in val_base:
                            val_base.append(V - 1)
                        if len(val_base) < (bs + 1):
                            val_base += [0] * ((bs + 1) - len(val_base))
                        tokens_dict = {"train": train_seq, "validation": val_base}

                        # Keep identity stoi/itos to avoid any re-derive bumps
                        self.gpt_stoi = {i: i for i in range(V)}
                        self.gpt_itos = {i: i for i in range(V)}

                        self.trainer = GptTrainer(
                            config=self.config, tokens=tokens_dict, model=None, k=k
                        )
                        model_path = getattr(self.trainer, "model_path", model_path)

                except Exception as e:
                    print(
                        f"{Colors.WARNING}[WARNING]{Colors.ENDC} Could not peek checkpoint vocab ({e}). Proceeding with current config."
                    )

                # Loader part
                if hasattr(self.trainer, "_load_state"):
                    self.trainer._load_state(model_path)
                    self.model = self.trainer.model
                elif hasattr(self.trainer, "load"):
                    self.model = self.trainer.load(model_path)
                else:
                    raise RuntimeError(
                        f"{Colors.FAIL}[ERROR]{Colors.ENDC} Trainer has no loader method (_load_state/load)."
                    )

                # keep k coherent for later
                if getattr(self, "max_k", None) is None:
                    self.max_k = k

            # --- Sampling knobs from __main__ (pipe.samplingggggg) ---
            s = getattr(self, "sampling", {}) or {}
            top_k = int(s.get("top_k", 40))
            top_p = float(s.get("top_p", 0.9))
            temperature = float(s.get("temperature", 0.9))

            # --- Encode prompt to ids using the SAME mapping used during training ---
            if prompt_tokens and isinstance(prompt_tokens[0], int):
                # already BPE int-ids
                prompt_ids = prompt_tokens
                # cap to block_size
                bs = int(getattr(self.config, "block_size", 64))
                if len(prompt_ids) > bs:
                    prompt_ids = prompt_ids[-bs:]
                if not prompt_ids:
                    prompt_ids = [0]

                self.decode_mode = (
                    "bpe" if hasattr(self.tokenizer, "decode") else "local"
                )
            else:
                if not hasattr(self, "gpt_stoi"):
                    if hasattr(self, "tokenizer") and hasattr(
                        self.tokenizer, "token_to_id"
                    ):
                        self.gpt_stoi = dict(self.tokenizer.token_to_id)
                        self.gpt_itos = dict(self.tokenizer.id_to_token)
                        self.decode_mode = "bpe"
                    else:
                        raise RuntimeError(
                            f"{Colors.FAIL}[ERROR]{Colors.ENDC} GPT stoi mapping not found; persist vocab or generate right after training."
                        )
                prompt_ids = [
                    self.gpt_stoi[t] for t in prompt_tokens if t in self.gpt_stoi
                ]
                # cap prompt to block_size to avoid oversize context
                bs = int(getattr(self.config, "block_size", 64))
                if len(prompt_ids) > bs:
                    prompt_ids = prompt_ids[-bs:]
                if not prompt_ids:
                    prompt_ids = [0]

                # Make sure the ids fit the emb loaded
                V = int(
                    getattr(
                        self.config,
                        "vocab_size",
                        len(getattr(self, "gpt_stoi", {})) or 0,
                    )
                )
                if V > 0:
                    prompt_ids = [i if i < V else 0 for i in prompt_ids]

            #  Build generator and sample (always run)
            gen = Generator(
                model=self.model,
                tokenizer=None,  # we pass raw ids; we decode below
                max_new_tokens=max_length,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                mode="auto",
            )
            out_ids = gen.generate(prompt_ids)

            #  Decode
            if getattr(self, "decode_mode", "bpe") == "bpe" and hasattr(
                self.tokenizer, "decode"
            ):
                return self.tokenizer.decode(out_ids)

            toks = [
                self.gpt_itos[i] for i in out_ids if i in getattr(self, "gpt_itos", {})
            ]
            gen_out = "".join(toks)
            p_gen_out = self._postprocess_text(gen_out)
            return p_gen_out

        raise NotImplementedError(
            f"{Colors.FAIL}[ERROR]{Colors.ENDC} Generation for model type '{self.model_type}' was not possible to implememt"
        )


# ------------------ CLEAN MAIN (pipeline) -------------------
if __name__ == "__main__":
    import argparse
    from types import SimpleNamespace

    # Data loader in case
    try:
        from llm_project.utils.dataloader import load_shakespeare
    except Exception as e:
        raise RuntimeError(
            f"{Colors.FAIL}[ERROR]{Colors.ENDC} Missing load_shakespeare; import it at top of pipeline.py"
        ) from e

    # Configs (fallbacks if import fails)
    try:
        from llm_project.models.configs.configs import (
            NgramConfig,
            NeuralConfig,
            GptConfig,
        )
    except Exception:

        class NgramConfig(SimpleNamespace):
            pass

        class NeuralConfig(SimpleNamespace):
            pass

        class GptConfig(SimpleNamespace):
            pass

    parser = argparse.ArgumentParser(description="Run LM pipeline")
    parser.add_argument(
        "--model",
        default="gpt",
        choices=ALL_MODEL_CHOICES,
        help="which model to run (GPT | Neural embeddings | Classic Ngram)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "generate", "compare"],
        default="train",
        help="train (default) | generate | compare (neural_slow vs neuralfast)",
    )
    parser.add_argument(
        "--compare_models",
        type=str,
        default="neural_slow,neuralfast",
        help="comma list among: neural_slow,neuralfast",
    )

    parser.add_argument("--device", default="cpu", help="cpu|cuda|mps")
    parser.add_argument("--max_k", type=int, default=800, help="BPE merges")
    parser.add_argument("--train_limit", type=int, default=100_000)
    parser.add_argument("--valid_limit", type=int, default=10_000)
    parser.add_argument(
        "--force_tokenizer",
        "--force_bpe",
        action="store_true",
        help="retrain BPE even if cached",
    )
    parser.add_argument(
        "--force_model",
        action="store_true",
        help="retrain model even if checkpoint exists",
    )

    # model knobs
    parser.add_argument("--n", type=int, default=3, help="ngram order")
    parser.add_argument("--embd_dim", type=int, default=256)
    parser.add_argument("--block_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=None)

    # generation args
    parser.add_argument("--prompt", type=str, default="", help="Prompt for generation")
    parser.add_argument(
        "--max_new_tokens", type=int, default=100, help="Tokens to generate"
    )
    parser.add_argument(
        "--top_k", type=int, default=40, help="Top-k sampling (0 to disable)"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling ∈ [0,1]"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.9, help="Softmax temperature (>0)"
    )

    args = parser.parse_args()

    # data
    train_text = load_shakespeare("train")
    valid_text = load_shakespeare("validation")

    # config
    if args.model in NGRAM_NAMES:
        cfg = NgramConfig(n=args.n, device=args.device)

    elif args.model in NEURAL_ALL:
        cfg = NeuralConfig(
            n=args.n,
            device=args.device,
            vocab_size=None,
            embd_dim=args.embd_dim,
            block_size=args.block_size,
        )
        for k in (
            "batch_size",
            "epochs",
            "lr",
            "patience",
            "weight_decay",
            "grad_clip",
        ):
            setattr(cfg, k, getattr(args, k))

    elif args.model in GPT_NAMES:
        cfg = GptConfig(
            vocab_size=0,
            n_heads=6,
            layer_dim=4,
            embd_dim=384,
            block_size=64,
            dropout=0.2,
            embd_pdrop=0.2,
            attn_pdrop=0.2,
            resid_pdrop=0.2,
            device=args.device,
        )
        setattr(cfg, "batch_size", args.batch_size)
        setattr(cfg, "max_iters", 5000)
        setattr(cfg, "eval_interval", 500)
        setattr(cfg, "eval_iters", 200)
        setattr(cfg, "learning_rate", 3e-4)
        setattr(cfg, "weight_decay", args.weight_decay)
        setattr(cfg, "log_interval", 500)
        setattr(cfg, "grad_clip", args.grad_clip)

    else:
        raise ValueError(f"Unknown model: {args.model}")

    pipe = LM_Pipeline(model_type=args.model, config=cfg, final=False)

    # expose sampling knobs to pipeline.generate
    pipe.sampling = {
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
    }

    ##############
    # MODE TRAIN #
    ##############
    if args.mode == "train":
        model, train_tokens, valid_tokens = pipe.train(
            train_text=train_text,
            valid_text=valid_text,
            max_k=args.max_k,
            force_retrain_tokenizer=args.force_tokenizer,
            force_retrain_model=args.force_model,
            train_limit=args.train_limit,
            valid_limit=args.valid_limit,
        )
        k = getattr(
            getattr(pipe, "tokenizer", None), "k", getattr(pipe, "max_k", "n/a")
        )

        # with a quick sample
        if args.model in NGRAM_NAMES:
            out = pipe.generate("To be or not to", max_length=70)
            print(f"\nSample of {Colors.OKCYAN}[{args.model}]{Colors.ENDC}\n")
            print(out)
            n_val = getattr(args, "n", "n/a")
            footer = f"|| n: {n_val} || merges k: {k} ||"
            pre_footer = len(footer) * "=" + "\n"
            print(pre_footer, footer)

        elif args.model in NEURAL_ALL or args.model in GPT_NAMES:
            out = pipe.generate("To be or not to", max_length=70)
            print(f"\nSample of {Colors.OKCYAN}[{args.model}]{Colors.ENDC}\n", out)
            footer = f"|| temperature: {args.temperature} || top k: {args.top_k} || top p: {args.top_p} || block_size: {args.block_size} || merges k: {k} ||"
            pre_footer = len(footer) * "=" + "\n"
            print(pre_footer, footer)

    #################
    # MODE GENERATE #
    #################
    elif args.mode == "generate":  # generate
        _ = pipe.prepare_tokens(
            train_text=train_text,
            max_k=args.max_k,
            force_retrain=bool(getattr(args, "force_tokenizer", False)),
            train_limit=args.train_limit,
            final=False,
        )
        out_text = pipe.generate(prompt=args.prompt, max_length=args.max_new_tokens)
        k = getattr(
            getattr(pipe, "tokenizer", None), "k", getattr(pipe, "max_k", "n/a")
        )
        print(f"\n=== Generated Text {Colors.OKCYAN}[{args.model}]{Colors.ENDC} ===\n")
        print(out_text)
        if args.model in NGRAM_NAMES:
            n_val = getattr(args, "n", "n/a")
            footer = f"|| n: {n_val} || merges k: {k} ||"
            pre_footer = len(footer) * "=" + "\n"
            print(pre_footer, footer)

        else:
            footer = f"|| temperature: {args.temperature} || top k: {args.top_k} || top p: {args.top_p} || block_size: {args.block_size} || merges k: {k} ||"
            pre_footer = len(footer) * "=" + "\n"
            print(pre_footer, footer)

    #################
    # MODE COMPARE  #
    #################
    elif args.mode == "compare":
        # Ensure tokenizer exists (and identity vocab)
        base = LM_Pipeline(
            model_type="ngram", config=NgramConfig(n=3, device=args.device)
        )
        _ = base.prepare_tokens(
            train_text=load_shakespeare("train"),
            max_k=args.max_k,
            force_retrain=False,
            train_limit=args.train_limit,
            final=False,
        )
        tok = base.tokenizer
        uniq = list(dict.fromkeys(tok.tokens))
        token_to_id = {tid: tid for tid in uniq}
        id2token = {tid: tid for tid in uniq}

        sampling = {
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
        }

        for m in [s.strip() for s in args.compare_models.split(",") if s.strip()]:
            if m not in (NEURAL_SLOW_NAMES | NEURAL_FAST_NAMES):
                print(
                    f"{Colors.WARNING}[SKIP]{Colors.ENDC} {m} is not a supported name/model for neural comparison"
                )
                continue

            cfg = NeuralConfig(
                n=3,
                device=args.device,
                vocab_size=None,
                embd_dim=args.embd_dim,
                block_size=args.block_size,
            )

            p = LM_Pipeline(model_type=m, config=cfg, final=False)
            p.tokenizer = tok
            p.token_to_id = token_to_id
            p.id2token = id2token
            p.unk_id = 0
            p.sampling = sampling

            out = p.generate(prompt=args.prompt, max_length=args.max_new_tokens)
            print(f"\n=== {m} ===\n{out}")
            print("====================================================")
            print(
                f"|| temperature: {args.temperature} || top k: {args.top_k} || top p: {args.top_p} || block_size: {args.block_size} || merges k: {k} ||\n"
            )
