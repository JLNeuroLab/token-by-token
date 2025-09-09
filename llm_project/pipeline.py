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

# OLD from llm_project.models.neural_ngrams.trainer import NeuralNgramTrainer
from llm_project.models.neural_embeddings.trainer import NeuralEmbedTrainer
from llm_project.models.gpt.trainer import GptTrainer

# -------------- MODEL IMPORTS ----------------
# from llm_project.models.neural_ngrams.model import NeuralNgram

# -------------- CONFIG IMPORTS ----------------
from llm_project.models.configs.configs import NgramConfig, NeuralConfig, GptConfig

# --------------GENERATOR IMPORT ----------------
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
            print(
                f"{Colors.OKCYAN}[DEBUG]{Colors.ENDC}: tokenizer.tokens length = {len(tokens)}"
            )
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
            self.max_k = int(max_k)
            try:
                setattr(self.tokenizer, "k", int(max_k))
            except Exception:
                pass

            if train_limit:
                tokens = tokens[:train_limit]
                self.tokenizer.tokens = tokens
            print(
                f"{Colors.OKGREEN}[OK]{Colors.ENDC}: loaded tokenizer.tokens length = {len(tokens)}"
            )
            return tokens

        if train_text is None:
            raise ValueError(
                f"{Colors.FAIL}[ERROR]{Colors.ENDC}: train_text must be provided to train a new BPE tokenizer."
            )
        # If dataloader returned list/tuple, flatten to a single string IDK LETS SEE
        # if isinstance(train_text, (list, tuple)):
        #    train_text = "".join(train_text)

        # Limit the training text before BPE to save time
        if train_limit:
            train_text = train_text[:train_limit]

        # CASE 3: Train a new tokenizer
        print("Training new BPE tokenizer...")
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
        # -----------------------------------------------------------------------

        # Fix UNK part

        # ----- Build vocab mapping ONLY for neural trainers (identity over BPE ids) -----
        mt = (self.model_type or "").lower()

        if mt in {"neuraln", "neural", "neural_slow"}:
            # tokens are already BPE integer ids; keep order, remove dups
            unique_ids = list(dict.fromkeys(tokens))
            self.vocab_tokens = unique_ids
            # identity mapping: id -> id
            self.token_to_id = {tid: tid for tid in unique_ids}
            self.id2token = {tid: tid for tid in unique_ids}
            # neuraln expects an unk_id; BPE shouldn’t produce UNK, use harmless default
            unk_id = getattr(self, "unk_id", 0)
            final_flag = getattr(self, "final", False)

            # optional: persist mapping next to tokenizer
            try:
                vocab_dir = get_model_path(
                    self.project_root,
                    "tokenizers",
                    subdir=self.subdir,
                    final=final_flag,
                )
                save_model(
                    {"vocab_tokens": unique_ids},
                    root=self.project_root,
                    category="tokenizers",
                    subdir=self.subdir,
                    filename=f"BPE_vocab_k{max_k}.pkl",
                )
            except Exception as e:
                print(
                    f"{Colors.WARNING}[WARN]{Colors.ENDC} Skipped saving vocab: {e}")
        # ----- end neural-only vocab block -----
        # Fix UNK part, end of block.

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
        """Setup and train the model."""
        model_type = self.model_type.lower()

        # Ensure vocab_size for trainers that need it (neuraln)
        if hasattr(self, "token_to_id"):
            self.config.vocab_size = len(self.token_to_id)

        ##########################
        # CLASSICAL N-GRAM MODEL #
        ##########################
        if model_type in {
            "ngram",
        }:
            self.trainer = NGramTrainer(
                config=self.config, model=None, tokens=train_tokens, k=max_k
            )
            self.trainer.train()
            self.trainer.final = self.final
            self.model = self.trainer.model
            # keep k/max_k coherent for filename glue and later saves
            if getattr(self, "max_k", None) is None and hasattr(self.trainer, "k"):
                self.max_k = self.trainer.k

        ##########################
        # CLASSICAL N-GRAM MODEL #
        ##########################

        elif model_type in {"neural_slow", "neuraln", "neural"}:
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
            self.trainer.set_vocab(
                self.token_to_id, self.id2token, self.unk_id)
            self.trainer.train_ids = train_ids
            self.trainer.val_ids = valid_ids

            # Lil Sanity check (UNK rate). Should be 0 on train and very little on val
            unk = self.unk_id
            ti, vi = self.trainer.train_ids or [], self.trainer.val_ids or []
            unk_train = sum(i == unk for i in ti) / max(1, len(ti))
            unk_val = sum(i == unk for i in vi) / max(1, len(vi))
            print(f"UNK train={unk_train:.2%}  UNK val={unk_val:.2%}")

            self.trainer.train(
                epochs=getattr(self.config, "epochs", 3),
                lr=getattr(self.config, "lr", 0.01),
                patience=getattr(self.config, "patience", 3),
                force_retrain=force_retrain,
            )
            self.model = self.trainer.model

        #############
        # GPT MODEL #
        #############

        elif model_type == "gpt":
            bs = int(getattr(self.config, "block_size", 64))

            # Choose source sequences (may be strings today)
            tr_seq = train_ids if train_ids is not None else train_tokens
            va_seq = valid_ids if valid_ids is not None else valid_tokens

            # --- ensure integer ids, prefer tokenizer's own mapping if present ---
            def _ensure_int_ids(seq, build=False):
                if seq is None:
                    return None
                if len(seq) > 0 and isinstance(seq[0], int):
                    # already ids
                    return seq

                # Try to use tokenizer's mapping (best, so we can decode with tokenizer.decode later)
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

        else:
            raise NotImplementedError(
                f"{Colors.FAIL}[ERROR]{Colors.ENDC} Model type '{self.model_type}' not implemented"
            )

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

        valid_tokens = self.tokenizer.BPE_segmenter(
            valid_text) if valid_text else None
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

    def old_generate(self, prompt, max_length=50, from_pretrained=False):
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
                    f"{Colors.FAIL}[ERROR]{Colors.ENDC} Tokenizer is not initialized. Cannot encode string prompt."
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
                        f"{Colors.WARNING}[!!!]{Colors.ENDC} No pre-trained N-gram model found in final folder. Using current model."
                    )

            else:
                raise NotImplementedError(
                    f"{Colors.WARNING}[!!!]{Colors.ENDC} No other model ready from retrieving pretrained models"
                )

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

        # --- Optional: load pretrained for N-gram only (as in your code) ---
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
                        f"{Colors.WARNING}[!!!]{Colors.ENDC} No pre-trained N-gram model found in final folder. Using current model."
                    )
            elif self.model_type.lower() in {"neural_slow", "ngramn", "neural"}:
                prompt_ids = [
                    self.token_to_id[t] for t in prompt_tokens if t in self.token_to_id
                ]
                generated_ids, generated_tokens, generated_text = self.model.generate(
                    prompt_ids,
                    max_new_tokens=max_length,
                    block_size=self.config.block_size,
                    id2token=self.id2token,
                    top_k=50,
                    top_p=0.9,
                    unk_id=self.unk_id,
                )
                return generated_text

            else:
                raise NotImplementedError(
                    f"{Colors.FAIL}[ERROR]{Colors.ENDC} No other model ready for pretrained loading"
                )

        # --- Dispatch by model type ---
        mt = self.model_type.lower()

        ##########################
        # CLASSICAL N-GRAM MODEL #
        ##########################

        if mt == "ngram":
            return self.model.generate_text(prompt_tokens, max_length=max_length)

        #######################
        # NEURAL N-GRAM MODEL #
        #######################
        elif mt in {"neuraln", "neural", "neural_slow"}:
            # Convert tokens -> ids
            prompt_ids = [
                self.token_to_id[t] for t in prompt_tokens if t in self.token_to_id
            ]
            unk_id = getattr(self, "unk_id", 0)
            generated_ids, generated_tokens, generated_text = self.model.generate(
                prompt_ids,
                max_new_tokens=max_length,
                block_size=self.config.block_size,
                id2token=self.id2token,
                top_k=50,
                top_p=0.9,
                unk_id=unk_id,
            )
            return generated_text

        #######################
        #      GPT  MODEL     #
        #######################

        elif mt == "gpt":
            # ---- infer k (BPE merges) for checkpoint naming ----
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
            # ----------------------------------------------------

            # Lazy-load GPT model from checkpoint if needed

            # ---- ensure vocab_size matches training (from tokenizer mapping) ----
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

            # ---- lazy-load GPT checkpoint (trainer *requires* a tokens dict) ----
            if self.model is None:
                bs = int(getattr(self.config, "block_size", 64))
                V = int(self.config.vocab_size)

                # COVER the vocab: ensure max id is V-1 so trainer infers the right vocab size.
                cover = list(range(V))  # [0, 1, ..., V-1]
                pad = max(0, (bs + 2) - len(cover))  # ensure length >= bs+2
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
                    self.trainer = GptTrainer(
                        self.config, tokens_dict, None, k=k)
                    print(
                        f"{Colors.OKCYAN}[MODEL]{Colors.ENDC} Program will load GPT from: {self.trainer.model_path}, vocab_size={self.config.vocab_size}"
                    )

                # Optional: sanity print to confirm the expected checkpoint path
                model_path = getattr(self.trainer, "model_path", None)
                if not model_path or not os.path.exists(model_path):
                    raise FileNotFoundError(
                        f"{Colors.FAIL}[ERROR]{Colors.ENDC} GPT checkpoint not found. Expected at: {model_path}. "
                        "Train first (or pass --force_model), then run --mode generate."
                    )

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

            # --- Sampling knobs from __main__ (pipe.sampling) ---
            s = getattr(self, "sampling", {}) or {}
            top_k = int(s.get("top_k", 40))
            top_p = float(s.get("top_p", 0.9))
            temperature = float(s.get("temperature", 0.9))

            # --- Encode prompt to ids using the SAME mapping used during training ---
            if prompt_tokens and isinstance(prompt_tokens[0], int):
                # already BPE int-ids
                prompt_ids = prompt_tokens
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

            # --- Build generator and sample (always run) ---
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

            # --- Decode ---
            if getattr(self, "decode_mode", "bpe") == "bpe" and hasattr(
                self.tokenizer, "decode"
            ):
                return self.tokenizer.decode(out_ids)

            toks = [
                self.gpt_itos[i] for i in out_ids if i in getattr(self, "gpt_itos", {})
            ]
            return "".join(toks)


# ------------------ NEW TEST -------------------
if __name__ == "__main__":
    import argparse
    from types import SimpleNamespace

    # --- Ensure required symbols exist ---
    # 1) Data loader
    try:
        from llm_project.utils.dataloader import load_shakespeare
    except Exception as e:
        raise RuntimeError(
            f"{Colors.FAIL}[ERROR]{Colors.ENDC} Missing load_shakespeare; import it at top of pipeline.py"
        ) from e

    # 2) Configs
    try:
        from llm_project.models.configs.configs import (
            NgramConfig,
            NeuralConfig,
            GptConfig,
        )  # noqa: F401
    except Exception:

        class NgramConfig(SimpleNamespace):
            pass

        class NeuralConfig(SimpleNamespace):
            pass

        class GptConfig(SimpleNamespace):
            pass

    # 3) Pipeline class resolution (works whether it's LM_Pipeline or Pipeline)
    _PipeClass = globals().get("LM_Pipeline") or globals().get("Pipeline")
    if _PipeClass is None:
        raise RuntimeError(
            f"{Colors.FAIL}[ERROR]{Colors.ENDC} Could not find pipeline class. Make sure a class named LM_Pipeline or Pipeline is defined above."
        )

    parser = argparse.ArgumentParser(description="Run LM pipeline")
    parser.add_argument(
        "--model",
        default="ngram",
        choices=["ngram", "neural", "neuraln", "gpt"],
        help="which model to run",
    )
    parser.add_argument("--device", default="cpu", help="cpu|cuda|mps")
    ad
    parser.add_argument("--max_k", type=int, default=800, help="BPE merges")
    parser.add_argument("--train_limit", type=int, default=100_000)
    parser.add_argument("--valid_limit", type=int, default=10_000)
    parser.add_argument(
        "--force_tokenizer", action="store_true", help="retrain BPE even if cached"
    )
    parser.add_argument(
        "--force_model",
        action="store_true",
        help="retrain model even if checkpoint exists",
    )

    # model-specific knobs (read by your config/trainers if supported)
    parser.add_argument("--n", type=int, default=3, help="ngram order")
    parser.add_argument("--embd_dim", type=int, default=256)
    parser.add_argument("--block_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=3)

    # ---- Mode: train | generate ----
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "generate"],
        default="train",
        help="train (default) or generate",
    )

    # ---- Generation args (used only when --mode generate) ----
    parser.add_argument("--prompt", type=str, default="",
                        help="Prompt for generation")
    parser.add_argument(
        "--max_new_tokens", type=int, default=100, help="Tokens to generate"
    )

    # Sampling knobs (plumbed into Pipeline.generate → GPT Generator)
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

    # --- Load data (use the signature your project actually has) ---
    # If your util is load_shakespeare("train"), use that; otherwise use version="train".
    train_text = load_shakespeare("train")
    valid_text = load_shakespeare("validation")

    # --- Pick config by model type (match your existing config classes) ---
    if args.model == "ngram":
        cfg = NgramConfig(n=args.n, device=args.device)

    elif args.model in {"neural", "neuraln"}:
        # If you have a NeuralConfig, set fields your trainer expects.
        cfg = NeuralConfig(
            n=args.n,
            device=args.device,
            vocab_size=None,
            embd_dim=args.embd_dim,
            block_size=args.block_size,
        )
        setattr(cfg, "batch_size", args.batch_size)
        setattr(cfg, "epochs", args.epochs)
        setattr(cfg, "lr", args.lr)
        setattr(cfg, "patience", args.patience)

    elif args.model == "gpt":
        # Tune these defaults if your GptTrainer expects different names
        cfg = GptConfig(
            vocab_size=0,  # trainer will infer from tokens
            n_heads=6,
            layer_dim=4,  # your project uses layer_dim as n_layer
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
        setattr(cfg, "weight_decay", 0.0)
        setattr(cfg, "log_interval", 500)
        setattr(cfg, "grad_clip", None)

    else:
        raise ValueError(f"Unknown model: {args.model}")

    # --- Build pipeline (use your actual class name) ---
    # If your class is named Pipeline (not LM_Pipeline), use that name instead.
    pipe = LM_Pipeline(model_type=args.model, config=cfg, final=False)

    # Make sampling available to generate(); Pipeline.generate will read these
    pipe.sampling = {
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
    }

    # ---- Dispatch ----
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

        # Quick sample after training
        if args.model in {"ngram", "neural", "neuraln"}:
            sample_prompt = "To be or not to"
            out = pipe.generate(sample_prompt, max_length=50)
            print(f"\n[Sample ({args.model})]\n", out)

        elif args.model == "gpt":
            sample_prompt = "The lake of the swams is"
            out = pipe.generate(prompt=sample_prompt, max_length=80)
            print("\n[Sample (gpt)]\n", out)

    else:  # args.mode == "generate"
        # 1) Ensure tokenizer is loaded (or trained if missing)
        #    This will set: self.tokenizer, and also self.max_k inside the pipeline.
        _ = pipe.prepare_tokens(
            train_text=train_text,
            max_k=args.max_k,
            force_retrain=bool(getattr(args, "force_tokenizer", False)),
            train_limit=args.train_limit,
            final=False,
        )

        # 2) Now we can safely generate (generate() will also load the GPT model if needed)
        out_text = pipe.generate(
            prompt=args.prompt,
            max_length=args.max_new_tokens,
        )
        print("\n=== Generated Text ===\n")
        print(f"Top-p: {self.top_p}. Top-k: {pipe.}")
        print(out_text)
