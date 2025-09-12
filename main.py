import torch
import platform
from shutil import get_terminal_size
from types import SimpleNamespace

# Terminal colors fallback
try:
    from llm_project.utils.debugg_utils import Colors
except ImportError:
    class Colors:
        OKCYAN = ""
        WARNING = ""
        FAIL = ""
        ENDC = ""

# Load data
try:
    from llm_project.utils.dataloader import load_shakespeare
except Exception as e:
    raise RuntimeError(f"{Colors.FAIL}[ERROR]{Colors.ENDC} Missing load_shakespeare") from e

# Model configs fallback
try:
    from llm_project.models.configs.configs import NgramConfig, NeuralConfig, NeuralFastConfig, GptConfig
except Exception:
    class NgramConfig(SimpleNamespace): pass
    class NeuralConfig(SimpleNamespace): pass
    class NeuralFastConfig(SimpleNamespace): pass
    class GptConfig(SimpleNamespace): pass

# Pipeline and constants
try:
    from llm_project.pipeline import LM_Pipeline
    from llm_project.pipeline import (
        NGRAM_NAMES, NEURAL_ALL, GPT_NAMES, NEURAL_SLOW_NAMES,
        NEURAL_FAST_NAMES, ALL_MODEL_CHOICES
    )
except Exception:
    raise RuntimeError("LM_Pipeline or constants not found")

# ---------------- Banner ----------------
def welcome_banner():
    banner = r"""
  ______      __                 __             ______      __
 /_  __/___  / /_____  ____     / /_  __  __   /_  __/___  / /_____  ____
  / / / __ \/ //_/ _ \/ __ \   / __ \/ / / /    / / / __ \/ //_/ _ \/ __ \
 / / / /_/ / ,< /  __/ / / /  / /_/ / /_/ /    / / / /_/ / ,< /  __/ / / /
/_/  \____/_/|_|\___/_/ /_/  /_.___/\__, /    /_/  \____/_/|_|\___/_/ /_/
                                    /____/
    """
    term_width = get_terminal_size().columns
    print("\n".join([line.center(term_width) for line in banner.splitlines()]))
    print(f"{'='*term_width}")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    print(f"PyTorch Device: {device_name}")
    print(f"OS: {platform.system()} {platform.release()}")
    print("Available Models: gpt | ngram | neural embeddings")
    print(f"{'='*term_width}\n")

# ---------------- Banner ----------------
import argparse
def show_intro():
    parser = argparse.ArgumentParser(
        description=(
            "Token-by-Token: a minimalist NLP playground for text generation\n"
            "Supports GPT, classic NGram, and Neural Embedding models."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:

Train examples
python main.py train --model gpt --max_iters 5000 --dropout 0.1 --embd_dim 128 --n_layer 4
python main.py train --model ngram --n 3 --tune_lambdas --force_retrain
python main.py train --model neural --n 3 --epochs 10 --batch_size 32 --block_size 8 --embedding_dim 16 --force_retrain

Generate examples
python main.py generate --model gpt   --prompt "To be or not to be" --max_new_tokens 100
python main.py generate --model ngram --prompt "Once upon a time"   --max_new_tokens 100
python main.py generate --model neural --prompt "Once upon a time"  --max_new_tokens 100

Available modes:
train     Train a model from scratch on Shakespeare
generate  Generate text using a trained model
compare   Compare different models (limited support)

Made with love and caffeine (and a lot of Ritalin).
        """
    )

    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version="Token-by-Token v1.0 by Larizza & Ruiz",
    )

    # Display the help/introduction and exit
    parser.print_help()
    print("\n")


# ---------------- Helpers ----------------
def choose(prompt, choices=None, default=None):
    while True:
        val = input(f"{prompt} [{default}]: ").strip()
        if val.lower() == "exit":
            return "EXIT"
        if not val and default is not None:
            return default
        if choices and val.lower() not in [c.lower() for c in choices]:
            print(f"Invalid choice. Options: {choices} or type 'exit' to quit")
        else:
            return val

def choose_int(prompt, default=None):
    while True:
        val = input(f"{prompt} [{default}]: ").strip()
        if val.lower() == "exit":
            return "EXIT"
        if not val and default is not None:
            return default
        try: return int(val)
        except ValueError: print("Please enter an integer or 'exit'")

def choose_float(prompt, default=None):
    while True:
        val = input(f"{prompt} [{default}]: ").strip()
        if val.lower() == "exit":
            return "EXIT"
        if not val and default is not None:
            return default
        try: return float(val)
        except ValueError: print("Please enter a float or 'exit'")

# ---------------- Minimal config helper ----------------
def minimal_cfg_for_model(model, device):
    if model in NGRAM_NAMES:
        return NgramConfig(device=device, n=3)
    elif model in NEURAL_SLOW_NAMES:
        return NeuralConfig(device=device, n=3, embd_dim=256, block_size=8)
    elif model in NEURAL_FAST_NAMES:
        return NeuralFastConfig(device=device, n=3, embd_dim=256, block_size=8)
    elif model in GPT_NAMES:
        return GptConfig(device=device, embd_dim=384, block_size=64)
    else:
        return SimpleNamespace(device=device)

# ---------------- Main Loop ----------------
def main():
    welcome_banner()
    show_intro()
    train_text = load_shakespeare("train")
    valid_text = load_shakespeare("validation")

    trained_pipe = None  # Keep reference to last trained model

    while True:
        config = {}

        mode = choose("Select mode (train / generate / compare / exit)", choices=["train", "generate", "compare", "exit"], default="train")
        if mode == "EXIT": break
        config["mode"] = mode

        model = choose("Choose model", choices=ALL_MODEL_CHOICES, default="ngram")
        if model == "EXIT": break
        config["model"] = model

        device = choose("Device (cpu / cuda)", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
        if device == "EXIT": break
        config["device"] = device

        if mode == "train":
            # --- Ask about force retrain model first ---
            force_model = choose("Force retrain model? (y/n)", choices=["y", "n"], default="n")
            config["force_model"] = force_model.lower() == "y"

            # --- Ask about force retrain tokenizer separately ---
            force_tokenizer = choose("Force retrain tokenizer? (y/n)", choices=["y", "n"], default="n")
            config["force_tokenizer"] = force_tokenizer.lower() == "y"

            # --- If user wants to force model retrain, optionally skip the rest ---
            if config["force_model"]:
                print("\nSkipping the rest of training parameters as you chose to force retrain the model.\n")
                cfg = minimal_cfg_for_model(model, device)
                pipe = LM_Pipeline(model_type=model, config=cfg, final=False)
                pipe.train(
                    train_text=train_text,
                    valid_text=valid_text,
                    max_k=800,
                    force_retrain_model=True,
                    force_retrain_tokenizer=config["force_tokenizer"],
                    train_limit=config.get("train_limit", 100_000),
                    valid_limit=config.get("valid_limit", 10_000)
                )
                trained_pipe = pipe
                continue

            if not config["force_model"]:
                # Ask all other training questions only if not forcing retrain
                config["n"] = choose_int("Order n (for ngram or neural)", default=3)
                config["embd_dim"] = choose_int("Embedding dimension (for neural/gpt)", default=256)
                config["block_size"] = choose_int("Block size (for neural)", default=8)
                config["batch_size"] = choose_int("Batch size (for neural)", default=32)
                config["epochs"] = choose_int("Number of epochs", default=3)
                config["lr"] = choose_float("Learning rate", default=0.01)
                config["patience"] = choose_int("Patience", default=3)
                config["train_limit"] = choose_int("Train limit", default=100_000)
                config["valid_limit"] = choose_int("Valid limit", default=10_000)

            # Setup pipeline config
            if model in NGRAM_NAMES:
                cfg = NgramConfig(n=config.get("n",3), device=config["device"])
            elif model in NEURAL_SLOW_NAMES:
                cfg = NeuralConfig(n=config.get("n",3), device=config["device"],
                                   vocab_size=None, embd_dim=config.get("embd_dim",256),
                                   block_size=config.get("block_size",8))
                for k in ["batch_size","epochs","lr","patience"]:
                    setattr(cfg,k,config.get(k, {"batch_size":32,"epochs":3,"lr":0.01,"patience":3}[k]))
            elif model in NEURAL_FAST_NAMES:
                cfg = NeuralFastConfig(n=config.get("n",3), device=config["device"],
                                       vocab_size=None, embd_dim=config.get("embd_dim",256),
                                       block_size=config.get("block_size",8))
                for k in ["batch_size","epochs","lr","patience"]:
                    setattr(cfg,k,config.get(k, {"batch_size":32,"epochs":3,"lr":0.01,"patience":3}[k]))
            elif model in GPT_NAMES:
                cfg = GptConfig(vocab_size=0, n_heads=6, layer_dim=4, embd_dim=384,
                                block_size=64, dropout=0.2, embd_pdrop=0.2,
                                attn_pdrop=0.2, resid_pdrop=0.2, device=config["device"])
                setattr(cfg,"batch_size",config.get("batch_size",32))
                setattr(cfg,"max_iters",5000)
                setattr(cfg,"learning_rate",3e-4)
            else:
                print(f"{Colors.WARNING}Unknown model {model}{Colors.ENDC}")
                continue

            pipe = LM_Pipeline(model_type=model, config=cfg, final=False)
            pipe.train(
                train_text=train_text, valid_text=valid_text,
                max_k=800,
                force_retrain_tokenizer=config.get("force_model",False),
                force_retrain_model=config.get("force_model",False),
                train_limit=config.get("train_limit",100_000),
                valid_limit=config.get("valid_limit",10_000)
            )
            trained_pipe = pipe  # keep reference to trained model
            print("\nTraining done! You can now generate text or train again.\n")

        elif mode == "generate":
            # Ask first about pretrained
            from_pretrained = choose("Load pretrained model? (y/n)", choices=["y","n"], default="y")
            pipe = None

            if from_pretrained.lower() == "y":
                pipe = LM_Pipeline(model_type=model, config=None, final=False)
                pipe.prepare_tokens(train_text=train_text, max_k=800, force_retrain=False, final=False)
            else:
                if trained_pipe is not None:
                    pipe = trained_pipe
                else:
                    print("No trained model in memory. Loading saved pretrained model instead.")
                    pipe = LM_Pipeline(model_type=model, config=None, final=False)
                    pipe.prepare_tokens(train_text=train_text, max_k=800, force_retrain=False, final=False)

            config["max_new_tokens"] = choose_int("Max new tokens", default=100)
            config["top_k"] = choose_int("top_k for generation", default=40)
            config["top_p"] = choose_float("top_p for generation", default=0.9)
            config["temperature"] = choose_float("temperature for generation", default=0.9)

            print("\nEnter prompts continuously. Type 'exit' to quit generation loop.\n")
            while True:
                prompt = input("Prompt: ").strip()
                if prompt.lower() == "exit":
                    break
                out = pipe.generate(
                    prompt=prompt,
                    max_length=config["max_new_tokens"],
                    from_pretrained=from_pretrained.lower()=="y"
                )
                print("\n=== Generated Text ===\n")
                print(out)
                print("\n---\n")

        elif mode == "compare":
            print("Compare mode not fully implemented in interactive mode yet.\n")

        print("\nType 'exit' at any prompt to quit the program.\n")


if __name__ == "__main__":
    main()
