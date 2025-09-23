import argparse
import torch
import platform
import sys
import shlex
import subprocess
from shutil import get_terminal_size
from types import SimpleNamespace

# Terminal colors fallback
from llm_project.utils.debugg_utils import Colors


# Load data
try:
    from llm_project.utils.dataloader import load_shakespeare
except Exception as e:
    raise RuntimeError(
        f"{Colors.FAIL}[ERROR]{Colors.ENDC} Missing load_shakespeare"
    ) from e

# Model configs fallback
try:
    from llm_project.models.configs.configs import (
        NgramConfig,
        NeuralConfig,
        NeuralFastConfig,
        GptConfig,
    )
except Exception:

    class NgramConfig(SimpleNamespace):
        pass

    class NeuralConfig(SimpleNamespace):
        pass

    class NeuralFastConfig(SimpleNamespace):
        pass

    class GptConfig(SimpleNamespace):
        pass


# Pipeline and constants
try:
    from llm_project.pipeline import LM_Pipeline
    from llm_project.pipeline import (
        NGRAM_NAMES,
        NEURAL_ALL,
        GPT_NAMES,
        NEURAL_SLOW_NAMES,
        NEURAL_FAST_NAMES,
        ALL_MODEL_CHOICES,
    )
except Exception:
    raise RuntimeError("LM_Pipeline or constants not found")

def _flatten(argv):
    result = []
    for x in argv:
        if isinstance(x, list):
            result.extend(_flatten(x)) 
        else:
            result.append(str(x))
    return result

def _pipeline_run(argv: list[str], echo: bool = True) -> int:
    """Runs: python -m llm_project.pipeline <argv...> (stream output)."""
    flat_argv = _flatten(argv)
    cmd = [sys.executable, "-m", "llm_project.pipeline"] + flat_argv
    if echo:
        print(
            f"{Colors.WARNING}Token console > {Colors.ENDC}",
            " ".join(shlex.quote(x) for x in cmd),
        )
    return subprocess.call(cmd)


def _normalize_manual_line(line: str):
    """
    Accepts:
      generate --model gpt ...
      train --model ngram ...
      compare --model neural_fast ...
      --model gpt --mode generate ...
      (optional) python -m llm_project.pipeline --model ...
    Returns argv for the pipeline CLI, or None if we already executed a full python command.
    """
    parts = shlex.split(line)
    if not parts:
        return []

    # Verb/inject --mode
    verb_map = {
        "generate": "generate",
        "gen": "generate",
        "g": "generate",
        "compare": "compare",
        "cmp": "compare",
        "c": "compare",
        "train": "train",
        "tr": "train",
        "t": "train",
    }
    if parts[0].lower() in verb_map:
        mode = verb_map[parts[0].lower()]
        return ["--mode", mode] + parts[1:]

    # Flags-only: just pass through (user must include --mode)
    return parts


def manual_mode_loop() -> int:
    print("\n[manual] Examples:")
    print(
        '  generate --model gpt --device cuda --max_k 800 --block_size 64 --prompt "ROMEO:" --max_new_tokens 120 --top_k 50 --top_p 0.95 --temperature 0.8'
    )
    print(
        "  train --model ngram --n 3 --max_k 800 --train_limit 100000 --valid_limit 10000"
    )
    print("  compare --max_k 800 [flags…]")
    print("Type 'back' to return, 'exit' to quit.\n")
    while True:
        line = input(
            f"{Colors.WARNING}Token console {Colors.OKBLUE}[Manual mode] {Colors.WARNING}> {Colors.ENDC}"
        ).strip()
        if not line:
            continue
        if line.lower() in {"exit", "quit", "q"}:
            return "exit"
        if line.lower() in {"back", "b"}:
            return "back"
        argv = _normalize_manual_line(line)
        if argv is None:
            # full `python -m` command already executed
            continue
        if not argv:
            print("Nothing to run.")
            continue
        _pipeline_run(argv)


# ---------------- Banner and menu ----------------


def welcome_banner():
    cc = Colors.OKCYAN
    cg = Colors.OKGREEN
    cr = Colors.FAIL
    cy = Colors.WARNING
    cv = Colors.HEADER
    cd = Colors.ENDC
    cb = Colors.OKBLUE

    s1 = cd
    s2 = cg
    s3 = cy
    # s4 = cd
    # s5 = cy
    de = cd

    ban_v2 = f"""
  {s1}██╗{s2}████████╗ ██████╗ ██╗  ██╗███████╗███╗   ██╗    {s3}██╗{s1}██╗
{s1} ██╔╝{s2}╚══██╔══╝██╔═══██╗██║ ██╔╝██╔════╝████╗  ██║   {s3}██╔╝{s1}╚██╗
{s1}██╔╝    {s2}██║   ██║   ██║█████╔╝ █████╗  ██╔██╗ ██║  {s3}██╔╝  {s1}╚██╗
{s1}╚██╗    {s2}██║   ██║   ██║██╔═██╗ ██╔══╝  ██║╚██╗██║ {s3}██╔╝   {s1}██╔╝
 {s1}╚██╗   {s2}██║   ╚██████╔╝██║  ██╗███████╗██║ ╚████║{s3}██╔╝   {s1}██╔╝
 {s1} ╚═╝   {s2}╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝{s3}╚═╝    {s1}╚═╝

               {s1}██╗ {s2}██████╗ ██╗   ██╗  {s3}██╗{s1}██╗
              {s1}██╔╝ {s2}██╔══██╗╚██╗ ██╔╝ {s3}██╔╝{s1}╚██╗
             {s1}██╔╝  {s2}██████╔╝ ╚████╔╝ {s3}██╔╝  {s1}╚██╗
             {s1}╚██╗  {s2}██╔══██╗  ╚██╔╝ {s3}██╔╝   {s1}██╔╝
              {s1}╚██╗ {s2}██████╔╝   ██║ {s3}██╔╝   {s1}██╔╝
               {s1}╚═╝ {s2}╚═════╝    ╚═╝ {s3}╚═╝    {s1}╚═╝

  {s1}██╗{s2}████████╗ ██████╗     {s3}██╗{s1}██╗        {s1}██╗{s2}██╗  ██╗███████╗███╗   ██╗    {s3}██╗{s1}██╗
 {s1}██╔╝{s2}╚══██╔══╝██╔═══██╗   {s3}██╔╝{s1}╚██╗      {s1}██╔╝{s2}██║ ██╔╝██╔════╝████╗  ██║   {s3}██╔╝{s1}╚██╗
{s1}██╔╝   {s2} ██║   ██║   ██║  {s3}██╔╝  {s1}╚██╗    {s1}██╔╝ {s2}█████╔╝ █████╗  ██╔██╗ ██║  {s3}██╔╝  {s1}╚██╗
{s1}╚██╗    {s2}██║   ██║   ██║ {s3}██╔╝   {s1}██╔╝    {s1}╚██╗ {s2}██╔═██╗ ██╔══╝  ██║╚██╗██║ {s3}██╔╝   {s1}██╔╝
 {s1}╚██╗   {s2}██║   ╚██████╔╝{s3}██╔╝   {s1}██╔╝      {s1}╚██╗{s2}██║  ██╗███████╗██║ ╚████║{s3}██╔╝   {s1}██╔╝
{s1}    ╚═╝   {s2}╚═╝    ╚═════╝ {s3}╚═╝    {s1}╚═╝        {s1}╚═╝{s2}╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝{s3}╚═╝    {s1}╚═╝{de}
    """

    banner = ban_v2

    term_width = get_terminal_size().columns
    print("\n".join([line.center(term_width) for line in banner.splitlines()]))
    print(f"{'=' * term_width}")
    device_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    )
    print(f"PyTorch Device: {device_name}")
    print(f"OS: {platform.system()} {platform.release()}")
    print("Available Models: gpt | ngram | neural embeddings")
    print(f"{'=' * term_width}\n")


def show_intro():
    parser = argparse.ArgumentParser(
        description=(
            f"{Colors.WARNING}Token-by-Token: a minimalist NLP playground for text generation\n"
            "Supports GPT, classic NGram, and Neural Embedding models (fast/torch and slow/numpy versions)."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=f"""
Examples:
{Colors.HEADER}
Train examples:
{Colors.WARNING}train{Colors.ENDC}  --model gpt --epochs 10 --batch_size 128 --block_size 64 --embd_dim 384 --n_layer 4 --n_heads 6 --lr 3e-4 --force_model
{Colors.WARNING}train{Colors.ENDC} --model ngram --n 3 --tune_lambdas --max_k 800 --force_model
{Colors.WARNING}train{Colors.ENDC} --model neural_fast --n 3 --epochs 10 --batch_size 128 --block_size 32 --embd_dim 384 --lr 3e-4 --patience 3 --force_model
{Colors.WARNING}train{Colors.ENDC} --model neural_slow --n 3 --epochs 10 --batch_size 64 --block_size 16 --embd_dim 256 --lr 3e-4 --patience 3 --force_model
{Colors.HEADER}
Generate examples:
{Colors.WARNING}generate{Colors.ENDC} --model gpt --prompt "The meaning of life is" --max_new_tokens 100 --top_k 50 --top_p 0.95 --temperature 0.9
{Colors.WARNING}generate{Colors.ENDC} --model ngram --prompt "To be or not to" --max_new_tokens 100 --top_k 40 
{Colors.WARNING}generate{Colors.ENDC} --model neural_fast --prompt "In a galaxy far away" --max_new_tokens 100 --top_k 50 --top_p 0.95 --temperature 0.9
{Colors.WARNING}generate{Colors.ENDC} --model neural --prompt "Once upon a time" --max_new_tokens 100 --top_k 50 --top_p 0.9 --temperature 0.9
{Colors.HEADER}
Available modes:
{Colors.WARNING}train{Colors.ENDC}     Train a model from scratch on Shakespeare
{Colors.WARNING}generate{Colors.ENDC}  Generate text using a trained model
{Colors.WARNING}compare{Colors.ENDC}   Compare different models (limited support)

{Colors.FAIL}Made with love and caffeine (and a lot of Ritalin){Colors.ENDC}.
        """,
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


# ---------------- Banner and menu ----------------


def assisted_wizard() -> str:
    # 1) core choices
    mode = choose("Mode", ["train", "generate", "compare"], "generate")
    if mode in (_BACK, _EXIT):
        return "back" if mode == _BACK else "exit"

    model = choose("Model", ["ngram", "neural_slow", "neural_fast", "gpt"], "ngram")
    if model in (_BACK, _EXIT):
        return "back" if model == _BACK else "exit"

    device = choose("Device", ["cpu", "cuda", "mps"], "cpu")
    if device in (_BACK, _EXIT):
        return "back" if device == _BACK else "exit"

    max_k = choose_int("BPE merges K", 800)
    if max_k in (_BACK, _EXIT):
        return "back" if max_k == _BACK else "exit"

    argv = ["--model", model, "--mode", mode, "--max_k", str(max_k)]
    if device in ("cuda", "mps"):
        argv += ["--device", device]

    # 2) per-mode options
    if mode == "train":
        if model == "ngram":
            n = choose_int("n-gram order n", 3)
            if n in (_BACK, _EXIT):
                return "back" if n == _BACK else "exit"
            train_limit = choose_int("train_limit", 100_000)
            if train_limit in (_BACK, _EXIT):
                return "back" if train_limit == _BACK else "exit"
            valid_limit = choose_int("valid_limit", 10_000)
            if valid_limit in (_BACK, _EXIT):
                return "back" if valid_limit == _BACK else "exit"
            force_model = choose("force_model?", ["y", "n"], "y")
            if force_model in (_BACK, _EXIT):
                return "back" if force_model == _BACK else "exit"
            force_tokenizer = choose("force_tokenizer?", ["y", "n"], "y")
            if force_tokenizer in (_BACK, _EXIT):
                return "back" if force_tokenizer == _BACK else "exit"
            argv += [
                "--n",
                str(n),
                "--train_limit",
                str(train_limit),
                "--valid_limit",
                str(valid_limit),
            ]
            if force_model == "y":
                argv += ["--force_model"]
            elif force_tokenizer == "y":
                argv += ["--force_tokenizer"]


        elif model in ("neural_slow", "neural_fast"):
            epochs = choose_int("epochs", 2)
            if epochs in (_BACK, _EXIT):
                return "back" if epochs == _BACK else "exit"
            batch = choose_int("batch_size", 32 if model == "neural_slow" else 64)
            if batch in (_BACK, _EXIT):
                return "back" if batch == _BACK else "exit"
            bs = choose_int("block_size", 8)
            if bs in (_BACK, _EXIT):
                return "back" if bs == _BACK else "exit"
            train_limit = choose_int("train_limit", 50_000)
            if train_limit in (_BACK, _EXIT):
                return "back" if train_limit == _BACK else "exit"
            valid_limit = choose_int("valid_limit", 10_000)
            if valid_limit in (_BACK, _EXIT):
                return "back" if valid_limit == _BACK else "exit"
            force_model = choose("force_model? (y/n)", ["y", "n"], "y")
            if force_model in (_BACK, _EXIT):
                return "back" if force_model == _BACK else "exit"
            force_tokenizer = choose("force_tokenizer?", ["y", "n"], "y")
            if force_tokenizer in (_BACK, _EXIT):
                return "back" if force_tokenizer == _BACK else "exit"
            argv += [
                "--epochs",
                str(epochs),
                "--batch_size",
                str(batch),
                "--block_size",
                str(bs),
                "--train_limit",
                str(train_limit),
                "--valid_limit",
                str(valid_limit),
            ]
            if force_model == "y":
                argv += ["--force_model"]
            elif force_tokenizer == "y":
                argv += ["--force_tokenizer"]

        elif model == "gpt":
            epochs = choose_int("epochs", 2)
            if epochs in (_BACK, _EXIT):
                return "back" if epochs == _BACK else "exit"
            batch = choose_int("batch_size", 64)
            if batch in (_BACK, _EXIT):
                return "back" if batch == _BACK else "exit"
            bs = choose_int("block_size", 8)
            if bs in (_BACK, _EXIT):
                return "back" if bs == _BACK else "exit"
            train_limit = choose_int("train_limit", 50_000)
            if train_limit in (_BACK, _EXIT):
                return "back" if train_limit == _BACK else "exit"
            valid_limit = choose_int("valid_limit", 10_000)
            if valid_limit in (_BACK, _EXIT):
                return "back" if valid_limit == _BACK else "exit"
            force_model = choose("force_model? (y/n)", ["y", "n"], "y")
            if force_model in (_BACK, _EXIT):
                return "back" if force_model == _BACK else "exit"
            force_tokenizer = choose("force_tokenizer?", ["y", "n"], "y")
            if force_tokenizer in (_BACK, _EXIT):
                return "back" if force_tokenizer == _BACK else "exit"
            argv += [
                "--epochs",
                str(epochs),
                "--batch_size",
                str(batch),
                "--block_size",
                str(bs),
                "--train_limit",
                str(train_limit),
                "--valid_limit",
                str(valid_limit),
            ]
            if force_model == "y":
                argv += ["--force_model"]
            elif force_tokenizer == "y":
                argv += ["--force_tokenizer"]

        
    elif mode == "compare":
        models = (
            input(
                f"{ASSIST_PROMPT}Models (CSV: neural_slow,neural_fast) [neural_slow,neural_fast]: "
            ).strip()
            or "neural_slow,neural_fast"
        )
        sp = _check_special(models)
        if sp:
            return "back" if sp == _BACK else "exit"
        prompt = (
            input(f"{ASSIST_PROMPT}Prompt [To be or not to]: ").strip()
            or "To be or not to"
        )
        sp = _check_special(prompt)
        if sp:
            return "back" if sp == _BACK else "exit"
        max_new_tokens = choose_int("max_new_tokens", 100)
        if max_new_tokens in (_BACK, _EXIT):
            return "back" if max_new_tokens == _BACK else "exit"
        bs = choose_int("block_size (neural context)", 8)
        if bs in (_BACK, _EXIT):
            return "back" if bs == _BACK else "exit"
        argv += [
            "--compare_models",
            models,
            "--prompt",
            prompt,
            "--max_new_tokens",
            str(max_new_tokens),
            "--block_size",
            str(bs),
        ]

    else:  # generate
        prompt = (
            input(f"{ASSIST_PROMPT}Prompt [To be or not to]: ").strip()
            or "To be or not to"
        )
        sp = _check_special(prompt)
        if sp:
            return "back" if sp == _BACK else "exit"
        max_new_tokens = choose_int("max_new_tokens", 100)
        if max_new_tokens in (_BACK, _EXIT):
            return "back" if max_new_tokens == _BACK else "exit"
        top_k = choose_int("top_k", 50)
        if top_k in (_BACK, _EXIT):
            return "back" if top_k == _BACK else "exit"
        top_p = choose_float("top_p", 0.95)
        if top_p in (_BACK, _EXIT):
            return "back" if top_p == _BACK else "exit"
        temperature = choose_float("temperature", 0.9)
        if temperature in (_BACK, _EXIT):
            return "back" if temperature == _BACK else "exit"
        bs = 64 if model == "gpt" else 8
        argv += [
            "--block_size",
            str(bs),
            "--prompt",
            prompt,
            "--max_new_tokens",
            str(max_new_tokens),
            "--top_k",
            str(top_k),
            "--top_p",
            str(top_p),
            "--temperature",
            str(temperature),
        ]

    print("\n[assist] running pipeline…\n")
    _pipeline_run(argv)
    return "stay"


# ---------------- Helpers ----------------


# Assisted mode prompt prefix
ASSIST_PROMPT = f"{Colors.WARNING}Token console {Colors.OKCYAN}[Assisted mode]{Colors.WARNING} > {Colors.ENDC}"
_BACK = "__BACK__"
_EXIT = "__EXIT__"

# So assisted mode handles exit well


def _check_special(s: str):
    lo = s.lower()
    if lo in {"back", "b"}:
        return _BACK
    if lo in {"exit", "quit", "q"}:
        return _EXIT
    return None


def choose(prompt, choices, default):
    s = input(f"{ASSIST_PROMPT}{prompt} {choices} [{default}]: ").strip()
    if not s:
        return default
    special = _check_special(s)
    if special:
        return special
    s = s.lower()
    return s if s in choices else default


def choose_int(prompt, default):
    s = input(f"{ASSIST_PROMPT}{prompt} [{default}]: ").strip()
    if not s:
        return default
    special = _check_special(s)
    if special:
        return special
    return int(s) if s.isdigit() else default


def choose_float(prompt, default):
    s = input(f"{ASSIST_PROMPT}{prompt} [{default}]: ").strip()
    if not s:
        return default
    special = _check_special(s)
    if special:
        return special
    try:
        return float(s)
    except ValueError:
        return default


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

#
# def main():
#     welcome_banner()
#     show_intro()
#     train_text = load_shakespeare("train")
#     valid_text = load_shakespeare("validation")
#
#     trained_pipe = None  # Keep reference to last trained model
#
#     while True:
#         config = {}
#
#         mode = choose(
#             "Select mode (train / generate / compare / exit)",
#             choices=["train", "generate", "compare", "exit"],
#             default="train",
#         )
#         if mode == "EXIT":
#             break
#         config["mode"] = mode
#
#         model = choose("Choose model", choices=ALL_MODEL_CHOICES, default="ngram")
#         if model == "EXIT":
#             break
#         config["model"] = model
#
#         device = choose(
#             "Device (cpu / cuda)",
#             choices=["cpu", "cuda"],
#             default="cuda" if torch.cuda.is_available() else "cpu",
#         )
#         if device == "EXIT":
#             break
#         config["device"] = device
#
#         if mode == "train":
#             # --- Ask about force retrain model first ---
#             force_model = choose(
#                 "Force retrain model? (y/n)", choices=["y", "n"], default="n"
#             )
#             config["force_model"] = force_model.lower() == "y"
#
#             # --- Ask about force retrain tokenizer separately ---
#             force_tokenizer = choose(
#                 "Force retrain tokenizer? (y/n)", choices=["y", "n"], default="n"
#             )
#             config["force_tokenizer"] = force_tokenizer.lower() == "y"
#
#             # --- If user wants to force model retrain, optionally skip the rest ---
#             if config["force_model"]:
#                 print(
#                     "\nSkipping the rest of training parameters as you chose to force retrain the model.\n"
#                 )
#                 cfg = minimal_cfg_for_model(model, device)
#                 pipe = LM_Pipeline(model_type=model, config=cfg, final=False)
#                 pipe.train(
#                     train_text=train_text,
#                     valid_text=valid_text,
#                     max_k=800,
#                     force_retrain_model=True,
#                     force_retrain_tokenizer=config["force_tokenizer"],
#                     train_limit=config.get("train_limit", 100_000),
#                     valid_limit=config.get("valid_limit", 10_000),
#                 )
#                 trained_pipe = pipe
#                 continue
#
#             if not config["force_model"]:
#                 # Ask all other training questions only if not forcing retrain
#                 config["n"] = choose_int("Order n (for ngram or neural)", default=3)
#                 config["embd_dim"] = choose_int(
#                     "Embedding dimension (for neural/gpt)", default=256
#                 )
#                 config["block_size"] = choose_int("Block size (for neural)", default=8)
#                 config["batch_size"] = choose_int("Batch size (for neural)", default=32)
#                 config["epochs"] = choose_int("Number of epochs", default=3)
#                 config["lr"] = choose_float("Learning rate", default=0.01)
#                 config["patience"] = choose_int("Patience", default=3)
#                 config["train_limit"] = choose_int("Train limit", default=100_000)
#                 config["valid_limit"] = choose_int("Valid limit", default=10_000)
#
#             # Setup pipeline config
#             if model in NGRAM_NAMES:
#                 cfg = NgramConfig(n=config.get("n", 3), device=config["device"])
#             elif model in NEURAL_SLOW_NAMES:
#                 cfg = NeuralConfig(
#                     n=config.get("n", 3),
#                     device=config["device"],
#                     vocab_size=None,
#                     embd_dim=config.get("embd_dim", 256),
#                     block_size=config.get("block_size", 8),
#                 )
#                 for k in ["batch_size", "epochs", "lr", "patience"]:
#                     setattr(
#                         cfg,
#                         k,
#                         config.get(
#                             k,
#                             {"batch_size": 32, "epochs": 3, "lr": 0.01, "patience": 3}[
#                                 k
#                             ],
#                         ),
#                     )
#             elif model in NEURAL_FAST_NAMES:
#                 cfg = NeuralFastConfig(
#                     n=config.get("n", 3),
#                     device=config["device"],
#                     vocab_size=None,
#                     embd_dim=config.get("embd_dim", 256),
#                     block_size=config.get("block_size", 8),
#                 )
#                 for k in ["batch_size", "epochs", "lr", "patience"]:
#                     setattr(
#                         cfg,
#                         k,
#                         config.get(
#                             k,
#                             {"batch_size": 32, "epochs": 3, "lr": 0.01, "patience": 3}[
#                                 k
#                             ],
#                         ),
#                     )
#             elif model in GPT_NAMES:
#                 cfg = GptConfig(
#                     vocab_size=0,
#                     n_heads=6,
#                     layer_dim=4,
#                     embd_dim=384,
#                     block_size=64,
#                     dropout=0.2,
#                     embd_pdrop=0.2,
#                     attn_pdrop=0.2,
#                     resid_pdrop=0.2,
#                     device=config["device"],
#                 )
#                 setattr(cfg, "batch_size", config.get("batch_size", 32))
#                 setattr(cfg, "max_iters", 5000)
#                 setattr(cfg, "learning_rate", 3e-4)
#             else:
#                 print(f"{Colors.WARNING}Unknown model {model}{Colors.ENDC}")
#                 continue
#
#             pipe = LM_Pipeline(model_type=model, config=cfg, final=False)
#             pipe.train(
#                 train_text=train_text,
#                 valid_text=valid_text,
#                 max_k=800,
#                 force_retrain_tokenizer=config.get("force_model", False),
#                 force_retrain_model=config.get("force_model", False),
#                 train_limit=config.get("train_limit", 100_000),
#                 valid_limit=config.get("valid_limit", 10_000),
#             )
#             trained_pipe = pipe  # keep reference to trained model
#             print("\nTraining done! You can now generate text or train again.\n")
#
#         elif mode == "generate":
#             # Ask first about pretrained
#             from_pretrained = choose(
#                 "Load pretrained model? (y/n)", choices=["y", "n"], default="y"
#             )
#             pipe = None
#
#             if from_pretrained.lower() == "y":
#                 pipe = LM_Pipeline(model_type=model, config=None, final=False)
#                 pipe.prepare_tokens(
#                     train_text=train_text, max_k=800, force_retrain=False, final=False
#                 )
#             else:
#                 if trained_pipe is not None:
#                     pipe = trained_pipe
#                 else:
#                     print(
#                         "No trained model in memory. Loading saved pretrained model instead."
#                     )
#                     pipe = LM_Pipeline(model_type=model, config=None, final=False)
#                     pipe.prepare_tokens(
#                         train_text=train_text,
#                         max_k=800,
#                         force_retrain=False,
#                         final=False,
#                     )
#
#             config["max_new_tokens"] = choose_int("Max new tokens", default=100)
#             config["top_k"] = choose_int("top_k for generation", default=40)
#             config["top_p"] = choose_float("top_p for generation", default=0.9)
#             config["temperature"] = choose_float(
#                 "temperature for generation", default=0.9
#             )
#
#             print(
#                 "\nEnter prompts continuously. Type 'exit' to quit generation loop.\n"
#             )
#             while True:
#                 prompt = input("Prompt: ").strip()
#                 if prompt.lower() == "exit":
#                     break
#                 out = pipe.generate(
#                     prompt=prompt,
#                     max_length=config["max_new_tokens"],
#                     from_pretrained=from_pretrained.lower() == "y",
#                 )
#                 print("\n=== Generated Text ===\n")
#                 print(out)
#                 print("\n---\n")
#
#         elif mode == "compare":
#             print("Compare mode not fully implemented in interactive mode yet.\n")
#
#         print("\nType 'exit' at any prompt to quit the program.\n")
#


def main():
    # in case flags are given first
    if len(sys.argv) > 1 and sys.argv[1].startswith("--"):
        raise SystemExit(_pipeline_run(sys.argv[1:]))

    # banner/intro
    try:
        welcome_banner()
        show_intro()
    except Exception:
        pass

    while True:
        mode = (
            input(
                f"{Colors.WARNING}Token console >{Colors.ENDC} Console mode (manual / assisted / exit) [manual if blank]: "
            )
            .strip()
            .lower()
            or "manual"
        )
        if mode in ("exit", "quit", "q"):
            break
        if mode == "manual":
            res = manual_mode_loop()
        elif mode == "assisted":
            res = assisted_wizard()
        else:
            print("Unknown option. Type 'manual', 'assisted', or 'exit'.")
            continue
        if res == "exit":
            break
    return 0


if __name__ == "__main__":
    main()
