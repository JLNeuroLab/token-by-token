import argparse
import os
import torch
import platform
from shutil import get_terminal_size

from llm_project.models.gpt.train import main as train_main
from llm_project.models.gpt.generator import Generator


def run_training(args):
    print("[INFO] Running training with arguments:", vars(args))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Override args inside train.py via monkeypatch or CLI config injection
    train_main(
        max_iters=args.max_iters,
        embd_dim=args.embd_dim,
        n_layer=args.n_layer,
        dropout=args.dropout,
        max_k=args.max_k,
        device=args.device,
    )


def run_generation(args):
    print("[INFO] Generating samples with model")
    out = Generator(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        prompt=args.prompt,
        device=args.device,
    )
    print("\n=== Generated Text ===\n")
    print(out)


def main():
    parser = argparse.ArgumentParser(
        description="Token-by-Token A minimalist GPT playground by Larizza & Ruiz",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python main.py train --max_iters 5000 --dropout 0.1
  python main.py generate --model_path saved_models/gpt_shakespeare.pth --prompt "To be or not to be"

Available modes:
  train     Train a GPT model from scratch on Shakespeare
  generate  Generate text using a trained model

Made with love and caffeine (and a lot of Ritalin).
""",
    )

    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version="Token-by-Token v1.0 by Larizza & Ruiz",
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Train mode
    train_parser = subparsers.add_parser("train", help="Train a GPT model")
    train_parser.add_argument("--max_iters", type=int, default=8000)
    train_parser.add_argument("--embd_dim", type=int, default=128)
    train_parser.add_argument("--n_layer", type=int, default=4)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--max_k", type=int, default=2000)
    train_parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    train_parser.set_defaults(func=run_training)

    # Generate mode
    gen_parser = subparsers.add_parser(
        "generate", help="Generate text using a trained model"
    )
    gen_parser.add_argument("--model_path", type=str, required=True)
    gen_parser.add_argument("--prompt", type=str, required=True)
    gen_parser.add_argument("--max_new_tokens", type=int, default=100)
    gen_parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    gen_parser.set_defaults(func=run_generation)

    if len(os.sys.argv) == 1:
        parser.print_help()
        exit(1)
    args = parser.parse_args()
    args.func(args)


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

    print(f"{'=' * term_width}")
    print(
        f"PyTorch Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}"
    )
    print(f"OS: {platform.system()} {platform.release()}")
    print("vocab Checkpoint: experiments/bpe_results/train_final_vocab.pkl")
    print(f"{'=' * term_width}\n")


if __name__ == "__main__":
    welcome_banner()
    main()
