import argparse
import os
import torch
import platform
from shutil import get_terminal_size
from llm_project.utils.dataloader import load_shakespeare

# -------------- Imports for GPT ----------------
from llm_project.models.gpt.train import main as GPT_Trainer
from llm_project.models.gpt.generator import Generator
# -------------- Imports for ngram --------------------
from llm_project.models.ngrams.trainer import NGramTrainer
# --------------- Imports for nerural ngram ---------------
from llm_project.models.neural_ngrams.trainer import NeuralNgramTrainer

def run_training(args):
    print("[INFO] Running training with arguments:", vars(args))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print_args_for_model(args)

    train_text = load_shakespeare("train")[:args.train_size]
    valid_text = load_shakespeare("validation")[:args.valid_size]


    if args.model == "gpt":
    # Override args inside train.py via monkeypatch or CLI config injection
        GPT_Trainer(
            max_iters=args.max_iters,
            embd_dim=args.embd_dim,
            n_layer=args.n_layer,
            dropout=args.dropout,
            max_k=args.max_k,
            device=args.device,
        )
    elif args.model == "ngram":
        ngram_trainer = NGramTrainer(model=None,
                                     tokens=None,
                                     n=args.n,
                                     max_k=args.max_k)
        ngram_trainer.train(tune_lambdas=args.tune_lambdas,
                            force_retrain=args.force_retrain)
        
    elif args.model == "neural":
        nngram_trainer = NeuralNgramTrainer(
            model=None,
            n=args.n,
            train_text=train_text,
            valid_text=valid_text,
            max_k=args.max_k,
            batch_size=args.batch_size,
            block_size=args.block_size,
            embedding_dim=args.embedding_dim,
        )
    
        nngram_trainer.fit(epochs=args.epochs,
                       lr=args.lr,
                        patience=args.patience,
                        force_retrain=args.force_retrain)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    


def run_generation(args):
    print("[INFO] Generating samples with model")
    print_args_for_model(args)

    if args.model == "gpt":
        out = Generator(
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
            prompt=args.prompt,
            device=args.device,
        )
        print("\n=== Generated Text ===\n")
        print(out)
    
    elif args.model == "ngram":
        ngram_trainer = NGramTrainer(model=None, 
                                     tokens=None, 
                                     n=args.n, 
                                     max_k=args.max_k)
        
        model, merges = ngram_trainer.train(force_retrain=False, tune_lambdas=False)

        prompt_tokens = ngram_trainer.bpe.BPE_segmenter(args.prompt)
        generated_tokens = model.generate_text(prompt_tokens, max_length=args.max_new_tokens)
        print("\n=== Generated Text ===\n")
        print("".join(generated_tokens))
    
    elif args.model == "neural":
        nngram_trainer = NeuralNgramTrainer(
            model=None,
            n=args.n,
            train_text=[],   
            valid_text=[],
            max_k=args.max_k,
            batch_size=args.batch_size,
            block_size=args.block_size,
            print_every=args.print_every,
            embedding_dim=args.embedding_dim,
            autoload=True
        )
        nngram_trainer.prepare_bpe(force_retrain=False)

        generated_ids, generated_tokens = nngram_trainer.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            stochastic=True,   
            stop_words={'.', '\n'}
        )

        print("\n=== Generated Text ===\n")
        print("".join(generated_tokens))
                

    else:
        raise ValueError(f"Unknown model: {args.model}")


def main():
    parser = argparse.ArgumentParser(
        description="Token-by-Token: a minimalist NLP playground for text generation\n"
                    "Supports GPT, classic NGram, and Neural NGram models.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
    Examples:

    Train examples
    python main.py train --model gpt --max_iters 5000 --dropout 0.1 --embd_dim 128 --n_layer 4
    python main.py train --model ngram --n 3 --tune_lambdas --force_retrain
    python main.py train --model neural --n 3 --epochs 10 --batch_size 32 --block_size 8 --embedding_dim 16 --force_retrain

    Generate examples
    python main.py generate --model gpt  --prompt "To be or not to be" --max_new_tokens 100
    python main.py generate --model ngram  --prompt "Once upon a time" --max_new_tokens 100
    python main.py generate --model neural --prompt "Once upon a time" --max_new_tokens 100

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

    # ---------------- Train mode --------------------
    train_parser = subparsers.add_parser("train", help="Train a GPT model")

    # Common args
    train_parser.add_argument(
    "--model", type=str,  choices=["gpt", "ngram", "neural"], default="gpt",
    help="Which model to train: gpt | ngram | neural"
    )
    
    train_parser.add_argument("--max_k", type=int, default=2000)
    train_parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    # GPT args
    train_parser.add_argument("--max_iters", type=int, default=8000)
    train_parser.add_argument("--embd_dim", type=int, default=128)
    train_parser.add_argument("--n_layer", type=int, default=4)
    train_parser.add_argument("--dropout", type=float, default=0.1)

    # ngram args
    train_parser.add_argument("--n", type=int, default=3, help="Order of the n-gram")
    train_parser.add_argument("--train_size", type=int, default=10000)
    train_parser.add_argument("--valid_size", type=int, default=1000)
    train_parser.add_argument("--tune_lambdas", action="store_true")
    train_parser.add_argument("--force_retrain", action="store_true")

    # neural ngram args
    train_parser.add_argument("--batch_size", type=int, default=32)
    train_parser.add_argument("--block_size", type=int, default=8)
    train_parser.add_argument("--embedding_dim", type=int, default=16)
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--lr", type=float, default=0.01)
    train_parser.add_argument("--patience", type=int, default=2)

    train_parser.set_defaults(func=run_training)

    #--------------------- Generate mode -----------------------
    # common args
    gen_parser = subparsers.add_parser(
        "generate", help="Generate text using a trained model"
    )
    gen_parser.add_argument(
    "--model", type=str, choices=["gpt", "ngram", "neural"], default="gpt",
        help="Which model to generate from: gpt | ngram | neural gram"
    )
    gen_parser.add_argument("--model_path", type=str, default=None,
                        help="Path to checkpoint (used only by GPT)")   # Need to review this mechanism
    gen_parser.add_argument("--prompt", type=str, required=True)
    gen_parser.add_argument("--max_new_tokens", type=int, default=100)
    gen_parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Ngram args
    gen_parser.add_argument("--n", type=int, default=3, help="Order of the n-gram (for ngram or neural)")

    # Neural NGram args
    gen_parser.add_argument("--max_k", type=int, default=2000, help="Maximum context size / k for NGram or Neural NGram")
    gen_parser.add_argument("--batch_size", type=int, default=32)
    gen_parser.add_argument("--block_size", type=int, default=8)
    gen_parser.add_argument("--embedding_dim", type=int, default=16)
    gen_parser.add_argument("--print_every", type=int, default=50)

    gen_parser.set_defaults(func=run_generation)

    if len(os.sys.argv) == 1:
        welcome_banner()
        parser.print_help()
        exit(1)
    args = parser.parse_args()
    welcome_banner()
    args.func(args)

def print_args_for_model(args):
    if args.model == "gpt":
        relevant = ["mode", "model", "max_iters", "embd_dim", "n_layer", "dropout", "max_k", "device"]
    elif args.model == "ngram":
        relevant = ["mode", "model", "n", "train_size", "valid_size", "tune_lambdas", "force_retrain", "max_k", "device"]
    elif args.model == "neural":
        relevant = ["mode", "model", "n", "max_k", "batch_size", "block_size", "embedding_dim",
                    "epochs", "lr", "patience", "print_every", "device"]
    else:
        relevant = vars(args).keys()  # fallback

    print("--- Provided Arguments ---")
    for k in relevant:
        if hasattr(args, k):
            print(f"{k}: {getattr(args, k)}")


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
    print("Available Models: gpt | ngram | neural")
    print("Default Checkpoints Directory: saved_models/")
    
    
    print(f"{'=' * term_width}\n")
    print(f"{'=' * term_width}\n")


if __name__ == "__main__":
    main()
