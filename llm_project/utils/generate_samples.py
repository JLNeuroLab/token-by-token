import argparse
import pathlib
import subprocess
from llm_project.utils.debugg_utils import Colors

parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, required=True)
args = parser.parse_args()
k = args.k

PROMPT = "Once upon a time"
MAX_NEW_TOKENS = "100"
MODELS = ["ngram", "neural", "gpt"]
OUTBASE = pathlib.Path("experiments/summary/generations")
OUTBASE.mkdir(parents=True, exist_ok=True)


# set to your best file
NEURAL_CKPT = pathlib.Path(
    "experiments/saved_models/neural_ngram/val=2.8311_epoch=1.pkl"
)
# Report defaults for GPT (train/generate must match shapes)
GPT_HP = {"embd_dim": "384", "n_layer": "4", "dropout": "0.2", "block_size": "64"}


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)  # no check=True
    # Return both stdout and stderr so we can write them to the file
    out = p.stdout or ""
    if p.stderr:
        out += "\n\n[STDERR]\n" + p.stderr
    if p.returncode != 0:
        print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} command failed: {' '.join(cmd)}")
    return out


for model in MODELS:
    cmd = [
        "python",
        "main.py",
        "generate",
        "--model",
        model,
        "--k",
        str(k),
        "--prompt",
        PROMPT,
        "--max_new_tokens",
        MAX_NEW_TOKENS,
    ]
    if model == "neural":
        if NEURAL_CKPT.exists():
            cmd += ["--neural_ckpt", str(NEURAL_CKPT)]
        else:
            print(
                f"{Colors.WARNING}[WARN]{Colors.ENDC} Neural checkpoint not found: {NEURAL_CKPT} (running without it)"
            )

    if model == "gpt":
        # These flags require that main.py's generate parser accepts them.
        cmd += [
            "--embd_dim",
            GPT_HP["embd_dim"],
            "--n_layer",
            GPT_HP["n_layer"],
            "--dropout",
            GPT_HP["dropout"],
            "--block_size",
            GPT_HP["block_size"],
        ]

    out = run(cmd)
    path = OUTBASE / f"{model}_k{k}.txt"
    path.write_text(out)
    print(f"{Colors.OKGREEN}[DONE]{Colors.ENDC}", path)
