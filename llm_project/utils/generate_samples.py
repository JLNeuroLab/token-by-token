import argparse
import pathlib
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, required=True)
args = parser.parse_args()
k = args.k

PROMPT = "Once upon a time"
MAX_NEW_TOKENS = "100"
MODELS = ["ngram", "neural", "gpt"]
OUTBASE = pathlib.Path("experiments/summary/generations")
OUTBASE.mkdir(parents=True, exist_ok=True)


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


for model in MODELS:
    out = run(
        [
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
    )
    path = OUTBASE / f"{model}_k{k}.txt"
    path.write_text(out)
    print("✅", path)
