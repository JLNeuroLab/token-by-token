import argparse
import csv
import math
import pathlib
import re
import subprocess
from llm_project.utils.debugg_utils import Colors

# --------------------
# Argparse CLI Options
# --------------------
parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, required=True)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--block_size", type=int, default=8)
args = parser.parse_args()

k = args.k
batch_size = args.batch_size
block_size = args.block_size

# --------------------
# Global config
# --------------------
MODELS = ["ngram", "neural", "gpt"]
OUTDIR = pathlib.Path("experiments/summary")
OUTDIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTDIR / f"ppl_table_k{k}.csv"

# --------------------
# Subprocess runner
# --------------------


def run(cmd, output_path=None):
    print(">>", " ".join(cmd))
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as proc:
        lines = []
        for line in proc.stdout:
            print(line, end="")  # live output
            lines.append(line)
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, proc.args)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
        return "".join(lines)  # important for parse_metrics!


# --------------------
# Extract validation perplexity
# --------------------


def parse_metrics(model, stdout):
    m = re.search(r"Validation perplexity:\s*([0-9.]+)", stdout, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"Validation loss:\s*([0-9.]+)", stdout, re.I)
    if m:
        return math.exp(float(m.group(1)))
    raise RuntimeError(f"Couldn't parse PPL for {model}. Output:\n{stdout}")


# --------------------
# Run experiments
# --------------------
rows = [("model", "k", "val_perplexity")]

for model in MODELS:
    # --- Train ---
    train_args = [
        "python",
        "main.py",
        "generate",
        "--model",
        model,
        "--k",
        str(k),
        "--prompt",
        "To be or not to be",
        "--max_new_tokens",
        "100",
        "--batch_size",
        str(batch_size),
        "--block_size",
        str(block_size),
    ]
    run(train_args)

    # --- Generate ---
    output_path = f"experiments/results/generation_{model}_k{k}.txt"
    generate_args = [
        "python",
        "main.py",
        "generate",
        "--model",
        model,
        "--k",
        str(k),
        "--prompt",
        "To be or not to be",
        "--max_new_tokens",
        "100",
        "--batch_size",
        str(batch_size),
        "--block_size",
        str(block_size),
    ]
    stdout = run(generate_args, output_path=output_path)

    # --- Parse metrics ---
    ppl = parse_metrics(model, stdout)
    rows.append((model, k, round(ppl, 2)))

# --------------------
# Save final table
# --------------------
with open(CSV_PATH, "w", newline="") as f:
    csv.writer(f).writerows(rows)

print(f"\n {Colors.OKCYAN}Saved:{Colors.ENDC} {CSV_PATH}")
