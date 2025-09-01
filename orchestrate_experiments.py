#!/usr/bin/env python
"""
orchestrate_experiments.py
Run all experiments from Python (no PowerShell needed).

Usage (from repo root):
  # Stage 1: sweep n-gram & neural over K list
  python orchestrate_experiments.py --stage1

  # Stage 2: GPT grid over your chosen TopKs (edit config below or pass --topks ...)
  python orchestrate_experiments.py --stage2 --max-iters 4000

  # Stage 3: sample generations from best of each (optional)
  python orchestrate_experiments.py --stage3 --best-k 2000

  # Everything
  python orchestrate_experiments.py --all

All logs go to: experiments/logs/
CSV summary for GPT goes to: experiments/logs/gpt_results.csv
A Markdown report is written to: experiments/REPORT.md
"""

from __future__ import annotations
import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional

# ---------------------------- Config (edit as needed) ----------------------------
DEFAULT_K_LIST = [800, 1200, 1600, 2000]   # Stage 1 sweep
DEFAULT_TOPKS  = [800, 1200, 2000]         # Stage 2 GPT grid (set after Stage 1)
GPT_GRID = [
    {"embd_dim": 128, "dropout": 0.1, "n_layer": 4},
    {"embd_dim": 192, "dropout": 0.2, "n_layer": 4},
]
DEFAULT_MAX_ITERS = 4000
PYTHON_EXE = sys.executable  # use current interpreter
# -------------------------------------------------------------------------------

REPO_ROOT = Path.cwd()
LOG_DIR = REPO_ROOT / "experiments" / "logs"
PLOTS_DIR = REPO_ROOT / "experiments" / "plots"
NG_PLOTS_SRC = REPO_ROOT / "experiments" / "saved_models" / "ngram"
NN_PLOTS_SRC = REPO_ROOT / "experiments" / "saved_models" / "neural_ngram"
GPT_PLOTS_DIR = PLOTS_DIR / "gpt"

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def run_and_log(cmd: List[str], log_path: Path) -> int:
    """Run command, tee stdout/err to file and console."""
    ensure_dir(log_path.parent)
    with log_path.open("w", encoding="utf-8", newline="") as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        proc.wait()
        return proc.returncode

def mirror_plots(src_dir: Path, dst_dir: Path, names: Iterable[str]) -> None:
    ensure_dir(dst_dir)
    for n in names:
        src = src_dir / n
        if src.exists():
            shutil.copy2(src, dst_dir / n)
            print(f"[plots] copied {src} -> {dst_dir / n}")
        else:
            print(f"[plots] missing {src} (skipped)")

def parse_last_gpt_val_loss(log_path: Path) -> Optional[float]:
    """
    Find the last line like:
      step 3500: train loss 4.36, val loss 4.97
    Return the val loss as float, or None if not found.
    """
    pat = re.compile(r"step\s+\d+:\s+train loss\s+([0-9.]+),\s+val loss\s+([0-9.]+)")
    last = None
    try:
        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = pat.search(line)
            if m:
                last = float(m.group(2))
    except FileNotFoundError:
        return None
    return last

def append_gpt_csv(csv_path: Path, k: int, emb: int, drop: float, iters: int, val_loss: Optional[float], log_file: Path) -> None:
    ensure_dir(csv_path.parent)
    new_file = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp","K","embd_dim","dropout","max_iters","val_loss","val_ppl","logfile"])
        ppl = math.exp(val_loss) if val_loss is not None else ""
        w.writerow([datetime.now().isoformat(timespec="seconds"), k, emb, drop, iters, val_loss if val_loss is not None else "", ppl, str(log_file)])

def write_report_md(report_path: Path, gpt_csv: Path, k_list: List[int], topks: List[int]) -> None:
    ensure_dir(report_path.parent)
    lines = []
    lines.append("# Experiment Report\n")
    lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_\n")
    lines.append("## Stages & Settings\n")
    lines.append(f"- Stage 1 K sweep: `{k_list}`\n- Stage 2 GPT TopKs: `{topks}`\n- GPT grid: " + ", ".join([f"(embd={g['embd_dim']}, drop={g['dropout']}, layers={g['n_layer']})" for g in GPT_GRID]) + "\n")
    lines.append("\n---\n\n## GPT Results (from logs)\n")
    if gpt_csv.exists():
        lines.append(f"CSV: `{gpt_csv.as_posix()}`\n\n")
        try:
            with gpt_csv.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                lines.append("| timestamp | K | embd_dim | dropout | max_iters | val_loss | val_ppl |\n")
                lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
                for r in rows:
                    lines.append(f"| {r['timestamp']} | {r['K']} | {r['embd_dim']} | {r['dropout']} | {r['max_iters']} | {r['val_loss']} | {r['val_ppl']} |\n")
        except Exception as e:
            lines.append(f"_Could not read CSV: {e}_\n")
    else:
        lines.append("_No GPT CSV yet._\n")

    lines.append("\n---\n\n## Plots\n")
    # N-gram
    lines.append("### Classic n-gram\n")
    lines.append("- `experiments/saved_models/ngram/lambda_perplexity.png`\n")
    lines.append("- `experiments/saved_models/ngram/vocabulary_growth.png`\n")
    # Neural n-gram
    lines.append("\n### Neural n-gram\n")
    lines.append("- `experiments/saved_models/neural_ngram/perplexity_curve.png`\n")
    lines.append("- `experiments/saved_models/neural_ngram/vocabulary_growth.png`\n")
    # GPT
    lines.append("\n### GPT\n")
    lines.append("- `experiments/plots/gpt/loss_curve.png`\n")
    lines.append("- `experiments/plots/gpt/val_perplexity.png`\n")
    lines.append("- `experiments/plots/gpt/generation_preview.txt`\n")

    report_path.write_text("".join(lines), encoding="utf-8")
    print(f"[report] wrote {report_path}")

def stage1(k_list: List[int]) -> None:
    print("=== Stage 1: sweeping K with classic & neural n-gram ===")
    ensure_dir(LOG_DIR); ensure_dir(PLOTS_DIR / "ngram"); ensure_dir(PLOTS_DIR / "neural_ngram")

    for k in k_list:
        # Classic n-gram
        log_ng = LOG_DIR / f"ngram_k{k}.log"
        cmd_ng = [PYTHON_EXE, "main.py", "train", "--model", "ngram", "--n", "3", "--max_k", str(k), "--tune_lambdas", "--force_retrain"]
        rc = run_and_log(cmd_ng, log_ng)
        if rc != 0: raise SystemExit(f"ngram K={k} failed with code {rc}")
        mirror_plots(NG_PLOTS_SRC, PLOTS_DIR / "ngram", ["lambda_perplexity.png","vocabulary_growth.png"])

        # Neural n-gram
        log_nn = LOG_DIR / f"neural_k{k}.log"
        cmd_nn = [PYTHON_EXE, "main.py", "train", "--model", "neural", "--n", "3", "--max_k", str(k), "--epochs", "8", "--batch_size", "32", "--block_size", "8", "--embedding_dim", "16", "--force_retrain"]
        rc = run_and_log(cmd_nn, log_nn)
        if rc != 0: raise SystemExit(f"neural K={k} failed with code {rc}")
        mirror_plots(NN_PLOTS_SRC, PLOTS_DIR / "neural_ngram", ["perplexity_curve.png","vocabulary_growth.png"])

    print("Stage 1 complete. Inspect plots and choose TopKs.")

def stage2(topks: List[int], max_iters: int) -> None:
    print("=== Stage 2: GPT grid on chosen TopKs ===")
    ensure_dir(LOG_DIR); ensure_dir(GPT_PLOTS_DIR)
    csv_path = LOG_DIR / "gpt_results.csv"

    for k in topks:
        for g in GPT_GRID:
            emb = g["embd_dim"]; drop = g["dropout"]; n_layer = g["n_layer"]
            tag = f"k{k}_e{emb}_d{str(drop).replace('.','p')}_L{n_layer}_I{max_iters}"
            log = LOG_DIR / f"gpt_{tag}.log"
            cmd = [PYTHON_EXE,"main.py","train","--model","gpt",
                   "--max_iters", str(max_iters),
                   "--embd_dim", str(emb),
                   "--n_layer", str(n_layer),
                   "--dropout", str(drop),
                   "--max_k", str(k),
                   "--force_retrain"]
            rc = run_and_log(cmd, log)
            if rc != 0: raise SystemExit(f"GPT K={k} emb={emb} drop={drop} failed with code {rc}")
            val_loss = parse_last_gpt_val_loss(log)
            append_gpt_csv(csv_path, k, emb, drop, max_iters, val_loss, log)

    print("Stage 2 complete. See experiments/logs/gpt_results.csv")

def stage3_generate(best_k: int, max_new_tokens: int = 120) -> None:
    print("=== Stage 3: qualitative generations ===")
    # neural
    log_nn = LOG_DIR / f"gen_neural_k{best_k}.log"
    cmd_nn = [PYTHON_EXE, "main.py", "generate", "--model", "neural",
              "--prompt", "Once upon a time", "--max_k", str(best_k),
              "--max_new_tokens", str(max_new_tokens)]
    run_and_log(cmd_nn, log_nn)
    # gpt
    for prompt in ["ROMEO:", "Once upon a time"]:
        tag = prompt.replace(" ", "_").replace(":", "")
        log_g = LOG_DIR / f"gen_gpt_{tag}.log"
        cmd_g = [PYTHON_EXE, "main.py", "generate", "--model", "gpt",
                 "--prompt", prompt, "--max_new_tokens", str(max_new_tokens)]
        run_and_log(cmd_g, log_g)
    # n-gram
    log_ng = LOG_DIR / f"gen_ngram_k{best_k}.log"
    cmd_ng = [PYTHON_EXE, "main.py", "generate", "--model", "ngram",
              "--prompt", "Once upon a time", "--max_new_tokens", str(max_new_tokens)]
    run_and_log(cmd_ng, log_ng)

def main():
    parser = argparse.ArgumentParser(description="Run token-by-token experiments from Python.")
    parser.add_argument("--stage1", action="store_true", help="Run Stage 1 K sweep for n-gram & neural n-gram.")
    parser.add_argument("--stage2", action="store_true", help="Run Stage 2 GPT grid on chosen TopKs.")
    parser.add_argument("--stage3", action="store_true", help="Run Stage 3 qualitative generations.")
    parser.add_argument("--all",    action="store_true", help="Run all stages sequentially.")
    parser.add_argument("--k-list", type=int, nargs="+", default=DEFAULT_K_LIST, help="K values for Stage 1 sweep.")
    parser.add_argument("--topks",  type=int, nargs="+", default=DEFAULT_TOPKS, help="TopKs for Stage 2 GPT grid.")
    parser.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS, help="max_iters for GPT Stage 2.")
    parser.add_argument("--best-k", type=int, default=2000, help="K to use for Stage 3 generations.")
    args = parser.parse_args()

    # ensure directories
    ensure_dir(LOG_DIR); ensure_dir(PLOTS_DIR)

    if args.all or args.stage1:
        stage1(args.k_list)

    if args.all or args.stage2:
        stage2(args.topks, args.max_iters)

    if args.all or args.stage3:
        stage3_generate(args.best_k)

    # Always (re)write a brief report stub
    write_report_md(REPO_ROOT / "experiments" / "REPORT.md",
                    LOG_DIR / "gpt_results.csv",
                    args.k_list, args.topks)

if __name__ == "__main__":
    main()
