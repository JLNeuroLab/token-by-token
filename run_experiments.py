import os
import subprocess
import json
import re
from datetime import datetime
import argparse
from llm_project.utils.debugg_utils import Colors

# --- 1. Define Your Experiments ---
# An expanded suite of 32 experiments to run overnight.
experiments = [
    # --- Baseline and Best Model ---
    {
        "name": "A_baseline_model",
        "params": {
            "MAX_ITERS": 5000,
            "embd_dim": 128,
            "n_layer": 4,
            "dropout": 0.1,
            "max_k": 650,
        },
    },
    {
        "name": "B_best_model_so_far",
        "params": {
            "MAX_ITERS": 8000,
            "embd_dim": 128,
            "n_layer": 4,
            "dropout": 0.2,
            "max_k": 2000,
        },
    },
    # --- Exploring Model Depth (n_layer) ---
    {
        "name": "C_deeper_model_more_dropout",
        "params": {
            "MAX_ITERS": 8000,
            "embd_dim": 128,
            "n_layer": 6,
            "dropout": 0.2,
            "max_k": 2000,
        },
    },
    {
        "name": "D_very_deep_model",
        "params": {
            "MAX_ITERS": 8000,
            "embd_dim": 128,
            "n_layer": 8,
            "dropout": 0.2,
            "max_k": 2000,
        },
    },
    {
        "name": "E_shallow_model",
        "params": {
            "MAX_ITERS": 5000,
            "embd_dim": 128,
            "n_layer": 2,
            "dropout": 0.1,
            "max_k": 2000,
        },
    },
    # --- Exploring Model Width (embd_dim) ---
    {
        "name": "F_wider_model",
        "params": {
            "MAX_ITERS": 8000,
            "embd_dim": 256,
            "n_layer": 4,
            "dropout": 0.2,
            "max_k": 2000,
        },
    },
    {
        "name": "G_very_wide_model",
        "params": {
            "MAX_ITERS": 8000,
            "embd_dim": 384,
            "n_layer": 4,
            "dropout": 0.2,
            "max_k": 2000,
        },
    },
    {
        "name": "H_narrow_model",
        "params": {
            "MAX_ITERS": 5000,
            "embd_dim": 64,
            "n_layer": 4,
            "dropout": 0.1,
            "max_k": 2000,
        },
    },
    # --- Exploring BPE Merges (max_k) ---
    {
        "name": "I_bpe_merges_1000",
        "params": {
            "MAX_ITERS": 5000,
            "embd_dim": 128,
            "n_layer": 4,
            "dropout": 0.1,
            "max_k": 1000,
        },
    },
    {
        "name": "J_bpe_merges_3000",
        "params": {
            "MAX_ITERS": 5000,
            "embd_dim": 128,
            "n_layer": 4,
            "dropout": 0.1,
            "max_k": 3000,
        },
    },
    {
        "name": "K_bpe_merges_4000",
        "params": {
            "MAX_ITERS": 5000,
            "embd_dim": 128,
            "n_layer": 4,
            "dropout": 0.1,
            "max_k": 4000,
        },
    },
    # --- Exploring Learning Rate ---
    {
        "name": "L_lower_learning_rate",
        "params": {
            "MAX_ITERS": 8000,
            "embd_dim": 128,
            "n_layer": 4,
            "dropout": 0.2,
            "LEARNING_RATE": 1e-4,
            "max_k": 2000,
        },
    },
    {
        "name": "M_higher_learning_rate",
        "params": {
            "MAX_ITERS": 5000,
            "embd_dim": 128,
            "n_layer": 4,
            "dropout": 0.1,
            "LEARNING_RATE": 5e-4,
            "max_k": 2000,
        },
    },
    # --- Exploring Batch Size ---
    {
        "name": "N_larger_batch_size",
        "params": {
            "MAX_ITERS": 5000,
            "embd_dim": 128,
            "n_layer": 4,
            "dropout": 0.1,
            "BATCH_SIZE": 64,
            "max_k": 2000,
        },
    },
    {
        "name": "O_smaller_batch_size",
        "params": {
            "MAX_ITERS": 5000,
            "embd_dim": 128,
            "n_layer": 4,
            "dropout": 0.1,
            "BATCH_SIZE": 16,
            "max_k": 2000,
        },
    },
    # --- Combination Experiments (trying to find a new best) ---
    {
        "name": "P_wide_and_deep",
        "params": {
            "MAX_ITERS": 8000,
            "embd_dim": 256,
            "n_layer": 6,
            "dropout": 0.2,
            "max_k": 2000,
        },
    },
    {
        "name": "Q_wide_and_more_dropout",
        "params": {
            "MAX_ITERS": 8000,
            "embd_dim": 256,
            "n_layer": 4,
            "dropout": 0.3,
            "max_k": 2000,
        },
    },
    {
        "name": "R_deep_and_more_dropout",
        "params": {
            "MAX_ITERS": 8000,
            "embd_dim": 128,
            "n_layer": 6,
            "dropout": 0.3,
            "max_k": 2000,
        },
    },
    {
        "name": "S_best_model_fewer_merges",
        "params": {
            "MAX_ITERS": 8000,
            "embd_dim": 128,
            "n_layer": 4,
            "dropout": 0.2,
            "max_k": 1000,
        },
    },
    {
        "name": "T_best_model_longer_training",
        "params": {
            "MAX_ITERS": 12000,
            "embd_dim": 128,
            "n_layer": 4,
            "dropout": 0.2,
            "max_k": 2000,
        },
    },
    # --- Added 12 More Experiments ---
    {
        "name": "U_wider_deeper_more_dropout",
        "params": {
            "MAX_ITERS": 10000,
            "embd_dim": 256,
            "n_layer": 6,
            "dropout": 0.3,
            "max_k": 2000,
        },
    },
    {
        "name": "V_very_wide_less_deep",
        "params": {
            "MAX_ITERS": 8000,
            "embd_dim": 384,
            "n_layer": 2,
            "dropout": 0.2,
            "max_k": 2000,
        },
    },
    {
        "name": "W_best_model_more_merges",
        "params": {
            "MAX_ITERS": 8000,
            "embd_dim": 128,
            "n_layer": 4,
            "dropout": 0.2,
            "max_k": 3000,
        },
    },
    {
        "name": "X_extreme_dropout",
        "params": {
            "MAX_ITERS": 10000,
            "embd_dim": 128,
            "n_layer": 4,
            "dropout": 0.4,
            "max_k": 2000,
        },
    },
    {
        "name": "Y_larger_block_size",
        "params": {
            "MAX_ITERS": 5000,
            "embd_dim": 128,
            "n_layer": 4,
            "dropout": 0.1,
            "BLOCK_SIZE": 128,
            "max_k": 2000,
        },
    },
    {
        "name": "Z_more_heads_deeper",
        "params": {
            "MAX_ITERS": 8000,
            "embd_dim": 128,
            "n_layer": 6,
            "n_head": 8,
            "dropout": 0.2,
            "max_k": 2000,
        },
    },
    {
        "name": "AA_wide_and_shallow",
        "params": {
            "MAX_ITERS": 8000,
            "embd_dim": 384,
            "n_layer": 2,
            "dropout": 0.2,
            "max_k": 2000,
        },
    },
    {
        "name": "AB_best_config_lower_lr",
        "params": {
            "MAX_ITERS": 10000,
            "embd_dim": 128,
            "n_layer": 4,
            "dropout": 0.2,
            "LEARNING_RATE": 1e-4,
            "max_k": 2000,
        },
    },
    {
        "name": "AC_very_small_model",
        "params": {
            "MAX_ITERS": 5000,
            "embd_dim": 64,
            "n_layer": 2,
            "dropout": 0.1,
            "max_k": 1000,
        },
    },
    {
        "name": "AD_wide_less_dropout",
        "params": {
            "MAX_ITERS": 8000,
            "embd_dim": 256,
            "n_layer": 4,
            "dropout": 0.1,
            "max_k": 2000,
        },
    },
    {
        "name": "AE_deep_less_dropout",
        "params": {
            "MAX_ITERS": 8000,
            "embd_dim": 128,
            "n_layer": 6,
            "dropout": 0.1,
            "max_k": 2000,
        },
    },
    {
        "name": "AF_best_model_longest_training",
        "params": {
            "MAX_ITERS": 15000,
            "embd_dim": 128,
            "n_layer": 4,
            "dropout": 0.2,
            "max_k": 2000,
        },
    },
]


# --- 2. Setup Paths ---
project_root = os.path.dirname(os.path.abspath(__file__))
train_script_path = os.path.join(
    project_root, "llm_project", "models", "gpt", "train.py"
)
results_file_path = os.path.join(project_root, "experiments", "tuning_results.json")
saved_models_dir = os.path.join(project_root, "saved_models")
bpe_cache_folder = os.path.join(
    project_root, "experiments", "bpe_results", "train_results"
)
merges_cache_file = os.path.join(bpe_cache_folder, "train_bpe_merges.pkl")
vocab_cache_file = os.path.join(bpe_cache_folder, "train_final_vocab.pkl")


os.makedirs(os.path.dirname(results_file_path), exist_ok=True)

# --- 3. Main Experiment Loop ---


def run_all_experiments(args):
    all_results = []

    if os.path.exists(results_file_path):
        with open(results_file_path, "r") as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} previous results.")

    for i, exp_config in enumerate(experiments):
        exp_name = exp_config["name"]
        params = exp_config["params"]

        print("-" * 80)
        print(
            f"{Colors.OKBLUE}[STARTING]{Colors.ENDC} Experiment {i + 1}/{len(experiments)}: {Colors.BOLD}{exp_name}{Colors.ENDC}"
        )
        print(f"   Parameters: {params}")
        if args.override_k:
            params["max_k"] = args.override_k

        if "max_k" in params:
            expected_cache_name = f"bpe_merges_k{params['max_k']}.pkl"
            merges_cache_file_specific = os.path.join(
                bpe_cache_folder, expected_cache_name
            )
            if not os.path.exists(merges_cache_file_specific):
                print(
                    "   New BPE merge value detected. Deleting old tokenizer cache..."
                )
                if os.path.exists(merges_cache_file):
                    os.remove(merges_cache_file)
                if os.path.exists(vocab_cache_file):
                    os.remove(vocab_cache_file)

        with open(train_script_path, "r") as f:
            original_script = f.read()

        modified_script = original_script
        for param_name, param_value in params.items():
            if param_name in ["dropout", "embd_pdrop", "attn_pdrop", "resid_pdrop"]:
                modified_script = re.sub(
                    r"(dropout\s*=\s*)[\d.]+", f"\\g<1>{param_value}", modified_script
                )
                modified_script = re.sub(
                    r"(embd_pdrop\s*=\s*)[\d.]+",
                    f"\\g<1>{param_value}",
                    modified_script,
                )
                modified_script = re.sub(
                    r"(attn_pdrop\s*=\s*)[\d.]+",
                    f"\\g<1>{param_value}",
                    modified_script,
                )
                modified_script = re.sub(
                    r"(resid_pdrop\s*=\s*)[\d.]+",
                    f"\\g<1>{param_value}",
                    modified_script,
                )
            else:
                pattern = re.compile(f"({param_name}\\s*=\s*)[\\d.e-]+")
                replacement = f"\\g<1>{param_value}"
                modified_script = pattern.sub(replacement, modified_script)

        # --- START: Real-time Output Fix ---
        # Use Popen to run the script as a subprocess and stream its output
        process = subprocess.Popen(
            ["python", "-c", modified_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line-buffered
            encoding="utf-8",
        )

        full_output = []

        # Read stdout line by line in real-time
        for line in process.stdout:
            print(line, end="")  # Print the progress to your terminal
            full_output.append(line)

        # Wait for the process to finish and capture any errors
        process.wait()
        error_output = process.stderr.read()
        # --- END: Real-time Output Fix ---

        if process.returncode == 0:
            final_loss = None
            for line in reversed(full_output):
                if "val loss" in line:
                    match = re.search(r"val loss ([\d.]+)", line)
                    if match:
                        final_loss = float(match.group(1))
                        break

            if final_loss is not None:
                print(
                    f"{Colors.OKGREEN}[SUCCESS]{Colors.ENDC} Experiment '{exp_name}' finished."
                )
                print(
                    f"   Final Validation Loss: {Colors.BOLD}{final_loss:.4f}{Colors.ENDC}"
                )

                model_filename = f"gpt_shakespeare_{exp_name}.pth"
                os.replace(
                    os.path.join(saved_models_dir, "gpt_shakespeare.pth"),
                    os.path.join(saved_models_dir, model_filename),
                )

                result = {
                    "experiment_name": exp_name,
                    "parameters": params,
                    "final_val_loss": final_loss,
                    "model_file": model_filename,
                    "timestamp": datetime.now().isoformat(),
                }
                all_results.append(result.copy())
            else:
                print(
                    f"{Colors.WARNING}[WARNING]{Colors.ENDC} Could not parse final validation loss for '{exp_name}'."
                )
        else:
            print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} Experiment '{exp_name}' failed!")
            print(error_output)

        with open(results_file_path, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"   Results saved to {results_file_path}")

    print("-" * 80)
    print(
        f"{Colors.OKGREEN}{Colors.BOLD}[COMPLETE] All experiments finished!{Colors.ENDC}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--override_k", type=int, help="Override max_k for BPE steps")
    args = parser.parse_args()
    run_all_experiments(args)
