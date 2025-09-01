import os
import subprocess
import json
from datetime import datetime
import argparse
from llm_project.utils.debugg_utils import Colors
import re
import shutil

# --- (File loading and path setup is unchanged) ---
experiments_json_path = "experiments.json"
try:
    with open(experiments_json_path, "r") as f:
        experiments = json.load(f)
    print(
        f"Successfully loaded {len(experiments)} experiments from {experiments_json_path}"
    )
except FileNotFoundError:
    print(f"Error: Could not find the experiments file at {experiments_json_path}")
    exit()
except json.JSONDecodeError:
    print(f"Error: The file {experiments_json_path} is not a valid JSON file.")
    exit()

project_root = os.path.dirname(os.path.abspath(__file__))
train_script_path = os.path.join(
    project_root, "llm_project", "models", "gpt", "train.py"
)
results_file_path = os.path.join(project_root, "experiments", "tuning_results.json")
saved_models_dir = os.path.join(project_root, "experiments", "saved_models")
bpe_cache_folder = os.path.join(
    project_root, "experiments", "bpe_results", "train_results"
)
plots_dir = os.path.join(project_root, "experiments", "plots")

os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
os.makedirs(saved_models_dir, exist_ok=True)
os.makedirs(bpe_cache_folder, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)


def find_existing_bpe_caches(cache_dir):
    trained_k = set()
    print("\n--- Checking for existing BPE caches ---")
    if not os.path.isdir(cache_dir):
        return trained_k

    pattern = re.compile(r"train_bpe_merges_k(\d+)\.pkl")
    for filename in os.listdir(cache_dir):
        match = pattern.match(filename)
        if match:
            k_value = int(match.group(1))
            trained_k.add(k_value)
            print(f"   Found existing BPE cache for k={k_value}")
    return trained_k


# The main function now accepts the list of experiments to run


def run_experiments_slice(args, experiments_to_run):
    all_results = []
    if os.path.exists(results_file_path):
        # Use a lock file to prevent race conditions when reading/writing the results
        # This is a simple approach; for more complex scenarios, a database would be better.
        try:
            with open(results_file_path, "r") as f:
                all_results = json.load(f)
            print(f"Loaded {len(all_results)} previous results.")
        except (json.JSONDecodeError, FileNotFoundError):
            all_results = []  # Start fresh if file is corrupt or gone

    completed_experiments = {
        res["experiment_name"]: res.get("parameters", {}) for res in all_results
    }

    trained_bpe_k_values = find_existing_bpe_caches(bpe_cache_folder)

    total_in_slice = len(experiments_to_run)
    for i, exp_config in enumerate(experiments_to_run):
        exp_name = exp_config["name"]
        params = exp_config["params"].copy()

        if args.override_k:
            params["max_k"] = args.override_k
        current_k = params.get("max_k", 200)

        if exp_name in completed_experiments:
            saved_params = completed_experiments[exp_name]
            if saved_params.get("max_k") == current_k:
                print(
                    f"\n--- Skipping completed experiment {i + 1}/{total_in_slice}: {exp_name} ---"
                )
                trained_bpe_k_values.add(current_k)
                continue

        print("\n" + "=" * 80)
        print(
            f"{Colors.OKBLUE}[STARTING]{Colors.ENDC} Experiment {i + 1}/{total_in_slice}: {Colors.BOLD}{exp_name}{Colors.ENDC}"
        )
        print(f"   Parameters: {params}")

        command = ["python", train_script_path]
        if current_k not in trained_bpe_k_values:
            print(f"   Note: Forcing BPE retrain for k={current_k}.")
            command.append("--force_retrain")
            trained_bpe_k_values.add(current_k)
        else:
            print(f"   Note: Using cached BPE for k={current_k}.")

        for param_name, param_value in params.items():
            command.append(f"--{param_name.lower()}")
            command.append(str(param_value))

        print("   Running command:", " ".join(command))

        try:
            result = subprocess.run(command, text=True, encoding="utf-8", check=False)

            if result.returncode == 0:
                print(
                    f"\n{Colors.OKGREEN}[SUCCESS]{Colors.ENDC} Experiment '{exp_name}' finished."
                )
                final_loss = "N/A (check plots)"

                # (Plot and model saving logic is unchanged)
                exp_plot_dir = os.path.join(plots_dir, exp_name)
                os.makedirs(exp_plot_dir, exist_ok=True)
                default_plot_dir = os.path.join(
                    project_root, "experiments", "plots", "gpt"
                )
                for plot_file in ["loss_curve.png", "val_perplexity.png"]:
                    src_path = os.path.join(default_plot_dir, plot_file)
                    dest_path = os.path.join(exp_plot_dir, plot_file)
                    if os.path.exists(src_path):
                        shutil.copy(src_path, dest_path)
                        print(f"   Saved plot to {dest_path}")

                generic_model_path = os.path.join(saved_models_dir, "gpt_final.pt")
                specific_model_path = os.path.join(
                    saved_models_dir, f"gpt_{exp_name}.pth"
                )
                if os.path.exists(generic_model_path):
                    os.rename(generic_model_path, specific_model_path)

                result_data = {
                    "experiment_name": exp_name,
                    "parameters": params,
                    "final_val_loss": final_loss,
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                print(
                    f"\n{Colors.FAIL}[ERROR]{Colors.ENDC} Experiment '{exp_name}' failed!"
                )

        except Exception as e:
            print(
                f"\n{Colors.FAIL}[CRITICAL ERROR]{Colors.ENDC} An exception occurred: {e}"
            )

        # Safely update the shared results file
        if os.path.exists(results_file_path):
            try:
                with open(results_file_path, "r") as f:
                    all_results = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                all_results = []

        all_results = [
            res for res in all_results if res.get("experiment_name") != exp_name
        ]
        if "result_data" in locals() and result.returncode == 0:
            all_results.append(result_data)

        with open(results_file_path, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"   Results saved to {results_file_path}")

    print("\n" + "=" * 80)
    print(
        f"{Colors.OKGREEN}{Colors.BOLD}[COMPLETE] All experiments in this slice have finished!{Colors.ENDC}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A slice runner for GPT experiments.")
    parser.add_argument(
        "--override_k", type=int, help="Override max_k for all experiments."
    )
    # --- New arguments to control the slice ---
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="The starting experiment index (1-based).",
    )
    parser.add_argument(
        "--end", type=int, default=None, help="The ending experiment index (inclusive)."
    )
    args = parser.parse_args()

    # --- Logic to slice the experiments list ---
    # Convert from 1-based user input to 0-based list index
    start_index = (args.start - 1) if args.start is not None else 0
    end_index = args.end if args.end is not None else len(experiments)

    # Ensure indices are within bounds
    start_index = max(0, start_index)
    end_index = min(len(experiments), end_index)

    experiments_to_run = experiments[start_index:end_index]

    print(
        f"--- Preparing to run {len(experiments_to_run)} experiments (from experiment {start_index + 1} to {end_index}) ---"
    )

    run_experiments_slice(args, experiments_to_run)

# How to Run Your Experiments in Parallel
#
# You have 12 experiments, so you can split them exactly in half.
#
# **In your first terminal(running `exp3.py`), run this command for the first half(experiments 1 to 6): **
# *(Note: You'll need to add the same `--start` and `--end` logic to `exp3.py` if you haven't already, or just stop it manually after the 6th run completes.)*
# ```bash
# python exp3.py - -end 6 - -override_k 200
# ```
#
# **In your second terminal(running the new script), run this command for the second half(experiments 7 to 12): **
# ```bash
# python run_slice.py - -start 7 - -override_k 200
