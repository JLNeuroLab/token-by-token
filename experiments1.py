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


def run_all_experiments(args):
    all_results = []
    if os.path.exists(results_file_path):
        with open(results_file_path, "r") as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} previous results.")

    completed_experiments = {
        res["experiment_name"]: res.get("parameters", {}) for res in all_results
    }

    trained_bpe_k_values = find_existing_bpe_caches(bpe_cache_folder)

    for i, exp_config in enumerate(experiments):
        exp_name = exp_config["name"]
        params = exp_config["params"].copy()

        if args.override_k:
            params["max_k"] = args.override_k
        current_k = params.get("max_k", 200)

        if exp_name in completed_experiments:
            saved_params = completed_experiments[exp_name]
            if saved_params.get("max_k") == current_k:
                print(
                    f"\n--- Skipping completed experiment {i + 1}/{len(experiments)}: {exp_name} ---"
                )
                trained_bpe_k_values.add(current_k)
                continue

        print("\n" + "=" * 80)
        print(
            f"{Colors.OKBLUE}[STARTING]{Colors.ENDC} Experiment {i + 1}/{len(experiments)}: {Colors.BOLD}{exp_name}{Colors.ENDC}"
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
            # =================== START OF THE FIX ===================
            # Use subprocess.run to execute the script and let it print directly.
            # This will show progress bars correctly but we can't capture output.
            result = subprocess.run(
                command,
                text=True,
                encoding="utf-8",
                check=False,  # Don't raise an exception on failure
            )
            # ==================== END OF THE FIX ====================

            if result.returncode == 0:
                print(
                    f"\n{Colors.OKGREEN}[SUCCESS]{Colors.ENDC} Experiment '{exp_name}' finished."
                )

                # Since we can't capture output, we can't parse the loss.
                # We will record the success and move on.
                final_loss = "N/A (check plots)"

                # Save plots
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

                # Rename model
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
                all_results.append(result_data)
            else:
                print(
                    f"\n{Colors.FAIL}[ERROR]{Colors.ENDC} Experiment '{exp_name}' failed with return code {result.returncode}!"
                )
                print("   See output above for details.")

        except Exception as e:
            print(
                f"\n{Colors.FAIL}[CRITICAL ERROR]{Colors.ENDC} An exception occurred: {e}"
            )

        # Update results file
        # Remove any old entry for this experiment before adding the new one
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
        f"{Colors.OKGREEN}{Colors.BOLD}[COMPLETE] All experiments finished!{Colors.ENDC}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple runner for GPT experiments.")
    parser.add_argument(
        "--override_k", type=int, help="Override max_k for all experiments."
    )
    args = parser.parse_args()
    run_all_experiments(args)
