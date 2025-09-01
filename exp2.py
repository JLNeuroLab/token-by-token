import os
import subprocess
import json
from datetime import datetime
import argparse
from llm_project.utils.debugg_utils import Colors
import re

# --- 1. Load Experiments from JSON File ---
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

# --- 2. Setup Paths ---
project_root = os.path.dirname(os.path.abspath(__file__))
train_script_path = os.path.join(
    project_root, "llm_project", "models", "gpt", "train.py"
)
results_file_path = os.path.join(project_root, "experiments", "tuning_results.json")
saved_models_dir = os.path.join(project_root, "experiments", "saved_models")

os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
os.makedirs(saved_models_dir, exist_ok=True)

# --- 3. Main Experiment Loop ---


def run_all_experiments(args):
    all_results = []
    if os.path.exists(results_file_path):
        with open(results_file_path, "r") as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} previous results.")

    # Store parameters of completed experiments to check for mismatches
    completed_experiments = {
        res["experiment_name"]: res.get("parameters", {}) for res in all_results
    }

    trained_bpe_k_values = set()

    for i, exp_config in enumerate(experiments):
        exp_name = exp_config["name"]
        # Use a copy so the original experiment list is not modified
        params = exp_config["params"].copy()

        # --- This block now correctly compares parameters before skipping ---
        if args.override_k:
            params["max_k"] = args.override_k

        if exp_name in completed_experiments:
            saved_params = completed_experiments[exp_name]
            # Check if the 'max_k' values are the same before skipping
            if saved_params.get("max_k") == params.get("max_k"):
                print(
                    f"\n--- Skipping completed experiment {i + 1}/{len(experiments)}: {exp_name} (parameters match) ---"
                )
                # Add the k from the skipped experiment to our set of trained BPEs
                trained_bpe_k_values.add(params.get("max_k"))
                continue
            else:
                print(
                    f"\n--- Re-running experiment {i + 1}/{len(experiments)}: {exp_name} ('max_k' parameter mismatch) ---"
                )
        # --- End of parameter comparison logic ---

        print("\n" + "=" * 80)
        print(
            f"{Colors.OKBLUE}[STARTING]{Colors.ENDC} Experiment {i + 1}/{len(experiments)}: {Colors.BOLD}{exp_name}{Colors.ENDC}"
        )
        print(f"   Parameters: {params}")

        command = ["python", train_script_path]

        current_k = params.get("max_k", 200)
        if current_k not in trained_bpe_k_values:
            print(f"   Note: First time seeing k={current_k}. Forcing BPE retrain.")
            command.append("--force_retrain")
            trained_bpe_k_values.add(current_k)
        else:
            print(
                f"   Note: BPE for k={current_k} is already cached. Skipping retrain."
            )

        for param_name, param_value in params.items():
            command.append(f"--{param_name.lower()}")
            command.append(str(param_value))

        print("   Running command:", " ".join(command))

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )

            full_output = []
            for line in process.stdout:
                print(line, end="")
                full_output.append(line)

            process.wait()
            error_output = process.stderr.read()

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
                        f"\n{Colors.OKGREEN}[SUCCESS]{Colors.ENDC} Experiment '{exp_name}' finished."
                    )
                    print(
                        f"   Final Validation Loss: {Colors.BOLD}{final_loss:.4f}{Colors.ENDC}"
                    )

                    generic_model_path = os.path.join(saved_models_dir, "gpt_final.pt")
                    specific_model_path = os.path.join(
                        saved_models_dir, f"gpt_{exp_name}.pth"
                    )
                    if os.path.exists(generic_model_path):
                        os.rename(generic_model_path, specific_model_path)
                        model_filename = os.path.basename(specific_model_path)
                    else:
                        model_filename = "Not found"

                    result = {
                        "experiment_name": exp_name,
                        "parameters": params,
                        "final_val_loss": final_loss,
                        "model_file": model_filename,
                        "timestamp": datetime.now().isoformat(),
                    }
                    all_results.append(result)
                else:
                    print(
                        f"\n{Colors.WARNING}[WARNING]{Colors.ENDC} Could not parse final validation loss for '{exp_name}'."
                    )
            else:
                print(
                    f"\n{Colors.FAIL}[ERROR]{Colors.ENDC} Experiment '{exp_name}' failed!"
                )
                print(error_output)

        except Exception as e:
            print(
                f"\n{Colors.FAIL}[CRITICAL ERROR]{Colors.ENDC} An exception occurred while running '{exp_name}': {e}"
            )

        # Filter out old results for experiments we just re-ran
        all_results = [
            res
            for res in all_results
            if not (
                res["experiment_name"] == exp_name
                and res["parameters"].get("max_k") != params.get("max_k")
            )
        ]

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
