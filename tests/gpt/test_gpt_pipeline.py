import subprocess
import re
import os

# Define the root of the project to find main.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MAIN_SCRIPT_PATH = os.path.join(PROJECT_ROOT, "main.py")
# Define a specific path for the test model to avoid conflicts
TEST_MODEL_DIR = os.path.join(PROJECT_ROOT, "experiments", "test_artifacts")
TEST_MODEL_PATH = os.path.join(TEST_MODEL_DIR, "gpt_test_model.pt")

# Create the directory for test artifacts if it doesn't exist
os.makedirs(TEST_MODEL_DIR, exist_ok=True)


def test_gpt_pipeline_evaluates_all_stages():
    """
    This is a full integration test that evaluates each major function:
    1.  TRAIN:      Runs a minimal GPT training session and saves the model.
    2.  GENERATE:   Uses the newly trained model to generate text.
    3.  PERPLEXITY: Parses the perplexity score from the generation output.
    """
    # In train.py, the model saves to a fixed path. We'll have to rename it.
    default_save_path = os.path.join(
        PROJECT_ROOT, "experiments", "saved_models", "gpt_final.pt"
    )

    try:
        # --- STAGE 1: TRAINING ---
        print("\n--- STAGE 1: Testing the 'train' command ---")
        train_command = [
            "python",
            MAIN_SCRIPT_PATH,
            "train",
            "--model",
            "gpt",
            "--force_retrain",
            "--max_iters",
            "15",
            "--embd_dim",
            "16",
            "--n_layer",
            "2",
            "--dropout",
            "0.1",
            "--max_k",
            "50",
        ]

        print(f"Running command: {' '.join(train_command)}")
        train_result = subprocess.run(
            train_command, capture_output=True, text=True, encoding="utf-8"
        )

        # Assertions for Training Stage
        assert train_result.returncode == 0, (
            f"Training script failed! Stderr:\n{train_result.stderr}"
        )
        assert "val loss" in train_result.stdout, (
            "Training output did not contain validation loss."
        )
        assert "Saving model to" in train_result.stdout, (
            "Training output did not confirm model saving."
        )

        # Move the trained model to our dedicated test path
        assert os.path.exists(default_save_path), (
            "The trained model was not found at the expected path."
        )
        os.rename(default_save_path, TEST_MODEL_PATH)
        print(f"--- STAGE 1 PASSED: Model trained and moved to {TEST_MODEL_PATH} ---")

        # --- STAGE 2 & 3: GENERATION and PERPLEXITY ---
        print("\n--- STAGE 2 & 3: Testing 'generate' command and Perplexity ---")
        generate_command = [
            "python",
            MAIN_SCRIPT_PATH,
            "generate",
            "--model",
            "gpt",
            "--model_path",
            TEST_MODEL_PATH,
            "--prompt",
            "O Romeo, Romeo!",
            "--max_new_tokens",
            "25",
        ]

        print(f"Running command: {' '.join(generate_command)}")
        generate_result = subprocess.run(
            generate_command, capture_output=True, text=True, encoding="utf-8"
        )

        # Assertions for Generation and Perplexity
        assert generate_result.returncode == 0, (
            f"Generation script failed! Stderr:\n{generate_result.stderr}"
        )

        # Check for generated text
        assert "=== Generated Text ===" in generate_result.stdout, (
            "Generation output is missing the generated text section."
        )

        # Check for and parse the perplexity score
        perplexity_match = re.search(
            r"Validation perplexity:\s*([\d.]+)", generate_result.stdout
        )
        assert perplexity_match is not None, (
            "Could not find perplexity score in the output."
        )

        perplexity_value = float(perplexity_match.group(1))
        assert perplexity_value > 0, (
            f"Perplexity score must be a positive number, but got {perplexity_value}."
        )

        print(f"Found Perplexity Score: {perplexity_value:.4f}")
        print("--- STAGE 2 & 3 PASSED: Text generated and perplexity is valid ---")

    finally:
        # --- Cleanup ---
        # This block will run even if the asserts fail, ensuring we clean up.
        print("\n--- Cleaning up test artifacts ---")
        if os.path.exists(TEST_MODEL_PATH):
            os.remove(TEST_MODEL_PATH)
            print(f"Removed test model: {TEST_MODEL_PATH}")
