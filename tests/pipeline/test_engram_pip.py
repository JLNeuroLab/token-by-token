import pytest
import os
import subprocess
import re
import shutil
from pathlib import Path
import pickle

# --- Test Setup: Define paths and create a fixture for cleanup ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
MAIN_SCRIPT_PATH = PROJECT_ROOT / "main.py"
TEST_ARTIFACTS_DIR = PROJECT_ROOT / "experiments" / "test_artifacts" / "ngram"
MODEL_SAVE_DIR = TEST_ARTIFACTS_DIR / "experiments" / "saved_models" / "ngram"


@pytest.fixture(scope="module")
def ngram_pipeline_setup():
    """Manages a clean test directory for N-gram artifacts."""
    if TEST_ARTIFACTS_DIR.exists():
        shutil.rmtree(TEST_ARTIFACTS_DIR)
    TEST_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    yield
    shutil.rmtree(TEST_ARTIFACTS_DIR)
    print(f"\nCleaned up test directory: {TEST_ARTIFACTS_DIR}")


# --- The End-to-End Test ---


def test_ngram_full_pipeline(ngram_pipeline_setup):
    """
    Verifies the entire n-gram workflow: train, save, load, and generate.
    Checks for tokenization success and parameter persistence.
    """
    N_VAL = 3
    K_VAL = 50

    # --- STAGE 1: TRAIN ---
    print("\n--- N-gram Test: Training Stage ---")
    train_command = [
        "python",
        str(MAIN_SCRIPT_PATH),
        "train",
        "--model",
        "ngram",
        "--n",
        str(N_VAL),
        "--k",
        str(K_VAL),
        "--force_retrain",
    ]
    # We need to run this from the project root for paths to work correctly.
    train_result = subprocess.run(
        train_command, capture_output=True, text=True, cwd=PROJECT_ROOT
    )

    # --- STAGE 2: ASSERT TRAINING & TOKENIZATION ---
    assert train_result.returncode == 0, (
        f"N-gram training failed!\nSTDERR:\n{train_result.stderr}"
    )

    # Check that BPE ran and created a vocabulary
    bpe_log_match = re.search(
        r"BPE training complete. Vocabulary size: (\d+)", train_result.stdout
    )
    assert bpe_log_match, "BPE vocabulary size not found in training log."
    vocab_size = int(bpe_log_match.group(1))
    assert vocab_size > 0, "BPE vocabulary size is zero."
    print(f"Verified BPE ran, vocab size: {vocab_size}")

    # Check that model artifacts were saved
    model_path = MODEL_SAVE_DIR / f"ngram_model_n{N_VAL}_k{K_VAL}.pkl"
    merges_path = MODEL_SAVE_DIR / f"BPE_merges_k{K_VAL}.pkl"
    assert model_path.exists(), "N-gram model file was not saved."
    assert merges_path.exists(), "N-gram BPE merges file was not saved."
    print("Verified training artifacts were saved.")

    # --- STAGE 3: ASSERT PARAMETER PERSISTENCE ---
    with open(model_path, "rb") as f:
        loaded_data = pickle.load(f)

    assert "n" in loaded_data, "Parameter 'n' not found in saved model file."
    assert loaded_data["n"] == N_VAL, (
        f"Saved parameter 'n' is {loaded_data['n']}, expected {N_VAL}."
    )
    print(f"Verified parameter persistence: 'n' is correctly saved as {N_VAL}.")

    # --- STAGE 4: GENERATE ---
    print("\n--- N-gram Test: Generation Stage ---")
    generate_command = [
        "python",
        str(MAIN_SCRIPT_PATH),
        "generate",
        "--model",
        "ngram",
        "--n",
        str(N_VAL),
        "--k",
        str(K_VAL),
        "--prompt",
        "hello world",
    ]
    generate_result = subprocess.run(
        generate_command, capture_output=True, text=True, cwd=PROJECT_ROOT
    )

    # --- STAGE 5: ASSERT GENERATION & EVALUATION ---
    assert generate_result.returncode == 0, (
        f"N-gram generation failed!\nSTDERR:\n{generate_result.stderr}"
    )
    assert "=== Generated Text ===" in generate_result.stdout, (
        "Generated text is missing from output."
    )

    ppl_match = re.search(r"Validation perplexity:\s*([\d.]+)", generate_result.stdout)
    assert ppl_match, "Perplexity score not found in generation output."
    perplexity = float(ppl_match.group(1))
    assert perplexity > 0, f"Invalid perplexity score: {perplexity}"
    print(f"Verified generation and evaluation. Perplexity: {perplexity:.2f}")
