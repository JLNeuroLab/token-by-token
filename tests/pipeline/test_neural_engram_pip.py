import pytest
import os
import subprocess
import re
import shutil
from pathlib import Path
import glob
import pickle

# --- Test Setup ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
MAIN_SCRIPT_PATH = PROJECT_ROOT / "main.py"
TEST_ARTIFACTS_DIR = PROJECT_ROOT / "experiments" / "test_artifacts" / "neural_ngram"
MODEL_SAVE_DIR = TEST_ARTIFACTS_DIR / "experiments" / "saved_models" / "neural_ngram"


@pytest.fixture(scope="module")
def neural_ngram_pipeline_setup():
    """Manages a clean test directory for Neural N-gram artifacts."""
    if TEST_ARTIFACTS_DIR.exists():
        shutil.rmtree(TEST_ARTIFACTS_DIR)
    TEST_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    yield
    shutil.rmtree(TEST_ARTIFACTS_DIR)
    print(f"\nCleaned up test directory: {TEST_ARTIFACTS_DIR}")


# --- The End-to-End Test ---


def test_neural_ngram_full_pipeline(neural_ngram_pipeline_setup):
    """
    Verifies the entire neural n-gram workflow, including checkpoint saving
    and loading, and parameter compatibility.
    """
    N_VAL = 3
    K_VAL = 50
    EMBED_DIM = 8

    # --- STAGE 1: TRAIN ---
    print("\n--- Neural N-gram Test: Training Stage ---")
    train_command = [
        "python",
        str(MAIN_SCRIPT_PATH),
        "train",
        "--model",
        "neural",
        "--n",
        str(N_VAL),
        "--k",
        str(K_VAL),
        "--embedding_dim",
        str(EMBED_DIM),
        "--epochs",
        "1",
        "--force_retrain",
    ]
    train_result = subprocess.run(
        train_command, capture_output=True, text=True, cwd=PROJECT_ROOT
    )

    # --- STAGE 2: ASSERT TRAINING & TOKENIZATION ---
    assert train_result.returncode == 0, (
        f"Neural n-gram training failed!\nSTDERR:\n{train_result.stderr}"
    )

    bpe_log_match = re.search(r"BPE ready with (\d+) tokens", train_result.stdout)
    assert bpe_log_match, "BPE token count not found in training log."
    vocab_size = int(bpe_log_match.group(1))
    assert vocab_size > 0, "BPE vocabulary size is zero."
    print(f"Verified BPE ran, vocab size: {vocab_size}")

    # Find the saved checkpoint file
    checkpoint_files = glob.glob(str(MODEL_SAVE_DIR / "val=*.pkl"))
    assert len(checkpoint_files) > 0, "Model checkpoint file was not saved."
    checkpoint_path = checkpoint_files[0]
    print(f"Verified training checkpoint was saved: {Path(checkpoint_path).name}")

    # --- STAGE 3: ASSERT PARAMETER PERSISTENCE ---
    with open(checkpoint_path, "rb") as f:
        loaded_data = pickle.load(f)

    assert "model_state" in loaded_data, "Checkpoint is missing 'model_state'."
    model_state = loaded_data["model_state"]
    assert "embeddings" in model_state, "Model state is missing 'embeddings'."

    embedding_shape = model_state["embeddings"].shape
    expected_shape = (vocab_size, EMBED_DIM)
    assert embedding_shape == expected_shape, (
        f"Saved embedding shape is {embedding_shape}, expected {expected_shape}."
    )
    print(f"Verified parameter persistence")
