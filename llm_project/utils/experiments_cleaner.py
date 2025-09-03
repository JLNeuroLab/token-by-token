import os
import shutil

# --- Configuration ---
# This script will remove all generated files and caches to give you a clean start.
# It is designed to be safe and will NOT delete your source code or data.

# Get the absolute path of the directory where the script is located
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Directories to be completely removed
DIRS_TO_DELETE = [
    ".hypothesis",
    ".pytest_cache",
    "llm_project.egg-info",
    "experiments/logs",
    "experiments/plots",
    "experiments/results",
    "experiments/saved_models",
    "experiments/summary",
    "experiments/test_artifacts",
    "saved_models",  # Old top-level saved models directory
]

# Specific files to be deleted if they exist
FILES_TO_DELETE = [
    "experiments/tuning_results.json",
    "experiments/tuning_results0.json",
    # Add any other specific log or output files at the root if needed
]

# Directories where we should delete specific file types (e.g., all __pycache__)
CLEANUP_TARGETS = [
    "llm_project",
    "tests",
]


def remove_item(path):
    """Safely removes a file or directory."""
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"Removed directory: {path}")
        elif os.path.isfile(path):
            os.remove(path)
            print(f"Removed file:      {path}")
    except FileNotFoundError:
        print(f"Skipped (not found): {path}")
    except Exception as e:
        print(f"Error removing {path}: {e}")


def main():
    """Runs the cleanup process."""
    print("--- Starting Project Cleanup ---")

    # 1. Remove entire directories
    print("\n[1] Deleting full directories...")
    for dir_name in DIRS_TO_DELETE:
        full_path = os.path.join(PROJECT_ROOT, dir_name)
        remove_item(full_path)

    # 2. Remove specific files
    print("\n[2] Deleting specific files...")
    for file_name in FILES_TO_DELETE:
        full_path = os.path.join(PROJECT_ROOT, file_name)
        remove_item(full_path)

    # 3. Clean up __pycache__ directories
    print("\n[3] Cleaning up __pycache__ directories...")
    for target_dir in CLEANUP_TARGETS:
        for root, dirs, _ in os.walk(os.path.join(PROJECT_ROOT, target_dir)):
            if "__pycache__" in dirs:
                pycache_path = os.path.join(root, "__pycache__")
                remove_item(pycache_path)

    print("\n--- Cleanup Complete ---")


if __name__ == "__main__":
    main()

# # How to Use It
# 1. ** Save the file ** above as `clean_project.py` in the main `token-by-token-ale_branch` folder.
# 2. ** Open your terminal ** in that same folder.
# 3. ** Run the script ** with the following command:
#
#     ```bash
#     python clean_project.py
