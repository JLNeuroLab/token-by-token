from llm_project.utils.file_manager import load_item

def load_shakespeare(version ="raw"):
    version_map = {
        "raw": ("data/raw", "Shakespeare_clean_full.txt"),
        "train": ("data/processed", "Shakespeare_clean_train.txt"),
        "validation": ("data/processed", "Shakespeare_clean_valid.txt"),
        "test": ("data/processed", "Shakespeare_clean_test.txt"),
    }
    if version not in version_map:
        raise ValueError(f"version name {version} not valid, choose amongst {list(version_map.keys())}")

    folder, fname = version_map[version]

    return load_item(folder, fname)